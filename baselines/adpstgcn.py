"""AdpSTGCN adapter requiring a separately obtained authorized model dependency."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.util import load_city

from scripts.metrics import (
    finite_mean,
    legacy_metric_all,
    legacy_metric_nz,
    legacy_metric_triplet,
    legacy_pearson,
)

import os


import re


import csv


import gc


import random


from collections import OrderedDict


from types import SimpleNamespace


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import Dataset, DataLoader


import torch.nn.functional as F


import matplotlib


import matplotlib.pyplot as plt


from scipy.sparse.linalg import eigs


DATA_ROOT = REPO_ROOT / "data"

RESULT_ROOT = REPO_ROOT / "runs/baselines/results"

MODEL_ROOT = REPO_ROOT / "runs/baselines/checkpoints"

PLOT_ROOT = REPO_ROOT / "runs/baselines/figures"

RUN_CITIES = ["Rome", "Milan", "Vienna", "SanFrancisco", "Seoul"]

TRAIN_SPLIT_RATIO = 0.6


VALIDATION_SPLIT_RATIO = 0.2


TEST_SPLIT_RATIO = 0.2


INPUT_STEPS = 28


OUTPUT_STEPS = 7


BATCH_SIZE = 1


GRAD_ACCUM_STEPS = 2


NUM_EPOCHS = 200


PATIENCE = 10


MIN_DELTA = 1e-4


BASE_SEED = 76


SKIP_COMPLETED = True


DEVICE = torch.device("cpu")  # Resolved explicitly by run_baseline.py.


def set_seed(seed=99):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixed at {seed}")


def load_city_layers(city, num_layers=4):
    if not 1 <= num_layers <= 4:
        raise ValueError("num_layers must be between 1 and 4")
    arrays, _ = load_city(DATA_ROOT, city)
    return [14, 15, 16, 17][:num_layers], arrays[:num_layers]


def normalize_data_levels(data_list, train_time_end, epsilon=1e-5):
    normalized = []
    params = []
    for data in data_list:
        train_slice = data[:train_time_end]
        mean = train_slice.mean()
        std = train_slice.std()
        if std < epsilon:
            std = torch.tensor(epsilon, dtype=data.dtype)
        normalized.append((data - mean) / std)
        params.append({"mean": mean, "std": std})
    return normalized, params


def inverse_normalize_data_levels(normalized_data_list, normalization_params_list):
    out = []
    for data, params in zip(normalized_data_list, normalization_params_list):
        out.append(data * params["std"].to(data.device) + params["mean"].to(data.device))
    return out


class MultiLayerSpatioTemporalDataset(Dataset):
    def __init__(self, data_list, input_steps, output_steps, time_start=0, time_end=None):
        self.data_list = data_list
        self.input_steps = input_steps
        self.output_steps = output_steps
        T = data_list[0].shape[0]
        if time_end is None:
            time_end = T
        max_start = time_end - (input_steps + output_steps)
        if max_start < time_start:
            raise ValueError(
                f"Time range [{time_start}, {time_end}) is too short for "
                f"input_steps={input_steps}, output_steps={output_steps}."
            )
        self.valid_indices = list(range(time_start, max_start + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.input_steps
        target_end = end + self.output_steps
        inputs = []
        targets = []
        for data in self.data_list:
            inputs.append(data[start:end])
            targets.append(data[end:target_end])
        return inputs, targets


def make_dataloaders(normalized_data_levels, total_timesteps):
    train_time_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    val_time_end = int(total_timesteps * (TRAIN_SPLIT_RATIO + VALIDATION_SPLIT_RATIO))

    train_dataset = MultiLayerSpatioTemporalDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, 0, train_time_end
    )
    val_dataset = MultiLayerSpatioTemporalDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, train_time_end, val_time_end
    )
    test_dataset = MultiLayerSpatioTemporalDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, val_time_end, total_timesteps
    )
    kwargs = dict(
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    return (
        DataLoader(train_dataset, **kwargs),
        DataLoader(val_dataset, **kwargs),
        DataLoader(test_dataset, **kwargs),
        train_time_end,
        val_time_end,
    )


def layers_list_to_model_input(layers, device=None):
    tensors = []
    B = T = C = None
    for idx, layer in enumerate(layers):
        x = layer if isinstance(layer, torch.Tensor) else torch.as_tensor(layer)
        if device is not None:
            x = x.to(device)
        if x.dim() != 5:
            raise ValueError(f"Layer {idx} must be (B,T,H,W,C), got {tuple(x.shape)}")
        b, t, h, w, c = x.shape
        if idx == 0:
            B, T, C = b, t, c
        elif (b, t, c) != (B, T, C):
            raise ValueError("Layer batch/time/channel sizes are inconsistent.")
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, h * w, C, T)
        tensors.append(x)
    return torch.cat(tensors, dim=1)


def model_input_to_layers_list(x, shapes):
    if x.dim() != 3:
        raise ValueError(f"x must be (B,N,C), got {tuple(x.shape)}")
    B, N, C = x.shape
    expected = sum(h * w for h, w in shapes)
    if N != expected:
        raise ValueError(f"N={N} does not match shapes total={expected}")
    out = []
    ptr = 0
    for h, w in shapes:
        n = h * w
        out.append(x[:, ptr : ptr + n, :].view(B, h, w, C).contiguous())
        ptr += n
    return out


def shapes_to_adjacency(shapes, self_loop=True, normalize=True):
    idx_maps = []
    offsets = []
    idx = 0
    for h, w in shapes:
        offsets.append(idx)
        idx_maps.append((np.arange(h * w).reshape(h, w) + idx).astype(np.int32))
        idx += h * w
    A = np.zeros((idx, idx), dtype=np.float32)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for layer_idx, (h, w) in enumerate(shapes):
        imap = idx_maps[layer_idx]
        for r in range(h):
            for c in range(w):
                u = int(imap[r, c])
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        A[u, int(imap[rr, cc])] = 1.0
    if self_loop:
        np.fill_diagonal(A, 1.0)
    if normalize:
        deg = A.sum(axis=1)
        deg[deg == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        A = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    return A


metric_triplet = legacy_metric_triplet


def result_is_complete(path):
    if not Path(path).exists():
        return False
    try:
        df = pd.read_csv(path)
        required = {"Layer", "Day", "MAE", "MAPE_floor1", "RMSE"}
        return (
            len(df) == 40
            and required.issubset(df.columns)
            and np.isfinite(df[["MAE", "MAPE_floor1", "RMSE"]].to_numpy(dtype=float)).all()
        )
    except Exception:
        return False


def save_results_csv(path, layers, prediction_steps, layer_metrics, overall_metrics):
    rows = []
    for i, layer in enumerate(layers):
        layer_mae, layer_mape, layer_rmse = [], [], []
        for d in range(prediction_steps):
            mae = float(np.mean(layer_metrics[i][d]["MAE"]))
            mape = float(np.mean(layer_metrics[i][d]["MAPE_floor1"]))
            rmse = float(np.mean(layer_metrics[i][d]["RMSE"]))
            rows.append(
                {
                    "Layer": layer,
                    "Day": f"Day {d + 1}",
                    "MAE": mae,
                    "MAPE_floor1": mape,
                    "RMSE": rmse,
                }
            )
            layer_mae.append(mae)
            layer_mape.append(mape)
            layer_rmse.append(rmse)
        rows.append(
            {
                "Layer": layer,
                "Day": "AVERAGE",
                "MAE": float(np.mean(layer_mae)),
                "MAPE_floor1": float(np.mean(layer_mape)),
                "RMSE": float(np.mean(layer_rmse)),
            }
        )
    overall_mae, overall_mape, overall_rmse = [], [], []
    for d in range(prediction_steps):
        mae = float(np.mean(overall_metrics[d]["MAE"]))
        mape = float(np.mean(overall_metrics[d]["MAPE_floor1"]))
        rmse = float(np.mean(overall_metrics[d]["RMSE"]))
        rows.append(
            {
                "Layer": "OVERALL",
                "Day": f"Day {d + 1}",
                "MAE": mae,
                "MAPE_floor1": mape,
                "RMSE": rmse,
            }
        )
        overall_mae.append(mae)
        overall_mape.append(mape)
        overall_rmse.append(rmse)
    rows.append(
        {
            "Layer": "OVERALL",
            "Day": "AVERAGE",
            "MAE": float(np.mean(overall_mae)),
            "MAPE_floor1": float(np.mean(overall_mape)),
            "RMSE": float(np.mean(overall_rmse)),
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    print(f"Saved result CSV: {path}")


def plot_loss(train_loss_history, val_loss_history, path, title):
    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train")
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


import sys


from baselines.vendor.adpstgcn.ada_STGCN import (
    Graph_Generator as AdpGraphGenerator,
    Embedding as AdpEmbedding,
    sample as AdpSample,
    gcn as AdpGCN,
)


class AdpSTGCNBaseline(nn.Module):
    """AdpSTGCN core adapted only where necessary for a fair OSM baseline.

    The uploaded source hard-codes the dynamic graph feature length as 12.
    This wrapper keeps the original components but makes that length match
    INPUT_STEPS, so AdpSTGCN uses the same 28-day history as other baselines.
    """

    def __init__(self, configs, supports):
        super().__init__()
        num_nodes = configs["n_nodes"]
        dropout = configs["dropout"]
        blocks = configs["n_blocks"]
        layers = configs["n_layers"]
        in_dim = configs["in_dim"]
        out_dim = configs["out_dim"]
        residual_channels = configs["d_model"]
        dilation_channels = configs["d_model"]
        skip_channels = configs["skip_channels"]
        end_channels = configs["end_channels"]
        kernel_size = configs["kernel_size"]
        seq_len = configs["seq_len"]

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.ndynadj = configs.get("num_dynamic_adj", 4)
        self.supports = supports
        self.supports_len = 0 if supports is None else len(supports)

        self.graph_generator = AdpGraphGenerator(
            "attention", seq_len, configs.get("graph_hidden", 32), self.ndynadj, num_nodes, dropout
        )
        self.embedding = AdpEmbedding(num_nodes, seq_len)
        self.sample = AdpSample()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.dyn = nn.ModuleList()

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(
                    AdpGCN(
                        dilation_channels, residual_channels, dropout, support_len=self.supports_len
                    )
                )
                self.dyn.append(
                    AdpGCN(
                        dilation_channels,
                        residual_channels,
                        dropout,
                        support_len=self.ndynadj,
                        dyn=True,
                    )
                )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1)
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1)
        )
        self.receptive_field = receptive_field

    def _build_static_support(self):
        support = []
        for a in self.supports:
            support.append(a.unsqueeze(0))
            support.append(torch.mm(a, a).unsqueeze(0))
        return torch.cat(support, dim=0)

    def forward(self, input_series):
        # input_series: (B, T, N), matching the original AdpSTGCN convention.
        input_tensor = input_series.transpose(1, 2).unsqueeze(1)
        static_support = self._build_static_support()

        graph_signal = self.embedding(input_tensor)
        dyn_adj = self.graph_generator(graph_signal)
        dyn = self.sample(dyn_adj)

        in_len = input_tensor.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input_tensor, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input_tensor
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[i](residual))
            gate_out = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_out * gate_out

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except Exception:
                skip = 0
            skip = s + skip

            x_dyn = self.dyn[i](x, dyn)
            x_static = self.gconv[i](x_dyn, static_support)
            x = x_static + residual[:, :, :, -x_static.size(3) :]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # Use the last temporal position for one-step forecasting.
        return x[:, 0, :, -1]


def layers_list_to_adpstgcn_series(layers, device=None):
    parts = []
    B = T = None
    for idx, layer in enumerate(layers):
        x = layer if isinstance(layer, torch.Tensor) else torch.as_tensor(layer)
        if device is not None:
            x = x.to(device)
        if x.dim() != 5:
            raise ValueError(f"Layer {idx} must be (B,T,H,W,C), got {tuple(x.shape)}")
        b, t, h, w, c = x.shape
        if c != 1:
            raise ValueError(f"AdpSTGCN baseline expects one value channel, got C={c}")
        if idx == 0:
            B, T = b, t
        elif (b, t) != (B, T):
            raise ValueError("Layer batch/time sizes are inconsistent.")
        parts.append(x[..., 0].contiguous().view(B, T, h * w))
    return torch.cat(parts, dim=2)


def adpstgcn_output_to_layers(preds, shapes):
    if preds.dim() == 2:
        preds = preds.unsqueeze(-1)
    if preds.dim() != 3:
        raise ValueError(f"preds must be (B,N) or (B,N,1), got {tuple(preds.shape)}")
    B, N, C = preds.shape
    expected = sum(h * w for h, w in shapes)
    if N != expected:
        raise ValueError(f"N={N} does not match shapes total={expected}")
    out = []
    ptr = 0
    for h, w in shapes:
        n = h * w
        out.append(preds[:, ptr : ptr + n, :].view(B, h, w, C).contiguous())
        ptr += n
    return out


def build_adpstgcn_supports(shapes):
    adj = shapes_to_adjacency(shapes, self_loop=True, normalize=True)
    support_fwd = torch.tensor(adj, dtype=torch.float32, device=DEVICE)
    support_bwd = torch.tensor(adj.T.copy(), dtype=torch.float32, device=DEVICE)
    return [support_fwd, support_bwd]


def run_adpstgcn_city(city):
    set_seed(BASE_SEED)
    name = f"AdpSTGCN_Baseline_{city}_Seed{BASE_SEED}"
    result_path = RESULT_ROOT / f"{city}_AdpSTGCN_Baseline_Seed{BASE_SEED}.csv"
    model_path = MODEL_ROOT / f"best_model_{name}_Seed{BASE_SEED}.pth"
    if globals().get("EVALUATION_CHECKPOINT") is not None:
        model_path = Path(globals()["EVALUATION_CHECKPOINT"])
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
    plot_path = PLOT_ROOT / f"{name}_loss_Seed{BASE_SEED}.png"
    if SKIP_COMPLETED and result_is_complete(result_path):
        print(f"[SKIP] Completed result exists: {result_path}")
        return

    layers, data_list = load_city_layers(city)
    total_timesteps = data_list[0].shape[0]
    shapes = [(int(x.shape[1]), int(x.shape[2])) for x in data_list]
    train_time_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    normalized, params = normalize_data_levels(data_list, train_time_end)
    train_loader, val_loader, test_loader, _, _ = make_dataloaders(normalized, total_timesteps)

    print(f"\n=== AdpSTGCN baseline | {city} ===")
    print(f"Layers={layers}, shapes={shapes}")
    print(f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    num_nodes = sum(h * w for h, w in shapes)
    supports = build_adpstgcn_supports(shapes)
    configs = {
        "device": DEVICE,
        "n_nodes": num_nodes,
        "dropout": 0.3,
        "n_blocks": 4,
        "n_layers": 2,
        "in_dim": 1,
        "out_dim": 1,
        "d_model": 32,
        "skip_channels": 128,
        "end_channels": 256,
        "kernel_size": 2,
        "seq_len": INPUT_STEPS,
        "graph_hidden": 32,
        "num_dynamic_adj": 4,
    }
    model = AdpSTGCNBaseline(configs, supports).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(0 if globals().get("EVALUATION_CHECKPOINT") else NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs_list_batch, targets_list_batch) in enumerate(train_loader):
            inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
            targets_one = [y[:, 0].to(DEVICE) for y in targets_list_batch]
            inputs_series = layers_list_to_adpstgcn_series(inputs_list_batch, DEVICE)
            target_series = layers_list_to_adpstgcn_series(
                [y.unsqueeze(1) for y in targets_one], DEVICE
            ).squeeze(1)

            preds = model(inputs_series)
            loss_raw = criterion(preds, target_series)
            loss = loss_raw / GRAD_ACCUM_STEPS
            loss.backward()
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss_raw.item()
        avg_train = train_loss / len(train_loader)
        train_loss_history.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_list_batch, targets_list_batch in val_loader:
                inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
                targets_one = [y[:, 0].to(DEVICE) for y in targets_list_batch]
                inputs_series = layers_list_to_adpstgcn_series(inputs_list_batch, DEVICE)
                target_series = layers_list_to_adpstgcn_series(
                    [y.unsqueeze(1) for y in targets_one], DEVICE
                ).squeeze(1)
                preds = model(inputs_series)
                val_loss += criterion(preds, target_series).item()
        avg_val = val_loss / len(val_loader)
        val_loss_history.append(avg_val)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] train={avg_train:.6f} val={avg_val:.6f}")

        if avg_val < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if not globals().get("EVALUATION_CHECKPOINT"):
        plot_loss(train_loss_history, val_loss_history, plot_path, f"{name} loss")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    prediction_steps = OUTPUT_STEPS
    layer_metrics = {
        i: {d: {"MAE": [], "MAPE_floor1": [], "RMSE": []} for d in range(prediction_steps)}
        for i in range(len(layers))
    }
    overall_metrics = {
        d: {"MAE": [], "MAPE_floor1": [], "RMSE": []} for d in range(prediction_steps)
    }

    with torch.no_grad():
        for inputs_list_batch, targets_list_batch in test_loader:
            current_inputs = [x.to(DEVICE) for x in inputs_list_batch]
            targets_list_batch = [y.to(DEVICE) for y in targets_list_batch]
            batch_preds_denorm = []
            for d in range(prediction_steps):
                inputs_series = layers_list_to_adpstgcn_series(current_inputs, DEVICE)
                preds = model(inputs_series)
                pred_layers = adpstgcn_output_to_layers(preds, shapes)
                step_denorm = []
                for i in range(len(layers)):
                    pred_denorm = inverse_normalize_data_levels([pred_layers[i]], [params[i]])[0]
                    step_denorm.append(pred_denorm)
                batch_preds_denorm.append(step_denorm)
                if d < prediction_steps - 1:
                    current_inputs = [
                        torch.cat([current_inputs[i][:, 1:], pred_layers[i].unsqueeze(1)], dim=1)
                        for i in range(len(layers))
                    ]

            for d in range(prediction_steps):
                true_flat_parts = []
                pred_flat_parts = []
                for i in range(len(layers)):
                    true_norm = targets_list_batch[i][:, d]
                    true_denorm = inverse_normalize_data_levels([true_norm], [params[i]])[0]
                    pred_denorm = batch_preds_denorm[d][i]
                    true_np = true_denorm.cpu().numpy()
                    pred_np = pred_denorm.cpu().numpy()
                    for k in range(true_np.shape[0]):
                        mae, mape, rmse = metric_triplet(true_np[k], pred_np[k])
                        layer_metrics[i][d]["MAE"].append(mae)
                        layer_metrics[i][d]["MAPE_floor1"].append(mape)
                        layer_metrics[i][d]["RMSE"].append(rmse)
                    true_flat_parts.append(true_np.reshape(true_np.shape[0], -1))
                    pred_flat_parts.append(pred_np.reshape(pred_np.shape[0], -1))
                true_concat = np.concatenate(true_flat_parts, axis=1)
                pred_concat = np.concatenate(pred_flat_parts, axis=1)
                for k in range(true_concat.shape[0]):
                    mae, mape, rmse = metric_triplet(true_concat[k], pred_concat[k])
                    overall_metrics[d]["MAE"].append(mae)
                    overall_metrics[d]["MAPE_floor1"].append(mape)
                    overall_metrics[d]["RMSE"].append(rmse)

    save_results_csv(result_path, layers, prediction_steps, layer_metrics, overall_metrics)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
