"""HSDGNN baseline adapted for multilevel OSM tile-access forecasting."""

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


BATCH_SIZE = 2


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


def _xavier_(t):
    if t is not None:
        nn.init.xavier_uniform_(t)


def _zeros_(t):
    if t is not None:
        nn.init.zeros_(t)


class HSDGNN_block(nn.Module):
    def __init__(self, num_nodes, input_dim, rnn_units, embed_dim):
        super(HSDGNN_block, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.gru1 = nn.GRU(embed_dim, rnn_units)
        self.gru2 = nn.GRU(rnn_units, rnn_units)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, rnn_units, rnn_units))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, rnn_units))
        self.diff = nn.Conv2d(
            rnn_units * 2, rnn_units, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )
        self.dropout = nn.Dropout(p=0.1)
        self.x_embedding = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(1, 16)),
                    ("sigmoid1", nn.Sigmoid()),
                    ("fc2", nn.Linear(16, 2)),
                    ("sigmoid2", nn.Sigmoid()),
                    ("fc3", nn.Linear(2, embed_dim)),
                ]
            )
        )
        self.fc1 = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(input_dim, 16)),
                    ("sigmoid1", nn.Sigmoid()),
                    ("fc2", nn.Linear(16, 2)),
                    ("sigmoid2", nn.Sigmoid()),
                    ("fc3", nn.Linear(2, embed_dim)),
                ]
            )
        )
        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(rnn_units, 16)),
                    ("sigmoid1", nn.Sigmoid()),
                    ("fc2", nn.Linear(16, 2)),
                    ("sigmoid2", nn.Sigmoid()),
                    ("fc3", nn.Linear(2, embed_dim)),
                ]
            )
        )
        _xavier_(self.weights_pool)
        _zeros_(self.bias_pool)

    def dynamic_dependency(self, output_of_gru, node_embeddings_all):
        # shape of output_of_gru: (T, B, N, rnn_units)
        # shape of node_embeddings_all:[(T, B, N, embed_dim),(N, embed_dim)]
        filter = self.fc2(output_of_gru)  # (T, B, N, embed_dim)
        nodevec = torch.tanh(torch.mul(node_embeddings_all[0], filter))  # (T, B, N, embed_dim)
        supports1 = torch.stack([torch.eye(self.num_nodes)] * output_of_gru.shape[0]).to(
            node_embeddings_all[0].device
        )  # (T, N, N)
        supports2 = F.relu(torch.matmul(nodevec, nodevec.permute(0, 1, 3, 2)))  # (T, B, N, N)
        x_g1 = torch.einsum("tnm,tbmc->tbnc", supports1, output_of_gru)  # (T, B, N, rnn_units)
        x_g2 = torch.einsum("tbnm,tbmc->tbnc", supports2, output_of_gru)
        x_g = x_g1 + x_g2
        weights = torch.einsum("nd,dcr->ncr", node_embeddings_all[1], self.weights_pool)
        bias = torch.matmul(node_embeddings_all[1], self.bias_pool)
        x_g = torch.einsum("tbnc,ncr->tbnr", x_g, weights) + bias  # T, B, N, rnn_units
        return x_g

    def forward(self, x, node_embeddings_all):
        # shape of x: (B, T, N, input_dim)
        # shape of node_embeddings_all:[(T, B, N, embed_dim),(N, embed_dim)]
        # shape of output: (B, T, N, rnn_units)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        B, T, N = x.shape[0], x.shape[1], x.shape[2]
        # GRU for each time series
        node_embed = self.x_embedding(x.unsqueeze(-1))  # (B, T, N, input_dim, embed_dim)
        supports_node1 = torch.eye(self.input_dim).to(node_embeddings_all[0].device)
        supports_node2 = F.relu(
            torch.matmul(node_embed, node_embed.permute(0, 1, 2, 4, 3))
        )  # (B, T, N, input_dim, input_dim)
        x1 = torch.einsum("ji,btni->btnj", supports_node1, x)
        x2 = torch.einsum("btnji,btni->btnj", supports_node2, x)  # (B, T, N, input_dim)
        input_for_fc1 = x1 + x2
        input_for_gru1 = self.fc1(input_for_fc1).permute(1, 0, 2, 3)
        h0 = torch.zeros(1, B * N, self.rnn_units).to(input_for_gru1.device)
        output_of_gru1, _ = self.gru1(input_for_gru1.reshape(T, B * N, self.embed_dim), h0)
        # GCN for each time step
        input_for_gru2 = []
        output_of_gru1 = output_of_gru1.reshape(T, B, N, self.rnn_units)
        input_for_gru2.append(output_of_gru1)
        diff1 = self.dynamic_dependency(output_of_gru1, node_embeddings_all)
        input_for_gru2.append(diff1)
        input_for_gru2 = self.diff(torch.cat(input_for_gru2, dim=3).permute(1, 3, 2, 0)).permute(
            3, 0, 2, 1
        )
        input_for_gru2 = self.dropout(input_for_gru2)
        # GRU2 for modeling the change of dependencies
        h0 = torch.zeros(1, B * N, self.rnn_units).to(input_for_gru2.device)
        output_of_gru2, _ = self.gru2(input_for_gru2.reshape(T, B * N, self.rnn_units), h0)
        return output_of_gru2.reshape(T, B, N, self.rnn_units).permute(1, 0, 2, 3)


class HSDGNN(nn.Module):
    def __init__(self, args):
        super(HSDGNN, self).__init__()
        self.batch_size = args.batch_size
        self.rnn_units = args.rnn_units
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.steps_per_day = args.steps_per_day
        self.node_embeddings = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim))
        self.T_i_D_emb = nn.Parameter(torch.empty(args.steps_per_day, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))
        self.encoder1 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)
        self.encoder2 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)
        self.end_conv1 = nn.Conv2d(
            1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True
        )
        self.end_conv1_b = nn.Conv2d(1, args.lag, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv2 = nn.Conv2d(
            1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True
        )
        self.dropout3 = nn.Dropout(p=0.1)
        self.encoder3 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)
        self.end_conv2_b = nn.Conv2d(1, args.lag, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv3 = nn.Conv2d(
            1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True
        )
        _xavier_(self.T_i_D_emb)
        _xavier_(self.D_i_W_emb)

    def forward(self, source):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # read input
        source1 = source[..., 0:-2]
        node_embedding = self.node_embeddings
        # add time information
        t_i_d_data = source[..., -2]
        T_i_D_emb = self.T_i_D_emb[
            (t_i_d_data * self.steps_per_day)
            .long()
            .clamp(0, self.steps_per_day - 1)
            .to(source.device)
        ]
        node_embedding = torch.mul(node_embedding, T_i_D_emb)
        d_i_w_data = source[..., -1]
        D_i_W_emb = self.D_i_W_emb[d_i_w_data.long().clamp(0, 6).to(source.device)]
        node_embedding = torch.mul(node_embedding, D_i_W_emb)
        node_embeddings_all = [node_embedding.permute(1, 0, 2, 3), self.node_embeddings]
        ################
        # Block 1
        output_1 = self.encoder1(source1, node_embeddings_all)
        output_1 = self.dropout1(output_1[:, -1:, :, :])
        # CNN based predictor
        output1 = self.end_conv1(output_1)
        # Residual
        source1_b = self.end_conv1_b(output_1)
        source2 = source1 - source1_b
        # Block 2
        output_2 = self.encoder2(source2, node_embeddings_all)
        output_2 = self.dropout2(output_2[:, -1:, :, :])
        # CNN based predictor
        output2 = self.end_conv2(output_2)
        # Residual
        source2_b = self.end_conv2_b(output_2)
        source3 = source2 - source2_b
        # Block 3
        output_3 = self.encoder3(source3, node_embeddings_all)
        output_3 = self.dropout3(output_3[:, -1:, :, :])
        # CNN based predictor
        output3 = self.end_conv3(output_3)
        return output1 + output2 + output3


def flatten_and_concat_layers(inputs_list):
    flat_list = []
    slices = []
    shapes = []
    start = 0
    for x in inputs_list:
        B, T, H, W, C = x.shape
        flat = x.view(B, T, H * W, C)
        flat_list.append(flat)
        end = start + H * W
        slices.append(slice(start, end))
        shapes.append((H, W))
        start = end
    return torch.cat(flat_list, dim=2), slices, shapes


def make_time_features(B, T, N, device):
    t_i_d = torch.zeros(B, T, N, dtype=torch.float32, device=device)
    d_i_w = torch.zeros(B, T, N, dtype=torch.long, device=device)
    return t_i_d, d_i_w


def build_model_input_from_list(inputs_list, device):
    x_flat, slices, shapes = flatten_and_concat_layers(inputs_list)
    B, T, N, _ = x_flat.shape
    t_i_d, d_i_w = make_time_features(B, T, N, device=device)
    values = x_flat[..., 0].to(device)
    source = torch.stack([values, t_i_d, d_i_w.float()], dim=-1)
    return source, slices, shapes


def unflatten_preds_to_layers(preds, slices, shapes):
    B, T_out, _, D = preds.shape
    out = []
    for slc, (H, W) in zip(slices, shapes):
        part = preds[:, :, slc, :].contiguous().view(B, T_out, H, W, D)
        out.append(part)
    return out


def stack_targets_list(targets_list):
    return torch.cat([t.view(t.shape[0], t.shape[1], -1, t.shape[-1]) for t in targets_list], dim=2)


def run_hsdgnn_city(city):
    set_seed(BASE_SEED)
    name = f"HSDGNN_Baseline_{city}"
    result_path = RESULT_ROOT / f"{city}_HSDGNN_Baseline_Seed{BASE_SEED}.csv"
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

    print(f"\n=== HSDGNN baseline | {city} ===")
    print(f"Layers={layers}, shapes={shapes}")
    print(f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    num_nodes = sum(h * w for h, w in shapes)
    args = SimpleNamespace(
        batch_size=BATCH_SIZE,
        rnn_units=32,
        num_nodes=num_nodes,
        input_dim=1,
        output_dim=1,
        lag=INPUT_STEPS,
        horizon=1,
        steps_per_day=1,
        embed_dim=8,
    )
    model = HSDGNN(args).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(0 if globals().get("EVALUATION_CHECKPOINT") else NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs_list_batch, targets_list_batch in train_loader:
            inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
            targets_one = [y[:, 0:1].to(DEVICE) for y in targets_list_batch]
            source, _, _ = build_model_input_from_list(inputs_list_batch, DEVICE)
            y = stack_targets_list(targets_one)
            optimizer.zero_grad(set_to_none=True)
            preds_raw = model(source)
            Bp, _, Np, _ = preds_raw.shape
            preds = preds_raw.view(Bp, args.horizon, args.output_dim, Np).permute(0, 1, 3, 2)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        train_loss_history.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_list_batch, targets_list_batch in val_loader:
                inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
                targets_one = [y[:, 0:1].to(DEVICE) for y in targets_list_batch]
                source, _, _ = build_model_input_from_list(inputs_list_batch, DEVICE)
                y = stack_targets_list(targets_one)
                preds_raw = model(source)
                Bp, _, Np, _ = preds_raw.shape
                preds = preds_raw.view(Bp, args.horizon, args.output_dim, Np).permute(0, 1, 3, 2)
                val_loss += criterion(preds, y).item()
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
                source, slices, flat_shapes = build_model_input_from_list(current_inputs, DEVICE)
                preds_raw = model(source)
                Bp, _, Np, _ = preds_raw.shape
                pred_flat = preds_raw.view(Bp, 1, args.output_dim, Np).permute(0, 1, 3, 2)
                pred_layers = unflatten_preds_to_layers(pred_flat, slices, flat_shapes)
                step_denorm = []
                for i in range(len(layers)):
                    pred_norm = pred_layers[i]
                    pred_denorm = inverse_normalize_data_levels([pred_norm], [params[i]])[0]
                    step_denorm.append(pred_denorm)
                batch_preds_denorm.append(step_denorm)
                if d < prediction_steps - 1:
                    current_inputs = [
                        torch.cat([current_inputs[i][:, 1:], pred_layers[i]], dim=1)
                        for i in range(len(layers))
                    ]

            for d in range(prediction_steps):
                true_flat_parts = []
                pred_flat_parts = []
                for i in range(len(layers)):
                    true_norm = targets_list_batch[i][:, d : d + 1]
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
