"""Graph WaveNet baseline adapted for multilevel OSM tile-access forecasting."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.util import load_city, temporal_features

from scripts.metrics import (
    finite_mean,
    legacy_metric_all,
    legacy_metric_nz,
    legacy_metric_triplet,
    legacy_pearson,
)

import os


import re


import gc


import random


from typing import List, Tuple, Union


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import Dataset, DataLoader


import matplotlib


import matplotlib.pyplot as plt


DATA_ROOT = REPO_ROOT / "data"

RESULT_ROOT = REPO_ROOT / "runs/baselines/results"

MODEL_ROOT = REPO_ROOT / "runs/baselines/checkpoints"

PLOT_ROOT = REPO_ROOT / "runs/baselines/figures"

MODEL_NAME = "GraphWaveNet"


RUN_CITIES = ["Rome", "Milan", "Vienna", "SanFrancisco", "Seoul"]

TRAIN_SPLIT_RATIO = 0.6


VALIDATION_SPLIT_RATIO = 0.2


INPUT_STEPS = 28


OUTPUT_STEPS = 7


BATCH_SIZE = 16


NUM_EPOCHS = 300


PATIENCE = 10


MIN_DELTA = 1e-4


BASE_SEED = 42


SKIP_COMPLETED = True


LEARNING_RATE = 1e-4


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


def _layer_sort_key(path):
    nums = re.findall(r"\d+", path.name)
    if not nums:
        raise ValueError(f"Layer file has no numeric layer id: {path}")
    return int(nums[0])


def load_city_layers(city, num_layers=4):
    if not 1 <= num_layers <= 4:
        raise ValueError("num_layers must be between 1 and 4")
    arrays, _ = load_city(DATA_ROOT, city)
    return [14, 15, 16, 17][:num_layers], arrays[:num_layers]


def normalize_data_levels(data_list, train_time_end, epsilon=1e-5):
    normalized, params = [], []
    for data in data_list:
        train_slice = data[:train_time_end]
        if train_slice.numel() == 0:
            raise ValueError("Empty training slice while normalizing.")
        mean = train_slice.mean()
        std = train_slice.std()
        if std < epsilon:
            std = torch.tensor(epsilon, dtype=data.dtype, device=data.device)
        normalized.append((data - mean) / std)
        params.append({"mean": mean, "std": std})
    return normalized, params


def inverse_normalize_data_levels(normalized_data_list, normalization_params_list):
    out = []
    for data, params in zip(normalized_data_list, normalization_params_list):
        mean = params["mean"].to(data.device)
        std = params["std"].to(data.device)
        out.append(data * std + mean)
    return out


class MultiLayerSpatioTemporalDataset(Dataset):
    def __init__(
        self,
        data_list,
        input_steps,
        output_steps,
        time_start=0,
        time_end=None,
        target_data_list=None,
    ):
        self.data_list = data_list
        self.target_data_list = data_list if target_data_list is None else target_data_list
        self.input_steps = input_steps
        self.output_steps = output_steps
        T = data_list[0].shape[0]
        if time_end is None:
            time_end = T
        max_start = time_end - (input_steps + output_steps)
        if max_start < time_start:
            raise ValueError(f"Time range [{time_start}, {time_end}) is too short.")
        self.valid_indices = list(range(time_start, max_start + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.input_steps
        target_end = end + self.output_steps
        inputs = [data[start:end] for data in self.data_list]
        targets = [data[end:target_end] for data in self.target_data_list]
        return inputs, targets


def make_dataloaders(normalized_data_levels, total_timesteps):
    feature_levels = temporal_features(
        normalized_data_levels, globals().get("INPUT_FEATURE_MODE", "raw")
    )
    train_time_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    val_time_end = int(total_timesteps * (TRAIN_SPLIT_RATIO + VALIDATION_SPLIT_RATIO))
    train_dataset = MultiLayerSpatioTemporalDataset(
        feature_levels,
        INPUT_STEPS,
        OUTPUT_STEPS,
        0,
        train_time_end,
        target_data_list=normalized_data_levels,
    )
    val_dataset = MultiLayerSpatioTemporalDataset(
        feature_levels,
        INPUT_STEPS,
        OUTPUT_STEPS,
        train_time_end,
        val_time_end,
        target_data_list=normalized_data_levels,
    )
    test_dataset = MultiLayerSpatioTemporalDataset(
        feature_levels,
        INPUT_STEPS,
        OUTPUT_STEPS,
        val_time_end,
        total_timesteps,
        target_data_list=normalized_data_levels,
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


def layers_list_to_model_input(
    layers: List[Union[np.ndarray, torch.Tensor]], device=None
) -> torch.Tensor:
    tensors = []
    B = T = C = None
    for idx, layer in enumerate(layers):
        x = torch.as_tensor(layer) if not isinstance(layer, torch.Tensor) else layer
        if device is not None:
            x = x.to(device)
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        if x.dim() != 5:
            raise ValueError(f"Layer {idx} should be (B,T,H,W,C), got {tuple(x.shape)}")
        b, t, h, w, c = x.shape
        if idx == 0:
            B, T, C = b, t, c
        elif (b, t, c) != (B, T, C):
            raise ValueError(f"Layer {idx} B/T/C mismatch: {(b, t, c)} vs {(B, T, C)}")
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(B, C, h * w, T)
        tensors.append(x)
    return torch.cat(tensors, dim=2)


def model_output_to_layers_list(x: Union[np.ndarray, torch.Tensor], shapes: List[Tuple[int, int]]):
    xt = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    if xt.dim() != 4:
        raise ValueError(f"GraphWaveNet output should be (B,OUT,N,T), got {tuple(xt.shape)}")
    B, C, N, T = xt.shape
    expected = sum(h * w for h, w in shapes)
    if N != expected:
        raise ValueError(f"Output node count {N} does not match shapes total {expected}")
    layers = []
    ptr = 0
    for h, w in shapes:
        n = h * w
        layer = xt[:, :, ptr : ptr + n, :].contiguous().view(B, C, h, w, T)
        layer = layer.permute(0, 4, 2, 3, 1).contiguous()
        layers.append(layer)
        ptr += n
    return layers


def shapes_to_adjacency(shapes: List[Tuple[int, int]], self_loop=True, normalize=True, device=None):
    idx_maps = []
    idx = 0
    for h, w in shapes:
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

    A_tensor = torch.as_tensor(A, dtype=torch.float32)
    if device is not None:
        A_tensor = A_tensor.to(device)
    return [A_tensor]


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        skip_mode="accumulate",
    ):
        super(gwnet, self).__init__()
        if skip_mode not in {"accumulate", "replace"}:
            raise ValueError("Unknown GWN skip implementation")
        self.skip_mode = skip_mode
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10).to(device), requires_grad=True
                ).to(device)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes).to(device), requires_grad=True
                ).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
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

                # 1x1 convolution for residual connection
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )

                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len,
                        )
                    )

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True
        )

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except:
                skip = 0
            skip = s if self.skip_mode == "replace" else s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


pearson_corr_np = legacy_pearson


metric_all = legacy_metric_all


metric_nz = legacy_metric_nz


def empty_metric_store(num_layers, prediction_steps):
    keys = ["MAE_ALL", "COR_ALL", "RMSE_ALL", "MAE_NZ", "COR_NZ", "RMSE_NZ", "Zero_FP"]
    return {
        "layer": {
            i: {d: {k: [] for k in keys} for d in range(prediction_steps)}
            for i in range(num_layers)
        },
        "overall": {d: {k: [] for k in keys} for d in range(prediction_steps)},
    }


def append_metrics(bucket, true_np, pred_np):
    mae_a, cor_a, rmse_a, zero_fp = metric_all(true_np, pred_np)
    mae_nz, cor_nz, rmse_nz = metric_nz(true_np, pred_np)
    bucket["MAE_ALL"].append(mae_a)
    bucket["COR_ALL"].append(cor_a)
    bucket["RMSE_ALL"].append(rmse_a)
    bucket["MAE_NZ"].append(mae_nz)
    bucket["COR_NZ"].append(cor_nz)
    bucket["RMSE_NZ"].append(rmse_nz)
    bucket["Zero_FP"].append(zero_fp)


def save_results_csv(path, layers, prediction_steps, metric_store):
    metric_cols = ["MAE_ALL", "COR_ALL", "RMSE_ALL", "MAE_NZ", "COR_NZ", "RMSE_NZ", "Zero_FP"]
    rows = []
    for i, layer in enumerate(layers):
        day_values = {k: [] for k in metric_cols}
        for d in range(prediction_steps):
            vals = {k: finite_mean(metric_store["layer"][i][d][k]) for k in metric_cols}
            rows.append({"Layer": layer, "Day": f"Day {d + 1}", **vals})
            for k in metric_cols:
                day_values[k].append(vals[k])
        rows.append(
            {
                "Layer": layer,
                "Day": "AVERAGE",
                **{k: finite_mean(day_values[k]) for k in metric_cols},
            }
        )
    overall_values = {k: [] for k in metric_cols}
    for d in range(prediction_steps):
        vals = {k: finite_mean(metric_store["overall"][d][k]) for k in metric_cols}
        rows.append({"Layer": "OVERALL", "Day": f"Day {d + 1}", **vals})
        for k in metric_cols:
            overall_values[k].append(vals[k])
    rows.append(
        {
            "Layer": "OVERALL",
            "Day": "AVERAGE",
            **{k: finite_mean(overall_values[k]) for k in metric_cols},
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    if len(df) != 40:
        raise RuntimeError(f"Unexpected result row count: {len(df)}")
    print(f"Saved result CSV: {path}")


def result_is_complete(path):
    if not Path(path).exists():
        return False
    try:
        df = pd.read_csv(path)
        required = {
            "Layer",
            "Day",
            "MAE_ALL",
            "COR_ALL",
            "RMSE_ALL",
            "MAE_NZ",
            "COR_NZ",
            "RMSE_NZ",
            "Zero_FP",
        }
        return len(df) == 40 and required.issubset(df.columns)
    except Exception:
        return False


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


def build_graphwavenet(num_nodes, supports):
    return gwnet(
        device=DEVICE,
        num_nodes=num_nodes,
        dropout=0.3,
        supports=supports,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=4 if globals().get("INPUT_FEATURE_MODE", "raw") == "decomposed" else 1,
        out_dim=OUTPUT_STEPS,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=128,
        end_channels=64,
        kernel_size=2,
        blocks=2,
        layers=globals().get("MODEL_LAYERS", 4),
        skip_mode=globals().get("MODEL_SKIP_MODE", "accumulate"),
    ).to(DEVICE)


def evaluate_graphwavenet(model, test_loader, layers, shapes, normalization_params):
    metric_store = empty_metric_store(len(layers), OUTPUT_STEPS)
    model.eval()
    with torch.no_grad():
        for inputs_list_batch, targets_list_batch in test_loader:
            inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
            targets_list_batch = [y.to(DEVICE) for y in targets_list_batch]
            inputs_gwn = layers_list_to_model_input(inputs_list_batch, DEVICE)
            preds_raw = model(inputs_gwn)
            pred_layers = model_output_to_layers_list(preds_raw, shapes)

            for d in range(OUTPUT_STEPS):
                batch_true_all_layers = []
                batch_pred_all_layers = []

                for i in range(len(layers)):
                    pred_day_norm = pred_layers[i][:, 0, :, :, d]
                    target_day_norm = targets_list_batch[i][:, d, :, :, 0]
                    pred_day_denorm = (
                        inverse_normalize_data_levels([pred_day_norm], [normalization_params[i]])[0]
                        .cpu()
                        .numpy()
                    )
                    target_day_denorm = (
                        inverse_normalize_data_levels([target_day_norm], [normalization_params[i]])[
                            0
                        ]
                        .cpu()
                        .numpy()
                    )

                    if pred_day_denorm.ndim == 2:
                        pred_day_denorm = pred_day_denorm[np.newaxis]
                        target_day_denorm = target_day_denorm[np.newaxis]

                    for k in range(target_day_denorm.shape[0]):
                        append_metrics(
                            metric_store["layer"][i][d], target_day_denorm[k], pred_day_denorm[k]
                        )

                    batch_true_all_layers.append(target_day_denorm)
                    batch_pred_all_layers.append(pred_day_denorm)

                true_concat = np.concatenate(
                    [x.reshape(x.shape[0], -1) for x in batch_true_all_layers], axis=1
                )
                pred_concat = np.concatenate(
                    [x.reshape(x.shape[0], -1) for x in batch_pred_all_layers], axis=1
                )
                for k in range(true_concat.shape[0]):
                    append_metrics(metric_store["overall"][d], true_concat[k], pred_concat[k])
    return metric_store


def run_graphwavenet_city(city):
    set_seed(BASE_SEED)
    name = f"{MODEL_NAME}_{city}"
    result_path = RESULT_ROOT / f"{city}_{MODEL_NAME}_Seed{BASE_SEED}.csv"
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
    total_timesteps = int(data_list[0].shape[0])
    shapes = [(int(x.shape[1]), int(x.shape[2])) for x in data_list]
    train_time_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    normalized, normalization_params = normalize_data_levels(data_list, train_time_end)
    train_loader, val_loader, test_loader, _, _ = make_dataloaders(normalized, total_timesteps)

    print(f"\n=== {MODEL_NAME} | {city} | Seed{BASE_SEED} ===")
    print(f"Layers={layers}, shapes={shapes}, timesteps={total_timesteps}")
    print(f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
    print(f"Result path: {result_path}")
    print(f"Best model path: {model_path}")

    num_nodes = sum(h * w for h, w in shapes)
    supports = shapes_to_adjacency(shapes, device=DEVICE)
    model = build_graphwavenet(num_nodes, supports)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_history, val_loss_history = [], []

    for epoch in range(0 if globals().get("EVALUATION_CHECKPOINT") else NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs_list_batch, targets_list_batch in train_loader:
            inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
            targets_list_batch = [y.to(DEVICE) for y in targets_list_batch]
            inputs_gwn = layers_list_to_model_input(inputs_list_batch, DEVICE)
            targets_gwn = layers_list_to_model_input(targets_list_batch, DEVICE).permute(0, 3, 2, 1)

            optimizer.zero_grad(set_to_none=True)
            preds = model(inputs_gwn)
            loss = criterion(preds, targets_gwn)
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
                targets_list_batch = [y.to(DEVICE) for y in targets_list_batch]
                inputs_gwn = layers_list_to_model_input(inputs_list_batch, DEVICE)
                targets_gwn = layers_list_to_model_input(targets_list_batch, DEVICE).permute(
                    0, 3, 2, 1
                )
                preds = model(inputs_gwn)
                val_loss += criterion(preds, targets_gwn).item()

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

    if not model_path.exists():
        raise RuntimeError(f"Best model was not saved: {model_path}")

    if not globals().get("EVALUATION_CHECKPOINT"):
        plot_loss(train_loss_history, val_loss_history, plot_path, f"{name} loss")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    metric_store = evaluate_graphwavenet(model, test_loader, layers, shapes, normalization_params)
    save_results_csv(result_path, layers, OUTPUT_STEPS, metric_store)

    del model, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
