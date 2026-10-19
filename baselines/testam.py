"""TESTAM baseline adapted for multilevel OSM tile-access forecasting."""

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


import gc


import random


import sys


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import Dataset, DataLoader


import matplotlib


import matplotlib.pyplot as plt


DATA_ROOT = REPO_ROOT / "data"

RESULT_ROOT = REPO_ROOT / "runs/baselines/results"

MODEL_ROOT = REPO_ROOT / "runs/baselines/checkpoints"

PLOT_ROOT = REPO_ROOT / "runs/baselines/figures"

RUN_CITIES = ["Rome", "Milan", "Vienna", "SanFrancisco", "Seoul"]

TRAIN_SPLIT_RATIO = 0.6


VALIDATION_SPLIT_RATIO = 0.2


INPUT_STEPS = 28


OUTPUT_STEPS = 7


BATCH_SIZE = 1


NUM_EPOCHS = 200


PATIENCE = 10


MIN_DELTA = 1e-4


BASE_SEED = 76


SKIP_COMPLETED = True


LEARNING_RATE = 1e-4


TESTAM_HIDDEN = 16


TESTAM_LAYERS = 1


TESTAM_DROPOUT = 0.3


TESTAM_MAX_TIME_INDEX = 7


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
    normalized, params = [], []
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


class MultiLayerIndexedDataset(Dataset):
    def __init__(self, data_list, input_steps, output_steps, time_start=0, time_end=None):
        self.data_list = data_list
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
        targets = [data[end:target_end] for data in self.data_list]
        return inputs, targets, torch.tensor(start, dtype=torch.long)


def make_indexed_dataloaders(normalized_data_levels, total_timesteps):
    train_time_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    val_time_end = int(total_timesteps * (TRAIN_SPLIT_RATIO + VALIDATION_SPLIT_RATIO))
    train_dataset = MultiLayerIndexedDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, 0, train_time_end
    )
    val_dataset = MultiLayerIndexedDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, train_time_end, val_time_end
    )
    test_dataset = MultiLayerIndexedDataset(
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


def flatten_layers_value(layers, device=None):
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
            raise ValueError(f"Expected one value channel, got C={c}")
        if idx == 0:
            B, T = b, t
        elif (b, t) != (B, T):
            raise ValueError("Layer batch/time sizes are inconsistent.")
        parts.append(x[..., 0].contiguous().view(B, T, h * w))
    return torch.cat(parts, dim=2)


def build_testam_input(layers, start_indices, device, offset=0):
    values = flatten_layers_value(layers, device=device)  # B,T,N
    B, T, N = values.shape
    start_indices = start_indices.to(device=device, dtype=torch.long).view(B, 1)
    t = torch.arange(T, device=device).view(1, T)
    dow = ((start_indices + offset + t) % TESTAM_MAX_TIME_INDEX).float() / float(
        TESTAM_MAX_TIME_INDEX
    )
    time_feat = dow.unsqueeze(-1).expand(B, T, N)
    return torch.stack([values, time_feat], dim=1)  # B,2,T,N -> fix below


def build_testam_input_bcnt(layers, start_indices, device, offset=0):
    x = build_testam_input(layers, start_indices, device, offset=offset)
    # x is B,2,T,N; TESTAM expects B,C,N,T.
    return x.permute(0, 1, 3, 2).contiguous()


def testam_output_to_layers(preds, shapes):
    if preds.dim() != 2:
        raise ValueError(f"preds must be (B,N), got {tuple(preds.shape)}")
    B, N = preds.shape
    expected = sum(h * w for h, w in shapes)
    if N != expected:
        raise ValueError(f"N={N} does not match total nodes={expected}")
    out = []
    ptr = 0
    for h, w in shapes:
        n = h * w
        out.append(preds[:, ptr : ptr + n].view(B, h, w, 1).contiguous())
        ptr += n
    return out


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
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
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


from baselines.vendor.testam.model import TESTAM


def testam_predict_one(model, x):
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    # TESTAM returns B,out_dim,N,T. Use the last time position as one-step output.
    return out[:, 0, :, -1]


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


def run_testam_city(city):
    set_seed(BASE_SEED)
    name = f"TESTAM_Baseline_{city}"
    result_path = RESULT_ROOT / f"{city}_TESTAM_Baseline_Seed{BASE_SEED}.csv"
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
    train_loader, val_loader, test_loader, _, _ = make_indexed_dataloaders(
        normalized, total_timesteps
    )

    print(f"\n=== TESTAM baseline | {city} ===")
    print(f"Layers={layers}, shapes={shapes}")
    print(f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    num_nodes = sum(h * w for h, w in shapes)
    model = TESTAM(
        num_nodes=num_nodes,
        dropout=TESTAM_DROPOUT,
        in_dim=2,
        out_dim=1,
        hidden_size=TESTAM_HIDDEN,
        layers=TESTAM_LAYERS,
        max_time_index=TESTAM_MAX_TIME_INDEX,
    ).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_history, val_loss_history = [], []

    for epoch in range(0 if globals().get("EVALUATION_CHECKPOINT") else NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs_list_batch, targets_list_batch, starts in train_loader:
            inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
            targets_one = [y[:, 0].to(DEVICE) for y in targets_list_batch]
            x_model = build_testam_input_bcnt(inputs_list_batch, starts, DEVICE, offset=0)
            target = flatten_layers_value(
                [y.unsqueeze(1) for y in targets_one], device=DEVICE
            ).squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            preds = testam_predict_one(model, x_model)
            loss = criterion(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        train_loss_history.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_list_batch, targets_list_batch, starts in val_loader:
                inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
                targets_one = [y[:, 0].to(DEVICE) for y in targets_list_batch]
                x_model = build_testam_input_bcnt(inputs_list_batch, starts, DEVICE, offset=0)
                target = flatten_layers_value(
                    [y.unsqueeze(1) for y in targets_one], device=DEVICE
                ).squeeze(1)
                preds = testam_predict_one(model, x_model)
                val_loss += criterion(preds, target).item()
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

    metric_store = empty_metric_store(len(layers), OUTPUT_STEPS)
    with torch.no_grad():
        for inputs_list_batch, targets_list_batch, starts in test_loader:
            current_inputs = [x.to(DEVICE) for x in inputs_list_batch]
            targets_list_batch = [y.to(DEVICE) for y in targets_list_batch]
            starts = starts.to(DEVICE)
            batch_preds_denorm = []
            for d in range(OUTPUT_STEPS):
                x_model = build_testam_input_bcnt(current_inputs, starts, DEVICE, offset=d)
                preds = testam_predict_one(model, x_model)
                pred_layers = testam_output_to_layers(preds, shapes)
                step_denorm = []
                for i in range(len(layers)):
                    pred_denorm = inverse_normalize_data_levels([pred_layers[i]], [params[i]])[0]
                    step_denorm.append(pred_denorm)
                batch_preds_denorm.append(step_denorm)
                if d < OUTPUT_STEPS - 1:
                    current_inputs = [
                        torch.cat([current_inputs[i][:, 1:], pred_layers[i].unsqueeze(1)], dim=1)
                        for i in range(len(layers))
                    ]

            for d in range(OUTPUT_STEPS):
                true_flat_parts, pred_flat_parts = [], []
                for i in range(len(layers)):
                    true_norm = targets_list_batch[i][:, d]
                    true_denorm = inverse_normalize_data_levels([true_norm], [params[i]])[0]
                    pred_denorm = batch_preds_denorm[d][i]
                    true_np = true_denorm.cpu().numpy()
                    pred_np = pred_denorm.cpu().numpy()
                    for k in range(true_np.shape[0]):
                        append_metrics(metric_store["layer"][i][d], true_np[k], pred_np[k])
                    true_flat_parts.append(true_np.reshape(true_np.shape[0], -1))
                    pred_flat_parts.append(pred_np.reshape(pred_np.shape[0], -1))
                true_concat = np.concatenate(true_flat_parts, axis=1)
                pred_concat = np.concatenate(pred_flat_parts, axis=1)
                for k in range(true_concat.shape[0]):
                    append_metrics(metric_store["overall"][d], true_concat[k], pred_concat[k])

    save_results_csv(result_path, layers, OUTPUT_STEPS, metric_store)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


RUN_SMOKE_TEST = False
