"""Configuration, causal input features, strict temporal splits, and graph utilities."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


def merge_config(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Circular configuration inheritance: {path}")
    seen.add(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent:
        config = merge_config(load_config(path.parent / parent, seen), config)
    return config


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu.")
    return device


def load_city(data_root, city, layers=(14, 15, 16, 17)):
    folder = Path(data_root) / city
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing city directory: {folder}. See data/README.md.")
    found = {}
    for path in folder.glob("*.npy"):
        # Actual files are City_14.npy; match the final integer, not digits in a city name.
        match = re.search(r"(\d+)$", path.stem)
        if match and int(match[1]) in layers:
            level = int(match[1])
            if level in found:
                raise ValueError(f"Ambiguous files for layer {level}: {found[level]} and {path}")
            found[level] = path
    if set(found) != set(layers):
        raise ValueError(f"{city}: expected layers {list(layers)}, found {sorted(found)}")
    arrays = []
    for level in layers:
        raw = np.load(found[level], allow_pickle=False)
        if raw.ndim == 3:
            raw = raw[..., None]
        if raw.ndim != 4 or raw.shape[-1] != 1:
            raise ValueError(f"{found[level].name}: expected [T,H,W,1] or [T,H,W], got {raw.shape}")
        if min(raw.shape) < 1 or not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError(f"{found[level].name}: access counts must be finite and nonnegative")
        arrays.append(torch.tensor(raw, dtype=torch.float32))
    if len({a.shape[0] for a in arrays}) != 1:
        raise ValueError("All layers must share the same ordered dates")
    shapes = [(int(x.shape[1]), int(x.shape[2])) for x in arrays]
    for coarse, fine in zip(shapes, shapes[1:]):
        if fine != (coarse[0] * 2, coarse[1] * 2):
            raise ValueError(f"Pyramid requires aligned 2x grids, got {coarse} -> {fine}")
    return arrays, shapes


def normalize_levels(arrays, train_end, parameters=None):
    """Per-level training-period mean and PyTorch sample standard deviation (correction=1)."""
    if parameters is None:
        parameters = []
        for x in arrays:
            part = x[:train_end]
            if part.numel() < 2:
                raise ValueError("Training range too small for sample standard deviation")
            mean, std = part.mean(), part.std().clamp_min(1e-5)
            parameters.append({"mean": float(mean), "std": float(std)})
    if len(parameters) != len(arrays):
        raise ValueError("Normalization metadata does not match layer count")
    normalized = [(x - p["mean"]) / p["std"] for x, p in zip(arrays, parameters)]
    return normalized, parameters


def temporal_features(arrays, mode="decomposed", period=7, trend_window=7):
    """Construct causal raw, running-trend, phase-periodic and residual channels."""
    if mode not in ("raw", "decomposed") or period < 1 or trend_window < 1:
        raise ValueError("Invalid temporal feature settings")
    if mode == "raw":
        return arrays
    result = []
    for raw in arrays:
        trend = torch.empty_like(raw)
        for t in range(len(raw)):
            trend[t] = raw[max(0, t - trend_window + 1) : t + 1].mean(dim=0)
        detrended = raw - trend
        periodic = torch.empty_like(raw)
        totals = [torch.zeros_like(raw[0]) for _ in range(period)]
        counts = [0] * period
        for t in range(len(raw)):
            phase = t % period
            totals[phase] = totals[phase] + detrended[t]
            counts[phase] += 1
            periodic[t] = totals[phase] / counts[phase]
        result.append(
            torch.cat([raw, trend, periodic, raw - trend - periodic], dim=-1).contiguous()
        )
    return result


class WindowDataset(Dataset):
    def __init__(self, features, targets, start, end, input_steps, output_steps):
        self.features, self.targets = features, targets
        self.input_steps, self.output_steps = input_steps, output_steps
        self.starts = list(range(start, end - input_steps - output_steps + 1))
        if not self.starts:
            raise ValueError(
                f"Range [{start},{end}) cannot hold {input_steps}+{output_steps} time steps"
            )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        mid, end = start + self.input_steps, start + self.input_steps + self.output_steps
        x = torch.cat(
            [v[start:mid].reshape(self.input_steps, -1, v.shape[-1]) for v in self.features], dim=1
        )
        y = torch.cat([v[mid:end].reshape(self.output_steps, -1, 1) for v in self.targets], dim=1)
        return x.permute(2, 1, 0).contiguous(), y.contiguous(), start


def prepare_data(config, data_root, city, normalization=None):
    arrays, shapes = load_city(data_root, city, tuple(config["layers"]))
    total = len(arrays[0])
    train_end = int(total * config["train_ratio"])
    val_end = int(total * (config["train_ratio"] + config["val_ratio"]))
    normalized, parameters = normalize_levels(arrays, train_end, normalization)
    features = temporal_features(
        normalized, config["features"], config["period"], config["trend_window"]
    )
    ranges = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, total)}
    datasets = {
        name: WindowDataset(
            features, normalized, a, b, config["input_steps"], config["output_steps"]
        )
        for name, (a, b) in ranges.items()
    }
    loaders = {
        name: DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False
        )
        for name, dataset in datasets.items()
    }
    metadata = {
        "city": city,
        "layers": config["layers"],
        "shapes": shapes,
        "time_length": total,
        "ranges": ranges,
        "normalization": parameters,
        "samples": {k: len(v) for k, v in datasets.items()},
    }
    return loaders, metadata


def shapes_to_adjacency(shapes, device="cpu"):
    """Eight-neighbor + self-loop, D^(-1/2) A D^(-1/2); layers remain disconnected."""
    n = sum(h * w for h, w in shapes)
    adj = np.zeros((n, n), dtype=np.float32)
    offset = 0
    for h, w in shapes:
        for row in range(h):
            for col in range(w):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if 0 <= row + dr < h and 0 <= col + dc < w:
                            adj[offset + row * w + col, offset + (row + dr) * w + col + dc] = 1.0
        offset += h * w
    inv = 1 / np.sqrt(adj.sum(axis=1).clip(min=1))
    # Algebraically identical to dense diagonal matrix multiplication, without cubic work.
    adj = (adj * inv[:, None]) * inv[None, :]
    return [torch.as_tensor(adj, dtype=torch.float32, device=device)]


def denormalize_nodes(values, metadata):
    output = values.copy()
    offset = 0
    for (h, w), params in zip(metadata["shapes"], metadata["normalization"]):
        stop = offset + h * w
        output[..., offset:stop] = values[..., offset:stop] * params["std"] + params["mean"]
        offset = stop
    return output
