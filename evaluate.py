"""Evaluate a validation-selected checkpoint and export aligned raw-count predictions."""

import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.engine import build_model
from scripts.metrics import compute_metrics
from scripts.util import (
    REPO_ROOT,
    denormalize_nodes,
    load_city,
    prepare_data,
    resolve_device,
    save_json,
)


def metrics_one(truth, prediction, nonzero=False):
    result = compute_metrics(truth, prediction, "NZ" if nonzero else "ALL")
    return np.array([result[name] for name in ("MAE", "RMSE", "COR")])


def finite_average(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else None


def metric_rows(truth, prediction, metadata):
    slices = [("overall", "all", slice(None))]
    offset = 0
    for layer, (h, w) in zip(metadata["layers"], metadata["shapes"]):
        slices.append(("layer", layer, slice(offset, offset + h * w)))
        offset += h * w
    rows = []
    for scope, layer, index in slices:
        for mask in ("ALL", "NZ"):
            values = np.array(
                [
                    [metrics_one(t, p, mask == "NZ") for t, p in zip(tw[:, index], pw[:, index])]
                    for tw, pw in zip(truth, prediction)
                ]
            )
            for step in range(truth.shape[1] + 1):
                selected = values if step == truth.shape[1] else values[:, step]
                row = {
                    "scope": scope,
                    "layer": layer,
                    "mask": mask,
                    "step": "mean" if step == truth.shape[1] else step + 1,
                }
                row.update(
                    {
                        name: finite_average(selected[..., i])
                        for i, name in enumerate(("MAE", "RMSE", "COR"))
                    }
                )
                rows.append(row)
    return rows


def evaluate(
    checkpoint_path,
    data_root,
    output_dir,
    device_name="auto",
    legacy_config=None,
    legacy_city=None,
    legacy_seed=None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if legacy_config is not None:
        if legacy_city is None or legacy_seed is None:
            raise ValueError("Legacy state-dict evaluation requires explicit config, city and seed")
        if not isinstance(checkpoint, dict) or "model_state" in checkpoint:
            raise ValueError("Expected an explicit bare legacy state dictionary")
        _, legacy_metadata = prepare_data(legacy_config, data_root, legacy_city)
        checkpoint = {
            "format_version": 1,
            "model_state": checkpoint,
            "config": legacy_config,
            "data_metadata": legacy_metadata,
            "city": legacy_city,
            "seed": legacy_seed,
            "epoch": None,
        }
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError(
            "Expected a checkpoint produced by this repository; historical raw state_dict requires explicit config/metadata conversion"
        )
    config, stored = checkpoint["config"], checkpoint["data_metadata"]
    city = checkpoint["city"]
    loaders, metadata = prepare_data(config, data_root, city, stored["normalization"])
    for key in ("shapes", "layers", "time_length", "ranges"):
        if json.dumps(metadata[key]) != json.dumps(stored[key]):
            raise ValueError(f"Dataset/checkpoint metadata differs for {key}")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {output_dir}")
    device = resolve_device(device_name)
    model = build_model(config, metadata["shapes"], device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    # Moran adjacency calibration averages over B: never mix forecast origins at inference.
    test_loader = DataLoader(loaders["test"].dataset, batch_size=1, shuffle=False, num_workers=0)
    predictions, starts = [], []
    with torch.no_grad():
        for x, _, start in test_loader:
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if config["amp"] and device.type == "cuda"
                else nullcontext()
            )
            with context:
                y = model(x.to(device))
            if y.shape[-1] != 1:
                raise ValueError(f"Expected final singleton temporal axis, got {tuple(y.shape)}")
            predictions.append(y[..., 0].float().cpu().numpy())
            starts.extend(start.tolist())
    prediction = denormalize_nodes(np.concatenate(predictions), metadata)
    if config["clip_negative_predictions"]:
        prediction = np.maximum(prediction, 0)
    # Retrieve exact original counts. Inverse-normalized targets can turn true zeros into tiny positives.
    raw, _ = load_city(data_root, city, tuple(config["layers"]))
    raw_flat = np.concatenate([x.numpy().reshape(len(x), -1) for x in raw], axis=1)
    input_steps, horizon = config["input_steps"], config["output_steps"]
    truth = np.stack([raw_flat[s + input_steps : s + input_steps + horizon] for s in starts])
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Prediction contains non-finite values")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        truth=truth,
        prediction=prediction,
        window_starts=np.asarray(starts, dtype=np.int64),
        target_time_indices=np.asarray(starts)[:, None] + input_steps + np.arange(horizon)[None, :],
    )
    metadata.update(
        {
            "seed": checkpoint["seed"],
            "configuration": config["name"],
            "checkpoint_epoch": checkpoint["epoch"],
            "input_steps": input_steps,
            "output_steps": horizon,
            "clip_negative_predictions": config["clip_negative_predictions"],
            "evaluation_batch_size": 1,
            "evaluation_amp": bool(config["amp"] and device.type == "cuda"),
            "model": "MP-STGNet",
            "forecast_mode": "direct_7",
            "input_features": config["features"],
            "truth_policy": "exact_raw_arrays",
            "checkpoint_source": str(Path(checkpoint_path).resolve()),
            "legacy_configuration_explicit": legacy_config is not None,
            "metric_aggregation": "window_horizon_mean",
        }
    )
    save_json(output_dir / "metadata.json", metadata)
    rows = metric_rows(truth, prediction, metadata)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics and {len(starts)} aligned test windows to {output_dir}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--legacy-state-dict",
        action="store_true",
        help="Require explicit --config, --city and --seed for a bare historical state dictionary",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--city")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    output = args.output_dir or args.checkpoint.parent / "evaluation"
    from scripts.util import load_config

    if args.legacy_state_dict and (args.config is None or args.city is None or args.seed is None):
        parser.error("--legacy-state-dict requires --config, --city and --seed")
    if not args.legacy_state_dict and any(
        v is not None for v in (args.config, args.city, args.seed)
    ):
        parser.error("--config/--city/--seed overrides require --legacy-state-dict")
    evaluate(
        args.checkpoint,
        args.data_root,
        output,
        args.device,
        load_config(args.config) if args.legacy_state_dict else None,
        args.city,
        args.seed,
    )


if __name__ == "__main__":
    main()
