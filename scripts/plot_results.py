"""Plot native run outputs: learning curves, prediction horizons and four-layer time series."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--comparison", type=Path, help="Optional second evaluation with exactly aligned targets"
    )
    args = parser.parse_args()
    metadata = json.loads((args.evaluation / "metadata.json").read_text(encoding="utf-8"))
    with np.load(args.evaluation / "predictions.npz", allow_pickle=False) as z:
        truth, prediction, times = z["truth"], z["prediction"], z["target_time_indices"]
    other, other_label = None, None
    if args.comparison:
        other_meta = json.loads((args.comparison / "metadata.json").read_text(encoding="utf-8"))
        with np.load(args.comparison / "predictions.npz", allow_pickle=False) as z:
            if not np.array_equal(z["truth"], truth) or not np.array_equal(
                z["target_time_indices"], times
            ):
                raise ValueError("Comparison targets/windows differ")
            other = z["prediction"]
        other_label = other_meta["configuration"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    # Use horizon 1 across windows; do not join repeated multi-horizon predictions into a fake time series.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    offset = 0
    for ax, layer, (h, w) in zip(axes.flat, metadata["layers"], metadata["shapes"]):
        sl = slice(offset, offset + h * w)
        ax.plot(times[:, 0], truth[:, 0, sl].sum(-1), color="tab:blue", label="Observed")
        ax.plot(
            times[:, 0],
            prediction[:, 0, sl].sum(-1),
            "--",
            color="tab:red",
            label=metadata["configuration"],
        )
        if other is not None:
            ax.plot(
                times[:, 0], other[:, 0, sl].sum(-1), "--", color="tab:green", label=other_label
            )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Target time index")
        ax.set_ylabel("Total access count")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
        offset += h * w
    fig.savefig(args.output_dir / "timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    with (args.evaluation / "metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for layer in metadata["layers"]:
            selected = [
                r
                for r in rows
                if r["scope"] == "layer"
                and r["layer"] == str(layer)
                and r["mask"] == "ALL"
                and r["step"] != "mean"
            ]
            ax.plot(
                [int(r["step"]) for r in selected],
                [float(r[metric]) for r in selected],
                marker="o",
                label=f"Layer {layer}",
            )
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel(metric)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.savefig(args.output_dir / "horizon_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    history = args.evaluation.parent / "history.csv"
    if history.is_file():
        with history.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for key in ("train_loss", "val_loss"):
            ax.plot([int(r["epoch"]) for r in rows], [float(r[key]) for r in rows], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized L1 loss")
        ax.legend()
        fig.savefig(args.output_dir / "learning_curve.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
