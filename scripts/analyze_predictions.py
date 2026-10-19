"""Paired seed/window analysis of aligned prediction archives; never treats windows as runs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_results import evaluation_identity, experiment_name


def window_errors(path):
    with np.load(path, allow_pickle=False) as data:
        truth = data["truth"].astype(np.float64)
        prediction = data["prediction"].astype(np.float64)
        times = data["target_time_indices"]
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError("Expected aligned [window,horizon,node] arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Prediction archive contains non-finite values")
    if times.shape != truth.shape[:2] or not np.issubdtype(times.dtype, np.integer):
        raise ValueError("Expected integer [window,horizon] target indices")
    if len(times) == 0 or (np.diff(times[:, 0]) <= 0).any():
        raise ValueError("Test windows must have unique chronological origins")
    error = prediction - truth
    signature = hashlib.sha256(truth.tobytes() + times.tobytes()).hexdigest()
    return {
        "MAE": np.abs(error).mean(axis=(1, 2)),
        "RMSE": np.sqrt(np.square(error).mean(axis=2)).mean(axis=1),
        "signature": signature,
        "pooled_RMSE": float(np.sqrt(np.square(error).mean())),
    }


def paired_summary(control, target, block_length=7, draws=10000, random_seed=20261018):
    control, target = np.asarray(control), np.asarray(target)
    if (
        control.shape != target.shape
        or control.ndim != 2
        or not np.isfinite(control).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Expected finite paired [training_run, chronological_window] arrays")
    seeds, windows = control.shape
    if seeds < 1 or windows < 1 or not isinstance(draws, int) or draws < 1:
        raise ValueError("At least one training run, window and bootstrap draw are required")
    if not isinstance(block_length, int) or block_length < 1:
        raise ValueError("Invalid block length")
    difference = control - target
    row = {
        "n_training_runs": seeds,
        "n_windows_per_run": windows,
        "block_length": block_length,
        "control_mean": float(control.mean()),
        "target_mean": float(target.mean()),
        "difference_control_minus_target": float(difference.mean()),
        "relative_improvement_pct_control": float(difference.mean() / control.mean() * 100)
        if control.mean()
        else np.nan,
        "relative_change_pct_target": float(difference.mean() / target.mean() * 100)
        if target.mean()
        else np.nan,
        "window_win_rate_pct": float((difference > 0).mean() * 100),
        "positive_run_directions": int((difference.mean(axis=1) > 0).sum()),
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "inference": "exploratory_unadjusted_percentile_CI; overlapping_windows_are_not_independent_replicates",
    }
    if seeds < 2:
        row["inference"] = "descriptive_only; one_training_run"
        return row
    if windows < block_length:
        row["inference"] = "descriptive_only; fewer_windows_than_requested_block"
        return row
    rng = np.random.default_rng(random_seed)
    bootstrap = []
    for begin in range(0, draws, 512):
        batch = min(512, draws - begin)
        seed_indices = rng.integers(0, seeds, size=(batch, seeds))
        blocks = rng.integers(
            0, windows - block_length + 1, size=(batch, int(np.ceil(windows / block_length)))
        )
        indices = (blocks[..., None] + np.arange(block_length)).reshape(batch, -1)[:, :windows]
        # A shared date-block draw across sampled seeds retains common calendar shocks.
        selected = difference[seed_indices[:, :, None], indices[:, None, :]]
        bootstrap.extend(selected.mean(axis=(1, 2)))
    row["ci95_low"], row["ci95_high"] = np.quantile(bootstrap, [0.025, 0.975])
    return row


def analyze(evaluations, output, draws=10000, metadata_paths=None):
    output = Path(output)
    for name in ("window_errors.csv", "paired_block_summary.csv", "missing_comparisons.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    output.mkdir(parents=True, exist_ok=True)
    records, index, seen = [], {}, set()
    sources = (
        metadata_paths
        if metadata_paths is not None
        else [p for base in evaluations for p in Path(base).rglob("metadata.json")]
    )
    for meta_path in sources:
        meta_path = Path(meta_path)
        archive = meta_path.with_name("predictions.npz")
        if not archive.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source_identity = evaluation_identity(meta)
        seed = source_identity["seed"]
        model = experiment_name(source_identity)
        errors = window_errors(archive)
        key = (meta["city"], model, seed)
        identity = (meta["city"], model, meta.get("run_id", str(seed)))
        if identity in seen:
            raise ValueError(f"Duplicate prediction source: {key}")
        seen.add(identity)
        if seed is not None:
            if key in index:
                raise ValueError(f"Multiple runs share a paired-training identity: {key}")
            index[key] = errors
        for metric in ("MAE", "RMSE"):
            for window, value in enumerate(errors[metric]):
                records.append(
                    {
                        "city": meta["city"],
                        "model": model,
                        "seed": seed,
                        "run_id": meta.get("run_id", str(seed)),
                        "window": window,
                        "metric": metric,
                        "value": value,
                        "source": str(archive),
                    }
                )
    comparisons = [
        ("gwn", "MP-STGNet"),
        ("gwn/decomposed", "MP-STGNet"),
        ("gwn", "gwn/decomposed"),
        ("gwn", "MP-STGNet/raw_only"),
        ("MP-STGNet/raw_only", "MP-STGNet"),
        ("MP-STGNet/no_moran", "MP-STGNet"),
        ("MP-STGNet/no_pyramid", "MP-STGNet"),
        ("MP-STGNet/no_both", "MP-STGNet"),
    ]
    summaries, missing = [], []
    cities = sorted({k[0] for k in index})
    for city in cities:
        for control_name, target_name in comparisons:
            common = sorted(
                {k[2] for k in index if k[:2] == (city, control_name)}
                & {k[2] for k in index if k[:2] == (city, target_name)}
            )
            if not common:
                missing.append({"city": city, "control": control_name, "target": target_name})
                continue
            for seed in common:
                if (
                    index[city, control_name, seed]["signature"]
                    != index[city, target_name, seed]["signature"]
                ):
                    raise ValueError(f"Comparison targets or target dates differ: {city}, {seed}")
            for metric in ("MAE", "RMSE"):
                a = np.stack([index[city, control_name, seed][metric] for seed in common])
                b = np.stack([index[city, target_name, seed][metric] for seed in common])
                for length in (7, 14):
                    row = paired_summary(a, b, length, draws)
                    row.update(
                        {
                            "city": city,
                            "control": control_name,
                            "target": target_name,
                            "metric": metric,
                            "paired_seeds": ",".join(map(str, common)),
                            "training_budget_note": "gwn/decomposed historical control is warm-started and fine-tuned; not a fully budget-matched causal contrast"
                            if "gwn/decomposed" in (control_name, target_name)
                            else "see_source_protocol",
                        }
                    )
                    summaries.append(row)
    pd.DataFrame(records).to_csv(output / "window_errors.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "paired_block_summary.csv", index=False)
    (output / "missing_comparisons.json").write_text(
        json.dumps(missing, indent=2), encoding="utf-8"
    )
    print(f"Analyzed {len(index)} seed-identified runs; {len(summaries)} paired summaries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--evaluations", nargs="+", type=Path)
    sources.add_argument(
        "--manifest",
        type=Path,
        help="The same explicit accepted-source manifest used by collect_results.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    paths = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        paths = []
        for entry in manifest["entries"]:
            path = Path(entry["path"])
            if not path.is_absolute():
                path = args.manifest.parent / path
            if path.is_file():
                paths.append(path.with_name("metadata.json"))
    analyze(args.evaluations or [], args.output_dir, args.draws, paths)


if __name__ == "__main__":
    main()
