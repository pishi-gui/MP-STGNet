"""Collect explicit legacy CSV manifests and canonical run outputs without inventing metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluation_identity(metadata):
    model = metadata.get("model", "MP-STGNet")
    seed = metadata.get("training_seed", metadata.get("seed"))
    return {
        "model": model,
        "configuration": metadata.get("configuration", "full" if model == "MP-STGNet" else model),
        "city": metadata["city"],
        "seed": seed,
        "run_id": metadata.get("run_id", str(seed) if seed is not None else "unknown"),
        "forecast_mode": metadata.get("forecast_mode", "unspecified"),
        "input_features": metadata.get("input_features", "unspecified"),
        "architecture_variant": metadata.get("architecture_variant", "unspecified"),
        "metric_aggregation": metadata.get("metric_aggregation", "unverified"),
        "truth_policy": metadata.get("truth_policy", "unverified"),
    }


def experiment_name(identity):
    model, configuration = identity["model"], identity["configuration"]
    if model == "MP-STGNet":
        return model if configuration == "full" else f"{model}/{configuration}"
    if model == "gwn" and identity.get("input_features") == "decomposed":
        return "gwn/decomposed"
    return model


def run_identity(identity):
    return tuple(
        identity.get(key, "unspecified") for key in ("model", "configuration", "city", "run_id")
    )


def normalize_csv(path, identity):
    frame = pd.read_csv(path)
    rows = []
    long_form = {"scope", "layer", "mask", "step"}.issubset(frame.columns)
    for record in frame.to_dict("records"):
        if long_form:
            scopes = [(record["scope"], record["layer"], record["mask"], record["step"], "")]
        elif {"Layer", "Day"}.issubset(frame.columns):
            layer = str(record["Layer"])
            scope = "overall" if layer.upper() == "OVERALL" else "layer"
            day = str(record["Day"])
            step = "mean" if day.upper() == "AVERAGE" else int(day.split()[-1])
            scopes = (
                [
                    (scope, "all" if scope == "overall" else layer, mask, step, "_" + mask)
                    for mask in ("ALL", "NZ")
                ]
                if "MAE_ALL" in frame.columns
                else [(scope, "all" if scope == "overall" else layer, "ALL", step, "")]
            )
        else:
            raise ValueError(f"Unrecognized metric schema: {path}")
        for scope, layer, mask, step, suffix in scopes:
            step = "mean" if str(step).lower() == "mean" else int(float(step))
            layer = "all" if scope == "overall" else int(float(layer))
            row = {
                **identity,
                "scope": scope,
                "layer": layer,
                "mask": mask,
                "step": step,
                "source_file": str(path.resolve()),
            }
            for metric in ("MAE", "RMSE", "COR"):
                row[metric] = record.get(metric + suffix, np.nan)
            row["missing_metrics"] = ",".join(k for k in ("MAE", "RMSE", "COR") if pd.isna(row[k]))
            rows.append(row)
    return rows


def collect(entries, output, expected=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.csv", "summary.csv", "missing_runs.json", "source_manifest.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    records, provenance, missing = [], [], []
    identities = set()
    for entry in entries:
        path = Path(entry["path"])
        identity = {key: value for key, value in entry.items() if key != "path"}
        identity.setdefault("run_id", str(identity.get("seed", "unknown")))
        identity.setdefault("configuration", "unspecified")
        identity.setdefault("metric_aggregation", "unverified_legacy")
        identity.setdefault("truth_policy", "unverified_legacy")
        key = run_identity(identity)
        if key in identities:
            raise ValueError(f"Duplicate run identity: {key}; choose one source explicitly")
        if not path.is_file():
            missing.append({**identity, "reason": "file_missing", "path": str(path)})
            continue
        identities.add(key)
        records.extend(normalize_csv(path, identity))
        provenance.append({**entry, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for item in expected:
        if run_identity(item) not in identities:
            missing.append({**item, "reason": "expected_run_not_available"})
    frame = pd.DataFrame(records)
    frame.to_csv(output / "comparison.csv", index=False)
    summaries = []
    if not frame.empty:
        grouping = [
            k
            for k in (
                "model",
                "configuration",
                "architecture_variant",
                "forecast_mode",
                "city",
                "input_features",
                "metric_aggregation",
                "truth_policy",
                "scope",
                "layer",
                "mask",
                "step",
            )
            if k in frame
        ]
        for key, group in frame.groupby(grouping, dropna=False, sort=False):
            row = dict(zip(grouping, key))
            row["n"] = group["run_id"].nunique()
            for metric in ("MAE", "RMSE", "COR"):
                values = pd.to_numeric(group[metric], errors="raise").dropna()
                row[metric + "_n"] = len(values)
                row[metric + "_mean"] = values.mean() if len(values) else np.nan
                row[metric + "_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    (output / "missing_runs.json").write_text(
        json.dumps(missing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "source_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Collected {len(identities)} runs, {len(frame)} rows; missing runs: {len(missing)}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON with explicit entries and optional expected run identities",
    )
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    entries, expected = [], []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        entries, expected = manifest["entries"], manifest.get("expected", [])
        for entry in entries:
            path = Path(entry["path"])
            if not path.is_absolute():
                entry["path"] = str(args.manifest.parent / path)
    else:
        for metadata_path in args.runs.rglob("metadata.json"):
            csv_path = metadata_path.with_name("metrics.csv")
            if not csv_path.exists():
                continue
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            entries.append({"path": str(csv_path), **evaluation_identity(meta)})
    if not entries:
        parser.error("No result sources found; use --manifest for explicit historical CSV files")
    collect(entries, args.output_dir, expected)


if __name__ == "__main__":
    main()
