"""Run a legacy baseline with isolated settings and explicit protocol/environment records."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from baselines.reporting import EvaluationRecorder
from scripts.metrics import observe_predictions
from scripts.util import REPO_ROOT, load_city, resolve_device, save_json


class BaselineUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineSpec:
    entry: str
    forecast_mode: str
    input_features: str = "raw"
    seed_first: bool = False
    required_source: str | None = None

    def invoke(self, module, city, seed):
        arguments = (seed, city) if self.seed_first else (city,)
        return getattr(module, self.entry)(*arguments)


BASELINES = {
    "gwn": BaselineSpec("run_graphwavenet_city", "direct_7"),
    "astgcn": BaselineSpec("run_astgcn_city", "recursive_7"),
    "hsdgnn": BaselineSpec(
        "run_hsdgnn_city", "recursive_7", input_features="raw + constant zero time placeholders"
    ),
    "adpstgcn": BaselineSpec(
        "run_adpstgcn_city", "recursive_7", required_source="vendor/adpstgcn/ada_STGCN.py"
    ),
    "testam": BaselineSpec("run_testam_city", "recursive_7", input_features="raw + time indices"),
    "pdformer": BaselineSpec("run_pdformer_city", "direct_7", seed_first=True),
}


def load_baseline(name):
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    spec = BASELINES[name]
    if spec.required_source:
        source = REPO_ROOT / "baselines" / spec.required_source
        if not source.is_file():
            raise BaselineUnavailableError(
                f"{name} is unavailable: {source.name} is not included in public Git. "
                "See THIRD_PARTY.md and obtain an authorized copy before using this baseline."
            )
    return spec, importlib.import_module(f"baselines.{name}")


@contextmanager
def temporary_settings(module, settings):
    """Restore module settings even if a run fails; not a threaded execution interface."""
    missing = object()
    previous = {key: getattr(module, key, missing) for key in settings}
    try:
        for key, value in settings.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                delattr(module, key)
            else:
                setattr(module, key, value)


def serializable_settings(module):
    result = {}
    for name, value in vars(module).items():
        if not name.isupper():
            continue
        if isinstance(value, (Path, torch.device)):
            result[name] = str(value)
        elif isinstance(value, (str, bool, int, float, list, tuple)) or value is None:
            result[name] = value
    return result


def run_baseline(
    name,
    city,
    seed=99,
    data_root=None,
    output_dir=None,
    device_name="auto",
    epochs=None,
    batch_size=None,
    dry_run=False,
    checkpoint=None,
    model_layers=None,
    features=None,
    legacy_skip_replace=False,
):
    spec, module = load_baseline(name)
    data_root = Path(data_root or REPO_ROOT / "data").resolve()
    device = resolve_device(device_name)
    settings = {"DATA_ROOT": data_root, "DEVICE": device, "SKIP_COMPLETED": False}
    settings["EVALUATION_CHECKPOINT"] = Path(checkpoint).resolve() if checkpoint else None
    if legacy_skip_replace:
        if name != "gwn" or checkpoint is None:
            raise ValueError(
                "--legacy-skip-replace requires GWN and an explicit historical checkpoint"
            )
        settings["MODEL_SKIP_MODE"] = "replace"
    if features is not None:
        if name != "gwn" or features not in {"raw", "decomposed"}:
            raise ValueError(
                "Explicit --features is supported only for GWN raw/decomposed controls"
            )
        settings["INPUT_FEATURE_MODE"] = features
    if model_layers is not None:
        if name != "gwn" or model_layers < 1:
            raise ValueError(
                "--model-layers is only supported for explicitly versioned GWN evaluation"
            )
        settings["MODEL_LAYERS"] = model_layers
    if checkpoint and output_dir is None:
        raise ValueError("Saved-checkpoint evaluation requires an explicit new --output-dir")
    if hasattr(module, "BASE_SEED"):
        settings["BASE_SEED"] = seed
    if epochs is not None:
        settings["NUM_EPOCHS"] = epochs
    if batch_size is not None:
        settings["BATCH_SIZE"] = batch_size
    with temporary_settings(module, settings):
        if module.NUM_EPOCHS < 1 or module.BATCH_SIZE < 1:
            raise ValueError("epochs and batch-size must be positive")
        arrays, shapes = load_city(data_root, city)
        total = len(arrays[0])
        train_end = int(total * module.TRAIN_SPLIT_RATIO)
        val_end = int(total * (module.TRAIN_SPLIT_RATIO + module.VALIDATION_SPLIT_RATIO))
        protocol = {
            "input_steps": module.INPUT_STEPS,
            "output_steps": module.OUTPUT_STEPS,
            "input_features": features or spec.input_features,
            "forecast_mode": spec.forecast_mode,
            "architecture_variant": (
                f"gwn_layers_{model_layers or 4}_skip_{'replace' if legacy_skip_replace else 'accumulate'}"
                if name == "gwn"
                else "exported_legacy_adaptation"
            ),
            "time_length": total,
            "shapes": shapes,
            "ranges": {
                "train": [0, train_end],
                "val": [train_end, val_end],
                "test": [val_end, total],
            },
            "metric_version": "shared_raw_metrics_v1",
            "metric_schema": "canonical long table plus retained legacy CSV",
            "truth_policy": "not_exported",
            "legacy_csv_truth_policy": "inverse_normalized_targets",
            "prediction_clipping": "nonnegative in legacy metric adapters",
        }
        protocol["expected_ranges"] = dict(protocol["ranges"])
        protocol.update(
            {"ranges_verified": False, "ranges_source": "expected_before_loader_verification"}
        )
        if dry_run:
            recorder = EvaluationRecorder(arrays, protocol)
            normalized, _ = module.normalize_data_levels(arrays, train_end)
            factory_name = "make_indexed_dataloaders" if name == "testam" else "make_dataloaders"
            recorder.verify_loaders(getattr(module, factory_name), normalized, total)
            print(f"{name}: city={city}, seed={seed}, device={device}, shapes={shapes}")
            print(f"Protocol: {spec.forecast_mode}; inputs={spec.input_features}")
            return protocol
        output = Path(output_dir or REPO_ROOT / "runs" / city / name / f"seed_{seed}").resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
        directories = {
            key: output / relative
            for key, relative in (
                ("RESULT_ROOT", "results"),
                ("MODEL_ROOT", "checkpoints"),
                ("PLOT_ROOT", "figures"),
                ("CACHE_ROOT", "cache"),
            )
        }
        for path in directories.values():
            path.mkdir(parents=True, exist_ok=True)
        with temporary_settings(module, directories):
            config = serializable_settings(module)
            config.update(
                {"city": city, "seed": seed, "data_root": str(data_root), "protocol": protocol}
            )
            save_json(output / "configuration.json", config)
            run = {
                "model": name,
                "city": city,
                "seed": seed,
                "device": str(device),
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "data_root": str(data_root),
                "protocol": protocol,
                "status": "running",
                "operation": "evaluate_existing_checkpoint" if checkpoint else "train_and_evaluate",
                "checkpoint_source": str(Path(checkpoint).resolve()) if checkpoint else None,
            }
            save_json(output / "run.json", run)
            try:
                recorder = EvaluationRecorder(arrays, protocol)
                factory_name = (
                    "make_indexed_dataloaders" if name == "testam" else "make_dataloaders"
                )
                original_factory = getattr(module, factory_name)

                def checked_factory(*args, **kwargs):
                    return recorder.verify_loaders(original_factory, *args, **kwargs)

                with temporary_settings(module, {factory_name: checked_factory}):
                    with observe_predictions(recorder.observe):
                        result = spec.invoke(module, city, seed)
                run["canonical_export"] = recorder.export(
                    output / "evaluation",
                    {
                        "model": name,
                        "city": city,
                        "seed": seed,
                        "configuration": f"{name}_decomposed" if features == "decomposed" else name,
                    },
                )
                if run["canonical_export"]:
                    protocol.update(
                        {
                            "truth_policy": "exact_raw_arrays",
                            "metric_aggregation": "window_horizon_mean",
                        }
                    )
            except Exception as error:
                run.update(
                    {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
                )
                save_json(output / "run.json", run)
                save_json(output / "configuration.json", config)
                raise
            run["status"] = "completed"
            save_json(output / "run.json", run)
            save_json(output / "configuration.json", config)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BASELINES, required=True)
    parser.add_argument("--city", default="Milan")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate existing weights without training; requires --output-dir",
    )
    parser.add_argument(
        "--model-layers",
        type=int,
        help="Explicit GWN layers per temporal block; default preserves current implementation",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check imports, data and protocol without training"
    )
    parser.add_argument(
        "--features",
        choices=["raw", "decomposed"],
        help="GWN input-matched control; raw remains default",
    )
    parser.add_argument(
        "--legacy-skip-replace",
        action="store_true",
        help="Use the verified historical GWN skip-replacement forward for saved weights only",
    )
    args = parser.parse_args()
    try:
        run_baseline(
            args.model,
            args.city,
            args.seed,
            args.data_root,
            args.output_dir,
            args.device,
            args.epochs,
            args.batch_size,
            args.dry_run,
            args.checkpoint,
            args.model_layers,
            args.features,
            args.legacy_skip_replace,
        )
    except (BaselineUnavailableError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
