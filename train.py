"""Train one city/seed/configuration; outputs are isolated and never silently overwritten."""

import argparse
import csv
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from scripts.engine import Trainer
from scripts.util import REPO_ROOT, load_config, prepare_data, resolve_device, save_json, set_seed


def train(config, city, seed, data_root, output_dir, device_name="auto"):
    set_seed(seed)
    device = resolve_device(device_name)
    loaders, metadata = prepare_data(config, data_root, city)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {output_dir}; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "data_metadata.json", metadata)
    save_json(
        output_dir / "run.json",
        {
            "city": city,
            "seed": seed,
            "device": str(device),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
        },
    )
    trainer = Trainer(config, metadata["shapes"], device)
    best, bad_epochs, history = float("inf"), 0, []
    for epoch in range(1, config["epochs"] + 1):
        begin = time.perf_counter()
        train_loss = trainer.run_epoch(loaders["train"], True)
        val_loss = trainer.run_epoch(loaders["val"], False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": time.perf_counter() - begin,
            }
        )
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best - config["min_delta"]:
            best, bad_epochs = val_loss, 0
            torch.save(
                {
                    "format_version": 1,
                    "model_state": trainer.model.state_dict(),
                    "config": config,
                    "data_metadata": metadata,
                    "city": city,
                    "seed": seed,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                output_dir / "best.pth",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                break
    if not (output_dir / "best.pth").exists():
        raise RuntimeError("No valid checkpoint was produced")
    return output_dir / "best.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.json")
    parser.add_argument("--city", default="Milan")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check real data, features and windows; do not instantiate/train the model",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if config["epochs"] < 1 or config["batch_size"] < 1:
        parser.error("epochs and batch-size must be positive")
    if args.dry_run:
        import json

        _, metadata = prepare_data(config, args.data_root, args.city)
        print(json.dumps(metadata, indent=2))
        return
    output = (
        args.output_dir or REPO_ROOT / "runs" / args.city / config["name"] / f"seed_{args.seed}"
    )
    checkpoint = train(config, args.city, args.seed, args.data_root, output, args.device)
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
