"""Launch the four principal ablations as independent processes with explicit execution."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cities", nargs="+", default=["Rome", "Milan", "Vienna", "SanFrancisco", "Seoul"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 76, 99])
    parser.add_argument(
        "--configs", nargs="+", default=["default", "no_moran", "no_pyramid", "no_both"]
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for config in args.configs:
        config_path = ROOT / "configs" / f"{config}.json"
        if not config_path.is_file():
            parser.error(f"Missing configuration: {config_path}")
        for city in args.cities:
            for seed in args.seeds:
                command = [
                    sys.executable,
                    str(ROOT / "train.py"),
                    "--config",
                    str(config_path),
                    "--city",
                    city,
                    "--seed",
                    str(seed),
                    "--data-root",
                    str(args.data_root),
                    "--device",
                    args.device,
                ]
                print(shlex.join(command), flush=True)
                if args.execute:
                    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
