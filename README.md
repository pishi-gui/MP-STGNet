# MP-STGNet

PyTorch implementation of MP-STGNet for multilevel OSM tile-access forecasting.

[中文说明](README_zh.md)

## Directory

```text
models/          MP-STGNet model
baselines/       Baseline models and third-party dependencies
configs/         Model and ablation configurations
data/            Prepared datasets for five cities
scripts/         Shared utilities and experiment/result scripts
train.py         Train MP-STGNet
evaluate.py      Evaluate a saved checkpoint and export predictions
run_baseline.py  Train and evaluate a baseline
test.py          Check the main model on synthetic data
```

Data are included for `Milan`, `Rome`, `Vienna`, `SanFrancisco` and `Seoul`,
with zoom levels 14–17. Files follow `data/<City>/<City>_<Layer>.npy`.

## Installation

Use Python 3.11 and install a CPU/CUDA-compatible PyTorch build. Dependencies
are listed in `requirements.txt`. Run the following from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Train the full model and evaluate its best checkpoint:

```bash
python -B train.py --config configs/default.json --city Milan --seed 99 --device auto
python -B evaluate.py --checkpoint runs/Milan/full/seed_99/best.pth --device auto
```

`auto` selects CUDA when available, otherwise CPU. Training outputs are saved
under `runs/Milan/full/seed_99/`; evaluation writes `metrics.csv`,
`metadata.json` and `predictions.npz` to its `evaluation/` subdirectory.
No pretrained checkpoints are bundled.

For an ablation, replace the configuration with `no_moran.json`,
`no_pyramid.json`, `no_both.json` or `raw_only.json`. For example:

```bash
python -B train.py --config configs/no_moran.json --city Milan --seed 99 --device auto
```

Train and evaluate a baseline:

```bash
python -B run_baseline.py --model pdformer --city Milan --seed 99 --device auto
```

Other bundled choices are `gwn`, `astgcn`, `hsdgnn` and `testam`.
Baseline outputs are saved under `runs/<City>/<model>/seed_<seed>/`.
AdpSTGCN needs a separately obtained authorised dependency; see
[THIRD_PARTY.md](THIRD_PARTY.md) for dependencies and permissions.

All examples use seed `99` for one run. For repeated experiments, use the same
set of seeds across models and cities, training separately for each seed.
Replace `--city` to select a dataset. For reruns, choose a new `--output-dir`
to avoid overwriting existing outputs.
