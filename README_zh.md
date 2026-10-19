# MP-STGNet

面向多层级OSM瓦片访问量预测的MP-STGNet代码。

[English](README.md)

## 文件目录

```text
models/          MP-STGNet模型
baselines/       基线模型及第三方依赖
configs/         主模型和消融实验配置
data/            五个城市的数据
scripts/         公共代码、实验调度和结果处理脚本
train.py         主模型训练入口
evaluate.py      加载权重、评估并导出预测
run_baseline.py  基线训练与评估入口
test.py          主模型的合成数据检查
```

数据包含`Milan`、`Rome`、`Vienna`、`SanFrancisco`、`Seoul`五个城市，
层级为14–17，路径为`data/<城市>/<城市>_<层级>.npy`。

## 环境安装

使用Python 3.11，先安装与CPU/CUDA设备匹配的PyTorch。
依赖统一列在`requirements.txt`中，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

## 快速训练与评估

训练完整模型，并评估保存的最佳权重：

```bash
python -B train.py --config configs/default.json --city Milan --seed 99 --device auto
python -B evaluate.py --checkpoint runs/Milan/full/seed_99/best.pth --device auto
```

`auto`在CUDA可用时使用GPU，否则使用CPU。训练结果保存在
`runs/Milan/full/seed_99/`，评估结果保存在其`evaluation/`子目录，
包括`metrics.csv`、`metadata.json`和`predictions.npz`。仓库不附带预训练权重。

消融实验只需将配置替换为`no_moran.json`、`no_pyramid.json`、
`no_both.json`或`raw_only.json`。例如：

```bash
python -B train.py --config configs/no_moran.json --city Milan --seed 99 --device auto
```

训练并评估基线模型：

```bash
python -B run_baseline.py --model pdformer --city Milan --seed 99 --device auto
```

还可选择`gwn`、`astgcn`、`hsdgnn`或`testam`。
基线结果保存在`runs/<城市>/<模型>/seed_<种子>/`。

以上命令统一以种子`99`演示单次运行。多次实验应对各模型和城市使用同一组种子，
逐次训练后汇总；更换城市只需修改`--city`。重复运行已有实验时，
请指定新的`--output-dir`，避免覆盖已有结果。
