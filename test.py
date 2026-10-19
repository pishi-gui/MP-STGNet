"""Check MP-STGNet on synthetic data without baseline training or paper reproduction."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from scripts.engine import build_model
import models
from models import MPSTGNet
from models.mp_stgnet import MPSTGNet as ModelImplementation
from train import train
from evaluate import evaluate
from scripts.util import (
    load_config,
    normalize_levels,
    prepare_data,
    shapes_to_adjacency,
    temporal_features,
)

ROOT = Path(__file__).resolve().parent


def small_config(name="default"):
    config = load_config(ROOT / f"configs/{name}.json")
    config.update({"epochs": 1, "batch_size": 64, "amp": False})
    config["model"].update(
        {
            "residual_channels": 4,
            "dilation_channels": 4,
            "skip_channels": 8,
            "end_channels": 8,
            "dropout": 0.0,
        }
    )
    return config


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_model_package_exports(self):
        self.assertIs(MPSTGNet, ModelImplementation)
        self.assertEqual(models.__all__, ["MPSTGNet"])
        self.assertFalse((ROOT / "model.py").exists())

    def test_features_are_causal_and_reconstruct_raw(self):
        torch.manual_seed(1)
        raw = torch.randn(40, 2, 2, 1)
        changed = raw.clone()
        changed[20:] += 100
        features = temporal_features([raw])[0]
        comparison = temporal_features([changed])[0]
        self.assertTrue(torch.equal(features[:20], comparison[:20]))
        torch.testing.assert_close(
            features[..., 0], features[..., 1] + features[..., 2] + features[..., 3]
        )

    def test_normalization_only_uses_training(self):
        original = torch.arange(40.0).reshape(40, 1, 1, 1)
        changed = original.clone()
        changed[20:] += 500
        _, first = normalize_levels([original], 20)
        _, second = normalize_levels([changed], 20)
        self.assertEqual(first, second)

    def test_graph_matches_symmetric_normalization(self):
        actual = shapes_to_adjacency([(2, 2), (4, 4)])[0]
        self.assertTrue(torch.equal(actual, actual.T))
        self.assertEqual(float(actual[:4, 4:].sum()), 0)
        self.assertTrue(torch.all(actual.diag() > 0))

    def test_model_configurations(self):
        shapes = [(1, 1), (2, 2), (4, 4), (8, 8)]
        for name in (
            "default",
            "no_moran",
            "no_pyramid",
            "no_both",
            "raw_only",
        ):
            with self.subTest(name=name):
                config = small_config(name)
                model = build_model(config, shapes, torch.device("cpu")).eval()
                channels = 1 if config["features"] == "raw" else 4
                x = torch.randn(2, channels, 85, 28, requires_grad=True)
                y = model(x)
                self.assertEqual(tuple(y.shape), (2, 7, 85, 1))
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(model(torch.zeros_like(x))).all())

    def test_train_save_reload_evaluate(self):
        with tempfile.TemporaryDirectory(prefix="mp_stgnet_test_") as temporary:
            base = Path(temporary)
            folder = base / "data/Synthetic"
            folder.mkdir(parents=True)
            rng = np.random.default_rng(3)
            for level, side in zip((14, 15, 16, 17), (1, 2, 4, 8)):
                values = rng.poisson(2, size=(180, side, side, 1)).astype(np.int32)
                values[:, 0, 0] = 0
                np.save(folder / f"Synthetic_{level}.npy", values)
            config = small_config()
            loaders, metadata = prepare_data(config, base / "data", "Synthetic")
            self.assertEqual(metadata["samples"], {"train": 74, "val": 2, "test": 2})
            self.assertEqual(loaders["val"].dataset.starts, [108, 109])
            self.assertEqual(loaders["test"].dataset.starts, [144, 145])
            checkpoint = train(config, "Synthetic", 99, base / "data", base / "run", "cpu")
            rows = evaluate(checkpoint, base / "data", base / "evaluation", "cpu")
            with np.load(base / "evaluation/predictions.npz", allow_pickle=False) as z:
                self.assertEqual(z["truth"].shape, (2, 7, 85))
                self.assertTrue((z["truth"][..., 0] == 0).all())
                self.assertTrue(np.isfinite(z["prediction"]).all())
                self.assertTrue((z["prediction"] >= 0).all())
            self.assertEqual(len(rows), 5 * 2 * 8)
            with self.assertRaises(FileExistsError):
                train(config, "Synthetic", 99, base / "data", base / "run", "cpu")


if __name__ == "__main__":
    unittest.main()
