"""Verify actual loaders and export aligned predictions from legacy evaluators."""

import csv
from pathlib import Path

import numpy as np
from torch.utils.data import SequentialSampler

from scripts.util import save_json


class EvaluationRecorder:
    def __init__(self, arrays, protocol):
        self.raw = np.concatenate([a.numpy().reshape(len(a), -1) for a in arrays], axis=1)
        self.protocol = protocol
        self.records = []
        self.test_starts = None
        self.batch_size = None

    def verify_loaders(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        observed = {}
        for name, loader in zip(("train", "val", "test"), result[:3]):
            dataset = loader.dataset
            starts = list(getattr(dataset, "valid_indices", getattr(dataset, "starts", [])))
            input_steps = int(dataset.input_steps)
            output_steps = int(dataset.output_steps)
            begin, end = self.protocol["expected_ranges"][name]
            expected = list(range(begin, end - input_steps - output_steps + 1))
            if (
                input_steps != self.protocol["input_steps"]
                or output_steps != self.protocol["output_steps"]
            ):
                raise ValueError(
                    f"Actual {name} input/output lengths differ from declared protocol"
                )
            if starts != expected:
                raise ValueError(f"Actual {name} window indices differ from the declared split")
            if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
                raise ValueError("Evaluation contract requires ordered, complete window traversal")
            observed[name] = [starts[0], starts[-1] + input_steps + output_steps]
            if name == "test":
                self.test_starts = starts
                self.batch_size = loader.batch_size
        self.protocol.update(
            {
                "ranges": observed,
                "ranges_verified": True,
                "ranges_source": "actual_loader_window_indices",
            }
        )
        return result

    def observe(self, truth, prediction):
        # Layer metrics contain fewer nodes; only capture the overall vector once.
        if np.asarray(truth).size == self.raw.shape[1]:
            self.records.append(
                (np.asarray(truth).reshape(-1).copy(), np.asarray(prediction).reshape(-1).copy())
            )

    def export(self, output, identity):
        if not self.records:
            return False
        if self.test_starts is None:
            raise ValueError("Prediction export requires verified actual test windows")
        horizon = self.protocol["output_steps"]
        if len(self.records) != len(self.test_starts) * horizon:
            raise ValueError("Observed evaluation count does not match test windows and horizons")
        truth_parts, prediction_parts = [], []
        cursor = 0
        for begin in range(0, len(self.test_starts), self.batch_size):
            batch = min(self.batch_size, len(self.test_starts) - begin)
            group = self.records[cursor : cursor + batch * horizon]
            for index, target in ((0, truth_parts), (1, prediction_parts)):
                values = np.stack([pair[index] for pair in group])
                target.append(values.reshape(horizon, batch, -1).transpose(1, 0, 2))
            cursor += batch * horizon
        reconstructed = np.concatenate(truth_parts)
        prediction = np.maximum(np.concatenate(prediction_parts), 0)
        input_steps = self.protocol["input_steps"]
        times = np.asarray(self.test_starts)[:, None] + input_steps + np.arange(horizon)[None, :]
        truth = self.raw[times]
        if not np.allclose(reconstructed, truth, rtol=2e-6, atol=0.01):
            raise ValueError("Legacy evaluation order/targets do not match exact raw targets")
        from evaluate import metric_rows

        metadata = {**identity, **self.protocol, "layers": [14, 15, 16, 17]}
        metadata.update(
            {
                "truth_policy": "exact_raw_arrays",
                "metric_aggregation": "window_horizon_mean",
                "clip_negative_predictions": True,
            }
        )
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "predictions.npz",
            truth=truth,
            prediction=prediction,
            window_starts=np.asarray(self.test_starts),
            target_time_indices=times,
        )
        rows = metric_rows(truth, prediction, metadata)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_json(output / "metadata.json", metadata)
        return True
