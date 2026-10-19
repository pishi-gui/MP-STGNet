"""Training/evaluation steps, following the Graph WaveNet-style engine interface."""

from contextlib import nullcontext

import torch
from models import MPSTGNet
from scripts.util import shapes_to_adjacency


def build_model(config, shapes, device):
    return MPSTGNet(
        device=device,
        num_nodes=sum(h * w for h, w in shapes),
        layer_shapes=[tuple(s) for s in shapes],
        supports=shapes_to_adjacency(shapes, device),
        in_dim=4 if config["features"] == "decomposed" else 1,
        out_dim=config["output_steps"],
        **config["model"],
    ).to(device)


class Trainer:
    def __init__(self, config, shapes, device):
        self.device = device
        self.model = build_model(config, shapes, device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        self.criterion = torch.nn.L1Loss()
        self.use_amp = config["amp"] and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.clip = config["grad_clip"]

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()

    def run_epoch(self, loader, training):
        self.model.train(training)
        losses = []
        with torch.set_grad_enabled(training):
            for x, y, _ in loader:
                x, y = x.to(self.device), y.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    prediction = self.model(x)
                    if prediction.shape != y.shape:
                        raise ValueError(
                            f"Model/target shapes differ: {prediction.shape} vs {y.shape}; check receptive field"
                        )
                    loss = self.criterion(prediction, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite loss; stopping instead of saving an invalid checkpoint"
                    )
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                losses.append(float(loss.detach()))
        # Preserve the source training/early-stopping convention: mean of batch losses.
        return sum(losses) / len(losses)
