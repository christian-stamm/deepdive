from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float

    def __str__(self) -> str:
        return f"loss={self.loss:.4f}, acc={self.accuracy:.4f}"


class ClassificationMeter:
    """Accumulates loss and accuracy over one full epoch."""

    def __init__(self) -> None:
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = targets.size(0)
        predictions = logits.argmax(dim=1)

        self.total_loss += float(loss.detach().item()) * batch_size
        self.total_correct += int((predictions == targets).sum().item())
        self.total_samples += batch_size

    def compute(self) -> EpochMetrics:
        if self.total_samples == 0:
            raise RuntimeError("Cannot compute metrics without samples.")

        return EpochMetrics(
            loss=self.total_loss / self.total_samples,
            accuracy=self.total_correct / self.total_samples,
        )
