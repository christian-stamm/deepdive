from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Data
    samples: int = 1_000
    channels: int = 3
    image_size: int = 64
    num_classes: int = 100
    validation_fraction: float = 0.2

    # Model
    model_depth: int = 1
    kernel_depth: int = 16
    kernel_size: int = 3
    negative_slope: float = 0.01

    # Optimization
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0

    # Runtime
    num_workers: int = 0
    use_amp: bool = True
    log_every_n_epochs: int = 1

    # Artifacts
    checkpoint_dir: Path = Path("checkpoints")
    best_checkpoint_name: str = "best_model.pt"
    last_checkpoint_name: str = "last_model.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.best_checkpoint_name

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.last_checkpoint_name
