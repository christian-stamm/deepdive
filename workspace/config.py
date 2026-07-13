from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Data
    path: Path = Path("data")

    # Model
    model_depth: int = 1
    kernel_depth: int = 16
    kernel_size: int = 3

    # Optimization
    epochs: int = 50
    batch_size: int = 64
    use_amp: bool = True
    learning_rate: float = 5e-4
    weight_decay: float = 0.0

    # Runtime
    num_workers: int = 2
    use_amp: bool = True
    log_every_n_epochs: int = 1

    # Artifacts
    checkpoint_dir: Path = Path("checkpoints")
    best_checkpoint_name: str = "best_model.pt"
    last_checkpoint_name: str = "last_model.pt"

    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def seed_everything(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.best_checkpoint_name

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.last_checkpoint_name

    def __str__(self) -> str:
        return "Config " + repr(self)

    def __repr__(self) -> str:
        state = asdict(self)
        state.update(
            {
                "Torch version:": torch.__version__,
                "CUDA supported:": torch.cuda.is_available(),
            }
        )

        return json.dumps(state, indent=4, default=str)
