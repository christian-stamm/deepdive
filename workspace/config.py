from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

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

    # Artifacts
    checkpoint_savedir: Path = Path("data/checkpoints")
    checkpoint_interval: int = 1
    checkpoint_restore: str = "last"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
