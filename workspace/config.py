import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torchvision import transforms


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Model
    layer_depth: int = 3
    kernel_depth: int = 64

    # Data
    dataset_rootdir: Path = Path("data/datasets")
    dataset_transform: torch.nn.Module = transforms.ToTensor()

    # Runtime
    num_workers: int = 4

    # Training
    epochs: int = 10
    batch_size: int = 2048
    learning_rate: float = 5e-4
    weight_decay: float = 0.0

    # Artifacts
    checkpoint_rootdir: Path = Path("data/checkpoints")
    checkpoint_interval: int = 1
    checkpoint_restore: str = "last"

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
