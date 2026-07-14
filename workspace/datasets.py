import struct
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .config import Config


def _read_idx_images(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        magic, num_images, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected IDX image magic number in {path}: {magic}")
        data = torch.frombuffer(handle.read(), dtype=torch.uint8)
    return data.view(num_images, rows, cols).clone()


def _read_idx_labels(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        magic, num_labels = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected IDX label magic number in {path}: {magic}")
        data = torch.frombuffer(handle.read(), dtype=torch.uint8)
    return data.view(num_labels).clone().long()


class MNISTDataset(Dataset):
    def __init__(
        self,
        config: Config,
        train: bool = True,
        transform: Any = None
    ):
        self.config = config
        self.transform = transform
        root = Path("data/datasets/MNISTDataset/raw")
        image_file = root / ("train-images.idx3-ubyte" if train else "t10k-images.idx3-ubyte")
        label_file = root / ("train-labels.idx1-ubyte" if train else "t10k-labels.idx1-ubyte")

        self.images = _read_idx_images(image_file)
        self.labels = _read_idx_labels(label_file)

        if len(self.images) != len(self.labels):
            raise ValueError("MNIST image and label counts do not match")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[index].unsqueeze(0).float() / 255.0
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    @property
    def loader(self) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
