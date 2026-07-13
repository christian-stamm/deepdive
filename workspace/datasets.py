from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import Config


class MNISTDataset(datasets.MNIST):
    def __init__(
        self,
        config: Config,
        train: bool = True,
        transform: Any = transforms.ToTensor(),
    ):
        super().__init__(
            root=config.path,
            train=train,
            download=True,
            transform=transform,
        )

        self.config = config

    @property
    def loader(self) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
