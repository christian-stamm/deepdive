from abc import abstractmethod

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


class DataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.train_ds: Dataset = None
        self.val_ds: Dataset = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = self._build_trainset()
            self.val_ds = self._build_valset()

    def train_dataloader(self) -> DataLoader:
        return self._build_dloader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_dloader(self.val_ds, shuffle=False)

    @abstractmethod
    def _build_trainset(self) -> Dataset:
        raise NotImplementedError("Subclasses must implement _build_trainset")

    @abstractmethod
    def _build_valset(self) -> Dataset:
        raise NotImplementedError("Subclasses must implement _build_valset")

    def _build_dloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup('fit') before requesting a dataloader.")

        workers = self.config.runtime.num_workers
        persistent = 0 < workers

        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=self.config.runtime.pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent,
        )
