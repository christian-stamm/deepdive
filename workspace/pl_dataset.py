from abc import abstractmethod
from pathlib import Path

import lightning.pytorch as pl
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
            path = self.config.dataset_rootdir
            self.train_ds = self._build_trainset()
            self.val_ds = self._build_valset()

    def train_dataloader(self) -> DataLoader:
        return self._build_dloader(self.train_ds)

    def val_dataloader(self) -> DataLoader:
        return self._build_dloader(self.val_ds)

    @abstractmethod
    def _build_trainset(self) -> Dataset:
        raise NotImplementedError("Subclasses must implement _build_trainset")

    @abstractmethod
    def _build_valset(self) -> Dataset:
        raise NotImplementedError("Subclasses must implement _build_valset")

    def _build_dloader(self, dataset: Dataset) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup('fit') before requesting a dataloader.")

        workers = self.config.num_workers
        persistent = workers > 0

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.dataset_shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=persistent,
        )
