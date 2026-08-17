import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split

from .seqset import SequenceSet


class DrivingSet(pl.LightningDataModule):
    def __init__(
        self,
        dataset: SequenceSet,
        ds_split: tuple = (0.80, 0.15, 0.05),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(dataset, SequenceSet):
            raise TypeError("dataset must be a SequenceSet")

        trainset, valset, testset = random_split(dataset, ds_split)
        self.trainset = trainset
        self.valset = valset
        self.testset = testset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()

    def setup(self, stage: str | None = None) -> None:
        print(f"Setting up {self.__class__.__name__} for stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        return self._build_dloader(self.trainset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_dloader(self.valset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_dloader(self.testset, shuffle=False)

    def _build_dloader(self, dataset: SequenceSet, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=0 < self.num_workers,
        )
