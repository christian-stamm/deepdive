import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .seqset import MultiDriveSequenceSet, SequenceSet


class DrivingSet(pl.LightningDataModule):
    def __init__(
        self,
        dataset: SequenceSet | MultiDriveSequenceSet,
        ds_split: tuple = (0.80, 0.15, 0.05),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(dataset, (SequenceSet, MultiDriveSequenceSet)):
            raise TypeError("dataset must be a SequenceSet or MultiDriveSequenceSet")
        if len(ds_split) != 3 or any(split < 0 for split in ds_split):
            raise ValueError("ds_split must contain three non-negative fractions")
        if sum(ds_split) > 1:
            raise ValueError("ds_split fractions must sum to at most 1")

        ranges = (
            dataset.drive_ranges()
            if isinstance(dataset, MultiDriveSequenceSet)
            else [range(len(dataset))]
        )
        train_indices = []
        val_indices = []
        test_indices = []
        for drive_range in ranges:
            start = drive_range.start
            total = len(drive_range)
            gap = dataset.seq_len
            split_total = sum(ds_split)
            available = max(0, total - 2 * gap)
            train_length = int(available * ds_split[0] / split_total)
            val_length = int(available * ds_split[1] / split_total)
            train_end = start + train_length
            val_start = min(start + total, train_end + gap)
            val_end = min(start + total, val_start + val_length)
            test_start = min(start + total, val_end + gap)
            train_indices.extend(range(start, train_end))
            val_indices.extend(range(val_start, val_end))
            test_indices.extend(range(test_start, start + total))

        trainset = Subset(dataset, train_indices)
        valset = Subset(dataset, val_indices)
        testset = Subset(dataset, test_indices)
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

    def _build_dloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=0 < self.num_workers,
        )
