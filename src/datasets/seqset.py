from __future__ import annotations

from collections import OrderedDict
from typing import Any

from torch.utils.data import Dataset, default_collate


class SequenceSet(Dataset):
    """
    Lazily creates fixed-length sliding windows over a map-style Dataset.

    Properties
    ----------
    - Each requested index of `source` is accessed at most once per iteration.
    - Memory usage is O(seq_len).
    - Overlapping samples are retained in a deque.
    - Each yielded item has a leading sequence dimension.
    - Only complete windows are yielded.

    Important
    ---------
    Use num_workers=0. Multiple DataLoader worker processes have independent
    dataset instances and independent caches, so they cannot guarantee that an
    underlying source index is read only once globally.
    """

    def __init__(
        self,
        source: Dataset,
        seq_len: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        if not isinstance(source, Dataset):
            raise TypeError("source must be a Torch Dataset")
        if seq_len <= 0:
            raise ValueError("seq_len must be greater than 0")
        if stride <= 0:
            raise ValueError("stride must be greater than 0")

        self.source = source
        self.seq_len = seq_len
        self.stride = stride
        self.cache = OrderedDict()

        if len(self) < 1:
            raise ValueError(
                f"Source dataset is too small for the given seq_len ({seq_len})"
            )

    def __len__(self) -> int:
        """Number of complete windows."""
        n = len(self.source)

        if n < self.seq_len:
            return 0

        samplerange = n - self.seq_len
        return 1 + samplerange // self.stride

    def __getitem__(self, index: int) -> Any:
        n = len(self)

        if n <= index:
            raise IndexError("Index out of bounds")

        start = index * self.stride
        end = start + self.seq_len

        batch = [None] * self.seq_len  # Preallocate list for efficiency

        for i in range(start, end):
            if i not in self.cache:
                self.cache[i] = self.source[i]

            batch[i - start] = self.cache[i]

        while self.seq_len < len(self.cache):
            self.cache.popitem(last=False)

        return default_collate(batch)
