from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_right
from pathlib import Path
from typing import Any, Sequence

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


class MultiDriveSequenceSet(Dataset):
    """Expose independent sequence windows from multiple drive directories."""

    def __init__(
        self,
        sources: Sequence[Dataset],
        seq_len: int,
        stride: int = 1,
        roots: Sequence[Path] | None = None,
    ) -> None:
        super().__init__()

        if not sources:
            raise ValueError("sources must contain at least one drive")

        self.datasets = []
        valid_roots = []
        for index, source in enumerate(sources):
            if isinstance(source, SequenceSet):
                sequence_set = source
            elif len(source) < seq_len:
                continue
            else:
                sequence_set = SequenceSet(source, seq_len, stride)
            self.datasets.append(sequence_set)
            if roots is not None:
                valid_roots.append(roots[index])
        self.seq_len = seq_len
        self.stride = stride
        self.roots = valid_roots

        self._ends = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self._ends.append(total)

        if total == 0:
            raise ValueError("No drive contains a complete sequence window")

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds")

        drive_index = bisect_right(self._ends, index)
        previous_end = 0 if drive_index == 0 else self._ends[drive_index - 1]
        return self.datasets[drive_index][index - previous_end]

    def drive_ranges(self) -> list[range]:
        ranges = []
        start = 0
        for end in self._ends:
            ranges.append(range(start, end))
            start = end
        return ranges
