
from pathlib import Path

import torch

from workspace.callback import Callback
from workspace.state import State




class Checkpointer(Callback):

    def __init__(
        self,
        directory: Path = Path(".data/checkpoints"),
        interval: int = 1,
        restore: str = "none",
        device: torch.device = torch.device("cpu"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.interval = interval
        self.restore = restore
        self.device = device

    @property
    def best(self) -> Path:
        return self.directory / "best.pt"

    @property
    def last(self) -> Path:
        return self.directory / "last.pt"

    def on_epoch_begin(self, state: State):
        if self.restore == "best":
            self._restore_checkpoint(state, self.best)
        elif self.restore == "last":
            self._restore_checkpoint(state, self.last)
        self.restore = "none"

    def on_epoch_end(self, state: State):
        save_best = state.is_best
        save_state = ((state.epoch + 1) % self.interval) == 0

        if save_state:
            self.save(self.last, state.to_dict())
        if save_best:
            self.save(self.best, state.to_dict())

    def _restore_checkpoint(self, state: State, name: Path):
        try:
            state_dict = self.load(name, device=self.device)
            state.from_dict(state_dict)
        except FileNotFoundError:
            print(f"No checkpoint found at {name}")
            return

    def save(self, path: Path, state: dict):
        print(f"Saving checkpoint to {path}")
        torch.save(state, path)

    def load(self, path: Path, device: torch.device):
        print(f"Loading checkpoint from {path}")
        return torch.load(path, map_location=device)
