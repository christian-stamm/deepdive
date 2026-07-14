from abc import abstractmethod

import torch
import random
from typing import Callable

from workspace.checkpoint import Checkpointer

from .callback import Callback
from .state import State
from .config import Config
from torch.utils.data import DataLoader
from tqdm import trange


@abstractmethod
def train_epoch(
    config: Config,
    state: State,
    trainloader: DataLoader,
) -> dict[str, float]:
    pass


@abstractmethod
def val_epoch(
    config: Config,
    state: State,
    valloader: DataLoader,
) -> dict[str, float]:
    pass


class Trainer:

    def __init__(
        self,
        config: Config,
        state: State,
        train_epoch_fn: Callable = train_epoch,
        val_epoch_fn: Callable = val_epoch,
        callbacks: list[Callback] | None = None,
    ):
        self.config = config
        self.state = state.to(config.device)
        self.seed_everything(config.seed)

        self._train_epoch = train_epoch_fn
        self._val_epoch = val_epoch_fn
        self._callbacks = callbacks if callbacks else []

        self._callbacks.append(
            Checkpointer(
                directory=config.checkpoint_savedir,
                interval=config.checkpoint_interval,
                restore=config.checkpoint_restore,
                device=config.device,
            )
        )

    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def fit(self, trainloader: DataLoader, valloader: DataLoader, epochs: int):
        progress_bar = trange(epochs, desc="Training", unit="epoch", leave=True)

        for epoch in progress_bar:
            self.state.epoch = epoch

            for callback in self._callbacks:
                callback.on_epoch_begin(self.state)
                callback.on_train_begin(self.state)

            train_metrics = self._train_epoch(self.config, self.state, trainloader)
            self.state.history.train.append(train_metrics)

            for callback in self._callbacks:
                callback.on_train_end(self.state)
                callback.on_val_begin(self.state)

            val_metrics = self._val_epoch(self.config, self.state, valloader)
            self.state.history.val.append(val_metrics)

            if val_metrics <= self.state.score:
                self.state.score = val_metrics

            for callback in self._callbacks:
                callback.on_val_end(self.state)
                callback.on_epoch_end(self.state)

            progress_bar.set_postfix(
                {
                    "train_loss": train_metrics,
                    "val_loss": val_metrics,
                }
            )
