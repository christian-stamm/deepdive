from dataclasses import dataclass, field

from torch import nn, optim
import torch

@dataclass
class History:
    train: list[dict[str, float]] = field(default_factory=list)
    val: list[dict[str, float]] = field(default_factory=list)


@dataclass
class State:
    model: nn.Module | None = None
    criterion: nn.Module | None = None
    optimizer: optim.Optimizer | None = None
    scheduler: optim.lr_scheduler.LRScheduler | None = None
    epoch: int = 0
    score: float = float("inf")
    history: History = field(default_factory=History)

    @property
    def is_best(self) -> bool:
        if not self.history.val:
            return False
        return self.history.val[-1] == self.score
    
    def to(self, device: torch.device) -> "State":
        if self.model is not None:
            self.model.to(device)
        if self.criterion is not None:
            self.criterion.to(device)
        return self


    def to_dict(self) -> dict:
        state = dict()

        if self.model is not None:
            state["model_state_dict"] = self.model.state_dict()
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        state.update(
            {
                "epoch": self.epoch,
                "best_metric": self.score,
                "train_history": self.history.train,
                "val_history": self.history.val,
            }
        )

        return state

    def from_dict(self, state_dict: dict) -> None:
        self.epoch = state_dict.get("epoch", 0)
        self.score = state_dict.get("best_metric", float("inf"))
        self.history.train = state_dict.get("train_history", [])
        self.history.val = state_dict.get("val_history", [])

        if self.model is not None and "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
