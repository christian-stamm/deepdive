from config import Config
from history import TrainingHistory
from torch import nn, optim


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler | None,
        criterion: nn.Module,
        train_step,
        val_step,
        config: Config,
        callbacks=None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.train_step = train_step
        self.val_step = val_step

        self.callbacks = callbacks or []
        self.history = TrainingHistory()

    def fit(
        self,
        trainloader,
        valloader,
        epochs,
    ):

        for epoch in range(epochs):

            self.history.epoch = epoch
            train_metrics = self.train_epoch(trainloader)
            val_metrics = self.validate(valloader)

            self.history.train.append(train_metrics)
            self.history.val.append(val_metrics)

            kill_request = False
            for callback in self.callbacks:
                kill_request |= callback.on_epoch_end(self)

            if kill_request:
                print("Training stopped by callback request.")
                break
