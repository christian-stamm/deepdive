from abc import abstractmethod

import lightning.pytorch as pl
import torch
from torch import nn, optim


class ClassifierNet(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        lr_scheduler_step_size: int = 5,
        lr_scheduler_gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()

        batch_size = labels.size(0)

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_acc",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx: int, dloader_idx: int = 0) -> torch.Tensor:
        images, _ = batch
        logits = self(images)
        return logits.argmax(dim=1)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.lr_scheduler_step_size,
            gamma=self.hparams.lr_scheduler_gamma,
        )

        return {
            "optimizer": self.optimizer,
            # "lr_scheduler": {
            #     "scheduler": self.scheduler,
            #     "interval": "epoch",
            #     "frequency": 1,
            # },
        }
