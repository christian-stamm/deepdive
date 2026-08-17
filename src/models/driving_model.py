from abc import abstractmethod

import lightning.pytorch as pl
import torch
from torch import nn, optim

from mp_config import Config


class DrivingModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        self.config = config

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        clouds: torch.Tensor,
        ego_state: torch.Tensor,
        ego_intent: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the training_step method.")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement the validation_step method."
        )

    @torch.no_grad()
    def predict_step(self, batch, batch_idx: int, dloader_idx: int = 0) -> torch.Tensor:
        images, clouds, ego_state, ego_intent = batch
        return self(images, clouds, ego_state, ego_intent)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.training.optimizer.learning_rate,
            weight_decay=self.config.training.optimizer.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.training.optimizer.scheduler.step_size,
            gamma=self.config.training.optimizer.scheduler.gamma,
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
