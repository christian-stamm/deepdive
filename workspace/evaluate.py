from __future__ import annotations

import logging

import torch
from torch import nn

from config import Config
from datasets import create_dataloaders
from models import AutoPilot
from trainer import Trainer, load_checkpoint
from train import build_model, configure_logging, get_device, seed_everything


def main() -> None:
    configure_logging()
    config = Config()
    seed_everything(config.seed)

    device = get_device()
    _, validation_loader = create_dataloaders(config)

    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    checkpoint = load_checkpoint(
        path=config.best_checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=False,
    )

    metrics = trainer.validate(validation_loader)

    logging.info("Loaded checkpoint from epoch: %s", checkpoint.get("epoch"))
    logging.info("Validation metrics: %s", metrics)


if __name__ == "__main__":
    main()
