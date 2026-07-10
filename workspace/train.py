from __future__ import annotations

import logging
import random

import torch
from torch import nn
from tqdm import trange

from config import Config
from datasets import create_dataloaders
from models import AutoPilot
from trainer import Trainer, save_checkpoint


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: Config, device: torch.device) -> AutoPilot:
    model = AutoPilot(
        in_channels=config.channels,
        image_size=config.image_size,
        num_classes=config.num_classes,
        model_depth=config.model_depth,
        kernel_depth=config.kernel_depth,
        kernel_size=config.kernel_size,
        negative_slope=config.negative_slope,
    )
    return model.to(device)


def main() -> None:
    configure_logging()
    config = Config()
    seed_everything(config.seed)

    device = get_device()
    logging.info("Using device: %s", device)

    train_loader, validation_loader = create_dataloaders(config)
    model = build_model(config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=config.use_amp,
        gradient_clip_norm=config.gradient_clip_norm,
    )

    best_validation_loss = float("inf")

    progress = trange(config.epochs, desc="Training", unit="epoch")
    for epoch_index in progress:
        epoch = epoch_index + 1

        train_metrics = trainer.train_epoch(train_loader)
        validation_metrics = trainer.validate(validation_loader)

        save_checkpoint(
            path=config.last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=validation_metrics.loss,
            validation_accuracy=validation_metrics.accuracy,
        )

        if validation_metrics.loss < best_validation_loss:
            best_validation_loss = validation_metrics.loss
            save_checkpoint(
                path=config.best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_metrics.loss,
                validation_accuracy=validation_metrics.accuracy,
            )

        progress.set_postfix(
            {
                "train_loss": f"{train_metrics.loss:.4f}",
                "train_acc": f"{train_metrics.accuracy:.4f}",
                "val_loss": f"{validation_metrics.loss:.4f}",
                "val_acc": f"{validation_metrics.accuracy:.4f}",
            }
        )

        if epoch % config.log_every_n_epochs == 0:
            logging.info(
                "Epoch %03d | train: %s | validation: %s",
                epoch,
                train_metrics,
                validation_metrics,
            )

    logging.info("Best checkpoint saved to: %s", config.best_checkpoint_path)


if __name__ == "__main__":
    main()
