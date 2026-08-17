import logging

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datasets.comma2k19.comma2k19 import CommaDataset
from datasets.driveset import DrivingSet
from datasets.seqset import SequenceSet
from models.minipilot.model import MiniPilot
from mp_config import Config


def load_cfg(path: str = "res/config.yaml") -> Config:
    config = Config()

    try:
        config = Config.from_yaml(path)
        print("Config file loaded successfully.")
    except FileNotFoundError:
        config.to_yaml(path)
        print("Config file not found. Saving default.")

    print("\nLoaded", config, "\n")
    return config


def setup_logger(log_dir: str = "res/logs") -> TensorBoardLogger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="deepdive",
    )

    return logger


def seed_everything(seed: int = 42) -> None:
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    logger = setup_logger()
    config = load_cfg()

    seed_everything(config.runtime.seed)
    torch.set_float32_matmul_precision("medium")

    checkpointer = ModelCheckpoint(
        dirpath=config.training.checkpoint.rootdir,
        filename="minipilot-{epoch:02d}",
        save_last=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="cuda",
        devices=[0],
        max_epochs=config.training.max_epochs,
        log_every_n_steps=config.runtime.logging.log_every_n_steps,
        callbacks=[checkpointer],
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=logger,
        precision="16-mixed",
    )

    model = MiniPilot(config)
    comset = CommaDataset(
        config.data.rootdir / config.data.sample,
        config.model.num_future_steps,
        transform=None,
    )
    seqset = SequenceSet(
        comset,
        seq_len=config.model.num_future_steps,
        stride=1,
    )

    datamodule = DrivingSet(
        seqset,
        ds_split=(0.8, 0.15, 0.05),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    trainer.fit(model, datamodule=datamodule)

    metrics = trainer.validate(
        model,
        datamodule=datamodule,
        ckpt_path="last",
        verbose=False,
        weights_only=False,
    )

    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()
