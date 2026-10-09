import logging

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datasets.comma2k19.comma2k19 import CommaDataset
from datasets.driveset import DrivingSet
from datasets.seqset import MultiDriveSequenceSet
from models.minipilot.model import MiniPilot
from mp_config import Config
from pathlib import Path


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
        name="my_model",
        save_dir=log_dir,
    )

    return logger


def seed_everything(seed: int = 42) -> None:
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_drives(config: Config) -> MultiDriveSequenceSet:
    if config.data.all_drives:
        roots = sorted(path.parent for path in config.data.rootdir.rglob("video.hevc"))
    else:
        roots = [config.data.rootdir / config.data.sample]

    if not roots:
        raise FileNotFoundError(f"No drive videos found under {config.data.rootdir}")

    datasets = [
        CommaDataset(root, config.model.num_future_steps, transform=None)
        for root in roots
    ]
    sequence_set = MultiDriveSequenceSet(
        datasets,
        seq_len=config.model.num_future_steps,
        stride=1,
        roots=roots,
    )
    print(f"Loaded {len(roots)} drives with {len(sequence_set)} sequence windows.")
    return sequence_set


def restore_weights(model: MiniPilot, checkpoint: Path | None) -> Path | None:
    if checkpoint is None:
        return None

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        print(
            "Checkpoint architecture differs from the current model. "
            "Loaded compatible weights and will start with a fresh optimizer."
        )
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        return None

    return checkpoint


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

    checkpoint = Path(config.training.checkpoint.rootdir) / config.training.checkpoint.restore
    checkpoint = checkpoint.with_suffix(".ckpt") if checkpoint.suffix != ".ckpt" else checkpoint
    if checkpoint.exists():
        print(f"Restoring from checkpoint: {checkpoint}")
    else:
        print(f"Checkpoint not found: {checkpoint}. Starting from scratch.")
        checkpoint = None

    use_cuda = config.runtime.device.startswith("cuda") and torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
        max_epochs=config.training.max_epochs,
        log_every_n_steps=config.runtime.logging.log_every_n_steps,
        callbacks=[checkpointer],
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=logger,
        precision=config.runtime.precision if use_cuda else 32,
    )

    model = MiniPilot(config)
    checkpoint = restore_weights(model, checkpoint)
    sequence_set = load_drives(config)
    if sequence_set.datasets[0].source.telemetry.shape[-1] != config.model.telemetry_dim:
        raise ValueError(
            f"Configured telemetry_dim={config.model.telemetry_dim}, "
            f"but dataset provides {sequence_set.datasets[0].source.telemetry.shape[-1]} features"
        )

    datamodule = DrivingSet(
        sequence_set,
        ds_split=(0.8, 0.15, 0.05),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)

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
