from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from config import Config


def create_synthetic_dataset(config: Config) -> TensorDataset:
    """Creates random images and random labels.

    This is intentionally synthetic. It is useful for testing the pipeline,
    but not for testing real generalization.
    """
    inputs = torch.rand(
        config.samples,
        config.channels,
        config.image_size,
        config.image_size,
    )
    labels = torch.randint(0, config.num_classes, (config.samples,))
    return TensorDataset(inputs, labels)


def split_dataset(config: Config, dataset: Dataset) -> tuple[Dataset, Dataset]:
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    validation_size = int(len(dataset) * config.validation_fraction)
    train_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(config.seed)

    return random_split(
        dataset,
        lengths=[train_size, validation_size],
        generator=generator,
    )


def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    dataset = create_synthetic_dataset(config)
    train_dataset, validation_dataset = split_dataset(config, dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader
