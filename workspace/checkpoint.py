from pathlib import Path

import torch


class CheckpointManager:

    def __init__(
        self,
        directory: Path,
    ):
        self.directory = directory
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        name: str,
        state: dict,
    ):

        torch.save(
            state,
            self.directory / name,
        )

    def load(
        self,
        path: Path,
        device,
    ):

        return torch.load(
            path,
            map_location=device,
        )
