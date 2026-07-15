import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _resolve_path(value: str | Path) -> Path:
        return Path(value).expanduser().resolve()


class RuntimeConfig(StrictModel):
    seed: int = 42
    pin_memory: bool = True
    num_workers: int = Field(default=0, ge=0)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    version: str = Field(default=torch.__version__)

    @model_validator(mode="after")
    def _normalize(self) -> "RuntimeConfig":
        max_worker = max(0, (os.cpu_count() or 1) - 1)
        num_workers = min(self.num_workers, max_worker)
        self.num_workers = num_workers
        return self


class DataConfig(StrictModel):
    rootdir: Path = Path("data/datasets")
    dataset: str = "mnist"

    @model_validator(mode="after")
    def _normalize(self) -> "DataConfig":
        self.rootdir = self._resolve_path(self.rootdir)
        return self


class ModelConfig(StrictModel):
    layer_depth: PositiveInt = Field(default=3, ge=1)
    kernel_depth: PositiveInt = Field(default=64, ge=1)


class SchedulerConfig(StrictModel):
    step_size: PositiveInt = 5
    gamma: PositiveFloat = 0.1


class OptimizerConfig(StrictModel):
    learning_rate: PositiveFloat = 5e-4
    weight_decay: float = Field(default=0.0, ge=0.0)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class CheckpointConfig(StrictModel):
    rootdir: Path = Path("data/checkpoints")
    interval: PositiveInt = 1
    restore: str = "last"

    @model_validator(mode="after")
    def _normalize(self) -> "CheckpointConfig":
        self.rootdir = self._resolve_path(self.rootdir)
        return self


class TrainingConfig(StrictModel):
    max_epochs: PositiveInt = 10
    batch_size: PositiveInt = Field(default=64, ge=1)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class LoggingConfig(StrictModel):
    level: str = "INFO"
    log_every_n_steps: PositiveInt = 10


class Config(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(payload, dict):
            raise TypeError("YAML config root must be a mapping")
        return cls.from_dict(payload)

    @classmethod
    def from_yaml_stack(cls, paths: list[str | Path]) -> "Config":
        if not paths:
            raise ValueError("paths must contain at least one YAML file")

        merged = OmegaConf.create()
        for path in paths:
            merged = OmegaConf.merge(merged, OmegaConf.load(path))

        payload = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(payload, dict):
            raise TypeError("Merged YAML config root must be a mapping")
        return cls.from_dict(payload)

    @classmethod
    def from_hydra(cls, cfg: Any) -> "Config":
        payload = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(payload, dict):
            raise TypeError("Hydra config root must be a mapping")
        return cls.from_dict(payload)

    def override(self, **overrides: Any) -> "Config":
        return self.model_copy(update=overrides, deep=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(self), encoding="utf-8")

    def __str__(self) -> str:
        return "Config " + repr(self)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, default=str)
