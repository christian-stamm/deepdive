from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    epoch: int = 0
    best_metric: float = float("inf")

    train: list[dict[str, float]] = field(default_factory=list)
    val: list[dict[str, float]] = field(default_factory=list)
