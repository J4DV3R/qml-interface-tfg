"""Configuration for quantum-circuit execution."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExecutionConfig:
    """Controls exact, sampled, and depolarizing circuit execution."""

    shots: int | None = None
    depolarizing_probability: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        if self.shots is not None and (
            not isinstance(self.shots, int) or isinstance(self.shots, bool) or self.shots < 1
        ):
            raise ValueError("shots must be None or a positive integer.")
        if (
            not math.isfinite(self.depolarizing_probability)
            or not 0.0 <= self.depolarizing_probability <= 1.0
        ):
            raise ValueError("depolarizing_probability must be between 0 and 1.")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or None.")

    @property
    def noisy(self) -> bool:
        return self.depolarizing_probability > 0.0
