"""Configuration for quantum-circuit execution."""

from dataclasses import dataclass
import math


NOISE_SIMULATIONS = (
    "density_matrix",
    "per_shot_trajectories",
    "legacy_shared_trajectory",
)


@dataclass(frozen=True)
class ExecutionConfig:
    """Controls exact, sampled, and depolarizing circuit execution.

    ``noise_simulation`` only matters when ``depolarizing_probability > 0``.
    ``"density_matrix"`` (default) evolves the exact mixed state, so noise is
    averaged correctly and every measured shot is an independent sample; it
    also allows ``shots=None`` to obtain exact noisy expectations.
    ``"per_shot_trajectories"`` re-executes one Pauli trajectory per shot.
    ``"legacy_shared_trajectory"`` reproduces the historical behavior (one
    Qibo noise trajectory per circuit execution shared by every shot) and is
    kept only to reproduce frozen results.
    """

    shots: int | None = None
    depolarizing_probability: float = 0.0
    seed: int | None = None
    noise_simulation: str = "density_matrix"

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
        if self.noise_simulation not in NOISE_SIMULATIONS:
            allowed = ", ".join(NOISE_SIMULATIONS)
            raise ValueError(f"noise_simulation must be one of: {allowed}.")

    @property
    def noisy(self) -> bool:
        return self.depolarizing_probability > 0.0
