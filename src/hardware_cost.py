"""Circuit and time budget estimates for running the variational QCNN on a QPU.

Circuit counts follow exactly the accounting of ``src/variational_training.py``:

* prediction / evaluation: 1 circuit per sample;
* GD / Adam (parameter-shift): ``1 + 2 * sum(occurrences)`` circuits per
  training sample and step (33 for the 4-qubit QCNN);
* SPSA: 2 circuits per training sample and step;
* sequential (Rotosolve/NFT-like exact reconstruction): ``sum(2 R + 1)`` over
  the weights, with ``R`` the number of occurrences of each weight (40 for the
  4-qubit QCNN);
* validation is evaluated once per step plus the initial step 0.

Times are **assumptions**, not measurements. The only figure published for the
BSC chips is the RES application guide's reference (a depth-1000, 10 000-shot
job takes about 20 s, most of it connection overhead; 1000 independent such
jobs take about 5.5 h). ``TimeModel`` keeps the per-job overhead and the
per-shot time separate so that both an unbatched upper bound and a batched
scenario can be reported; every report states which model was used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

OPTIMIZERS = ("gd", "adam", "spsa", "sequential")


def weight_occurrences_4q() -> list[int]:
    """Occurrences of each shared weight per layer for the 4-qubit QCNN."""
    return [3, 1]


def circuits_per_sample_step(optimizer: str, occurrences: list[int] | None = None) -> int:
    """Optimization circuits per training sample and step."""
    occurrences = occurrences or weight_occurrences_4q()
    weights_per_layer = 4
    if optimizer in ("gd", "adam"):
        return 1 + 2 * weights_per_layer * sum(occurrences)
    if optimizer == "spsa":
        return 2
    if optimizer == "sequential":
        return sum(weights_per_layer * (2 * r + 1) for r in occurrences)
    raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}.")


@dataclass(frozen=True)
class TimeModel:
    """Wall-time model of a QPU job queue (all values are assumptions).

    ``seconds_per_job`` is the fixed overhead of one submission/connection,
    ``seconds_per_shot`` the repetition time of one shot and
    ``circuits_per_job`` how many circuits are sent per submission.
    """

    name: str
    seconds_per_job: float
    seconds_per_shot: float
    circuits_per_job: int
    source: str

    def hours(self, circuits: int, shots: int) -> float:
        jobs = -(-circuits // self.circuits_per_job)  # ceil
        return (jobs * self.seconds_per_job + circuits * shots * self.seconds_per_shot) / 3600.0


TIME_MODELS = {
    "unbatched-upper-bound": TimeModel(
        name="unbatched-upper-bound",
        seconds_per_job=20.0,
        seconds_per_shot=0.0,
        circuits_per_job=1,
        source="RES guide reference: ~20 s per job (depth 1000, 10k shots), dominated by overhead; "
               "each circuit treated as its own job. Upper bound.",
    ),
    "batched-assumption": TimeModel(
        name="batched-assumption",
        seconds_per_job=20.0,
        seconds_per_shot=200e-6,
        circuits_per_job=500,
        source="Assumption, not a BSC figure: 20 s overhead per connection, 500 circuits per "
               "connection and 200 microseconds per shot (typical superconducting repetition time). "
               "Must be replaced by measured values once access is granted.",
    ),
}


@dataclass(frozen=True)
class CostEstimate:
    scenario: str
    description: str
    circuits: int
    shots_per_circuit: int
    hours: dict[str, float]

    @property
    def total_shots(self) -> int:
        return self.circuits * self.shots_per_circuit

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_shots"] = self.total_shots
        return data


def _hours(circuits: int, shots: int) -> dict[str, float]:
    return {name: model.hours(circuits, shots) for name, model in TIME_MODELS.items()}


def readout_calibration_circuits(measured_qubits: int = 1) -> int:
    """|0> and |1> preparations per measured qubit."""
    return 2 * measured_qubits


def inference_cost(
    scenario: str,
    *,
    n_validation: int,
    n_test: int,
    checkpoints: int,
    shots: int,
    calibration_repeats: int = 1,
) -> CostEstimate:
    """Evaluate trained checkpoints on validation and test samples."""
    circuits = checkpoints * (n_validation + n_test) + calibration_repeats * readout_calibration_circuits()
    return CostEstimate(
        scenario,
        f"Inference of {checkpoints} checkpoint(s) on {n_validation} validation + {n_test} test samples, "
        f"plus readout calibration ({calibration_repeats}x).",
        circuits,
        shots,
        _hours(circuits, shots),
    )


def training_cost(
    scenario: str,
    *,
    optimizer: str,
    n_train: int,
    n_validation: int,
    steps: int,
    shots: int,
    runs: int = 1,
    minibatch: int | None = None,
) -> CostEstimate:
    """Train ``runs`` models for ``steps`` steps (validation at every step and step 0)."""
    samples = minibatch if minibatch is not None else n_train
    per_run = steps * samples * circuits_per_sample_step(optimizer) + (steps + 1) * n_validation
    circuits = runs * per_run
    batch = f", minibatch {minibatch}" if minibatch is not None else ""
    return CostEstimate(
        scenario,
        f"{runs} training run(s), {optimizer}, {steps} steps, {samples} samples per step{batch}, "
        f"validation ({n_validation}) every step.",
        circuits,
        shots,
        _hours(circuits, shots),
    )


SPLIT_SIZES = {
    # From the persisted partition summaries (70/15/15 stratified).
    "breast-cancer": {"train": 398, "validation": 85, "test": 86},
    "iris": {"train": 70, "validation": 15, "test": 15},
}


def default_scenarios(shots: int = 1000) -> list[CostEstimate]:
    """Scenarios of the BSC plan (phase 3 inference, phase 4 short training)."""
    bc, iris = SPLIT_SIZES["breast-cancer"], SPLIT_SIZES["iris"]
    return [
        inference_cost("H1-smoke", n_validation=0, n_test=iris["test"], checkpoints=1, shots=shots),
        inference_cost("H1-iris", n_validation=iris["validation"], n_test=iris["test"], checkpoints=8,
                       shots=shots, calibration_repeats=4),
        inference_cost("H1-breast-cancer", n_validation=bc["validation"], n_test=bc["test"], checkpoints=8,
                       shots=shots, calibration_repeats=4),
        training_cost("H2-iris-sequential-10-steps", optimizer="sequential", n_train=iris["train"],
                      n_validation=iris["validation"], steps=10, shots=shots),
        training_cost("H2-iris-spsa-100-steps-mb10", optimizer="spsa", n_train=iris["train"],
                      n_validation=iris["validation"], steps=100, shots=shots, minibatch=10),
        training_cost("H2-iris-adam-100-steps (reference, not proposed)", optimizer="adam",
                      n_train=iris["train"], n_validation=iris["validation"], steps=100, shots=shots),
        training_cost("simulation-equivalent: BC adam 100 steps (not proposed)", optimizer="adam",
                      n_train=bc["train"], n_validation=bc["validation"], steps=100, shots=shots),
    ]


__all__ = [
    "CostEstimate",
    "OPTIMIZERS",
    "SPLIT_SIZES",
    "TIME_MODELS",
    "TimeModel",
    "circuits_per_sample_step",
    "default_scenarios",
    "inference_cost",
    "readout_calibration_circuits",
    "training_cost",
    "weight_occurrences_4q",
]
