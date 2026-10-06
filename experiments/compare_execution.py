"""Reproducible sweep comparing exact, sampled, and noisy QCNN execution."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time

import torch

from src.execution import ExecutionConfig
from src.classical_baseline import run_logistic_baseline
from src.datasets import SplitConfig, load_binary_iris_splits
from src.hybrid_model import QCNNModel
from src.metrics import classification_metrics
from src.training import TrainingConfig, train_qcnn


@dataclass(frozen=True)
class SweepConfig:
    """Experimental factors for the execution comparison."""

    shots_values: tuple[int, ...] = (100, 500, 1000)
    noise_values: tuple[float, ...] = (0.0, 0.01, 0.05)
    seeds: tuple[int, ...] = (1, 2, 3)
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    dataset_seed: int = 42

    def __post_init__(self):
        if not self.seeds:
            raise ValueError("seeds must not be empty.")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in self.seeds):
            raise ValueError("seeds must contain integers.")
        if any(not isinstance(shots, int) or isinstance(shots, bool) or shots < 1
               for shots in self.shots_values):
            raise ValueError("shots_values must contain positive integers.")
        if any(not 0.0 <= noise <= 1.0 for noise in self.noise_values):
            raise ValueError("noise_values must be between 0 and 1.")
        SplitConfig(
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            seed=self.dataset_seed,
        )


def _condition_name(shots: int | None, noise: float) -> str:
    if shots is None:
        return "exact"
    return f"shots_{shots}_noise_{noise:g}"


def _aggregate(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        grouped.setdefault(str(run["name"]), []).append(run)

    summaries = []
    for name, condition_runs in grouped.items():
        summary = {
            "name": name,
            "execution": condition_runs[0]["execution"],
            "training": condition_runs[0]["training"],
            "repetitions": len(condition_runs),
        }
        for metric in (
            "initial_loss",
            "final_loss",
            "validation_loss",
            "elapsed_seconds",
        ):
            values = [float(run[metric]) for run in condition_runs]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.pstdev(values)
        for split in ("validation", "test"):
            for metric in ("accuracy", "precision", "recall", "f1"):
                values = [
                    float(run[f"{split}_metrics"][metric])
                    for run in condition_runs
                ]
                summary[f"{split}_{metric}_mean"] = statistics.fmean(values)
                summary[f"{split}_{metric}_std"] = statistics.pstdev(values)
        summaries.append(summary)
    return summaries


def run_sweep(
    sweep_config: SweepConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> dict[str, object]:
    """Run all conditions and return individual runs plus aggregate statistics."""
    sweep = sweep_config or SweepConfig()
    training = training_config or TrainingConfig(
        epochs=30,
        learning_rate=0.03,
        optimizer="adam",
    )
    dataset = load_binary_iris_splits(
        seed=sweep.dataset_seed,
        train_fraction=sweep.train_fraction,
        validation_fraction=sweep.validation_fraction,
        test_fraction=sweep.test_fraction,
    )
    baseline = run_logistic_baseline(
        dataset.train_features,
        dataset.train_targets,
        dataset.test_features,
        dataset.test_targets,
        seed=sweep.dataset_seed,
    )

    torch.manual_seed(19)
    reference_model = QCNNModel(n_qubits=4)
    initial_weights = reference_model.weights.detach().clone()
    conditions = [(None, 0.0)]
    conditions.extend(
        (shots, noise)
        for shots in sweep.shots_values
        for noise in sweep.noise_values
    )

    runs = []
    for shots, noise in conditions:
        repetition_seeds = (None,) if shots is None else sweep.seeds
        for seed in repetition_seeds:
            execution = ExecutionConfig(
                shots=shots,
                depolarizing_probability=noise,
                seed=seed,
            )
            model = QCNNModel(n_qubits=4, execution_config=execution)
            with torch.no_grad():
                model.weights.copy_(initial_weights)

            start = time.perf_counter()
            losses = train_qcnn(
                model,
                dataset.train_features,
                dataset.train_targets,
                training,
            )
            with torch.no_grad():
                validation_predictions = model(dataset.validation_features)
                test_predictions = model(dataset.test_features)
            validation_loss = float(
                torch.nn.functional.mse_loss(
                    validation_predictions,
                    dataset.validation_targets,
                ).item()
            )
            validation_metrics = classification_metrics(
                validation_predictions,
                dataset.validation_targets,
            )
            test_metrics = classification_metrics(test_predictions, dataset.test_targets)
            elapsed_seconds = time.perf_counter() - start

            runs.append(
                {
                    "name": _condition_name(shots, noise),
                    "seed": seed,
                    "execution": asdict(execution),
                    "training": asdict(training),
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                    "validation_loss": validation_loss,
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "elapsed_seconds": elapsed_seconds,
                }
            )

    return {
        "sweep": asdict(sweep),
        "runs": runs,
        "summaries": _aggregate(runs),
        "baseline": baseline,
    }


def save_results(results: dict[str, object], path: str | Path) -> None:
    """Persist experiment results as UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    experiment_results = run_sweep()
    if arguments.output is not None:
        save_results(experiment_results, arguments.output)
    print(json.dumps(experiment_results, indent=2))
