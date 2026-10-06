"""Run the focused experiments used to close the first QCNN version."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from src.classical_baseline import run_logistic_baseline
from src.datasets import (
    load_binary_breast_cancer_splits,
    load_binary_iris_splits,
)
from src.execution import ExecutionConfig
from src.hybrid_model import QCNNModel
from src.metrics import classification_metrics
from src.training import TrainingConfig, train_qcnn


def run_experiment(
    dataset_name: str,
    epochs: tuple[int, ...],
    encoding: str,
    shots: int | None,
    noise: float,
    seeds: tuple[int, ...],
    dataset_seed: int = 42,
    n_components: int = 4,
) -> dict[str, object]:
    if dataset_name == "iris":
        dataset = load_binary_iris_splits(seed=dataset_seed)
    elif dataset_name == "breast_cancer":
        dataset = load_binary_breast_cancer_splits(
            seed=dataset_seed,
            n_components=n_components,
        )
    else:
        raise ValueError("dataset must be 'iris' or 'breast_cancer'.")

    baseline = run_logistic_baseline(
        dataset.train_features,
        dataset.train_targets,
        dataset.test_features,
        dataset.test_targets,
        seed=dataset_seed,
    )
    results = []
    for initialization_seed in seeds:
        torch.manual_seed(initialization_seed)
        reference = QCNNModel(n_qubits=n_components, encoding=encoding)
        initial_weights = reference.weights.detach().clone()
        for epoch_count in epochs:
            model = QCNNModel(
                n_qubits=n_components,
                execution_config=ExecutionConfig(
                    shots=shots,
                    depolarizing_probability=noise,
                    seed=initialization_seed if shots is not None else None,
                ),
                encoding=encoding,
            )
            with torch.no_grad():
                model.weights.copy_(initial_weights)
            start = time.perf_counter()
            history = train_qcnn(
                model,
                dataset.train_features,
                dataset.train_targets,
                TrainingConfig(
                    epochs=epoch_count,
                    learning_rate=0.03,
                    optimizer="adam",
                ),
            )
            with torch.no_grad():
                validation_predictions = model(dataset.validation_features)
                test_predictions = model(dataset.test_features)
            results.append(
                {
                    "initialization_seed": initialization_seed,
                    "epochs": epoch_count,
                    "loss_history": history,
                    "validation_loss": float(
                        torch.nn.functional.mse_loss(
                            validation_predictions,
                            dataset.validation_targets,
                        ).item()
                    ),
                    "validation_metrics": classification_metrics(
                        validation_predictions,
                        dataset.validation_targets,
                    ),
                    "test_metrics": classification_metrics(
                        test_predictions,
                        dataset.test_targets,
                    ),
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
            print(
                f"{dataset_name} encoding={encoding} shots={shots} noise={noise} "
                f"seed={initialization_seed} epochs={epoch_count} completed"
            )
    return {
        "dataset": {
            "name": dataset_name,
            "seed": dataset_seed,
            "n_components": n_components,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "model": {
            "n_qubits": n_components,
            "encoding": encoding,
            "execution": {
                "shots": shots,
                "depolarizing_probability": noise,
                "seeds": list(seeds) if shots is not None else [None],
            },
        },
        "training": {"learning_rate": 0.03, "optimizer": "adam"},
        "results": results,
        "baseline": baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iris", "breast_cancer"), default="breast_cancer")
    parser.add_argument("--encoding", choices=("angle", "phase"), default="angle")
    parser.add_argument("--epochs", type=int, nargs="+", default=[30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[19])
    parser.add_argument("--shots", type=int)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    results = run_experiment(
        dataset_name=arguments.dataset,
        epochs=tuple(arguments.epochs),
        encoding=arguments.encoding,
        shots=arguments.shots,
        noise=arguments.noise,
        seeds=tuple(arguments.seeds),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {arguments.output}")


if __name__ == "__main__":
    main()
