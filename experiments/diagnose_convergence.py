"""Diagnose QCNN convergence without the cost of shots or noise."""

import argparse
import json
from pathlib import Path
import time

import torch

from src.datasets import load_binary_iris_splits
from src.hybrid_model import QCNNModel
from src.metrics import classification_metrics
from src.training import TrainingConfig, train_qcnn


def run_diagnosis(
    epoch_values: tuple[int, ...] = (3, 10, 20, 30),
    learning_rate: float = 0.03,
) -> dict[str, object]:
    """Train fixed-initialization exact QCNNs for several epoch counts."""
    dataset = load_binary_iris_splits(seed=42)
    torch.manual_seed(19)
    reference = QCNNModel(n_qubits=4)
    initial_weights = reference.weights.detach().clone()
    results = []

    for index, epochs in enumerate(epoch_values, start=1):
        model = QCNNModel(n_qubits=4)
        with torch.no_grad():
            model.weights.copy_(initial_weights)
        start = time.perf_counter()
        history = train_qcnn(
            model,
            dataset.train_features,
            dataset.train_targets,
            TrainingConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                optimizer="adam",
            ),
        )
        with torch.no_grad():
            validation_predictions = model(dataset.validation_features)
            test_predictions = model(dataset.test_features)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "epochs": epochs,
                "training": {
                    "learning_rate": learning_rate,
                    "optimizer": "adam",
                },
                "loss_history": history,
                "final_validation_loss": float(
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
                "elapsed_seconds": elapsed,
            }
        )
        print(f"[{index}/{len(epoch_values)}] epochs={epochs} completed in {elapsed:.1f}s")

    return {
        "dataset": {
            "name": "binary_iris",
            "seed": 42,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "model": {
            "n_qubits": 4,
            "execution": "exact",
            "initialization_seed": 19,
        },
        "results": results,
    }


def save_results(results: dict[str, object], path: str | Path) -> None:
    """Save convergence results as UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/qcnn-convergence.json"))
    arguments = parser.parse_args()
    diagnosis = run_diagnosis()
    save_results(diagnosis, arguments.output)
    print(f"saved: {arguments.output}")
