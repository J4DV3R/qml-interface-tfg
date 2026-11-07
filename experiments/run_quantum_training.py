"""CLI for a short NumPy/Qibo discrete-QGD QCNN run.

The dataset loader returns Torch tensors for compatibility with the legacy
pipeline. Conversion to NumPy happens only at this runner boundary; the
quantum-training module itself has no Torch dependency. The test partition is
used once, after validation has selected the best checkpoint.
"""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets import (
    load_binary_breast_cancer_splits,
    load_binary_iris_splits,
)
from src.execution import ExecutionConfig
from src.qcnn_engine import QCNNConfig, initialize_weights, predict
from src.quantum_training import (
    QuantumTrainingConfig,
    derive_execution_seed,
    train_quantum_qcnn,
)
from src.quantum_update import QGDConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RESULTS = REPOSITORY_ROOT / "results" / "initial-hybrid-adam"


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Convert loader tensors to arrays at the classical/quantum boundary."""
    return tensor.detach().cpu().numpy().astype(float, copy=False)


def _binary_metrics(predictions, targets) -> dict[str, float]:
    prediction_values = np.asarray(predictions, dtype=float).reshape(-1)
    target_values = np.asarray(targets, dtype=float).reshape(-1)
    if prediction_values.shape != target_values.shape or not len(target_values):
        raise ValueError("test predictions and targets must have equal non-empty shape.")
    if not np.isfinite(prediction_values).all() or not np.isfinite(target_values).all():
        raise ValueError("test predictions and targets must be finite.")
    if not np.all((target_values == -1.0) | (target_values == 1.0)):
        raise ValueError("binary test targets must use the -1/+1 encoding.")

    predicted_labels = np.where(prediction_values >= 0.0, 1.0, -1.0)
    positive = target_values == 1.0
    predicted_positive = predicted_labels == 1.0
    true_positive = int(np.count_nonzero(positive & predicted_positive))
    false_positive = int(np.count_nonzero(~positive & predicted_positive))
    false_negative = int(np.count_nonzero(positive & ~predicted_positive))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": float(np.mean(predicted_labels == target_values)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": float(target_values.size),
    }


def _load_splits(dataset: str, dataset_seed: int):
    if dataset == "iris":
        return load_binary_iris_splits(seed=dataset_seed)
    if dataset == "breast-cancer":
        return load_binary_breast_cancer_splits(seed=dataset_seed)
    raise ValueError("dataset must be 'iris' or 'breast-cancer'.")


def build_result_payload(
    *,
    dataset: str,
    dataset_seed: int,
    initialization_seed: int,
    execution_seed: int,
    initial_weights: np.ndarray,
    model_config: QCNNConfig,
    qgd_config: QGDConfig,
    training_config: QuantumTrainingConfig,
    training_result,
    test_predictions: np.ndarray,
    test_targets: np.ndarray,
    test_execution_seed: int,
) -> dict[str, Any]:
    """Assemble the JSON payload after the single final test prediction pass."""
    target_values = np.asarray(test_targets, dtype=float)
    predictions = np.asarray(test_predictions, dtype=float)
    test_loss = float(np.mean((predictions - target_values.reshape(predictions.shape)) ** 2))
    circuit_counts = copy.deepcopy(training_result.circuit_executions_estimate)
    test_circuit_count = int(predictions.shape[0])
    training_total = int(circuit_counts["total_training_circuit_executions_estimate"])
    circuit_counts["final_test_evaluation"] = {
        "logical_batch_evaluations": 1,
        "samples_per_batch": test_circuit_count,
        "shots_per_circuit": training_config.shots,
        "execution_seed": int(test_execution_seed),
        "circuit_executions_estimate": test_circuit_count,
    }
    circuit_counts["total_including_final_test_circuit_executions_estimate"] = (
        training_total + test_circuit_count
    )

    return {
        "dataset": {
            "name": dataset,
            "seed": int(dataset_seed),
            "split_fractions": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
            "preprocessing": "fit on train only; loader returns normalized features",
        },
        "seeds": {
            "dataset": int(dataset_seed),
            "initialization": int(initialization_seed),
            "execution": int(execution_seed),
            "execution_seed_policy": (
                "NumPy SeedSequence([execution_seed, component_id, step]) yields "
                "a stable uint32 seed per gradient/evaluation component."
            ),
        },
        "model": {
            "n_qubits": model_config.n_qubits,
            "n_outputs": model_config.n_outputs,
            "encoding": model_config.encoding,
            "depolarizing_probability": (
                model_config.execution.depolarizing_probability
            ),
        },
        "initial_weights": np.asarray(initial_weights, dtype=float).tolist(),
        "qgd": {
            **asdict(qgd_config),
            "finite_register_semantics": (
                "weights are updated on a cyclic grid modulo n_levels; gradients "
                "outside [gradient_min, gradient_max] saturate at the boundary. "
                "This is not gradient clipping."
            ),
        },
        "training": {
            **asdict(training_config),
            "learning_rate": qgd_config.learning_rate,
            "steps_completed": training_result.steps_completed,
            "best_step": training_result.best_step,
            "best_validation_loss": float(
                training_result.history[training_result.best_step - 1][
                    "validation_loss"
                ]
            ),
            "elapsed_seconds": float(training_result.elapsed_seconds),
            "history": [dict(step) for step in training_result.history],
        },
        "best_weights": training_result.best_weights.tolist(),
        "final_weights": training_result.final_weights.tolist(),
        "test": {
            "evaluated_once_after_checkpoint_selection": True,
            "checkpoint_step": training_result.best_step,
            "execution_seed": int(test_execution_seed),
            "shots": training_config.shots,
            "samples": test_circuit_count,
            "mse_loss": test_loss,
            "metrics": _binary_metrics(predictions, target_values),
        },
        "circuit_executions_estimate": circuit_counts,
        "classical_quantum_boundary": {
            "initial_gradient": (
                "host computes parameter-shift differences from exact, noiseless "
                "Qibo simulator circuits; this is classical orchestration and "
                "postprocessing, not a quantum gradient register."
            ),
            "later_gradients": (
                "Qibo circuits are sampled with the configured shots; the host "
                "reads expectations and assembles parameter-shift gradients."
            ),
            "weight_update": (
            "one reversible Qibo permutation circuit per scalar weight; "
            "the exact simulator state vector is decoded classically (no shots)."
            ),
            "control_and_selection": (
                "loss calculation, validation checkpoint selection, stopping, "
                "and test metric postprocessing are classical."
            ),
        },
    }


def _resolve_new_output_path(output: Path) -> Path:
    resolved = output if output.is_absolute() else REPOSITORY_ROOT / output
    resolved = resolved.resolve()
    historical = HISTORICAL_RESULTS.resolve()
    try:
        resolved.relative_to(historical)
    except ValueError:
        pass
    else:
        raise ValueError(
            "output cannot be inside results/initial-hybrid-adam; historical "
            "results are frozen."
        )
    if resolved.exists():
        raise ValueError(f"output already exists; refusing to overwrite: {resolved}")
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short pure-NumPy/Qibo QCNN training increment."
    )
    parser.add_argument(
        "--dataset",
        choices=("iris", "breast-cancer"),
        default="iris",
        help="binary dataset (default: iris)",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="depolarizing probability for shot-based gradients/evaluations",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--qgd-bits", type=int, default=5)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--encoding", choices=("angle", "phase"), default="angle")
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=19)
    parser.add_argument("--execution-seed", type=int, default=101)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="required new JSON path; relative paths are rooted at the repository",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = _resolve_new_output_path(args.output)
        training_config = QuantumTrainingConfig(
            max_steps=args.max_steps,
            shots=args.shots,
            execution_seed=args.execution_seed,
            patience=args.patience,
        )
        qgd_config = QGDConfig(
            n_bits=args.qgd_bits,
            learning_rate=args.learning_rate,
        )
        execution_config = ExecutionConfig(
            depolarizing_probability=args.noise,
        )
    except ValueError as exc:
        parser.error(str(exc))

    splits = _load_splits(args.dataset, args.dataset_seed)
    # Torch is intentionally touched only here, at the loader-to-NumPy boundary.
    train_features = _tensor_to_numpy(splits.train_features)
    train_targets = _tensor_to_numpy(splits.train_targets)
    validation_features = _tensor_to_numpy(splits.validation_features)
    validation_targets = _tensor_to_numpy(splits.validation_targets)

    model_config = QCNNConfig(
        n_qubits=train_features.shape[1],
        n_outputs=1,
        encoding=args.encoding,
        execution=execution_config,
    )
    initial_weights = initialize_weights(
        model_config.n_qubits,
        n_outputs=model_config.n_outputs,
        seed=args.initialization_seed,
    )
    training_result = train_quantum_qcnn(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        initial_weights,
        model_config,
        qgd_config,
        training_config,
    )

    # The test partition is evaluated exactly once, only after validation has
    # selected the checkpoint. Conversion is deferred to this boundary too.
    test_features = _tensor_to_numpy(splits.test_features)
    test_targets = _tensor_to_numpy(splits.test_targets)
    # One predict call executes one circuit per sample.
    test_execution_seed = derive_execution_seed(
        training_config.execution_seed,
        "test_evaluation",
        training_result.best_step,
    )
    test_execution = QCNNConfig(
        n_qubits=model_config.n_qubits,
        n_outputs=model_config.n_outputs,
        encoding=model_config.encoding,
        execution=ExecutionConfig(
            shots=training_config.shots,
            depolarizing_probability=(
                model_config.execution.depolarizing_probability
            ),
            seed=test_execution_seed,
        ),
    )
    test_predictions = predict(
        training_result.best_weights,
        test_features,
        test_execution,
    )
    payload = build_result_payload(
        dataset=args.dataset,
        dataset_seed=args.dataset_seed,
        initialization_seed=args.initialization_seed,
        execution_seed=training_config.execution_seed,
        initial_weights=initial_weights,
        model_config=model_config,
        qgd_config=qgd_config,
        training_config=training_config,
        training_result=training_result,
        test_predictions=test_predictions,
        test_targets=test_targets,
        test_execution_seed=test_execution_seed,
    )
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as output_file:
            output_file.write(serialized)
    except FileExistsError:
        parser.error(f"output appeared during execution; refusing to overwrite: {output_path}")
    print(f"Wrote quantum training JSON to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
