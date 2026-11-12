"""Reproducible paired comparison or isolated QGD run for the same QCNN.

Paired runs share one exact initialization array and one QCNN noise executor
between QGD and Adam. QGD-only mode omits Adam entirely. Test data are not
accessed by screening runs and are evaluated once per selected checkpoint only
in confirmation runs. This module does not execute an experiment when imported.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as torch_functional

from src.circuit_execution import (
    QiboSimulatorExecutor,
    ResourceProfile,
    qibo_local_depolarizing_sensitivity_transform,
)
from src.datasets import (
    load_binary_breast_cancer_splits,
    load_binary_iris_splits,
)
from src.execution import ExecutionConfig
from src.hybrid_model import QCNNModel
from src.qcnn_engine import QCNNConfig, initialize_weights, predict
from src.quantum_training import (
    QuantumTrainingConfig,
    derive_execution_seed,
    train_quantum_qcnn,
)
from src.quantum_update import QGDConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ADAM_RESULTS = REPOSITORY_ROOT / "results" / "initial-hybrid-adam"
QCNN_NOISE_GATES = ("RY", "RZ", "CNOT")
QCNN_NOISE_MODEL = (
    "independent local single-qubit depolarizing channel after every RY, RZ, "
    "or CNOT gate, on each qubit touched by that gate"
)
UPDATE_NOISE_MODEL = (
    "independent local single-qubit depolarizing channel after every "
    "non-measurement logical gate in the QGD arithmetic update circuit"
)


@dataclass(frozen=True)
class AdamTrainingResult:
    """Adam history and its validation-selected checkpoint."""

    best_weights: np.ndarray
    final_weights: np.ndarray
    best_epoch: int
    history: tuple[dict[str, Any], ...]
    elapsed_seconds: float
    model: QCNNModel
    circuit_executions_estimate: dict[str, Any] = field(default_factory=dict)


def _parse_shots(value: str) -> int | None:
    if value.strip().lower() in {"exact", "none"}:
        return None
    try:
        shots = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shots must be a positive integer or 'exact'."
        ) from exc
    if shots < 1:
        raise argparse.ArgumentTypeError(
            "shots must be a positive integer or 'exact'."
        )
    return shots


def _tensor_to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(float, copy=False)


def _load_splits(
    dataset: str,
    dataset_seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    n_components: int,
):
    common = {
        "seed": dataset_seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }
    if dataset == "iris":
        return load_binary_iris_splits(**common)
    if dataset == "breast-cancer":
        return load_binary_breast_cancer_splits(
            **common, n_components=n_components
        )
    raise ValueError("dataset must be 'iris' or 'breast-cancer'.")


def _collect_run_provenance(command_arguments: list[str]) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        working_tree_status = [line for line in status.splitlines() if line.strip()]
        metadata_error = None
    except (OSError, subprocess.CalledProcessError) as exc:
        revision = None
        working_tree_status = []
        metadata_error = str(exc)

    if command_arguments:
        command_argv = [
            sys.executable,
            "-m",
            "experiments.compare_qgd_adam",
            *command_arguments,
        ]
    else:
        original_argv = getattr(sys, "orig_argv", None)
        command_argv = list(original_argv) if original_argv else list(sys.argv)
    return {
        "git_revision": revision,
        "working_tree_dirty": (
            None if metadata_error is not None else bool(working_tree_status)
        ),
        "working_tree_status": working_tree_status,
        "git_metadata_error": metadata_error,
        "command_argv": command_argv,
    }


def _resolve_new_output_path(output: Path) -> Path:
    resolved = output if output.is_absolute() else REPOSITORY_ROOT / output
    resolved = resolved.resolve()
    historical = HISTORICAL_ADAM_RESULTS.resolve()
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


def _binary_metrics(predictions, targets) -> dict[str, float]:
    prediction_values = np.asarray(predictions, dtype=float).reshape(-1)
    target_values = np.asarray(targets, dtype=float).reshape(-1)
    if prediction_values.shape != target_values.shape or not target_values.size:
        raise ValueError("predictions and targets must have equal non-empty shape.")
    if not np.isfinite(prediction_values).all() or not np.isfinite(target_values).all():
        raise ValueError("predictions and targets must be finite.")
    if not np.all((target_values == -1.0) | (target_values == 1.0)):
        raise ValueError("binary targets must use the -1/+1 encoding.")
    predicted_labels = np.where(prediction_values >= 0.0, 1.0, -1.0)
    actual_positive = target_values == 1.0
    predicted_positive = predicted_labels == 1.0
    true_positive = int(np.count_nonzero(actual_positive & predicted_positive))
    false_positive = int(np.count_nonzero(~actual_positive & predicted_positive))
    false_negative = int(np.count_nonzero(actual_positive & ~predicted_positive))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = (
        true_positive / recall_denominator if recall_denominator else 0.0
    )
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


def _partition_metrics(predictions, targets) -> dict[str, Any]:
    prediction_values = np.asarray(predictions, dtype=float)
    target_values = np.asarray(targets, dtype=float).reshape(prediction_values.shape)
    return {
        "mse_loss": float(np.mean((prediction_values - target_values) ** 2)),
        "metrics": _binary_metrics(prediction_values, target_values),
    }


def _set_model_execution(
    model: QCNNModel,
    *,
    shots: int | None,
    seed: int,
) -> None:
    model.execution_config = ExecutionConfig(shots=shots, seed=seed)


def _prepare_shared_initial_weights(weights: np.ndarray) -> np.ndarray:
    """Return one shared array exactly representable by Adam's float32 weights.

    The result is promoted back to float64 for the NumPy/QGD API. Thus the
    values Adam stores and the values QGD receives are numerically identical.
    """
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("initial_weights must contain only finite values.")
    return values.astype(np.float32).astype(np.float64)


def _build_initialized_adam_model(
    initial_weights: np.ndarray,
    *,
    n_qubits: int,
    encoding: str,
    shots: int | None,
    execution_seed: int,
    executor,
) -> QCNNModel:
    """Build Adam's model and reject any lossy parameter initialization."""
    initial_values = np.asarray(initial_weights, dtype=float)
    model = QCNNModel(
        n_qubits=n_qubits,
        n_outputs=1,
        execution_config=ExecutionConfig(shots=shots, seed=execution_seed),
        encoding=encoding,
        executor=executor,
    )
    parameter_values = torch.as_tensor(
        initial_values,
        dtype=model.weights.dtype,
        device=model.weights.device,
    )
    effective_values = parameter_values.detach().cpu().numpy()
    if not np.array_equal(effective_values, initial_values):
        raise ValueError(
            "initial_weights must be exactly representable by the Adam parameter dtype; "
            "use _prepare_shared_initial_weights first."
        )
    with torch.no_grad():
        model.weights.copy_(parameter_values)
    return model


def _logical_weight_occurrences(n_qubits: int) -> int:
    active_qubits = n_qubits
    pair_occurrences = 0
    while active_qubits > 1:
        pair_occurrences += active_qubits - 1
        active_qubits //= 2
    return 4 * pair_occurrences


def _adam_circuit_execution_estimate(
    *, n_qubits: int, train_size: int, validation_size: int, epochs: int, shots: int | None
) -> dict[str, Any]:
    weight_occurrences = _logical_weight_occurrences(n_qubits)
    gradient_circuits = train_size * (1 + 2 * weight_occurrences)
    post_update_train_circuits = train_size
    validation_circuits = validation_size
    per_epoch_total = (
        gradient_circuits + post_update_train_circuits + validation_circuits
    )
    training_total = epochs * per_epoch_total
    return {
        "accounting": (
            "Structural estimate only: one base circuit per training sample, "
            "two circuits per parameter occurrence and sample for parameter-shift, "
            "plus post-update train and validation prediction circuits."
        ),
        "instrumented": False,
        "logical_weight_occurrences_per_sample": weight_occurrences,
        "per_epoch": {
            "gradient_and_training_forward_circuit_executions_estimate": gradient_circuits,
            "post_update_train_evaluation_circuit_executions_estimate": post_update_train_circuits,
            "validation_evaluation_circuit_executions_estimate": validation_circuits,
            "total_circuit_executions_estimate": per_epoch_total,
            "shots_per_circuit": shots,
            "shots_total_estimate": (
                None if shots is None else per_epoch_total * shots
            ),
        },
        "epochs": epochs,
        "total_training_circuit_executions_estimate": training_total,
        "total_training_shots_estimate": (
            None if shots is None else training_total * shots
        ),
    }


def _evaluate_adam_partition(
    model: QCNNModel,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    shots: int | None,
    execution_seed: int,
) -> dict[str, Any]:
    _set_model_execution(model, shots=shots, seed=execution_seed)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(feature_tensor).detach().cpu().numpy()
    return _partition_metrics(predictions, targets)


def train_adam_with_validation(
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    initial_weights: np.ndarray,
    *,
    n_qubits: int,
    encoding: str,
    shots: int | None,
    execution_seed: int,
    epochs: int,
    learning_rate: float,
    executor,
) -> AdamTrainingResult:
    """Train with one persistent Adam optimizer and select only on validation."""
    initial_values = np.asarray(initial_weights, dtype=float)
    if initial_values.shape != (int(math.log2(n_qubits)), 4):
        raise ValueError("initial_weights do not match the configured QCNN shape.")
    if not np.isfinite(initial_values).all():
        raise ValueError("initial_weights must contain only finite values.")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite.")

    train_x = np.asarray(train_features, dtype=float)
    train_y = np.asarray(train_targets, dtype=float).reshape(-1, 1)
    validation_x = np.asarray(validation_features, dtype=float)
    validation_y = np.asarray(validation_targets, dtype=float).reshape(-1, 1)
    if train_x.ndim != 2 or train_x.shape[1] != n_qubits or not len(train_x):
        raise ValueError("train_features must have non-empty shape (batch, n_qubits).")
    if train_y.shape != (len(train_x), 1):
        raise ValueError("train_targets must have one binary label per sample.")
    if (
        validation_x.ndim != 2
        or validation_x.shape[1] != n_qubits
        or not len(validation_x)
    ):
        raise ValueError(
            "validation_features must have non-empty shape (batch, n_qubits)."
        )
    if validation_y.shape != (len(validation_x), 1):
        raise ValueError("validation_targets must have one label per sample.")

    initial_seed = derive_execution_seed(execution_seed, "sampled_gradient", 1)
    model = _build_initialized_adam_model(
        initial_values,
        n_qubits=n_qubits,
        encoding=encoding,
        shots=shots,
        execution_seed=initial_seed,
        executor=executor,
    )
    train_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    target_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, Any]] = []
    best_weights: np.ndarray | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    weight_occurrences_per_sample = _logical_weight_occurrences(n_qubits)
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        gradient_seed = derive_execution_seed(
            execution_seed, "sampled_gradient", epoch
        )
        _set_model_execution(model, shots=shots, seed=gradient_seed)
        model.train()
        optimizer.zero_grad()
        optimization_loss = torch_functional.mse_loss(
            model(train_tensor), target_tensor
        )
        optimization_loss.backward()
        optimizer.step()

        train_seed = derive_execution_seed(
            execution_seed, "train_evaluation", epoch
        )
        validation_seed = derive_execution_seed(
            execution_seed, "validation_evaluation", epoch
        )
        train_record = _evaluate_adam_partition(
            model,
            train_x,
            train_y,
            shots=shots,
            execution_seed=train_seed,
        )
        validation_record = _evaluate_adam_partition(
            model,
            validation_x,
            validation_y,
            shots=shots,
            execution_seed=validation_seed,
        )
        if not math.isfinite(train_record["mse_loss"]) or not math.isfinite(
            validation_record["mse_loss"]
        ):
            raise RuntimeError("Adam training produced a non-finite loss.")
        gradient_circuits = len(train_x) * (
            1 + 2 * weight_occurrences_per_sample
        )
        post_update_train_circuits = len(train_x)
        validation_circuits = len(validation_x)
        epoch_circuits = (
            gradient_circuits + post_update_train_circuits + validation_circuits
        )
        record = {
            "epoch": epoch,
            "optimization_loss_before_update": float(optimization_loss.detach().item()),
            "train_loss": train_record["mse_loss"],
            "train_metrics": train_record["metrics"],
            "validation_loss": validation_record["mse_loss"],
            "validation_metrics": validation_record["metrics"],
            "gradient_execution_seed": gradient_seed,
            "train_evaluation_execution_seed": train_seed,
            "validation_evaluation_execution_seed": validation_seed,
            "shots": shots,
            "circuit_executions_estimate": {
                "gradient_and_training_forward": gradient_circuits,
                "post_update_train_evaluation": post_update_train_circuits,
                "validation_evaluation": validation_circuits,
                "total": epoch_circuits,
                "shots_total_estimate": (
                    None if shots is None else epoch_circuits * shots
                ),
            },
        }
        history.append(record)
        if record["validation_loss"] < best_validation_loss:
            best_validation_loss = record["validation_loss"]
            best_weights = model.weights.detach().cpu().numpy().copy()
            best_epoch = epoch

    elapsed_seconds = time.perf_counter() - start_time
    final_weights = model.weights.detach().cpu().numpy().copy()
    if best_weights is None:
        raise RuntimeError("Adam training completed without a validation checkpoint.")
    circuit_executions_estimate = _adam_circuit_execution_estimate(
        n_qubits=n_qubits,
        train_size=len(train_x),
        validation_size=len(validation_x),
        epochs=len(history),
        shots=shots,
    )
    with torch.no_grad():
        model.weights.copy_(torch.as_tensor(best_weights, dtype=model.weights.dtype))
    return AdamTrainingResult(
        best_weights=best_weights,
        final_weights=final_weights,
        best_epoch=best_epoch,
        history=tuple(history),
        elapsed_seconds=float(elapsed_seconds),
        model=model,
        circuit_executions_estimate=circuit_executions_estimate,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paired QGD/Adam comparison or an isolated QGD QCNN run."
    )
    parser.add_argument("--dataset", choices=("iris", "breast-cancer"), default="iris")
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--initialization-seed", type=int, default=19)
    parser.add_argument("--execution-seed", type=int, default=101)
    parser.add_argument("--encoding", choices=("angle", "phase"), default="angle")
    parser.add_argument(
        "--shots",
        type=_parse_shots,
        default=100,
        help="finite shots or 'exact' (exact mode requires p_model=p_update=0)",
    )
    parser.add_argument(
        "--noise-simulation",
        choices=("exact-channel", "legacy-shared-trajectory"),
        default="exact-channel",
        help=(
            "exact-channel (default): density matrix for QCNN circuits and one "
            "Pauli trajectory per shot for QGD update circuits; "
            "legacy-shared-trajectory reproduces the November 2026 screens"
        ),
    )
    parser.add_argument("--p-model", type=float, default=0.0)
    parser.add_argument("--p-update", type=float, default=0.0)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="default is 2 with --screening and 30 otherwise; screening is fixed to 2",
    )
    parser.add_argument("--adam-learning-rate", type=float, default=0.03)
    parser.add_argument("--qgd-learning-rate", type=float, default=0.1)
    parser.add_argument("--qgd-bits", type=int, default=5)
    parser.add_argument(
        "--screening",
        action="store_true",
        help="omit test completely; intended for screens, not confirmation metrics",
    )
    parser.add_argument(
        "--qgd-only",
        action="store_true",
        help="run QGD without training/evaluating Adam; requires --p-model 0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="required new JSON path; existing and historical files are never overwritten",
    )
    return parser


def _effective_epochs(args: argparse.Namespace) -> int:
    """Resolve the budget and lock screening runs to the protocol's two steps."""
    if args.epochs is None:
        return 2 if args.screening else 30
    if not isinstance(args.epochs, int) or isinstance(args.epochs, bool) or args.epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if args.screening and args.epochs != 2:
        raise ValueError("screening runs are fixed to exactly 2 epochs/steps.")
    return args.epochs


def build_result_payload(
    *,
    dataset: str,
    dataset_seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    n_components: int,
    initialization_seed: int,
    execution_seed: int,
    encoding: str,
    shots: int | None,
    p_model: float,
    p_update: float,
    adam_learning_rate: float,
    qgd_config: QGDConfig,
    epochs: int,
    initial_weights: np.ndarray,
    adam_result: AdamTrainingResult | None,
    qgd_result,
    adam_test: dict[str, Any] | None,
    qgd_test: dict[str, Any] | None,
    provenance: dict[str, Any],
    screening: bool,
    qgd_only: bool,
) -> dict[str, Any]:
    if qgd_only == (adam_result is not None):
        raise ValueError("Adam result presence must match paired versus QGD-only mode.")
    qcnn_noise = {
        "probability": float(p_model),
        "model": QCNN_NOISE_MODEL if p_model else "none",
        "eligible_gate_names": list(QCNN_NOISE_GATES),
        "scope": "QCNN circuits only: forward, parameter-shift, train/validation, and final test when present",
        "methods": ["QGD"] if qgd_only else ["Adam", "QGD"],
        "measurement_noise": False,
        "calibrated_hardware_noise": False,
    }
    update_noise = {
        "probability": float(p_update),
        "model": UPDATE_NOISE_MODEL if p_update else "none",
        "eligible_gate_names": "all non-measurement logical gates",
        "scope": "QGD arithmetic update circuits only; never applied to QCNN circuits or Adam",
        "methods": ["QGD"],
        "measurement_noise": False,
        "calibrated_hardware_noise": False,
    }
    method_configuration = {
        "qgd": {
            **asdict(qgd_config),
            "moving_gradient_codes": qgd_config.moving_gradient_codes,
            "static_update": qgd_config.is_static,
            "finite_register_semantics": (
                "modular weight-grid subtraction; out-of-range gradients saturate; "
                "not weight clipping"
            ),
        },
    }
    methods: dict[str, Any] = {}
    if not qgd_only:
        method_configuration["adam"] = {
            "optimizer": "Adam",
            "learning_rate": float(adam_learning_rate),
            "optimizer_persists_across_epochs": True,
        }
        methods["adam"] = {
            "epochs_completed": len(adam_result.history),
            "elapsed_seconds": float(adam_result.elapsed_seconds),
            "history": [dict(item) for item in adam_result.history],
            "circuit_executions_estimate": adam_result.circuit_executions_estimate,
            "checkpoint": {
                "selection": "minimum validation MSE",
                "epoch": int(adam_result.best_epoch),
                "validation_loss": float(
                    adam_result.history[adam_result.best_epoch - 1]["validation_loss"]
                ),
                "weights": adam_result.best_weights.tolist(),
            },
            "final_weights": adam_result.final_weights.tolist(),
            "test": adam_test or {"evaluated": False, "reason": "screening"},
        }
    methods["qgd"] = {
        "steps_completed": int(qgd_result.steps_completed),
        "elapsed_seconds": float(qgd_result.elapsed_seconds),
        "history": [dict(item) for item in qgd_result.history],
        "checkpoint": {
            "selection": "minimum validation MSE",
            "step": int(qgd_result.best_step),
            "validation_loss": float(
                qgd_result.history[qgd_result.best_step - 1]["validation_loss"]
            ),
            "weights": qgd_result.best_weights.tolist(),
        },
        "final_weights": qgd_result.final_weights.tolist(),
        "circuit_executions_estimate": qgd_result.circuit_executions_estimate,
        "test": qgd_test or {"evaluated": False, "reason": "screening"},
    }
    return {
        "schema_version": 1,
        "run_mode": "qgd_only" if qgd_only else "paired_qgd_adam",
        "dataset": {
            "name": dataset,
            "seed": int(dataset_seed),
            "split_fractions": {
                "train": float(train_fraction),
                "validation": float(validation_fraction),
                "test": float(test_fraction),
            },
            "n_components": n_components if dataset == "breast-cancer" else None,
            "preprocessing": "normalization and, for Breast Cancer, standardization/PCA fitted on train only",
        },
        "provenance": provenance,
        "seeds": {
            "dataset": int(dataset_seed),
            "initialization": int(initialization_seed),
            "execution": int(execution_seed),
            "policy": "dataset, initialization, and execution seeds are configured independently; execution sub-seeds derive from NumPy SeedSequence",
        },
        "model": {
            "n_qubits": int(2 ** initial_weights.shape[0]),
            "n_outputs": 1,
            "encoding": encoding,
            "loss": "MSE against labels -1/+1",
            "initial_weights_shared_exactly": not qgd_only,
            "initial_weights_representation": (
                "initialized values quantized to float32 then promoted to float64; "
                + (
                    "Adam stores the same exactly representable values as float32 and QGD receives float64"
                    if not qgd_only
                    else "QGD receives the float64 representation of those quantized values"
                )
            ),
            "noise": {"p_model": qcnn_noise, "p_update": update_noise},
            "execution": {
                "shots": shots,
                "mode": "exact_noiseless_qcnn" if shots is None else "finite_counts",
                "update_mode": (
                    "one deterministic simulator shot"
                    if shots is None
                    else "finite counts using the configured shot budget"
                ),
            },
        },
        "configuration": {
            "epochs_or_qgd_steps": int(epochs),
            "methods": method_configuration,
            "screening": bool(screening),
            "qgd_only": bool(qgd_only),
        },
        "initial_weights": np.asarray(initial_weights, dtype=float).tolist(),
        "methods": methods,
        "test_policy": (
            "screening: test is not evaluated"
            if screening
            else (
                "one test prediction pass for QGD after its validation checkpoint is selected"
                if qgd_only
                else "one test prediction pass per method, after both validation checkpoints are selected"
            )
        ),
        "classical_quantum_boundary": {
            **(
                {}
                if qgd_only
                else {
                    "adam": "PyTorch Adam updates QCNN weights classically; its forward and parameter-shift circuits use the shared QCNN executor."
                }
            ),
            "qgd": "QGD's arithmetic update is a measured Qibo circuit; host computes gradients, losses, validation selection, and loop control.",
            "hardware_status": "local Qibo simulator only; no QPU provider, transpilation, or hardware execution is claimed",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(command_arguments)
    try:
        epochs = _effective_epochs(args)
        output_path = _resolve_new_output_path(args.output)
        # Construction validates the probabilities before any dataset is loaded.
        legacy_noise = args.noise_simulation == "legacy-shared-trajectory"
        model_noise_simulation = (
            "legacy_shared_trajectory" if legacy_noise else "density_matrix"
        )
        update_noise_simulation = (
            "legacy_shared_trajectory" if legacy_noise else "per_shot_trajectories"
        )
        model_transform = (
            qibo_local_depolarizing_sensitivity_transform(
                args.p_model,
                gate_names=QCNN_NOISE_GATES,
                simulation=model_noise_simulation,
            )
            if args.p_model != 0.0
            else None
        )
        update_transform = (
            qibo_local_depolarizing_sensitivity_transform(
                args.p_update, simulation=update_noise_simulation
            )
            if args.p_update != 0.0
            else None
        )
        qgd_config = QGDConfig(
            n_bits=args.qgd_bits,
            learning_rate=args.qgd_learning_rate,
        )
        if qgd_config.is_static:
            print(
                "WARNING: static QGD configuration "
                f"(n_bits={qgd_config.n_bits}, learning_rate={qgd_config.learning_rate}): "
                "0 gradient codes move a weight; QGD only quantizes the weights.",
                file=sys.stderr,
            )
        if not args.qgd_only and (
            not math.isfinite(args.adam_learning_rate)
            or args.adam_learning_rate <= 0
        ):
            raise ValueError("adam_learning_rate must be positive and finite.")
        if args.shots is None and (args.p_model != 0.0 or args.p_update != 0.0):
            raise ValueError("exact mode requires p_model=p_update=0.")
        if args.qgd_only and args.p_model != 0.0:
            raise ValueError("--qgd-only requires --p-model 0; it is not a paired comparison.")
        # Validate split fractions using the same data-loader contract.
        fractions = (args.train_fraction, args.validation_fraction, args.test_fraction)
        if (
            any(not math.isfinite(value) or value <= 0.0 or value >= 1.0 for value in fractions)
            or abs(sum(fractions) - 1.0) > 1e-9
        ):
            raise ValueError("split fractions must be in (0, 1) and sum to 1.")
        if args.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        provenance = _collect_run_provenance(command_arguments)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    splits = _load_splits(
        args.dataset,
        args.dataset_seed,
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
        args.n_components,
    )
    train_x = _tensor_to_numpy(splits.train_features)
    train_y = _tensor_to_numpy(splits.train_targets)
    validation_x = _tensor_to_numpy(splits.validation_features)
    validation_y = _tensor_to_numpy(splits.validation_targets)

    model_executor = None
    update_executor = None
    if args.shots is not None:
        model_executor = QiboSimulatorExecutor(
            capabilities=ResourceProfile(
                metadata={
                    "adapter": "qibo-local-simulator",
                    "scope": (
                        "QGD QCNN circuits only"
                        if args.qgd_only
                        else "shared QCNN circuits for Adam and QGD"
                    ),
                    "noise_probability": args.p_model,
                    "noise_gate_names": list(QCNN_NOISE_GATES),
                }
            ),
            noise_transform=model_transform,
        )
        update_executor = QiboSimulatorExecutor(
            capabilities=ResourceProfile(
                metadata={
                    "adapter": "qibo-local-simulator",
                    "scope": "QGD arithmetic update circuits only",
                    "noise_probability": args.p_update,
                }
            ),
            noise_transform=update_transform,
        )

    # One float32-quantized array is shared; Adam can store it without a lossy cast.
    model_config = QCNNConfig(
        n_qubits=train_x.shape[1],
        n_outputs=1,
        encoding=args.encoding,
        execution=ExecutionConfig(shots=args.shots, seed=args.execution_seed),
        executor=model_executor,
    )
    initial_weights = _prepare_shared_initial_weights(
        initialize_weights(
            model_config.n_qubits,
            n_outputs=model_config.n_outputs,
            seed=args.initialization_seed,
        )
    )
    adam_result = None
    if not args.qgd_only:
        adam_result = train_adam_with_validation(
            train_x,
            train_y,
            validation_x,
            validation_y,
            initial_weights,
            n_qubits=model_config.n_qubits,
            encoding=args.encoding,
            shots=args.shots,
            execution_seed=args.execution_seed,
            epochs=epochs,
            learning_rate=args.adam_learning_rate,
            executor=model_executor,
        )
    qgd_result = train_quantum_qcnn(
        train_x,
        train_y,
        validation_x,
        validation_y,
        initial_weights,
        model_config,
        qgd_config,
        QuantumTrainingConfig(
            max_steps=epochs,
            shots=args.shots,
            execution_seed=args.execution_seed,
        ),
        update_executor=update_executor,
    )

    adam_test = None
    qgd_test = None
    if not args.screening:
        test_x = _tensor_to_numpy(splits.test_features)
        test_y = _tensor_to_numpy(splits.test_targets)
        if adam_result is not None:
            adam_test_seed = derive_execution_seed(
                args.execution_seed, "test_evaluation", adam_result.best_epoch
            )
            adam_test_started = time.perf_counter()
            adam_test_metrics = _evaluate_adam_partition(
                adam_result.model,
                test_x,
                test_y,
                shots=args.shots,
                execution_seed=adam_test_seed,
            )
            adam_test = {
                "evaluated": True,
                "evaluated_once_after_checkpoint_selection": True,
                "checkpoint_epoch": adam_result.best_epoch,
                "execution_seed": adam_test_seed,
                "shots": args.shots,
                "circuit_executions_estimate": len(test_x),
                "shots_total_estimate": (
                    None if args.shots is None else len(test_x) * args.shots
                ),
                **adam_test_metrics,
                "elapsed_seconds": float(time.perf_counter() - adam_test_started),
            }

        qgd_test_seed = derive_execution_seed(
            args.execution_seed, "test_evaluation", qgd_result.best_step
        )
        test_model_config = QCNNConfig(
            n_qubits=model_config.n_qubits,
            n_outputs=model_config.n_outputs,
            encoding=model_config.encoding,
            execution=ExecutionConfig(
                shots=args.shots,
                depolarizing_probability=0.0,
                seed=qgd_test_seed,
            ),
            executor=model_executor,
        )
        qgd_test_started = time.perf_counter()
        qgd_predictions = predict(
            qgd_result.best_weights,
            test_x,
            test_model_config,
        )
        qgd_test = {
            "evaluated": True,
            "evaluated_once_after_checkpoint_selection": True,
            "checkpoint_step": qgd_result.best_step,
            "execution_seed": qgd_test_seed,
            "shots": args.shots,
            "circuit_executions_estimate": len(test_x),
            "shots_total_estimate": (
                None if args.shots is None else len(test_x) * args.shots
            ),
            **_partition_metrics(qgd_predictions, test_y),
            "elapsed_seconds": float(time.perf_counter() - qgd_test_started),
        }

    payload = build_result_payload(
        dataset=args.dataset,
        dataset_seed=args.dataset_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        n_components=args.n_components,
        initialization_seed=args.initialization_seed,
        execution_seed=args.execution_seed,
        encoding=args.encoding,
        shots=args.shots,
        p_model=args.p_model,
        p_update=args.p_update,
        adam_learning_rate=args.adam_learning_rate,
        qgd_config=qgd_config,
        epochs=epochs,
        initial_weights=initial_weights,
        adam_result=adam_result,
        qgd_result=qgd_result,
        adam_test=adam_test,
        qgd_test=qgd_test,
        provenance=provenance,
        screening=args.screening,
        qgd_only=args.qgd_only,
    )
    payload["model"]["noise"]["simulation"] = {
        "p_model": model_noise_simulation if args.p_model else None,
        "p_update": update_noise_simulation if args.p_update else None,
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as output_file:
            output_file.write(serialized)
    except FileExistsError:
        parser.error(f"output appeared during execution; refusing to overwrite: {output_path}")
    output_label = "QGD-only JSON" if args.qgd_only else "QGD/Adam comparison JSON"
    print(f"Wrote {output_label} to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
