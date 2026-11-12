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
import subprocess
import sys
from typing import Any

import numpy as np

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
from src.qcnn_engine import QCNNConfig, initialize_weights, predict
from src.quantum_training import (
    QuantumTrainingConfig,
    derive_execution_seed,
    train_quantum_qcnn,
)
from src.quantum_update import QGDConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RESULTS = REPOSITORY_ROOT / "results" / "initial-hybrid-adam"
SIMULATOR_NOISE_MODEL = (
    "independent single-qubit depolarizing channel on each qubit touched by "
    "each non-measurement logical gate (including controlled-X gates)"
)
QIBO_BUILTIN_NOISE_MODEL = (
    "Qibo depolarizing error after RY, RZ and CNOT gates; no arithmetic-update "
    "coverage is implied"
)


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Convert loader tensors to arrays at the classical/quantum boundary."""
    return tensor.detach().cpu().numpy().astype(float, copy=False)


def _parse_shots(value: str) -> int | None:
    """Accept a positive shot count or ``exact`` for simulator-only mode."""
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


def _collect_run_provenance(command_arguments: list[str]) -> dict[str, Any]:
    """Capture Git revision, dirty paths, and effective CLI invocation."""
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        working_tree_status = [
            line for line in status_result.stdout.splitlines() if line.strip()
        ]
        git_metadata_error = None
        revision = revision_result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError) as exc:
        revision = None
        working_tree_status = []
        git_metadata_error = str(exc)

    if command_arguments:
        command_argv = [
            sys.executable,
            "-m",
            "experiments.run_quantum_training",
            *command_arguments,
        ]
    else:
        original_argv = getattr(sys, "orig_argv", None)
        command_argv = list(original_argv) if original_argv else list(sys.argv)

    return {
        "git_revision": revision,
        "working_tree_dirty": (
            None if git_metadata_error is not None else bool(working_tree_status)
        ),
        "working_tree_status": working_tree_status,
        "git_metadata_error": git_metadata_error,
        "command_argv": command_argv,
    }


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
    provenance: dict[str, Any] | None = None,
    simulator_noise_probability: float | None = None,
    simulator_noise_model: str | None = None,
    simulator_noise_applied_to: tuple[str, ...] | None = None,
    measured_execution_requested: bool | None = None,
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
    requested_noise = (
        model_config.execution.depolarizing_probability
        if simulator_noise_probability is None
        else float(simulator_noise_probability)
    )
    if simulator_noise_model is not None:
        noise_description = simulator_noise_model
    elif requested_noise == 0:
        noise_description = "none"
    elif model_config.executor is None:
        noise_description = QIBO_BUILTIN_NOISE_MODEL
    else:
        noise_description = "executor-defined; see declared resource metadata"
    measured_execution_enabled = model_config.executor is not None
    if measured_execution_requested is None:
        measured_execution_requested = measured_execution_enabled
    if not isinstance(measured_execution_requested, bool):
        raise ValueError("measured_execution_requested must be a bool or None.")
    if measured_execution_enabled:
        if isinstance(model_config.executor, QiboSimulatorExecutor):
            executor_mode = (
                "qibo_simulator_measured_with_sensitivity_noise"
                if model_config.executor.noise_transform is not None
                else "qibo_simulator_measured"
            )
        else:
            executor_mode = type(model_config.executor).__name__
    else:
        executor_mode = "default_qibo_simulator"
    initial_gradient_mode = (
        "parameter_shift_shots"
        if measured_execution_enabled
        else "exact_initialization"
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
        "provenance": copy.deepcopy(
            provenance
            if provenance is not None
            else {
                "git_revision": None,
                "working_tree_dirty": None,
                "working_tree_status": [],
                "git_metadata_error": "not supplied to payload builder",
                "command_argv": None,
            }
        ),
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
            "measured_execution": measured_execution_enabled,
            "measured_execution_requested": measured_execution_requested,
            "executor_mode": executor_mode,
            "depolarizing_probability": requested_noise,
            "execution_config_depolarizing_probability": (
                model_config.execution.depolarizing_probability
            ),
            "simulator_noise": {
                "requested_probability": requested_noise,
                "model": noise_description,
                "scope": (
                    "simulator-only sensitivity model; not calibrated or claimed "
                    "to represent physical hardware noise"
                ),
                "applied_to": (
                    list(simulator_noise_applied_to or ())
                ),
            },
        },
        "initial_weights": np.asarray(initial_weights, dtype=float).tolist(),
        "qgd": {
            **asdict(qgd_config),
            "moving_gradient_codes": qgd_config.moving_gradient_codes,
            "static_update": qgd_config.is_static,
            "finite_register_semantics": (
                "weights are updated on a cyclic grid modulo n_levels; gradients "
                "outside [gradient_min, gradient_max] saturate at the boundary. "
                "This is not gradient clipping."
            ),
        },
        "training": {
            **asdict(training_config),
            "initial_gradient_mode": initial_gradient_mode,
            "initial_gradient_shots": (
                training_config.shots if measured_execution_enabled else None
            ),
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
                if model_config.executor is None
                else "host computes parameter-shift differences from measured "
                "executor circuits using finite shots; this is classical "
                "orchestration and postprocessing, not a quantum gradient register."
            ),
            "later_gradients": (
                "Qibo circuits use configured finite shots, or exact simulation "
                "when shots is null; the host reads expectations and assembles "
                "parameter-shift gradients."
            ),
            "weight_update": (
                "one measured arithmetic Qibo circuit per scalar weight; host "
                "encodes the gradient register and decodes the unique modal "
                "bitstring from counts. The update circuit applies the modular "
                "subtraction; update shots are recorded in the resource estimate. "
                "No state-vector access is used by this path."
            ),
            "control_and_selection": (
                "loss calculation, validation checkpoint selection, stopping, "
                "and test metric postprocessing are classical."
            ),
            "hardware_status": (
                "The CLI uses the local Qibo simulator. No vendor adapter, "
                "transpilation, connectivity validation, or hardware execution "
                "is selected or claimed."
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
    parser.add_argument(
        "--shots",
        type=_parse_shots,
        default=100,
        help="positive finite-shot budget, or 'exact' for noiseless simulator mode",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help=(
            "simulator-only local depolarizing sensitivity probability "
            "(not a physical noise calibration)"
        ),
    )
    parser.add_argument(
        "--measured-execution",
        action="store_true",
        help=(
            "use a measured Qibo executor for the initial gradient and all "
            "predictions even when --noise is zero"
        ),
    )
    parser.add_argument(
        "--noise-simulation",
        choices=("exact-channel", "legacy-shared-trajectory"),
        default="exact-channel",
        help=(
            "exact-channel (default): density matrix for QCNN circuits and one "
            "independent Pauli trajectory per shot for QGD update circuits; "
            "legacy-shared-trajectory reproduces the November 2026 profiles "
            "(one Qibo trajectory per circuit execution shared by all shots)"
        ),
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
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(command_arguments)
    if args.shots is None and (args.noise > 0.0 or args.measured_execution):
        parser.error(
            "--shots exact cannot be combined with --measured-execution or non-zero --noise."
        )
    try:
        output_path = _resolve_new_output_path(args.output)
        provenance = _collect_run_provenance(command_arguments)
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
        requested_noise_config = ExecutionConfig(
            depolarizing_probability=args.noise
        )
    except ValueError as exc:
        parser.error(str(exc))

    if qgd_config.is_static:
        print(
            "WARNING: static QGD configuration "
            f"(n_bits={qgd_config.n_bits}, learning_rate={qgd_config.learning_rate}): "
            "0 gradient codes move a weight; the run only quantizes the weights.",
            file=sys.stderr,
        )
    measured_execution_enabled = (
        args.measured_execution or requested_noise_config.noisy
    )
    legacy_noise = args.noise_simulation == "legacy-shared-trajectory"
    qcnn_noise_simulation = (
        "legacy_shared_trajectory" if legacy_noise else "density_matrix"
    )
    update_noise_simulation = (
        "legacy_shared_trajectory" if legacy_noise else "per_shot_trajectories"
    )
    simulator_executor = None
    update_executor = None
    if measured_execution_enabled:
        def _executor(simulation: str, scope: str):
            return QiboSimulatorExecutor(
                capabilities=ResourceProfile(
                    metadata={
                        "adapter": "qibo-local-simulator",
                        "scope": scope,
                        "noise_model": (
                            SIMULATOR_NOISE_MODEL
                            if requested_noise_config.noisy
                            else "none"
                        ),
                        "noise_probability": (
                            requested_noise_config.depolarizing_probability
                        ),
                        "noise_simulation": (
                            simulation if requested_noise_config.noisy else None
                        ),
                        "execution_mode": "measured-counts",
                    }
                ),
                noise_transform=(
                    qibo_local_depolarizing_sensitivity_transform(
                        requested_noise_config.depolarizing_probability,
                        simulation=simulation,
                    )
                    if requested_noise_config.noisy
                    else None
                ),
            )

        simulator_executor = _executor(qcnn_noise_simulation, "QCNN circuits")
        update_executor = _executor(
            update_noise_simulation, "QGD arithmetic update circuits"
        )
    # Noise is supplied by the shared executor on both QCNN and QGD circuits;
    # leave the built-in QCNN-only model disabled to avoid double application.
    execution_config = ExecutionConfig(depolarizing_probability=0.0)

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
        executor=simulator_executor,
    )
    initial_weights = initialize_weights(
        model_config.n_qubits,
        n_outputs=model_config.n_outputs,
        seed=args.initialization_seed,
    )
    try:
        training_result = train_quantum_qcnn(
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            initial_weights,
            model_config,
            qgd_config,
            training_config,
            update_executor=update_executor,
        )
    except ValueError as exc:
        parser.error(str(exc))

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
        executor=model_config.executor,
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
        provenance=provenance,
        simulator_noise_probability=requested_noise_config.depolarizing_probability,
        simulator_noise_model=(
            SIMULATOR_NOISE_MODEL
            if requested_noise_config.noisy
            else "none"
        ),
        simulator_noise_applied_to=(
            ("QCNN circuits", "QGD arithmetic update circuits")
            if requested_noise_config.noisy
            else ()
        ),
        measured_execution_requested=args.measured_execution,
    )
    payload["model"]["simulator_noise"]["simulation"] = (
        {
            "qcnn_circuits": qcnn_noise_simulation,
            "qgd_update_circuits": update_noise_simulation,
        }
        if requested_noise_config.noisy
        else None
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
