"""NumPy/Qibo training orchestration for the discrete-QGD QCNN increment.

This module is deliberately independent of PyTorch. In the local simulator
path, an exact parameter-shift gradient initializes the first update; later
gradients use the configured shots, or remain exact when ``shots=None``. An
injected measured-circuit executor requires finite shots from the first
gradient. Weight updates use measured reversible arithmetic circuits from
``quantum_update``. Measurement decoding, losses, validation-based checkpoint
selection, early stopping, and loop control remain classical host operations.
This is therefore not an all-quantum training procedure.

Circuit-execution counts are estimates derived from the circuit structure, not
runtime instrumentation. For every gradient evaluation, the engine executes
one unshifted forward circuit per training sample and two circuits for every
parameter occurrence and sample. A logical shared weight has one occurrence
per adjacent pair in each convolution layer. The reversible update executes
one Qibo circuit per scalar weight. Post-update train and validation losses
each execute one prediction circuit per sample.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import math
import time
from typing import Any
import warnings

import numpy as np

if __package__:
    from .circuit_execution import CircuitExecutor, QiboSimulatorExecutor
    from .execution import ExecutionConfig
    from .qcnn_engine import (
        QCNNConfig,
        initialize_weights,
        parameter_shift_gradient,
        predict,
    )
    from .quantum_update import (
        QGDConfig,
        estimate_update_resources,
        quantum_update_step,
    )
else:  # Allows running this file from the src directory during development.
    from circuit_execution import CircuitExecutor, QiboSimulatorExecutor
    from execution import ExecutionConfig
    from qcnn_engine import (
        QCNNConfig,
        initialize_weights,
        parameter_shift_gradient,
        predict,
    )
    from quantum_update import QGDConfig, estimate_update_resources, quantum_update_step


_UINT32_MODULUS = 1 << 32
_DEFAULT_UPDATE_EXECUTOR = object()
_SEED_PURPOSES = {
    "initial_gradient": 0,
    "sampled_gradient": 1,
    "train_evaluation": 2,
    "validation_evaluation": 3,
    "test_evaluation": 4,
    "weight_update": 5,
}


@dataclass(frozen=True)
class QuantumTrainingConfig:
    """Budget and sampling settings for :func:`train_quantum_qcnn`.

    ``max_steps`` counts QGD weight updates. ``shots`` is used for predictions
    and parameter-shift gradients after initialization. ``None`` selects exact,
    noiseless local simulation for all QCNN predictions/gradients; the measured
    arithmetic update still uses one simulator shot because its API returns
    counts. An injected adapter requires finite shots. The QGD learning rate
    and finite-register settings belong to ``QGDConfig``.
    """

    max_steps: int = 1
    shots: int | None = 100
    execution_seed: int = 101
    patience: int | None = None

    def __post_init__(self):
        if (
            not isinstance(self.max_steps, (int, np.integer))
            or isinstance(self.max_steps, (bool, np.bool_))
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer.")
        if self.shots is not None and (
            not isinstance(self.shots, (int, np.integer))
            or isinstance(self.shots, (bool, np.bool_))
            or self.shots < 1
        ):
            raise ValueError("shots must be None or a positive integer.")
        if self.shots is not None:
            object.__setattr__(self, "shots", int(self.shots))
        if (
            not isinstance(self.execution_seed, (int, np.integer))
            or isinstance(self.execution_seed, (bool, np.bool_))
            or not 0 <= self.execution_seed < _UINT32_MODULUS
        ):
            raise ValueError("execution_seed must be an integer in [0, 2**32).")
        if self.patience is not None and (
            not isinstance(self.patience, (int, np.integer))
            or isinstance(self.patience, (bool, np.bool_))
            or self.patience < 1
        ):
            raise ValueError("patience must be None or a positive integer.")


@dataclass(frozen=True)
class QuantumTrainingResult:
    """Training output; validation chooses ``best_weights`` and test is absent."""

    best_weights: np.ndarray
    final_weights: np.ndarray
    best_step: int
    steps_completed: int
    history: tuple[dict[str, Any], ...]
    elapsed_seconds: float
    circuit_executions_estimate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this result."""
        return {
            "best_weights": self.best_weights.tolist(),
            "final_weights": self.final_weights.tolist(),
            "best_step": int(self.best_step),
            "steps_completed": int(self.steps_completed),
            "history": [copy.deepcopy(item) for item in self.history],
            "elapsed_seconds": float(self.elapsed_seconds),
            "circuit_executions_estimate": copy.deepcopy(
                self.circuit_executions_estimate
            ),
        }


def derive_execution_seed(
    execution_seed: int,
    component: str,
    step: int = 0,
) -> int:
    """Derive a stable uint32 seed for one logical execution component.

    NumPy's ``SeedSequence`` is used with explicit entropy components; no
    Python hash or ambient random-generator state participates in derivation.
    A single resulting seed is passed to each engine call, so that its
    unshifted forward pass and all parameter-shift circuits use the same
    deterministic random stream for that gradient evaluation.
    """
    if (
        not isinstance(execution_seed, (int, np.integer))
        or isinstance(execution_seed, (bool, np.bool_))
        or not 0 <= execution_seed < _UINT32_MODULUS
    ):
        raise ValueError("execution_seed must be an integer in [0, 2**32).")
    if component not in _SEED_PURPOSES:
        raise ValueError(f"unsupported execution seed component: {component!r}.")
    if (
        not isinstance(step, (int, np.integer))
        or isinstance(step, (bool, np.bool_))
        or step < 0
    ):
        raise ValueError("step must be a non-negative integer.")
    state = np.random.SeedSequence(
        [int(execution_seed), _SEED_PURPOSES[component], int(step)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


@contextmanager
def _preserve_numpy_random_state():
    """Keep the engine's legacy global-NumPy seeding local to this operation."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def _as_finite_array(values, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_partition(
    features,
    targets,
    *,
    name: str,
    n_qubits: int,
    n_outputs: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_values = _as_finite_array(features, f"{name}_features")
    if (
        feature_values.ndim != 2
        or feature_values.shape[0] == 0
        or feature_values.shape[1] != n_qubits
    ):
        raise ValueError(
            f"{name}_features must have non-empty shape (batch, {n_qubits})."
        )
    if np.any((feature_values < 0.0) | (feature_values > 1.0)):
        raise ValueError(f"{name}_features must be normalized to [0, 1].")

    target_values = _as_finite_array(targets, f"{name}_targets")
    if n_outputs == 1 and target_values.ndim == 1:
        target_values = target_values.reshape(-1, 1)
    expected_shape = (feature_values.shape[0], n_outputs)
    if target_values.shape != expected_shape:
        if n_outputs == 1:
            raise ValueError(
                f"{name}_targets must have shape ({expected_shape[0]},) "
                f"or {expected_shape}."
            )
        raise ValueError(f"{name}_targets must have shape {expected_shape}.")
    if not np.all((target_values == -1.0) | (target_values == 1.0)):
        raise ValueError(f"{name}_targets must use the -1/+1 encoding.")
    return feature_values, target_values


def _parameter_occurrences_per_sample(model_config: QCNNConfig) -> int:
    """Count occurrences across all four shared weights in all QCNN layers."""
    active_count = model_config.n_qubits
    pair_count = 0
    for _ in range(model_config.n_layers):
        pair_count += active_count - 1
        if active_count > model_config.n_outputs:
            active_count //= 2
    return 4 * pair_count


def _binary_metrics(predictions, targets) -> dict[str, float]:
    """Calculate binary classification metrics from one prediction pass."""
    predicted_values = np.asarray(predictions, dtype=float).reshape(-1)
    target_values = np.asarray(targets, dtype=float).reshape(-1)
    predicted_labels = np.where(predicted_values >= 0.0, 1.0, -1.0)
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
    }


def _evaluate_partition(weights, features, targets, model_config):
    """Return loss and classification metrics from one circuit pass per sample."""
    predictions = predict(weights, features, model_config)
    loss = float(np.mean((predictions - targets) ** 2))
    return loss, _binary_metrics(predictions, targets)


def _make_execution_model(
    model_config: QCNNConfig,
    *,
    shots: int | None,
    seed: int,
    exact_noiseless: bool = False,
) -> QCNNConfig:
    noise_probability = (
        0.0 if exact_noiseless else model_config.execution.depolarizing_probability
    )
    return QCNNConfig(
        n_qubits=model_config.n_qubits,
        n_outputs=model_config.n_outputs,
        encoding=model_config.encoding,
        execution=ExecutionConfig(
            shots=shots,
            depolarizing_probability=noise_probability,
            seed=seed,
        ),
        executor=model_config.executor,
    )


def _circuit_count_estimate(
    *,
    model_config: QCNNConfig,
    train_size: int,
    validation_size: int,
    steps: int,
    sampled_gradient_calls: int,
    shots: int | None,
    initial_gradient_shots: int | None,
    update_shots: int,
    qgd_config: QGDConfig,
    resource_profile,
) -> dict[str, Any]:
    occurrences = _parameter_occurrences_per_sample(model_config)
    shifts_per_gradient = 2 * occurrences * train_size
    base_forward = train_size
    gradient_call_circuits = base_forward + shifts_per_gradient
    weight_count = model_config.n_layers * 4

    initial_gradient = {
        "logical_gradient_evaluations": 1,
        "execution_mode": "exact_simulator" if initial_gradient_shots is None else "finite_shots",
        "base_prediction_logical_passes": 1,
        "base_prediction_circuit_executions_estimate": base_forward,
        "parameter_shift_circuit_executions_estimate": shifts_per_gradient,
        "shots_per_circuit": initial_gradient_shots,
        "circuit_executions_estimate": gradient_call_circuits,
    }
    subsequent_gradients = {
        "logical_gradient_evaluations": sampled_gradient_calls,
        "execution_mode": "exact_simulator" if shots is None else "finite_shots",
        "base_prediction_logical_passes": sampled_gradient_calls,
        "base_prediction_circuit_executions_estimate": (
            sampled_gradient_calls * base_forward
        ),
        "parameter_shift_circuit_executions_estimate": (
            sampled_gradient_calls * shifts_per_gradient
        ),
        "shots_per_circuit": shots,
        "circuit_executions_estimate": (
            sampled_gradient_calls * gradient_call_circuits
        ),
    }
    update_circuits = steps * weight_count
    train_eval_circuits = steps * train_size
    validation_eval_circuits = steps * validation_size
    components = {
        "initial_gradient": initial_gradient,
        "subsequent_gradients": subsequent_gradients,
        "qgd_weight_updates": {
            "logical_update_steps": steps,
            "circuits_per_step": weight_count,
            "shots_per_circuit": update_shots,
            "circuit_executions_estimate": update_circuits,
            "qubits_per_circuit": 2 * qgd_config.n_bits,
            "shots_total_estimate": update_circuits * update_shots,
            "circuit_resources": estimate_update_resources(
                qgd_config,
                n_weights=weight_count,
                shots=update_shots,
                resource_profile=resource_profile,
            ),
        },
        "post_update_train_evaluation": {
            "logical_batch_evaluations": steps,
            "samples_per_batch": train_size,
            "shots_per_circuit": shots,
            "circuit_executions_estimate": train_eval_circuits,
        },
        "validation_evaluation": {
            "logical_batch_evaluations": steps,
            "samples_per_batch": validation_size,
            "shots_per_circuit": shots,
            "circuit_executions_estimate": validation_eval_circuits,
        },
    }
    total = sum(
        component["circuit_executions_estimate"]
        for component in components.values()
    )
    return {
        "accounting": (
            "Estimated Qibo circuit executions; one circuit per sample for a "
            "base/prediction pass, two per scalar parameter occurrence and "
            "sample for parameter shift, and one per scalar weight for QGD. "
            "This is structural accounting, not runtime instrumentation."
        ),
        "logical_weight_occurrences_per_sample": occurrences,
        "shifted_circuits_per_gradient": shifts_per_gradient,
        "train_samples": train_size,
        "validation_samples": validation_size,
        "steps_completed": steps,
        "qcnn_qubits_per_circuit": model_config.n_qubits,
        "declared_resource_profile": resource_profile.to_dict(),
        "components": components,
        "total_training_circuit_executions_estimate": total,
        "total_weight_update_shots_estimate": update_circuits * update_shots,
    }


def train_quantum_qcnn(
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    initial_weights,
    model_config: QCNNConfig,
    qgd_config: QGDConfig,
    training_config: QuantumTrainingConfig,
    *,
    update_executor: CircuitExecutor | None | object = _DEFAULT_UPDATE_EXECUTOR,
) -> QuantumTrainingResult:
    """Train a QCNN using exact or measured parameter-shift gradients.

    Without an injected executor, the first parameter-shift gradient is exact
    and noiseless; later gradients and predictions use ``training_config.shots``
    (or remain exact when it is ``None``). With an injected measured executor,
    finite shots are required from the first gradient onward. Every scalar
    weight update executes a measured arithmetic circuit and decodes counts;
    an update uses one shot in exact-simulator mode, otherwise the configured
    shot budget. ``update_executor`` controls only these arithmetic circuits.
    If omitted, it defaults to ``model_config.executor`` for backwards
    compatibility; pass ``None`` explicitly to use an unmodified local
    simulator for updates while retaining a separate QCNN executor. Validation
    loss alone chooses the returned best checkpoint.

    This API intentionally has no test partition argument. Test evaluation is
    reserved for a separate final evaluation after training.
    """
    if not isinstance(model_config, QCNNConfig):
        raise TypeError("model_config must be a QCNNConfig instance.")
    if not isinstance(qgd_config, QGDConfig):
        raise TypeError("qgd_config must be a QGDConfig instance.")
    if not isinstance(training_config, QuantumTrainingConfig):
        raise TypeError("training_config must be a QuantumTrainingConfig instance.")
    if model_config.executor is not None and training_config.shots is None:
        raise ValueError("an injected circuit executor requires finite shots.")
    actual_update_executor = (
        model_config.executor
        if update_executor is _DEFAULT_UPDATE_EXECUTOR
        else update_executor
    )
    if actual_update_executor is not None and not isinstance(
        actual_update_executor, CircuitExecutor
    ):
        raise TypeError("update_executor must implement CircuitExecutor or be None.")
    if actual_update_executor is not None and training_config.shots is None:
        raise ValueError("an injected update executor requires finite shots.")
    if model_config.execution.noisy and model_config.executor is None:
        raise ValueError(
            "QGD arithmetic updates need an injected noise_transform for noisy "
            "runs; the built-in depolarizing setting alone covers QCNN gates."
        )
    if model_config.execution.noisy and model_config.executor is not None:
        raise ValueError(
            "configure noise through the injected executor and set the built-in "
            "depolarizing_probability to zero."
        )
    if qgd_config.is_static:
        warnings.warn(
            "QGD configuration is static: every gradient code has a zero weight "
            "shift, so weights are only quantized and never move "
            f"(n_bits={qgd_config.n_bits}, learning_rate={qgd_config.learning_rate}).",
            RuntimeWarning,
            stacklevel=2,
        )

    train_x, train_y = _validate_partition(
        train_features,
        train_targets,
        name="train",
        n_qubits=model_config.n_qubits,
        n_outputs=model_config.n_outputs,
    )
    validation_x, validation_y = _validate_partition(
        validation_features,
        validation_targets,
        name="validation",
        n_qubits=model_config.n_qubits,
        n_outputs=model_config.n_outputs,
    )
    weights = _as_finite_array(initial_weights, "initial_weights")
    expected_weight_shape = (model_config.n_layers, 4)
    if weights.shape != expected_weight_shape:
        raise ValueError(f"initial_weights must have shape {expected_weight_shape}.")
    weights = weights.copy()

    update_shots = training_config.shots if training_config.shots is not None else 1
    resource_profile = (
        actual_update_executor.capabilities
        if actual_update_executor is not None
        else QiboSimulatorExecutor().capabilities
    )
    if training_config.shots is not None:
        model_resource_profile = (
            model_config.executor.capabilities
            if model_config.executor is not None
            else QiboSimulatorExecutor().capabilities
        )
        model_resource_profile.validate_request(
            model_config.n_qubits, training_config.shots
        )
    resource_profile.validate_request(2 * qgd_config.n_bits, update_shots)

    start_time = time.perf_counter()
    initial_gradient_seed = derive_execution_seed(
        training_config.execution_seed, "initial_gradient", 0
    )
    initial_gradient_shots = (
        training_config.shots if model_config.executor is not None else None
    )
    exact_model = _make_execution_model(
        model_config,
        shots=initial_gradient_shots,
        seed=initial_gradient_seed,
        exact_noiseless=model_config.executor is None,
    )
    # The engine seeds the legacy global NumPy RNG internally. Preserve the
    # caller's generator state while retaining a stable stream for this gradient.
    with _preserve_numpy_random_state():
        gradient = parameter_shift_gradient(
            weights, train_x, train_y, exact_model
        )

    history: list[dict[str, Any]] = []
    best_weights: np.ndarray | None = None
    best_validation_loss = math.inf
    best_step = 0
    stale_steps = 0
    sampled_gradient_calls = 0
    gradient_mode = (
        "exact_initialization"
        if model_config.executor is None
        else "parameter_shift_shots"
    )
    gradient_seed = initial_gradient_seed
    gradient_shots: int | None = initial_gradient_shots

    for step in range(1, training_config.max_steps + 1):
        update_seed = derive_execution_seed(
            training_config.execution_seed, "weight_update", step
        )
        updated_flat, update_report = quantum_update_step(
            weights.reshape(-1),
            gradient.reshape(-1),
            qgd_config,
            shots=update_shots,
            executor=actual_update_executor,
            seed=update_seed,
            return_report=True,
        )
        updated = _as_finite_array(updated_flat, "updated_weights")
        if updated.shape != (weights.size,):
            raise RuntimeError("quantum_update_step returned an unexpected shape.")
        weights = updated.reshape(expected_weight_shape)

        train_seed = derive_execution_seed(
            training_config.execution_seed, "train_evaluation", step
        )
        validation_seed = derive_execution_seed(
            training_config.execution_seed, "validation_evaluation", step
        )
        train_execution = _make_execution_model(
            model_config, shots=training_config.shots, seed=train_seed
        )
        validation_execution = _make_execution_model(
            model_config, shots=training_config.shots, seed=validation_seed
        )
        with _preserve_numpy_random_state():
            train_loss, train_metrics = _evaluate_partition(
                weights, train_x, train_y, train_execution
            )
        with _preserve_numpy_random_state():
            validation_loss, validation_metrics = _evaluate_partition(
                weights, validation_x, validation_y, validation_execution
            )
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise RuntimeError("training produced a non-finite loss.")

        record = {
            "step": step,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "gradient_mode": gradient_mode,
            "gradient": gradient.reshape(-1).tolist(),
            "gradient_saturated_fraction": float(
                np.mean(
                    (gradient < qgd_config.gradient_min)
                    | (gradient > qgd_config.gradient_max)
                )
            ),
            "weights_after_update": weights.reshape(-1).tolist(),
            "gradient_execution_seed": gradient_seed,
            "gradient_shots": gradient_shots,
            "update_execution_seed": update_seed,
            "update_shots_per_weight": update_shots,
            "update_mode_probabilities": list(update_report["mode_probabilities"]),
            "update_minimum_mode_probability": update_report[
                "minimum_mode_probability"
            ],
            "train_evaluation_execution_seed": train_seed,
            "validation_evaluation_execution_seed": validation_seed,
        }
        history.append(record)

        # Checkpoint selection is based exclusively on the validation loss.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = weights.copy()
            best_step = step
            stale_steps = 0
        else:
            stale_steps += 1

        should_stop = (
            training_config.patience is not None
            and stale_steps >= training_config.patience
        )
        if step == training_config.max_steps or should_stop:
            break

        gradient_seed = derive_execution_seed(
            training_config.execution_seed, "sampled_gradient", step
        )
        sampled_execution = _make_execution_model(
            model_config,
            shots=training_config.shots,
            seed=gradient_seed,
        )
        with _preserve_numpy_random_state():
            gradient = parameter_shift_gradient(
                weights, train_x, train_y, sampled_execution
            )
        gradient = _as_finite_array(gradient, "gradient")
        if gradient.shape != expected_weight_shape:
            raise RuntimeError("parameter_shift_gradient returned an unexpected shape.")
        sampled_gradient_calls += 1
        gradient_mode = (
            "parameter_shift_exact"
            if training_config.shots is None
            else "parameter_shift_shots"
        )
        gradient_shots = training_config.shots

    if best_weights is None:  # max_steps is positive; defensive only.
        raise RuntimeError("training completed without a validation checkpoint.")

    elapsed_seconds = time.perf_counter() - start_time
    count_estimate = _circuit_count_estimate(
        model_config=model_config,
        train_size=len(train_x),
        validation_size=len(validation_x),
        steps=len(history),
        sampled_gradient_calls=sampled_gradient_calls,
        shots=training_config.shots,
        initial_gradient_shots=initial_gradient_shots,
        update_shots=update_shots,
        qgd_config=qgd_config,
        resource_profile=resource_profile,
    )
    return QuantumTrainingResult(
        best_weights=best_weights.copy(),
        final_weights=weights.copy(),
        best_step=best_step,
        steps_completed=len(history),
        history=tuple(copy.deepcopy(record) for record in history),
        elapsed_seconds=float(elapsed_seconds),
        circuit_executions_estimate=count_estimate,
    )


__all__ = [
    "QuantumTrainingConfig",
    "QuantumTrainingResult",
    "derive_execution_seed",
    "initialize_weights",
    "train_quantum_qcnn",
]
