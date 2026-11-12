"""Variational (hybrid quantum-classical) training of the QCNN without PyTorch.

This is the standard execution model for parameterized circuits on current
hardware: the quantum device only evaluates circuits (expectation values from
measured shots) and a classical optimizer updates the weights. Four update
rules are provided, all using the same circuit evaluations and the same
validation-only checkpoint selection:

``gd``
    Gradient descent with parameter-shift gradients: ``w <- w - lr * g``.
``adam``
    Adam (Kingma & Ba) on parameter-shift gradients; same formulas as
    ``torch.optim.Adam`` with default betas/epsilon, but implemented in NumPy.
``spsa``
    Simultaneous-perturbation stochastic approximation (Spall): two loss
    evaluations per step, whatever the number of weights.
``sequential``
    Sequential exact coordinate minimization (Rotosolve / NFT family). Each
    shared weight enters ``R`` RY/RZ gates, so every Z expectation is a
    trigonometric polynomial of degree ``R`` in that weight (also under the
    depolarizing/readout noise used here, which does not depend on the
    weight). ``2R + 1`` equally spaced evaluations reconstruct it exactly for
    every sample; the MSE is then minimized over a fine grid and the weight is
    set to that minimizer. No learning rate is needed.

The training API has no test argument: the caller evaluates test once, after
the validation checkpoint is selected. Step 0 (initial weights) is a valid
checkpoint candidate. Circuit executions are counted structurally: one per
sample for a prediction, ``1 + 2 * occurrences`` per sample for a
parameter-shift gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Protocol

import numpy as np

if __package__:
    from .execution import ExecutionConfig
    from .fast_qcnn import FastQCNN, FastQCNNConfig, sample_expectations, weight_occurrences
    from .qcnn_engine import QCNNConfig, parameter_shift_gradient, predict
else:  # pragma: no cover - direct execution from src/
    from execution import ExecutionConfig
    from fast_qcnn import FastQCNN, FastQCNNConfig, sample_expectations, weight_occurrences
    from qcnn_engine import QCNNConfig, parameter_shift_gradient, predict


OPTIMIZERS = ("gd", "adam", "spsa", "sequential")


# ---------------------------------------------------------------- evaluators
class Evaluator(Protocol):
    n_qubits: int
    occurrences: list[int]

    def predict(self, weights, features, *, shots, rng) -> np.ndarray: ...

    def predict_variants(self, weight_variants, features, *, shots, rng) -> np.ndarray: ...

    def gradient(self, weights, features, targets, *, shots, rng) -> tuple[np.ndarray, int]: ...


class FastEvaluator:
    """Vectorized NumPy evaluator (exact, noisy density matrix, or shots)."""

    backend = "fast-numpy"

    def __init__(self, config: FastQCNNConfig):
        if config.n_outputs != 1:
            raise ValueError("variational training supports n_outputs=1 (binary).")
        self.config = config
        self.model = FastQCNN(config)
        self.n_qubits = config.n_qubits
        self.occurrences = weight_occurrences(config.n_qubits, config.n_outputs)

    def predict(self, weights, features, *, shots, rng):
        exact = self.model.expectations(weights, features)[:, 0]
        return sample_expectations(exact, shots, rng)

    def predict_variants(self, weight_variants, features, *, shots, rng):
        variants = np.asarray(weight_variants, dtype=float)
        features = np.asarray(features, dtype=float)
        n_variants, n_samples = len(variants), len(features)
        angles = np.repeat(self.model.angles_from_weights(variants), n_samples, axis=0)
        feats = np.tile(features, (n_variants, 1))
        exact = self.model.expectations_from_angles(angles, feats)[:, 0]
        return sample_expectations(exact, shots, rng).reshape(n_variants, n_samples)

    def gradient(self, weights, features, targets, *, shots, rng):
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float)
        prediction = self.predict(weights, features, shots=shots, rng=rng)
        upstream = 2.0 * (prediction - targets) / len(targets)
        angles, feats, index, n_shift = self.model.shift_rows(weights, features)
        shifted = sample_expectations(
            self.model.expectations_from_angles(angles, feats)[:, 0], shots, rng
        ).reshape(len(features), n_shift)
        gradient = np.zeros_like(weights)
        for column, (layer, _pair, weight, sign) in enumerate(index):
            gradient[layer, weight] += 0.5 * sign * float(upstream @ shifted[:, column])
        circuits = len(features) * (1 + n_shift)
        return gradient, circuits


class QiboEvaluator:
    """Reference evaluator through the Qibo engine (slow; same semantics)."""

    backend = "qibo"

    def __init__(self, n_qubits: int, encoding: str = "angle", depolarizing_probability: float = 0.0,
                 readout_error: float = 0.0, executor=None):
        if readout_error:
            raise ValueError("readout_error is only modeled by the fast evaluator.")
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.depolarizing_probability = float(depolarizing_probability)
        self.executor = executor
        self.occurrences = weight_occurrences(n_qubits, 1)

    def _config(self, shots, rng) -> QCNNConfig:
        seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
        return QCNNConfig(
            self.n_qubits,
            1,
            self.encoding,
            execution=ExecutionConfig(
                shots=shots,
                depolarizing_probability=0.0 if self.executor else self.depolarizing_probability,
                seed=seed,
            ),
            executor=self.executor,
        )

    def predict(self, weights, features, *, shots, rng):
        return predict(weights, features, self._config(shots, rng))[:, 0]

    def predict_variants(self, weight_variants, features, *, shots, rng):
        return np.stack(
            [self.predict(w, features, shots=shots, rng=rng) for w in weight_variants]
        )

    def gradient(self, weights, features, targets, *, shots, rng):
        gradient = parameter_shift_gradient(weights, features, targets, self._config(shots, rng))
        circuits = len(features) * (1 + 2 * 4 * sum(self.occurrences))
        return gradient, circuits


# ---------------------------------------------------------------- metrics
def binary_metrics(predictions, targets) -> dict[str, float]:
    """Accuracy/precision/recall/F1 for -1/+1 labels, threshold at 0."""
    predicted = np.where(np.asarray(predictions, dtype=float).reshape(-1) >= 0.0, 1.0, -1.0)
    actual = np.asarray(targets, dtype=float).reshape(-1)
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual != 1) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted != 1)))
    tn = int(np.sum((actual != 1) & (predicted != 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(np.mean(predicted == actual)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(
            0.5 * (recall + (tn / (tn + fp) if tn + fp else 0.0))
        ),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "support": int(actual.size),
    }


def mse(predictions, targets) -> float:
    return float(np.mean((np.asarray(predictions).reshape(-1) - np.asarray(targets).reshape(-1)) ** 2))


# ---------------------------------------------------------------- config
@dataclass(frozen=True)
class VariationalTrainingConfig:
    optimizer: str
    max_steps: int = 100
    learning_rate: float = 0.05
    shots: int | None = None
    patience: int | None = None
    max_optimization_circuits: int | None = None
    evaluate_train_every: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    spsa_a: float = 0.2
    spsa_c: float = 0.1
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_stability: float | None = None
    sequential_grid_points: int = 2048

    def __post_init__(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of: {', '.join(OPTIMIZERS)}.")
        for name in ("max_steps", "evaluate_train_every", "sequential_grid_points"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("shots", "patience", "max_optimization_circuits"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be None or a positive integer.")
        for name in ("learning_rate", "adam_epsilon", "spsa_a", "spsa_c", "spsa_alpha", "spsa_gamma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("adam_beta1", "adam_beta2"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        if self.spsa_stability is not None and (not math.isfinite(self.spsa_stability) or self.spsa_stability < 0):
            raise ValueError("spsa_stability must be None or non-negative.")

    def effective_hyperparameters(self) -> dict[str, Any]:
        if self.optimizer in ("gd", "adam"):
            out = {"learning_rate": self.learning_rate}
            if self.optimizer == "adam":
                out.update(beta1=self.adam_beta1, beta2=self.adam_beta2, epsilon=self.adam_epsilon)
            return out
        if self.optimizer == "spsa":
            return {
                "a": self.spsa_a, "c": self.spsa_c, "alpha": self.spsa_alpha,
                "gamma": self.spsa_gamma,
                "A": self.spsa_stability if self.spsa_stability is not None else 0.1 * self.max_steps,
            }
        return {"grid_points": self.sequential_grid_points, "learning_rate": None}


@dataclass
class VariationalTrainingResult:
    initial_weights: np.ndarray
    best_weights: np.ndarray
    final_weights: np.ndarray
    best_step: int
    best_validation_loss: float
    steps_completed: int
    stop_reason: str
    history: list[dict[str, Any]] = field(default_factory=list)
    optimization_circuits: int = 0
    evaluation_circuits: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_weights": self.initial_weights.tolist(),
            "best_weights": self.best_weights.tolist(),
            "final_weights": self.final_weights.tolist(),
            "best_step": self.best_step,
            "best_validation_loss": self.best_validation_loss,
            "steps_completed": self.steps_completed,
            "stop_reason": self.stop_reason,
            "optimization_circuits": self.optimization_circuits,
            "evaluation_circuits": self.evaluation_circuits,
            "elapsed_seconds": self.elapsed_seconds,
            "history": self.history,
        }


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _sequential_sweep(weights, train_x, train_y, evaluator, config, rng):
    """One Rotosolve/NFT-style sweep over every shared weight."""
    weights = weights.copy()
    circuits = 0
    targets = np.asarray(train_y, dtype=float).reshape(-1)
    grid = 2.0 * np.pi * np.arange(config.sequential_grid_points) / config.sequential_grid_points
    for layer in range(weights.shape[0]):
        degree = evaluator.occurrences[layer]
        points = 2 * degree + 1
        offsets = 2.0 * np.pi * np.arange(points) / points
        frequencies = np.arange(-degree, degree + 1)
        basis_at_points = np.exp(-1j * np.outer(frequencies, offsets)) / points  # (K, M)
        basis_on_grid = np.exp(1j * np.outer(grid, frequencies))  # (G, K)
        for index in range(4):
            theta0 = weights[layer, index]
            variants = np.repeat(weights[None], points, axis=0)
            variants[:, layer, index] = theta0 + offsets
            values = evaluator.predict_variants(variants, train_x, shots=config.shots, rng=rng)  # (M, N)
            circuits += points * len(train_x)
            coefficients = basis_at_points @ values  # (K, N)
            reconstructed = np.real(basis_on_grid @ coefficients)  # (G, N)
            losses = np.mean((reconstructed - targets[None, :]) ** 2, axis=1)
            weights[layer, index] = _wrap(theta0 + grid[int(np.argmin(losses))])
    return weights, circuits


def train_variational_qcnn(
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    initial_weights,
    evaluator: Evaluator,
    config: VariationalTrainingConfig,
    rng: np.random.Generator,
) -> VariationalTrainingResult:
    """Train with the configured optimizer; select the checkpoint on validation only."""
    if not isinstance(config, VariationalTrainingConfig):
        raise TypeError("config must be a VariationalTrainingConfig instance.")
    train_x = np.asarray(train_features, dtype=float)
    train_y = np.asarray(train_targets, dtype=float).reshape(-1)
    val_x = np.asarray(validation_features, dtype=float)
    val_y = np.asarray(validation_targets, dtype=float).reshape(-1)
    for name, x, y in (("train", train_x, train_y), ("validation", val_x, val_y)):
        if x.ndim != 2 or x.shape[1] != evaluator.n_qubits or len(x) != len(y) or not len(x):
            raise ValueError(f"{name} partition has an inconsistent shape.")
        if np.any((x < 0) | (x > 1)) or not np.all(np.isin(y, (-1.0, 1.0))):
            raise ValueError(f"{name} features must be in [0, 1] and labels in -1/+1.")
    weights = np.asarray(initial_weights, dtype=float).copy()
    if weights.shape != (len(evaluator.occurrences), 4) or not np.isfinite(weights).all():
        raise ValueError("initial_weights have an unexpected shape or non-finite values.")

    start = time.perf_counter()
    shots = config.shots
    optimization_circuits = 0
    evaluation_circuits = 0
    adam_m = np.zeros_like(weights)
    adam_v = np.zeros_like(weights)
    stability = (
        config.spsa_stability if config.spsa_stability is not None else 0.1 * config.max_steps
    )

    def evaluate(step, current, extra):
        nonlocal evaluation_circuits
        record = {"step": step, **extra}
        if step == 0 or step % config.evaluate_train_every == 0:
            train_pred = evaluator.predict(current, train_x, shots=shots, rng=rng)
            evaluation_circuits += len(train_x)
            record["train_loss"] = mse(train_pred, train_y)
            record["train_metrics"] = binary_metrics(train_pred, train_y)
        val_pred = evaluator.predict(current, val_x, shots=shots, rng=rng)
        evaluation_circuits += len(val_x)
        record["validation_loss"] = mse(val_pred, val_y)
        record["validation_metrics"] = binary_metrics(val_pred, val_y)
        record["optimization_circuits"] = optimization_circuits
        record["evaluation_circuits"] = evaluation_circuits
        record["weights"] = current.reshape(-1).tolist()
        return record

    history = [evaluate(0, weights, {})]
    best_weights, best_loss, best_step = weights.copy(), history[0]["validation_loss"], 0
    stale = 0
    stop_reason = "max_steps"

    for step in range(1, config.max_steps + 1):
        extra: dict[str, Any] = {}
        if config.optimizer in ("gd", "adam"):
            gradient, circuits = evaluator.gradient(weights, train_x, train_y, shots=shots, rng=rng)
            optimization_circuits += circuits
            extra["gradient_norm"] = float(np.linalg.norm(gradient))
            if config.optimizer == "gd":
                weights = weights - config.learning_rate * gradient
            else:
                adam_m = config.adam_beta1 * adam_m + (1 - config.adam_beta1) * gradient
                adam_v = config.adam_beta2 * adam_v + (1 - config.adam_beta2) * gradient**2
                m_hat = adam_m / (1 - config.adam_beta1**step)
                v_hat = adam_v / (1 - config.adam_beta2**step)
                weights = weights - config.learning_rate * m_hat / (np.sqrt(v_hat) + config.adam_epsilon)
        elif config.optimizer == "spsa":
            a_k = config.spsa_a / (stability + step) ** config.spsa_alpha
            c_k = config.spsa_c / step**config.spsa_gamma
            delta = rng.choice((-1.0, 1.0), size=weights.shape)
            plus = mse(evaluator.predict(weights + c_k * delta, train_x, shots=shots, rng=rng), train_y)
            minus = mse(evaluator.predict(weights - c_k * delta, train_x, shots=shots, rng=rng), train_y)
            optimization_circuits += 2 * len(train_x)
            estimate = (plus - minus) / (2.0 * c_k) * delta
            extra["gradient_norm"] = float(np.linalg.norm(estimate))
            weights = weights - a_k * estimate
        else:
            weights, circuits = _sequential_sweep(weights, train_x, train_y, evaluator, config, rng)
            optimization_circuits += circuits

        if not np.isfinite(weights).all():
            raise RuntimeError("training produced non-finite weights.")
        record = evaluate(step, weights, extra)
        history.append(record)
        if record["validation_loss"] < best_loss:
            best_weights, best_loss, best_step, stale = weights.copy(), record["validation_loss"], step, 0
        else:
            stale += 1
        if config.patience is not None and stale >= config.patience:
            stop_reason = "patience"
            break
        if (
            config.max_optimization_circuits is not None
            and optimization_circuits >= config.max_optimization_circuits
        ):
            stop_reason = "circuit_budget"
            break

    return VariationalTrainingResult(
        initial_weights=np.asarray(initial_weights, dtype=float).copy(),
        best_weights=best_weights,
        final_weights=weights.copy(),
        best_step=best_step,
        best_validation_loss=float(best_loss),
        steps_completed=len(history) - 1,
        stop_reason=stop_reason,
        history=history,
        optimization_circuits=optimization_circuits,
        evaluation_circuits=evaluation_circuits,
        elapsed_seconds=time.perf_counter() - start,
    )


__all__ = [
    "OPTIMIZERS",
    "FastEvaluator",
    "QiboEvaluator",
    "VariationalTrainingConfig",
    "VariationalTrainingResult",
    "binary_metrics",
    "mse",
    "train_variational_qcnn",
]
