"""Derivative-free optimizers for quantum-training objective functions.

The module is intentionally independent of the QCNN engine.  It only requires
a callable ``objective(weights) -> float`` and keeps a hard evaluation budget,
which is the relevant unit when every objective call executes a quantum
circuit.
"""

from dataclasses import dataclass
import math
import time
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, minimize


_ALLOWED_METHODS = frozenset({"cobyla", "spsa", "nelder-mead"})


class _BudgetExhausted(Exception):
    """Internal signal used to stop SciPy without exceeding the budget."""


def _finite_float(value, name, *, positive=False):
    """Validate a real scalar and return it as a Python float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration shared by the derivative-free optimizers.

    ``max_evaluations`` counts every real call to the objective, including the
    initial point.  SPSA consumes two evaluations per iteration, so it may
    leave the last evaluation unused when the remaining budget is odd.
    """

    method: str
    max_evaluations: int
    seed: int = 0
    bounds: tuple[float, float] = (-np.pi, np.pi)
    rhobeg: float = 0.1
    tol: float = 1e-4
    spsa_a: float = 0.1
    spsa_c: float = 0.1
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_A: float = 10.0

    def __post_init__(self):
        if not isinstance(self.method, str) or self.method not in _ALLOWED_METHODS:
            allowed = ", ".join(sorted(_ALLOWED_METHODS))
            raise ValueError(f"method must be one of: {allowed}.")

        if (
            not isinstance(self.max_evaluations, (int, np.integer))
            or isinstance(self.max_evaluations, (bool, np.bool_))
            or self.max_evaluations < 1
        ):
            raise ValueError("max_evaluations must be a positive integer.")
        object.__setattr__(self, "max_evaluations", int(self.max_evaluations))

        if (
            not isinstance(self.seed, (int, np.integer))
            or isinstance(self.seed, (bool, np.bool_))
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")
        object.__setattr__(self, "seed", int(self.seed))

        if isinstance(self.bounds, (str, bytes)):
            raise ValueError("bounds must be a pair (lower, upper).")
        try:
            lower, upper = self.bounds
        except (TypeError, ValueError) as exc:
            raise ValueError("bounds must be a pair (lower, upper).") from exc
        lower = _finite_float(lower, "bounds[0]")
        upper = _finite_float(upper, "bounds[1]")
        if not lower < upper:
            raise ValueError("bounds must satisfy lower < upper.")
        object.__setattr__(self, "bounds", (lower, upper))

        object.__setattr__(
            self, "rhobeg", _finite_float(self.rhobeg, "rhobeg", positive=True)
        )
        object.__setattr__(self, "tol", _finite_float(self.tol, "tol", positive=True))

        for name in (
            "spsa_a",
            "spsa_c",
            "spsa_alpha",
            "spsa_gamma",
            "spsa_A",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, positive=True),
            )


@dataclass
class OptimizationResult:
    """Outcome of a derivative-free optimization run.

    ``weights`` is the best point actually evaluated, not necessarily the last
    iterate.  ``loss_history`` has exactly ``evaluations`` entries, one per
    real objective call, in call order.
    """

    weights: np.ndarray
    loss_history: list[float]
    best_loss: float
    evaluations: int
    elapsed_seconds: float
    method: str


class _ObjectiveRecorder:
    """Count, log, and validate objective calls while enforcing the budget."""

    def __init__(self, objective, max_evaluations, initial_weights):
        self._objective = objective
        self._max_evaluations = max_evaluations
        self._loss_history = []
        self._best_loss = math.inf
        self._best_weights = np.array(initial_weights, dtype=float, copy=True)

    @property
    def loss_history(self):
        return self._loss_history

    @property
    def evaluations(self):
        return len(self._loss_history)

    @property
    def best_loss(self):
        return self._best_loss

    @property
    def best_weights(self):
        return np.array(self._best_weights, dtype=float, copy=True)

    def __call__(self, weights):
        if self.evaluations >= self._max_evaluations:
            raise _BudgetExhausted

        candidate = np.asarray(weights, dtype=float)
        if candidate.ndim != 1:
            raise ValueError("optimizer supplied weights that are not one-dimensional.")
        value = self._evaluate(candidate.copy())

        self._loss_history.append(value)
        if value < self._best_loss:
            self._best_loss = value
            self._best_weights = candidate.copy()
        return value

    def _evaluate(self, candidate):
        try:
            raw_value = self._objective(candidate)
        except _BudgetExhausted:
            raise

        try:
            as_array = np.asarray(raw_value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "objective must return a finite real scalar."
            ) from exc
        if as_array.ndim != 0:
            raise ValueError("objective must return a scalar, not an array.")
        value = float(as_array)
        if not math.isfinite(value):
            raise ValueError("objective must return a finite scalar.")
        return value


def _run_cobyla(recorder, start, config):
    """Run COBYLA and swallow only the internal budget-stop signal."""
    n_weights = start.size
    remaining = config.max_evaluations - recorder.evaluations
    options = {
        # ``maxfun`` is the historical COBYLA option name and the one kept in
        # the public contract.  SciPy 1.18 renamed the functional cap to
        # ``maxiter`` (maximum function evaluations), so both are provided.
        "maxfun": config.max_evaluations,
        "maxiter": max(remaining, n_weights + 2),
        "rhobeg": config.rhobeg,
        "tol": config.tol,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown solver options.*",
            category=OptimizeWarning,
        )
        try:
            minimize(
                recorder,
                np.array(start, dtype=float, copy=True),
                method="COBYLA",
                bounds=[config.bounds] * n_weights,
                options=options,
            )
        except _BudgetExhausted:
            pass


def _run_nelder_mead(recorder, start, config):
    """Run Nelder-Mead with the remaining budget as its function cap."""
    remaining = config.max_evaluations - recorder.evaluations
    try:
        minimize(
            recorder,
            np.array(start, dtype=float, copy=True),
            method="Nelder-Mead",
            bounds=[config.bounds] * start.size,
            options={
                "maxfev": max(remaining, 1),
                "xatol": config.tol,
                "fatol": config.tol,
            },
        )
    except _BudgetExhausted:
        pass


def _run_spsa(recorder, start, config):
    """Run simultaneous perturbation stochastic approximation (SPSA)."""
    rng = np.random.default_rng(config.seed)
    lower, upper = config.bounds
    weights = np.array(start, dtype=float, copy=True)
    iteration = 0

    # Two objective calls are needed per iteration.  If only one evaluation
    # remains, it is deliberately left unused.
    while config.max_evaluations - recorder.evaluations >= 2:
        direction = (
            rng.integers(0, 2, size=weights.size).astype(float) * 2.0 - 1.0
        )
        a_k = config.spsa_a / (iteration + 1.0 + config.spsa_A) ** config.spsa_alpha
        c_k = config.spsa_c / (iteration + 1.0) ** config.spsa_gamma

        plus = np.clip(weights + c_k * direction, lower, upper)
        minus = np.clip(weights - c_k * direction, lower, upper)
        loss_plus = recorder(plus)
        loss_minus = recorder(minus)

        gradient = (loss_plus - loss_minus) / (2.0 * c_k * direction)
        weights = np.clip(weights - a_k * gradient, lower, upper)
        iteration += 1


def _build_result(recorder, config, elapsed_seconds):
    history = list(recorder.loss_history)
    return OptimizationResult(
        weights=recorder.best_weights,
        loss_history=history,
        best_loss=float(min(history)),
        evaluations=len(history),
        elapsed_seconds=float(elapsed_seconds),
        method=config.method,
    )


def optimize(objective, initial_weights, config):
    """Minimize ``objective`` from ``initial_weights`` with a hard budget.

    Parameters
    ----------
    objective:
        Callable accepting a one-dimensional ``numpy.ndarray`` and returning a
        finite real scalar.
    initial_weights:
        Non-empty, finite, one-dimensional starting point.  Values outside
        ``config.bounds`` are clipped to the feasible interval before the first
        evaluation.
    config:
        Validated :class:`OptimizerConfig`.

    Returns
    -------
    OptimizationResult
        Best evaluated point, complete loss history, and evaluation count.
    """
    if not isinstance(config, OptimizerConfig):
        raise TypeError("config must be an OptimizerConfig instance.")
    if not callable(objective):
        raise TypeError("objective must be callable.")

    try:
        start = np.asarray(initial_weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "initial_weights must be a finite one-dimensional array."
        ) from exc
    if start.ndim != 1:
        raise ValueError("initial_weights must be one-dimensional.")
    if start.size == 0:
        raise ValueError("initial_weights must not be empty.")
    if not np.all(np.isfinite(start)):
        raise ValueError("initial_weights must contain only finite values.")

    lower, upper = config.bounds
    start = np.clip(np.array(start, dtype=float, copy=True), lower, upper)

    started_at = time.perf_counter()
    recorder = _ObjectiveRecorder(objective, config.max_evaluations, start)
    recorder(start)

    if recorder.evaluations < config.max_evaluations:
        if config.method == "cobyla":
            _run_cobyla(recorder, start, config)
        elif config.method == "nelder-mead":
            _run_nelder_mead(recorder, start, config)
        else:
            _run_spsa(recorder, start, config)

    elapsed_seconds = time.perf_counter() - started_at
    return _build_result(recorder, config, elapsed_seconds)


__all__ = ["OptimizationResult", "OptimizerConfig", "optimize"]
