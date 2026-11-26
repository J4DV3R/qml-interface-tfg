"""QML algorithm plugins selectable in the QML node.

Add an algorithm by subclassing :class:`QMLAlgorithm` and decorating it with
:func:`register_algorithm`; the QML node lists it and builds its settings form
from ``parameter_schema()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..data import Splits
from ..node import RunContext
from ..parameters import Param


@dataclass
class FitOutcome:
    """What an algorithm returns after training.

    ``predict(x)`` gives real-valued outputs (sign = class). It is called by the
    QML node on test once, then on validation and train, at the checkpoint the
    algorithm selected using validation only.
    """

    predict: Any
    history: list[dict[str, Any]]
    checkpoint: dict[str, Any]
    model_state: dict[str, Any]
    cost: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class QMLAlgorithm(ABC):
    algorithm_id: str = ""
    label: str = ""
    description: str = ""
    n_features: int | None = None  # required input features (None = any)

    @abstractmethod
    def parameter_schema(self, values: Mapping[str, Any]) -> list[Param]:
        """Hyperparameters (may depend on the current values)."""

    def cross_validate(self, values: Mapping[str, Any]) -> list[str]:
        return []

    @abstractmethod
    def fit(self, splits: Splits, params: Mapping[str, Any], context: RunContext) -> FitOutcome:
        """Train using train and validation only; never touch ``splits.test_*``."""


_ALGORITHMS: dict[str, QMLAlgorithm] = {}


def register_algorithm(cls):
    instance = cls()
    if not instance.algorithm_id or instance.algorithm_id in _ALGORITHMS:
        raise ValueError(f"invalid or duplicate algorithm id {instance.algorithm_id!r}")
    _ALGORITHMS[instance.algorithm_id] = instance
    return cls


def algorithms() -> list[QMLAlgorithm]:
    _ensure_builtin()
    return list(_ALGORITHMS.values())


def get_algorithm(algorithm_id: str) -> QMLAlgorithm:
    _ensure_builtin()
    try:
        return _ALGORITHMS[algorithm_id]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm {algorithm_id!r}") from exc


def _ensure_builtin() -> None:
    from . import qcnn  # noqa: F401


__all__ = ["FitOutcome", "QMLAlgorithm", "algorithms", "get_algorithm", "register_algorithm"]
