"""Node types, typed ports, run context and the node registry.

A *node type* is a plugin: it declares typed input/output ports, a parameter
schema and a ``run`` method. Registering it makes it available in the palette,
in templates and in the headless runner; no view code is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .parameters import Param, ParameterError, validate_values


class Cancelled(Exception):
    """Raised inside a node when the user cancels the execution."""


@dataclass(frozen=True)
class PortSpec:
    name: str
    data_type: str
    label: str


class RunContext:
    """What a running node may use: progress, log and cancellation."""

    def __init__(self, progress: Callable[[float, str], None] | None = None,
                 log: Callable[[str], None] | None = None,
                 cancelled: Callable[[], bool] | None = None):
        self._progress = progress or (lambda fraction, message: None)
        self._log = log or (lambda message: None)
        self._cancelled = cancelled or (lambda: False)

    def progress(self, fraction: float, message: str = "") -> None:
        self._progress(max(0.0, min(1.0, float(fraction))), message)

    def log(self, message: str) -> None:
        self._log(message)

    @property
    def is_cancelled(self) -> bool:
        return bool(self._cancelled())

    def check_cancelled(self) -> None:
        if self.is_cancelled:
            raise Cancelled()


class NodeType:
    """Base class for node plugins."""

    type_id: str = ""
    title: str = ""
    category: str = "General"
    description: str = ""
    inputs: Sequence[PortSpec] = ()
    outputs: Sequence[PortSpec] = ()

    def parameter_schema(self, values: Mapping[str, Any] | None = None) -> list[Param]:
        """Schema for the current values (it may depend on them, e.g. the chosen algorithm)."""
        return []

    def default_parameters(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        # Resolve dynamic schemas iteratively (a choice may add fields).
        for _ in range(3):
            schema = self.parameter_schema(values)
            values = {param.name: values.get(param.name, param.default) for param in schema}
        return values

    def validate_parameters(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Validate (and complete with defaults) the parameter values."""
        schema = self.parameter_schema(values)
        known = {param.name for param in schema}
        # Values for fields that the current choice hides are dropped, not errors.
        filtered = {k: v for k, v in values.items() if k in known}
        validated = validate_values(schema, filtered)
        problems = self.cross_validate(validated)
        if problems:
            raise ParameterError("\n".join(problems))
        return validated

    def cross_validate(self, values: Mapping[str, Any]) -> list[str]:
        """Constraints involving several parameters."""
        return []

    def run(self, inputs: Mapping[str, Any], params: Mapping[str, Any], context: RunContext) -> dict[str, Any]:
        raise NotImplementedError

    def input_port(self, name: str) -> PortSpec:
        return next(p for p in self.inputs if p.name == name)

    def output_port(self, name: str) -> PortSpec:
        return next(p for p in self.outputs if p.name == name)


_NODE_TYPES: dict[str, NodeType] = {}


def register_node(cls):
    """Class decorator: instantiate and register a node type."""
    instance = cls()
    if not instance.type_id:
        raise ValueError("node types need a type_id")
    if instance.type_id in _NODE_TYPES:
        raise ValueError(f"duplicate node type {instance.type_id}")
    _NODE_TYPES[instance.type_id] = instance
    return cls


def get_node_type(type_id: str) -> NodeType:
    _ensure_builtin()
    try:
        return _NODE_TYPES[type_id]
    except KeyError as exc:
        raise KeyError(f"unknown node type {type_id!r}") from exc


def node_types() -> list[NodeType]:
    _ensure_builtin()
    return sorted(_NODE_TYPES.values(), key=lambda n: (n.category, n.title))


def _ensure_builtin() -> None:
    from . import nodes  # noqa: F401  (registers the built-in node types)
