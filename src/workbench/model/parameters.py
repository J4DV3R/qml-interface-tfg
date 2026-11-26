"""Declarative parameter schemas.

Nodes and algorithms describe their settings with these classes. The view
builds its forms from them and the model validates values with them, so adding
an algorithm or a node never requires interface code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


class ParameterError(ValueError):
    """A parameter value is invalid."""


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    default: Any = None
    help: str = ""
    # Show the field only when other parameters take one of these values.
    visible_when: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    group: str = "General"

    kind = "param"

    def validate(self, value: Any) -> Any:
        return value

    def is_visible(self, values: Mapping[str, Any]) -> bool:
        return all(values.get(name) in allowed for name, allowed in self.visible_when.items())

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "label": self.label, "default": self.default,
                "help": self.help, "group": self.group}


@dataclass(frozen=True)
class IntParam(Param):
    minimum: int | None = None
    maximum: int | None = None
    optional: bool = False  # None allowed (e.g. "no patience")

    kind = "int"

    def validate(self, value: Any) -> Any:
        if value is None or value == "":
            if self.optional:
                return None
            raise ParameterError(f"«{self.label}» es obligatorio.")
        if isinstance(value, bool):
            raise ParameterError(f"«{self.label}» debe ser un entero.")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"«{self.label}» debe ser un entero.") from exc
        if isinstance(value, float) and not value.is_integer():
            raise ParameterError(f"«{self.label}» debe ser un entero.")
        if self.minimum is not None and number < self.minimum:
            raise ParameterError(f"«{self.label}» debe ser ≥ {self.minimum}.")
        if self.maximum is not None and number > self.maximum:
            raise ParameterError(f"«{self.label}» debe ser ≤ {self.maximum}.")
        return number


@dataclass(frozen=True)
class FloatParam(Param):
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 4
    step: float = 0.01

    kind = "float"

    def validate(self, value: Any) -> Any:
        if isinstance(value, bool):
            raise ParameterError(f"«{self.label}» debe ser un número.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"«{self.label}» debe ser un número.") from exc
        if not math.isfinite(number):
            raise ParameterError(f"«{self.label}» debe ser finito.")
        if self.minimum is not None and number < self.minimum:
            raise ParameterError(f"«{self.label}» debe ser ≥ {self.minimum}.")
        if self.maximum is not None and number > self.maximum:
            raise ParameterError(f"«{self.label}» debe ser ≤ {self.maximum}.")
        return number


@dataclass(frozen=True)
class ChoiceParam(Param):
    # Sequence of (value, label) pairs.
    choices: Sequence[tuple[Any, str]] = ()

    kind = "choice"

    def validate(self, value: Any) -> Any:
        allowed = [choice for choice, _ in self.choices]
        if value not in allowed:
            raise ParameterError(f"«{self.label}»: valor {value!r} no permitido.")
        return value


@dataclass(frozen=True)
class BoolParam(Param):
    kind = "bool"

    def validate(self, value: Any) -> Any:
        if not isinstance(value, bool):
            raise ParameterError(f"«{self.label}» debe ser verdadero o falso.")
        return value


@dataclass(frozen=True)
class StringParam(Param):
    allow_empty: bool = True
    is_path: bool = False

    kind = "string"

    def validate(self, value: Any) -> Any:
        text = "" if value is None else str(value)
        if not text and not self.allow_empty:
            raise ParameterError(f"«{self.label}» es obligatorio.")
        return text


def defaults(schema: Sequence[Param]) -> dict[str, Any]:
    return {param.name: param.default for param in schema}


def validate_values(schema: Sequence[Param], values: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated values (defaults filled in); raise ParameterError with all problems."""
    result: dict[str, Any] = {}
    errors: list[str] = []
    merged = {**defaults(schema), **dict(values)}
    for param in schema:
        try:
            result[param.name] = param.validate(merged.get(param.name, param.default))
        except ParameterError as exc:
            errors.append(str(exc))
    unknown = set(values) - {param.name for param in schema}
    if unknown:
        errors.append("Parámetros desconocidos: " + ", ".join(sorted(unknown)))
    if errors:
        raise ParameterError("\n".join(errors))
    return result
