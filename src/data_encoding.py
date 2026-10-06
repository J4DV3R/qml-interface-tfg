"""Data-encoding circuits used by the QML models."""

from collections.abc import Sequence

import numpy as np
from qibo import gates
from qibo.models import Circuit


def _validated_features(features: Sequence[float]) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("features must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("features must be normalized to the [0, 1] interval.")
    return values


def _validated_shift(
    shifted_feature: tuple[int, float] | None,
    feature_count: int,
) -> tuple[int, float]:
    if shifted_feature is None:
        return -1, 0.0
    feature_index, shift = shifted_feature
    if not 0 <= feature_index < feature_count:
        raise ValueError("shifted feature index is out of range.")
    if not np.isfinite(shift):
        raise ValueError("feature shift must be finite.")
    return feature_index, shift


def add_angle_encoding(
    circuit: Circuit,
    features: Sequence[float],
    shifted_feature: tuple[int, float] | None = None,
) -> Circuit:
    """Append RY angle encoding, mapping each feature x to RY(pi * x)."""
    values = _validated_features(features)
    if len(values) > circuit.nqubits:
        raise ValueError("the circuit must have at least one qubit per feature.")
    feature_index, shift = _validated_shift(shifted_feature, len(values))

    for qubit, value in enumerate(values):
        theta = np.pi * value
        if qubit == feature_index:
            theta += shift
        circuit.add(gates.RY(qubit, theta=float(theta)))
    return circuit


def angle_encoding(features: Sequence[float]) -> Circuit:
    """Create a circuit using RY angle encoding for normalized features."""
    values = _validated_features(features)
    circuit = Circuit(len(values))
    return add_angle_encoding(circuit, values)


def add_phase_encoding(
    circuit: Circuit,
    features: Sequence[float],
    shifted_feature: tuple[int, float] | None = None,
) -> Circuit:
    """Append phase encoding using H followed by RZ(2*pi*x) on each qubit."""
    values = _validated_features(features)
    if len(values) > circuit.nqubits:
        raise ValueError("the circuit must have at least one qubit per feature.")
    feature_index, shift = _validated_shift(shifted_feature, len(values))

    for qubit, value in enumerate(values):
        circuit.add(gates.H(qubit))
        theta = 2.0 * np.pi * value
        if qubit == feature_index:
            theta += shift
        circuit.add(gates.RZ(qubit, theta=float(theta)))
    return circuit


def encoding_scale(encoding: str) -> float:
    """Return the angle scale used by a supported feature encoding."""
    if encoding == "angle":
        return np.pi
    if encoding == "phase":
        return 2.0 * np.pi
    raise ValueError("encoding must be 'angle' or 'phase'.")


def phase_encoding(features: Sequence[float]) -> Circuit:
    """Create a circuit using phase encoding for normalized features."""
    values = _validated_features(features)
    circuit = Circuit(len(values))
    return add_phase_encoding(circuit, values)


if __name__ == "__main__":
    print(angle_encoding([0.0, 0.5, 1.0]).draw())
    print(phase_encoding([0.0, 0.5, 1.0]).draw())
