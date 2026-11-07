"""Discrete QGD weight updates using a reversible Qibo permutation circuit.

The update is performed modulo the weight grid.  This is intentionally
reversible: the gradient register is preserved, so the map
``(theta, grad) -> (theta - lr * grad mod n_levels, grad)`` is a permutation.
Clipping cannot be used in a unitary map over the same two registers because
it is many-to-one.  The modulo boundary is a documented deviation from the
initial clipped-update proposal; angle parameters are periodic over a 2*pi
range, making the cyclic weight grid a natural prototype.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from qibo import gates
from qibo.models import Circuit


@dataclass(frozen=True)
class QGDConfig:
    n_bits: int = 5
    weight_min: float = -np.pi
    weight_max: float = np.pi
    gradient_min: float = -1.0
    gradient_max: float = 1.0
    learning_rate: float = 0.1

    def __post_init__(self):
        if (not isinstance(self.n_bits, int) or isinstance(self.n_bits, bool)
                or not 2 <= self.n_bits <= 5):
            raise ValueError("n_bits must be an integer between 2 and 5.")
        for name in ("weight_min", "weight_max", "gradient_min", "gradient_max", "learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if self.weight_min >= self.weight_max:
            raise ValueError("weight_min must be less than weight_max.")
        if self.gradient_min >= self.gradient_max:
            raise ValueError("gradient_min must be less than gradient_max.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

    @property
    def n_levels(self) -> int:
        return 1 << self.n_bits


def _encode(value, minimum, maximum, config):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be finite.")
    value = float(np.clip(value, minimum, maximum))
    return int(np.rint((value - minimum) * (config.n_levels - 1) / (maximum - minimum)))


def _decode(code, minimum, maximum, config):
    if not isinstance(code, (int, np.integer)) or isinstance(code, (bool, np.bool_)) or not 0 <= code < config.n_levels:
        raise ValueError("code must be an integer within the configured grid.")
    return float(minimum + int(code) * (maximum - minimum) / (config.n_levels - 1))


def encode_weight(value, config):
    return _encode(value, config.weight_min, config.weight_max, config)


def decode_weight(code, config):
    return _decode(code, config.weight_min, config.weight_max, config)


def encode_gradient(value, config):
    """Quantize a gradient, saturating values outside the configured range.

    Saturation is part of this finite-register prototype and changes the
    effective update when a true gradient exceeds the configured interval.
    Callers should track or inspect the fraction of saturated gradients when
    evaluating QGD; this encoding does not preserve out-of-range magnitudes.
    """
    return _encode(value, config.gradient_min, config.gradient_max, config)


def decode_gradient(code, config):
    return _decode(code, config.gradient_min, config.gradient_max, config)


def _validated_arrays(weights, gradients):
    try:
        weight_values = np.asarray(weights, dtype=float)
        gradient_values = np.asarray(gradients, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights and gradients must be finite one-dimensional arrays.") from exc
    if weight_values.ndim != 1 or gradient_values.ndim != 1:
        raise ValueError("weights and gradients must be one-dimensional.")
    if weight_values.shape != gradient_values.shape:
        raise ValueError("weights and gradients must have equal length.")
    if not np.isfinite(weight_values).all() or not np.isfinite(gradient_values).all():
        raise ValueError("weights and gradients must contain only finite values.")
    return weight_values, gradient_values


def classical_grid_update(weights, gradients, config):
    """Reference cyclic-grid update, used to verify the circuit result."""
    weight_values, gradient_values = _validated_arrays(weights, gradients)
    updated = np.empty_like(weight_values)
    for index, (weight, gradient) in enumerate(zip(weight_values, gradient_values)):
        theta_code = encode_weight(weight, config)
        gradient_code = encode_gradient(gradient, config)
        delta = int(np.rint(config.learning_rate * decode_gradient(gradient_code, config)
                             * (config.n_levels - 1) / (config.weight_max - config.weight_min)))
        updated[index] = decode_weight((theta_code - delta) % config.n_levels, config)
    return updated


@lru_cache(maxsize=8)
def _cached_unitary(n_bits, weight_min, weight_max, gradient_min, gradient_max, learning_rate):
    config = QGDConfig(n_bits, weight_min, weight_max, gradient_min, gradient_max, learning_rate)
    levels = config.n_levels
    dimension = levels * levels
    unitary = np.zeros((dimension, dimension), dtype=complex)
    for gradient_code in range(levels):
        gradient = decode_gradient(gradient_code, config)
        delta = int(np.rint(config.learning_rate * gradient * (levels - 1)
                             / (config.weight_max - config.weight_min)))
        for theta_code in range(levels):
            # Qibo indexes basis states with qubit 0 as the most significant
            # bit: the weight register occupies the high-order b bits.
            source = (theta_code << n_bits) + gradient_code
            target_theta = (theta_code - delta) % levels
            target = (target_theta << n_bits) + gradient_code
            unitary[target, source] = 1.0
    return unitary


def build_update_unitary(config):
    """Return the cached reversible lookup permutation (size 2**(2b))."""
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    return _cached_unitary(config.n_bits, config.weight_min, config.weight_max,
                           config.gradient_min, config.gradient_max, config.learning_rate)


def quantum_update_step(weights, gradients, config):
    """Apply one Qibo update per weight and decode its exact output state.

    The lookup table is a precomputed unitary permutation, not a circuit for
    arithmetic with carry. The simulator returns a state vector, whose
    deterministic computational-basis output is decoded classically; no shots
    or measurement gate are used for this update. The gradient code remains as
    garbage to make the transformation reversible.
    """
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    weight_values, gradient_values = _validated_arrays(weights, gradients)
    unitary = build_update_unitary(config)
    result = np.empty_like(weight_values)
    for index, (weight, gradient) in enumerate(zip(weight_values, gradient_values)):
        theta_code = encode_weight(weight, config)
        gradient_code = encode_gradient(gradient, config)
        circuit = Circuit(2 * config.n_bits)
        for bit in range(config.n_bits):
            qubit = config.n_bits - 1 - bit
            if (theta_code >> bit) & 1:
                circuit.add(gates.X(qubit))
            if (gradient_code >> bit) & 1:
                circuit.add(gates.X(2 * config.n_bits - 1 - bit))
        circuit.add(gates.Unitary(unitary, *range(2 * config.n_bits)))
        state = circuit.execute().state()
        output_index = int(np.argmax(np.abs(state) ** 2))
        if not np.isclose(abs(state[output_index]), 1.0, atol=1e-10):
            raise RuntimeError("QGD circuit did not produce a deterministic basis state.")
        result[index] = decode_weight(output_index >> config.n_bits, config)
    return result


__all__ = ["QGDConfig", "build_update_unitary", "classical_grid_update",
           "decode_gradient", "decode_weight", "encode_gradient", "encode_weight",
           "quantum_update_step"]
