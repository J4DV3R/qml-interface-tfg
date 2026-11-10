"""Measured, reversible Qibo circuits for discrete QGD weight updates.

The new execution path builds an arithmetic circuit from controlled-X gates,
measures the output registers, and decodes the most frequent output bitstring.
It does not require a dense ``Unitary`` gate or access to a state vector. The
gradient register is preserved, making
``(theta, grad) -> (theta - lr * grad mod n_levels, grad)`` reversible. This
modular rule is not gradient clipping; out-of-range gradients saturate while
being encoded.

The arithmetic schedule is synthesized on the host from the fixed QGD grid
configuration and covers every gradient-register basis value. The actual
weight/gradient data enter as computational-basis inputs, so the per-update
subtraction is carried out by the circuit. A backend adapter still needs to
translate/decompose the multi-controlled gates; no hardware transpiler or
backend is selected here.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from qibo import gates
from qibo.models import Circuit

if __package__:
    from .circuit_execution import (
        CircuitExecutor,
        QiboSimulatorExecutor,
        ResourceProfile,
        validate_counts,
    )
else:
    from circuit_execution import (
        CircuitExecutor,
        QiboSimulatorExecutor,
        ResourceProfile,
        validate_counts,
    )


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
    if weight_values.size == 0:
        raise ValueError("weights and gradients must not be empty.")
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


def _delta_for_gradient_code(gradient_code: int, config: QGDConfig) -> int:
    gradient = decode_gradient(gradient_code, config)
    scaled_delta = (
        config.learning_rate
        * gradient
        * (config.n_levels - 1)
        / (config.weight_max - config.weight_min)
    )
    if not math.isfinite(scaled_delta):
        raise ValueError("learning_rate and gradient range produce a non-finite update.")
    return int(np.rint(scaled_delta)) % config.n_levels


def _append_arithmetic_update(circuit: Circuit, config: QGDConfig) -> None:
    """Append a reversible controlled subtractor for every gradient code.

    Qibo numbers qubit 0 as the most-significant bit. The weight and gradient
    registers each occupy ``n_bits`` contiguous qubits; bit lists below are in
    least-significant-first order to make the ripple-borrow schedule explicit.
    Multi-controlled X gates are logical operations and may require backend
    decomposition before physical execution.
    """
    bits = config.n_bits
    weight_qubits = [bits - 1 - bit for bit in range(bits)]
    gradient_qubits = [2 * bits - 1 - bit for bit in range(bits)]

    for gradient_code in range(config.n_levels):
        # Turn the selected gradient value into the all-ones control word.
        zero_gradient_qubits = [
            gradient_qubits[bit]
            for bit in range(bits)
            if not ((gradient_code >> bit) & 1)
        ]
        for qubit in zero_gradient_qubits:
            circuit.add(gates.X(qubit))

        delta = _delta_for_gradient_code(gradient_code, config)
        for start_bit in range(bits):
            if not ((delta >> start_bit) & 1):
                continue
            # Subtracting 2**start_bit borrows while all lower affected bits
            # are zero. Apply high-to-low so every control sees its input value.
            for target_bit in range(bits - 1, start_bit - 1, -1):
                target = weight_qubits[target_bit]
                lower_bits = weight_qubits[start_bit:target_bit]
                zero_lower_bits = list(lower_bits)
                for qubit in zero_lower_bits:
                    circuit.add(gates.X(qubit))
                controls = (*gradient_qubits, *lower_bits)
                circuit.add(gates.X(target).controlled_by(*controls))
                for qubit in reversed(zero_lower_bits):
                    circuit.add(gates.X(qubit))

        for qubit in reversed(zero_gradient_qubits):
            circuit.add(gates.X(qubit))


def build_qgd_update_circuit(config: QGDConfig, *, measure: bool = True) -> Circuit:
    """Build the unprepared arithmetic circuit for one weight/gradient pair."""
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    if not isinstance(measure, bool):
        raise TypeError("measure must be a bool.")
    circuit = Circuit(2 * config.n_bits)
    _append_arithmetic_update(circuit, config)
    if measure:
        circuit.add(gates.M(*range(2 * config.n_bits)))
    return circuit


def build_update_circuit(
    weight_code: int,
    gradient_code: int,
    config: QGDConfig,
) -> Circuit:
    """Prepare one pair of registers, apply the generic update, and measure."""
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    _decode(weight_code, config.weight_min, config.weight_max, config)
    _decode(gradient_code, config.gradient_min, config.gradient_max, config)
    bits = config.n_bits
    circuit = Circuit(2 * bits)
    for bit in range(bits):
        if (weight_code >> bit) & 1:
            circuit.add(gates.X(bits - 1 - bit))
        if (gradient_code >> bit) & 1:
            circuit.add(gates.X(2 * bits - 1 - bit))
    _append_arithmetic_update(circuit, config)
    circuit.add(gates.M(*range(2 * bits)))
    return circuit


@lru_cache(maxsize=8)
def _cached_unitary(n_bits, weight_min, weight_max, gradient_min, gradient_max, learning_rate):
    """Legacy dense reference permutation; not used by measured updates."""
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
    """Return the legacy dense reference permutation (size ``2**(2b)``).

    Kept for API compatibility and small-register checks. The production update
    functions below do not call this helper or construct a dense unitary.
    """
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    return _cached_unitary(config.n_bits, config.weight_min, config.weight_max,
                           config.gradient_min, config.gradient_max, config.learning_rate)


def decode_update_counts(
    counts,
    config: QGDConfig,
    *,
    shots: int,
    minimum_mode_probability: float = 0.5,
    expected_gradient_code: int | None = None,
) -> tuple[float, float]:
    """Decode the modal weight code; reject ties or a low-confidence mode.

    Under noise, all measured registers (including the preserved gradient)
    may be corrupted. The policy selects the unique most frequent complete
    bitstring and raises ``RuntimeError`` if its frequency is below the
    configured threshold. It cannot detect a confidently wrong mode.
    """
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    if (
        isinstance(minimum_mode_probability, bool)
        or not isinstance(minimum_mode_probability, (int, float, np.number))
        or not math.isfinite(minimum_mode_probability)
        or not 0.0 < minimum_mode_probability <= 1.0
    ):
        raise ValueError("minimum_mode_probability must be in (0, 1].")
    if expected_gradient_code is not None:
        _decode(
            expected_gradient_code,
            config.gradient_min,
            config.gradient_max,
            config,
        )
    normalized = validate_counts(counts, n_bits=2 * config.n_bits, shots=shots)
    ranked = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise RuntimeError("QGD output counts have no unique modal bitstring.")
    confidence = ranked[0][1] / shots
    if confidence < minimum_mode_probability:
        raise RuntimeError(
            "QGD output modal probability "
            f"{confidence:.6f} is below the required "
            f"{minimum_mode_probability:.6f}."
        )
    output_bits = ranked[0][0]
    measured_gradient_code = int(output_bits[config.n_bits :], 2)
    if expected_gradient_code is not None and measured_gradient_code != expected_gradient_code:
        raise RuntimeError("measured QGD gradient register differs from its input code.")
    weight_code = int(output_bits[: config.n_bits], 2)
    return decode_weight(weight_code, config), float(confidence)


def _derive_weight_seed(seed: int | None, index: int) -> int | None:
    if seed is None:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer or None.")
    return int(
        np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint32)[0]
    )


def _executor_and_profile(executor: CircuitExecutor | None):
    if executor is None:
        executor = QiboSimulatorExecutor()
    if not isinstance(executor, CircuitExecutor):
        raise TypeError("executor must implement CircuitExecutor.")
    if not isinstance(executor.capabilities, ResourceProfile):
        raise TypeError("executor.capabilities must be a ResourceProfile instance.")
    return executor


def _validate_update_shots(shots: int | None, executor: CircuitExecutor | None) -> int:
    if shots is None:
        if executor is not None:
            raise ValueError("an injected circuit executor requires an explicit finite shots value.")
        return 1
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be None or a positive integer.")
    return shots


def quantum_update_weight(
    weight: float,
    gradient: float,
    config: QGDConfig,
    *,
    shots: int | None = None,
    executor: CircuitExecutor | None = None,
    seed: int | None = None,
    minimum_mode_probability: float = 0.5,
    return_report: bool = False,
):
    """Update one scalar weight with a measured circuit and return its result.

    Omitting ``shots`` preserves the former no-argument simulator call by using
    one deterministic simulator shot. An injected executor requires an explicit
    finite-shot budget. Sampled counts use the unique-mode confidence policy.
    """
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    if not isinstance(return_report, bool):
        raise TypeError("return_report must be a bool.")
    shots = _validate_update_shots(shots, executor)
    weight_value, gradient_value = _validated_arrays([weight], [gradient])
    executor = _executor_and_profile(executor)
    executor.capabilities.validate_request(2 * config.n_bits, shots)
    circuit = build_update_circuit(
        encode_weight(weight_value[0], config),
        encode_gradient(gradient_value[0], config),
        config,
    )
    counts = executor.execute_counts(circuit, shots=shots, seed=seed)
    updated_weight, confidence = decode_update_counts(
        counts,
        config,
        shots=shots,
        minimum_mode_probability=minimum_mode_probability,
        expected_gradient_code=encode_gradient(gradient_value[0], config),
    )
    if not return_report:
        return updated_weight
    report = estimate_update_resources(config, n_weights=1, shots=shots)
    report["mode_probabilities"] = [confidence]
    report["minimum_mode_probability"] = float(minimum_mode_probability)
    report["decoding_policy"] = "unique modal complete bitstring; reject ties/low confidence"
    report["resource_profile"] = executor.capabilities.to_dict()
    return updated_weight, report


def estimate_update_resources(
    config: QGDConfig,
    *,
    n_weights: int,
    shots: int,
    resource_profile: ResourceProfile | None = None,
) -> dict:
    """Estimate untranspiled logical Qibo resources for measured updates.

    The depth is a serial upper bound from the queued logical instructions;
    no routing, native-gate decomposition or backend transpilation is modeled.
    """
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    if not isinstance(n_weights, int) or isinstance(n_weights, bool) or n_weights < 1:
        raise ValueError("n_weights must be a positive integer.")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer.")
    arithmetic_instructions = len(build_qgd_update_circuit(config, measure=False).queue)
    # At most one input X per qubit; the generic arithmetic circuit includes
    # the final measurement instruction when requested.
    instruction_upper_bound = arithmetic_instructions + 2 * config.n_bits + 1
    result = {
        "resource_estimate_scope": (
            "Untranspiled Qibo logical instructions; serial depth upper bound. "
            "Excludes backend routing/decomposition, calibration, queueing, and "
            "provider-specific overhead."
        ),
        "qubits_per_weight_circuit": 2 * config.n_bits,
        "shots_per_weight_circuit": shots,
        "weight_circuits": n_weights,
        "total_shots_estimate": n_weights * shots,
        "arithmetic_gate_count_estimate": arithmetic_instructions,
        "gate_count_per_circuit_upper_bound": instruction_upper_bound,
        "serialized_depth_upper_bound_estimate": instruction_upper_bound,
    }
    if resource_profile is not None:
        if not isinstance(resource_profile, ResourceProfile):
            raise TypeError("resource_profile must be a ResourceProfile instance.")
        resource_profile.validate_request(2 * config.n_bits, shots)
        result["resource_profile"] = resource_profile.to_dict()
    return result


def quantum_update_step(
    weights,
    gradients,
    config,
    *,
    shots: int | None = None,
    executor: CircuitExecutor | None = None,
    seed: int | None = None,
    minimum_mode_probability: float = 0.5,
    return_report: bool = False,
):
    """Apply one measured Qibo update per weight and decode modal counts.

    With a noiseless simulator, one shot is sufficient because the update
    circuit maps each computational-basis input deterministically. On noisy
    execution, finite counts can corrupt the output; the most frequent complete
    bitstring is selected, ties/low-confidence modes are rejected, and a
    confidently wrong mode remains possible. No state-vector API is used.
    """
    if not isinstance(config, QGDConfig):
        raise TypeError("config must be a QGDConfig instance.")
    if not isinstance(return_report, bool):
        raise TypeError("return_report must be a bool.")
    shots = _validate_update_shots(shots, executor)
    weight_values, gradient_values = _validated_arrays(weights, gradients)
    executor = _executor_and_profile(executor)
    executor.capabilities.validate_request(2 * config.n_bits, shots)
    result = np.empty_like(weight_values)
    confidences = []
    for index, (weight, gradient) in enumerate(zip(weight_values, gradient_values)):
        theta_code = encode_weight(weight, config)
        gradient_code = encode_gradient(gradient, config)
        circuit = build_update_circuit(theta_code, gradient_code, config)
        circuit_seed = _derive_weight_seed(seed, index)
        counts = executor.execute_counts(circuit, shots=shots, seed=circuit_seed)
        result[index], confidence = decode_update_counts(
            counts,
            config,
            shots=shots,
            minimum_mode_probability=minimum_mode_probability,
            expected_gradient_code=gradient_code,
        )
        confidences.append(confidence)
    if not return_report:
        return result
    report = estimate_update_resources(
        config,
        n_weights=len(weight_values),
        shots=shots,
        resource_profile=executor.capabilities,
    )
    report["mode_probabilities"] = confidences
    report["minimum_mode_probability"] = float(minimum_mode_probability)
    report["decoding_policy"] = "unique modal complete bitstring; reject ties/low confidence"
    return result, report


__all__ = [
    "QGDConfig",
    "build_qgd_update_circuit",
    "build_update_circuit",
    "build_update_unitary",
    "classical_grid_update",
    "decode_gradient",
    "decode_update_counts",
    "decode_weight",
    "encode_gradient",
    "encode_weight",
    "estimate_update_resources",
    "quantum_update_step",
    "quantum_update_weight",
]
