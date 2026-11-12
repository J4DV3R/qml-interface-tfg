"""Pure Qibo/NumPy QCNN engine without PyTorch.

This module owns the circuit construction, execution and parameter-shift
differentiation of the one-dimensional QCNN. It deliberately does not import
PyTorch: the legacy hybrid model delegates its simulation here, while the new
non-hybrid training path can evaluate losses and gradients directly with NumPy.

The circuit layout is identical to the historical implementation:

* angle encoding ``RY(pi * x)`` or phase encoding ``H + RZ(2 * pi * x)``;
* translation-shared convolution filters over adjacent active qubits;
* logical pooling (the Qibo register is not physically reduced);
* Pauli-Z expectation values of the surviving qubits.
"""

from dataclasses import dataclass, field
import math

import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibo.noise import DepolarizingError, NoiseModel

if __package__:
    from .circuit_execution import (
        CircuitExecutor,
        QiboSimulatorExecutor,
        qibo_local_depolarizing_sensitivity_transform,
        validate_counts,
    )
    from .data_encoding import add_angle_encoding, add_phase_encoding
    from .execution import ExecutionConfig
    from .pooling import add_pooling_layer
    from .qcnn_circuit import add_conv_layer
else:
    from circuit_execution import (
        CircuitExecutor,
        QiboSimulatorExecutor,
        qibo_local_depolarizing_sensitivity_transform,
        validate_counts,
    )
    from data_encoding import add_angle_encoding, add_phase_encoding
    from execution import ExecutionConfig
    from pooling import add_pooling_layer
    from qcnn_circuit import add_conv_layer


@dataclass(frozen=True)
class QCNNConfig:
    """Static configuration of a QCNN circuit and its execution.

    The validation mirrors ``QCNNModel.__init__`` from the legacy hybrid path:
    ``n_qubits`` must be a power of two of at least two, ``n_outputs`` must be a
    smaller power of two, and their ratio must be a power of two so that every
    convolution layer can halve the number of active qubits.
    """

    n_qubits: int
    n_outputs: int = 1
    encoding: str = "angle"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    executor: CircuitExecutor | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.n_qubits, int) or self.n_qubits < 2:
            raise ValueError("n_qubits must be an integer of at least 2.")
        if self.n_qubits & (self.n_qubits - 1):
            raise ValueError(
                "n_qubits must be a power of two for this pooling layout."
            )
        if not isinstance(self.n_outputs, int) or self.n_outputs < 1:
            raise ValueError("n_outputs must be a positive integer.")
        if self.n_outputs >= self.n_qubits or self.n_outputs & (self.n_outputs - 1):
            raise ValueError(
                "n_outputs must be a power of two smaller than n_qubits."
            )

        layer_ratio = self.n_qubits // self.n_outputs
        if self.n_qubits % self.n_outputs or layer_ratio & (layer_ratio - 1):
            raise ValueError("n_qubits / n_outputs must be a power of two.")

        if self.encoding not in ("angle", "phase"):
            raise ValueError("encoding must be 'angle' or 'phase'.")
        if not isinstance(self.execution, ExecutionConfig):
            raise TypeError("execution must be an ExecutionConfig instance.")
        if self.executor is not None and not isinstance(self.executor, CircuitExecutor):
            raise TypeError("executor must implement CircuitExecutor or be None.")

    @property
    def n_layers(self) -> int:
        """Number of convolution layers needed to reach ``n_outputs``."""
        return int(math.log2(self.n_qubits // self.n_outputs))


def expectation_z(state: np.ndarray, n_qubits: int, qubit: int) -> float:
    """Return the Pauli-Z expectation value of ``qubit`` for a state vector.

    Qibo state vectors index computational basis states with qubit 0 as the
    most significant bit; hence its bit position is ``n_qubits - 1 - qubit``.
    """
    probabilities = np.abs(np.asarray(state).reshape(-1)) ** 2
    return expectation_z_from_probabilities(probabilities, n_qubits, qubit)


def expectation_z_from_probabilities(probabilities, n_qubits: int, qubit: int) -> float:
    """Pauli-Z expectation of ``qubit`` from computational-basis probabilities."""
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    basis_states = np.arange(probabilities.size)
    bit_position = n_qubits - 1 - qubit
    signs = 1.0 - 2.0 * ((basis_states >> bit_position) & 1)
    return float(np.dot(probabilities, signs).real)


QCNN_NOISE_GATE_NAMES = ("RY", "RZ", "CNOT")


def depolarizing_noise_model(probability: float) -> NoiseModel:
    """Build the legacy depolarizing noise model applied after RY, RZ and CNOT.

    Kept for ``noise_simulation="legacy_shared_trajectory"`` only. Qibo
    0.1.11 samples this model once per circuit execution (shared by all
    shots) and applies a two-qubit channel after CNOT.
    """
    noise_model = NoiseModel()
    error = DepolarizingError(probability)
    for gate in (gates.RY, gates.RZ, gates.CNOT):
        noise_model.add(error, gate)
    return noise_model


def _build_circuit_and_active(
    features,
    weights,
    config: QCNNConfig,
    shifted_feature: tuple[int, float] | None = None,
    shifted_weight: tuple[int, int, int, float] | None = None,
) -> tuple[Circuit, list[int]]:
    if not isinstance(config, QCNNConfig):
        raise TypeError("config must be a QCNNConfig instance.")

    circuit = Circuit(config.n_qubits)
    if config.encoding == "angle":
        add_angle_encoding(circuit, features, shifted_feature)
    elif config.encoding == "phase":
        add_phase_encoding(circuit, features, shifted_feature)
    else:  # Defensive: QCNNConfig already validates this.
        raise ValueError("encoding must be 'angle' or 'phase'.")

    active_qubits = list(range(config.n_qubits))
    for layer_index, layer_weights in enumerate(weights):
        layer_shift = None
        if shifted_weight is not None and shifted_weight[0] == layer_index:
            _, pair_index, weight_index, shift = shifted_weight
            layer_shift = (pair_index, weight_index, shift)
        add_conv_layer(circuit, active_qubits, layer_weights, layer_shift)

        if len(active_qubits) > config.n_outputs:
            keep = active_qubits[::2]
            discard = active_qubits[1::2]
            add_pooling_layer(circuit, keep, discard)
            active_qubits = keep

    return circuit, active_qubits


def _final_active_qubits(n_qubits: int, n_outputs: int) -> list[int]:
    """Return the active-qubit layout after the configured logical pooling."""
    active_qubits = list(range(n_qubits))
    while len(active_qubits) > n_outputs:
        active_qubits = active_qubits[::2]
    return active_qubits


def build_qcnn_circuit(
    features,
    weights,
    config: QCNNConfig,
    shifted_feature: tuple[int, float] | None = None,
    shifted_weight: tuple[int, int, int, float] | None = None,
) -> Circuit:
    """Build the QCNN circuit for one sample, optionally shifting one gate.

    ``shifted_feature`` is ``(feature_index, shift)`` and shifts the encoding
    angle of that feature. ``shifted_weight`` is
    ``(layer_index, pair_index, weight_index, shift)`` and shifts the selected
    occurrence of the shared convolution filter. These hooks implement the
    parameter-shift rule without duplicating circuit logic.
    """
    circuit, _ = _build_circuit_and_active(
        features,
        weights,
        config,
        shifted_feature=shifted_feature,
        shifted_weight=shifted_weight,
    )
    return circuit


def _execute_expectations(
    circuit: Circuit,
    active_qubits: list[int],
    config: QCNNConfig,
    execution_seed: int | None = None,
) -> np.ndarray:
    execution = config.execution
    if execution.shots is None:
        if config.executor is not None:
            raise ValueError(
                "an injected measured-circuit executor requires positive shots; "
                "shots=None is reserved for exact local simulation."
            )
        if execution.noisy:
            if execution.noise_simulation != "density_matrix":
                raise ValueError(
                    "exact noisy expectations require noise_simulation="
                    "'density_matrix'; trajectory modes need a positive shots value."
                )
            noisy = qibo_local_depolarizing_sensitivity_transform(
                execution.depolarizing_probability,
                gate_names=QCNN_NOISE_GATE_NAMES,
                simulation="density_matrix",
            )(circuit)
            rho = np.asarray(noisy.execute().state())
            probabilities = np.real(np.diagonal(rho))
            return np.asarray(
                [
                    expectation_z_from_probabilities(
                        probabilities, config.n_qubits, qubit
                    )
                    for qubit in active_qubits
                ],
                dtype=float,
            )
        state = circuit.execute().state()
        return np.asarray(
            [
                expectation_z(state, config.n_qubits, qubit)
                for qubit in active_qubits
            ],
            dtype=float,
        )

    circuit.add(gates.M(*active_qubits))
    executor = config.executor
    if executor is None:
        noise_transform = None
        if execution.noisy:
            if execution.noise_simulation == "legacy_shared_trajectory":
                # Historical behavior: Qibo NoiseModel, one shared trajectory.
                noise_model = depolarizing_noise_model(
                    execution.depolarizing_probability
                )
                noise_transform = noise_model.apply
            else:
                noise_transform = qibo_local_depolarizing_sensitivity_transform(
                    execution.depolarizing_probability,
                    gate_names=QCNN_NOISE_GATE_NAMES,
                    simulation=execution.noise_simulation,
                )
        executor = QiboSimulatorExecutor(noise_transform=noise_transform)
    elif execution.noisy:
        raise ValueError(
            "depolarizing_probability is the built-in Qibo simulator model; "
            "configure noise through the injected executor instead."
        )

    executor.capabilities.validate_request(config.n_qubits, execution.shots)

    frequencies = validate_counts(
        executor.execute_counts(
            circuit,
            shots=execution.shots,
            seed=(execution.seed if execution_seed is None else execution_seed),
        ),
        n_bits=len(active_qubits),
        shots=execution.shots,
    )
    expectations = np.zeros(len(active_qubits), dtype=float)
    for bitstring, count in frequencies.items():
        for index, bit in enumerate(bitstring):
            expectations[index] += (1.0 if bit == "0" else -1.0) * count
    return expectations / execution.shots


def _simulate_with_config(
    features,
    weights,
    config: QCNNConfig,
    shifted_feature: tuple[int, float] | None = None,
    shifted_weight: tuple[int, int, int, float] | None = None,
    execution_seed: int | None = None,
) -> np.ndarray:
    circuit = build_qcnn_circuit(
        features,
        weights,
        config,
        shifted_feature=shifted_feature,
        shifted_weight=shifted_weight,
    )
    active_qubits = _final_active_qubits(config.n_qubits, config.n_outputs)
    return _execute_expectations(
        circuit, active_qubits, config, execution_seed=execution_seed
    )


def _derive_circuit_seed(seed: int | None, *coordinates: int) -> int | None:
    """Derive independent deterministic streams for circuits in one call."""
    if seed is None:
        return None
    entropy = [int(seed) % (1 << 32), *(int(value) for value in coordinates)]
    return int(
        np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0]
    )


def simulate_qcnn(
    features,
    weights,
    n_qubits: int,
    n_outputs: int,
    shifted_feature: tuple[int, float] | None = None,
    shifted_weight: tuple[int, int, int, float] | None = None,
    encoding: str = "angle",
    execution_config: ExecutionConfig = ExecutionConfig(),
    executor: CircuitExecutor | None = None,
) -> np.ndarray:
    """Simulate one sample with the legacy-compatible parameter order.

    This is the reusable low-level entry point used by the hybrid wrapper. It
    does not seed the random generator; the caller controls sampling through
    ``execution_config`` and global NumPy seeding, exactly as the legacy model.
    """
    config = QCNNConfig(
        n_qubits=n_qubits,
        n_outputs=n_outputs,
        encoding=encoding,
        execution=execution_config,
        executor=executor,
    )
    return _simulate_with_config(
        features,
        weights,
        config,
        shifted_feature=shifted_feature,
        shifted_weight=shifted_weight,
    )


def _validated_weights(weights, config: QCNNConfig) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    expected_shape = (config.n_layers, 4)
    if values.shape != expected_shape:
        raise ValueError(f"weights must have shape {expected_shape}.")
    if not np.isfinite(values).all():
        raise ValueError("weights must contain only finite values.")
    return values


def _validated_features(features, n_qubits: int) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != n_qubits:
        raise ValueError(f"features must have shape (batch, {n_qubits}).")
    if values.shape[0] == 0:
        raise ValueError("features must contain at least one sample.")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("features must be normalized to the [0, 1] interval.")
    return values


def _validated_targets(targets, batch_size: int, n_outputs: int) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    if n_outputs == 1:
        if values.ndim == 1 and values.shape[0] == batch_size:
            values = values.reshape(batch_size, 1)
        elif values.ndim != 2 or values.shape != (batch_size, 1):
            raise ValueError(
                f"targets must have shape ({batch_size},) or ({batch_size}, 1)."
            )
    else:
        if values.ndim != 2 or values.shape != (batch_size, n_outputs):
            raise ValueError(
                f"targets must have shape ({batch_size}, {n_outputs})."
            )
    if not np.isfinite(values).all():
        raise ValueError("targets must contain only finite values.")
    return values


def _predict_validated(
    weight_values: np.ndarray,
    feature_values: np.ndarray,
    config: QCNNConfig,
) -> np.ndarray:
    if config.execution.seed is not None:
        np.random.seed(config.execution.seed)
    return np.stack(
        [
            _simulate_with_config(
                sample,
                weight_values,
                config,
                execution_seed=_derive_circuit_seed(
                    config.execution.seed, 0, sample_index
                ),
            )
            for sample_index, sample in enumerate(feature_values)
        ]
    )


def predict(weights, features, config: QCNNConfig) -> np.ndarray:
    """Return Z expectations with shape ``(batch, n_outputs)``.

    Exact, sampled and depolarizing executions are supported. Sampling uses the
    global NumPy generator, so a non-``None`` ``execution.seed`` makes every
    call reproducible. In exact mode a noisy configuration raises ``ValueError``.
    """
    if not isinstance(config, QCNNConfig):
        raise TypeError("config must be a QCNNConfig instance.")
    weight_values = _validated_weights(weights, config)
    feature_values = _validated_features(features, config.n_qubits)
    return _predict_validated(weight_values, feature_values, config)


def mse_loss(weights, features, targets, config: QCNNConfig) -> float:
    """Mean squared error averaged over every batch and output element."""
    if not isinstance(config, QCNNConfig):
        raise TypeError("config must be a QCNNConfig instance.")
    weight_values = _validated_weights(weights, config)
    feature_values = _validated_features(features, config.n_qubits)
    target_values = _validated_targets(
        targets,
        feature_values.shape[0],
        config.n_outputs,
    )
    prediction = _predict_validated(weight_values, feature_values, config)
    return float(np.mean((prediction - target_values) ** 2))


def parameter_shift_gradient(
    weights,
    features,
    targets,
    config: QCNNConfig,
) -> np.ndarray:
    """Return the MSE gradient with shape ``(n_layers, 4)``.

    Every appearance of a shared convolution weight contributes to the same
    gradient entry. The upstream factor is ``2 * (prediction - target) / N``
    with ``N = batch_size * n_outputs``, matching PyTorch's mean MSE.
    """
    if not isinstance(config, QCNNConfig):
        raise TypeError("config must be a QCNNConfig instance.")
    weight_values = _validated_weights(weights, config)
    feature_values = _validated_features(features, config.n_qubits)
    target_values = _validated_targets(
        targets,
        feature_values.shape[0],
        config.n_outputs,
    )

    prediction = _predict_validated(weight_values, feature_values, config)
    total_elements = prediction.size
    upstream = 2.0 * (prediction - target_values) / total_elements

    shift = np.pi / 2.0
    gradient = np.zeros_like(weight_values, dtype=float)

    for batch_index, sample in enumerate(feature_values):
        active_count = config.n_qubits
        for layer_index in range(config.n_layers):
            for pair_index in range(active_count - 1):
                for weight_index in range(4):
                    positive = _simulate_with_config(
                        sample,
                        weight_values,
                        config,
                        shifted_weight=(
                            layer_index,
                            pair_index,
                            weight_index,
                            shift,
                        ),
                        execution_seed=_derive_circuit_seed(
                            config.execution.seed,
                            1,
                            batch_index,
                            layer_index,
                            pair_index,
                            weight_index,
                            0,
                        ),
                    )
                    negative = _simulate_with_config(
                        sample,
                        weight_values,
                        config,
                        shifted_weight=(
                            layer_index,
                            pair_index,
                            weight_index,
                            -shift,
                        ),
                        execution_seed=_derive_circuit_seed(
                            config.execution.seed,
                            1,
                            batch_index,
                            layer_index,
                            pair_index,
                            weight_index,
                            1,
                        ),
                    )
                    derivative = 0.5 * (positive - negative)
                    gradient[layer_index, weight_index] += np.dot(
                        upstream[batch_index],
                        derivative,
                    )
            if active_count > config.n_outputs:
                active_count //= 2

    return gradient


def initialize_weights(
    n_qubits: int,
    n_outputs: int = 1,
    seed: int | None = None,
    low: float = -0.25,
    high: float = 0.25,
) -> np.ndarray:
    """Create reproducible convolution weights with shape ``(n_layers, 4)``."""
    config = QCNNConfig(n_qubits=n_qubits, n_outputs=n_outputs)
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool)
    ):
        raise ValueError("seed must be an integer or None.")
    if (
        not math.isfinite(low)
        or not math.isfinite(high)
        or low >= high
    ):
        raise ValueError("low and high must be finite with low < high.")
    generator = np.random.default_rng(seed)
    return generator.uniform(low, high, size=(config.n_layers, 4))


if __name__ == "__main__":
    sample = np.array([0.1, 0.5, 0.9, 0.2], dtype=float)
    sample_weights = initialize_weights(4, seed=7)
    print("Z expectations:", predict(sample_weights, [sample], QCNNConfig(4)))
    print("MSE:", mse_loss(sample_weights, [sample], [1.0], QCNNConfig(4)))
    print(
        "Weight gradients:",
        parameter_shift_gradient(
            sample_weights,
            [sample],
            [1.0],
            QCNNConfig(4),
        ),
    )
