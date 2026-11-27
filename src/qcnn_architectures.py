"""QCNN architectures as gate programs, with a batched simulator and a Qibo builder.

Three architectures share the data encoding (angle ``RY(pi*x)`` or phase
``H + RZ(2*pi*x)``), the Z readout of one surviving qubit and the weight layout
``(n_layers, params_per_layer)`` (weights shared by every pair of a layer):

``v1``
    The original project circuit (``qcnn_circuit`` + ``pooling``): per adjacent
    pair ``RY RZ CNOT(l,r) RY RZ CNOT(r,l)`` and pooling ``CNOT(discard, keep)``.
    4 weights per layer. The last convolution CNOT and the pooling CNOT are the
    same gate and cancel, so the output ignores x3 and the layer-2 weights
    w1-w3 (technical log, section 37). Kept unchanged for reproducibility.
``v1-param-pool``
    Same convolution and layout as ``v1``; each pooling becomes
    ``RY(a) on keep, RY(b) on discard, CNOT(discard, keep), RY(-a) on keep``
    (the minimal one-CNOT version of Qibo's parametrized pooling). The
    rotation between the two CNOTs breaks the cancellation. 6 weights per layer.
``qibo``
    The ``qibo.models.QuantumCNN`` ansatz (default two-qubit ansatz, qibo main
    branch): 15-parameter convolution ``R(a) R(b) RZZ RYY RXX R(a) R(b)`` with
    ``R = RX RY RZ`` on pairs ``(0,1),(2,3),...`` then the ring
    ``(1,2),(3,4),...,(n-1,0)`` (only the first pass with 2 active qubits),
    and the 6-parameter pooling ``R(sink) R(source) CNOT(source, sink)
    R(sink)^-1`` from the first half of the active qubits (source) to the
    second half (sink). The readout qubit is ``n-1``. 21 weights per layer.
    Differences with Qibo's class, documented on purpose: no trainable bias,
    our data encoding instead of an initial-state vector, ⟨Z⟩ trained with MSE
    on -1/+1 labels by our optimizers instead of ``qibo.optimizers``.

Every parametrized gate is ``exp(-i * angle * P / 2)`` with ``angle = coeff *
weight`` and ``coeff`` in ``{+1, -1}`` and a Pauli product ``P``, so the
parameter-shift rule with ``±pi/2`` applies per occurrence and the output is a
trigonometric polynomial of integer degree (number of occurrences) in each
weight. The simulator reuses :class:`fast_qcnn.FastQCNN` (state vector, or
exact density matrix with local depolarizing noise after every gate on the
touched qubits, plus symmetric readout error). Qubit 0 is the most
significant bit, as in Qibo.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

if __package__:
    from .fast_qcnn import SHIFT, FastQCNN, FastQCNNConfig, qcnn_layout
else:  # pragma: no cover - direct execution from src/
    from fast_qcnn import SHIFT, FastQCNN, FastQCNNConfig, qcnn_layout


ARCHITECTURES = ("v1", "v1-param-pool", "qibo")
ROTATIONS_1Q = ("rx", "ry", "rz")
ROTATIONS_2Q = ("rxx", "ryy", "rzz")

_PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class Gate:
    """One gate. Parametrized gates use ``weights[layer, param] * coeff``."""

    kind: str
    qubits: tuple[int, ...]
    layer: int = -1
    param: int = -1
    coeff: float = 1.0

    @property
    def parametrized(self) -> bool:
        return self.kind != "cnot"


@dataclass(frozen=True)
class Ansatz:
    name: str
    n_qubits: int
    n_layers: int
    params_per_layer: int
    gates: tuple[Gate, ...]
    readout_qubit: int
    param_labels: tuple[str, ...]

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.n_layers, self.params_per_layer)

    @property
    def occurrences(self) -> list[int]:
        """Indexes (into ``gates``) of the parametrized gate occurrences."""
        return [i for i, gate in enumerate(self.gates) if gate.parametrized]

    @property
    def weight_degrees(self) -> np.ndarray:
        """Structural trigonometric degree of the output in each weight."""
        degrees = np.zeros(self.weight_shape, dtype=int)
        for index in self.occurrences:
            gate = self.gates[index]
            degrees[gate.layer, gate.param] += 1
        return degrees

    @property
    def cnot_count(self) -> int:
        return sum(gate.kind == "cnot" for gate in self.gates)

    def occurrence_label(self, index: int) -> str:
        gate = self.gates[index]
        qubits = "".join(str(q) for q in gate.qubits)
        sign = "-" if gate.coeff < 0 else ""
        return f"layer{gate.layer}-{gate.kind}{qubits}-{sign}{self.param_labels[gate.param]}"

    def summary(self) -> dict:
        return {
            "architecture": self.name,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "weight_shape": list(self.weight_shape),
            "trainable_weights": int(np.prod(self.weight_shape)),
            "parametrized_occurrences": len(self.occurrences),
            "cnot_count": self.cnot_count,
            "readout_qubit": self.readout_qubit,
            "param_labels": list(self.param_labels),
        }


def _check_qubits(n_qubits: int) -> int:
    if not isinstance(n_qubits, int) or n_qubits < 2 or n_qubits & (n_qubits - 1):
        raise ValueError("n_qubits must be a power of two of at least 2.")
    return int(math.log2(n_qubits))


def _v1_convolution(layer: int, left: int, right: int) -> list[Gate]:
    return [
        Gate("ry", (left,), layer, 0),
        Gate("rz", (left,), layer, 1),
        Gate("cnot", (left, right)),
        Gate("ry", (right,), layer, 2),
        Gate("rz", (right,), layer, 3),
        Gate("cnot", (right, left)),
    ]


def _build_v1(n_qubits: int, parametrized_pool: bool) -> Ansatz:
    n_layers = _check_qubits(n_qubits)
    layers, final_active = qcnn_layout(n_qubits, 1)
    gates: list[Gate] = []
    for layer, (pairs, pooling) in enumerate(layers):
        for left, right in pairs:
            gates += _v1_convolution(layer, left, right)
        for discard, keep in pooling:
            if parametrized_pool:
                gates += [
                    Gate("ry", (keep,), layer, 4),
                    Gate("ry", (discard,), layer, 5),
                    Gate("cnot", (discard, keep)),
                    Gate("ry", (keep,), layer, 4, -1.0),
                ]
            else:
                gates.append(Gate("cnot", (discard, keep)))
    labels = ("w0", "w1", "w2", "w3") + (("pool_keep", "pool_discard") if parametrized_pool else ())
    return Ansatz(
        name="v1-param-pool" if parametrized_pool else "v1",
        n_qubits=n_qubits,
        n_layers=n_layers,
        params_per_layer=len(labels),
        gates=tuple(gates),
        readout_qubit=final_active[0],
        param_labels=labels,
    )


def _qibo_one_qubit(layer: int, qubit: int, first: int, coeff: float = 1.0, inverse: bool = False) -> list[Gate]:
    gates = [Gate(kind, (qubit,), layer, first + k, coeff) for k, kind in enumerate(ROTATIONS_1Q)]
    return gates[::-1] if inverse else gates


def _qibo_two_qubit_unitary(layer: int, a: int, b: int) -> list[Gate]:
    return (
        _qibo_one_qubit(layer, a, 0)
        + _qibo_one_qubit(layer, b, 3)
        + [Gate("rzz", (a, b), layer, 6), Gate("ryy", (a, b), layer, 7), Gate("rxx", (a, b), layer, 8)]
        + _qibo_one_qubit(layer, a, 9)
        + _qibo_one_qubit(layer, b, 12)
    )


def _build_qibo(n_qubits: int) -> Ansatz:
    n_layers = _check_qubits(n_qubits)
    qubits = list(range(n_qubits))
    gates: list[Gate] = []
    for layer in range(n_layers):
        conv_start = int(n_qubits - n_qubits / 2**layer)
        pool_start = int(n_qubits - n_qubits / 2 ** (layer + 1))
        bits = qubits[conv_start:]
        for a, b in zip(bits[0::2], bits[1::2]):
            gates += _qibo_two_qubit_unitary(layer, a, b)
        if len(bits) > 2:
            for a, b in zip(bits[1::2], bits[2::2] + [bits[0]]):
                gates += _qibo_two_qubit_unitary(layer, a, b)
        for source, sink in zip(qubits[conv_start:pool_start], qubits[pool_start:]):
            gates += _qibo_one_qubit(layer, sink, 15)
            gates += _qibo_one_qubit(layer, source, 18)
            gates.append(Gate("cnot", (source, sink)))
            gates += _qibo_one_qubit(layer, sink, 15, coeff=-1.0, inverse=True)
    labels = tuple(f"conv{k}" for k in range(15)) + tuple(f"pool{k}" for k in range(6))
    return Ansatz("qibo", n_qubits, n_layers, 21, tuple(gates), n_qubits - 1, labels)


def build_ansatz(architecture: str, n_qubits: int = 4) -> Ansatz:
    if architecture == "v1":
        return _build_v1(n_qubits, parametrized_pool=False)
    if architecture == "v1-param-pool":
        return _build_v1(n_qubits, parametrized_pool=True)
    if architecture == "qibo":
        return _build_qibo(n_qubits)
    raise ValueError(f"architecture must be one of: {', '.join(ARCHITECTURES)}.")


def initialize_ansatz_weights(ansatz: Ansatz, seed: int | None, low: float = -0.25, high: float = 0.25) -> np.ndarray:
    """Uniform initial weights; for ``v1`` identical to ``qcnn_engine.initialize_weights``."""
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("low and high must be finite with low < high.")
    return np.random.default_rng(seed).uniform(low, high, size=ansatz.weight_shape)


# ---------------------------------------------------------------- simulator
def _rotation_1q(kind: str, theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    pauli = _PAULI[kind[1]]
    return c[:, None, None] * np.eye(2) - 1j * s[:, None, None] * pauli


def _rotation_2q(kind: str, theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    pauli = np.kron(_PAULI[kind[1]], _PAULI[kind[2]])
    return c[:, None, None] * np.eye(4) - 1j * s[:, None, None] * pauli


class AnsatzQCNN(FastQCNN):
    """Batched simulator of any :class:`Ansatz`. Every input row is one circuit."""

    def __init__(self, ansatz: Ansatz, encoding: str = "angle", depolarizing_probability: float = 0.0,
                 readout_error: float = 0.0):
        super().__init__(FastQCNNConfig(ansatz.n_qubits, 1, encoding, depolarizing_probability, readout_error))
        self.ansatz = ansatz
        self._occurrences = ansatz.occurrences

    def _apply_2q(self, state, matrices, q1, q2, density):
        n, dim = self.n, self.dim

        def side(flat, mats):
            rows, extra = flat.shape[0], flat.shape[2]
            tensor = np.moveaxis(flat.reshape((rows,) + (2,) * n + (extra,)), (1 + q1, 1 + q2), (1, 2))
            shape = tensor.shape
            tensor = np.einsum("rij,rjk->rik", mats, tensor.reshape(rows, 4, -1)).reshape(shape)
            return np.moveaxis(tensor, (1, 2), (1 + q1, 1 + q2)).reshape(rows, dim, extra)

        if not density:
            return side(state[:, :, None], matrices)[:, :, 0]
        state = side(state, matrices)  # U rho
        return side(state.transpose(0, 2, 1), matrices.conj()).transpose(0, 2, 1)  # (U rho) U^dagger

    def angles_from_weights(self, weights) -> np.ndarray:
        """Occurrence angles ``(rows, n_occurrences)`` from weights ``(rows, L, P)`` or ``(L, P)``."""
        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 2:
            weights = weights[None]
        if weights.shape[1:] != self.ansatz.weight_shape:
            raise ValueError(f"weights must have shape {self.ansatz.weight_shape}.")
        gates = [self.ansatz.gates[i] for i in self._occurrences]
        layers = np.array([g.layer for g in gates])
        params = np.array([g.param for g in gates])
        coeffs = np.array([g.coeff for g in gates])
        return weights[:, layers, params] * coeffs

    def expectations_from_angles(self, angles, features) -> np.ndarray:
        angles = np.asarray(angles, dtype=float)
        features = np.asarray(features, dtype=float)
        rows = features.shape[0]
        if features.shape != (rows, self.n):
            raise ValueError(f"features must have shape (rows, {self.n}).")
        if angles.shape != (rows, len(self._occurrences)):
            raise ValueError("angles have an unexpected shape.")
        density = self.config.noisy
        state = self._encode(features, density)
        column = 0
        for gate in self.ansatz.gates:
            if gate.kind == "cnot":
                state = self._apply_cnot(state, gate.qubits[0], gate.qubits[1], density)
            else:
                theta = angles[:, column]
                column += 1
                if gate.kind in ROTATIONS_1Q:
                    state = self._apply_1q(state, _rotation_1q(gate.kind, theta), gate.qubits[0], density)
                else:
                    state = self._apply_2q(state, _rotation_2q(gate.kind, theta), *gate.qubits, density)
            state = self._noise(state, gate.qubits, density)
        return self._measure_z(state, (self.ansatz.readout_qubit,), density)

    def expectations(self, weights, features) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        angles = np.broadcast_to(self.angles_from_weights(weights), (len(features), len(self._occurrences)))
        return self.expectations_from_angles(angles, features)

    def shift_rows(self, weights, features):
        """Rows of every ``±pi/2`` parameter-shift circuit of every sample.

        Returns ``(angles, features, index, n_shift)`` with ``index`` listing
        ``(layer, param, coeff, sign)`` per shifted circuit of one sample.
        """
        features = np.asarray(features, dtype=float)
        base = self.angles_from_weights(weights)[0]
        rows_angles, index = [], []
        for column, gate_index in enumerate(self._occurrences):
            gate = self.ansatz.gates[gate_index]
            for sign in (1, -1):
                shifted = base.copy()
                shifted[column] += sign * SHIFT
                rows_angles.append(shifted)
                index.append((gate.layer, gate.param, gate.coeff, sign))
        n_shift = len(rows_angles)
        angles = np.tile(np.stack(rows_angles), (len(features), 1))
        return angles, np.repeat(features, n_shift, axis=0), index, n_shift

    @property
    def occurrences_per_sample(self) -> int:
        return len(self._occurrences)


# ---------------------------------------------------------------- Qibo
def build_qibo_circuit(ansatz: Ansatz, weights, features, encoding: str = "angle", angles=None):
    """Qibo circuit of one sample (reference implementation and hardware path).

    ``angles`` (optional) overrides the occurrence angles, e.g. for parameter shift.
    """
    from qibo import gates as qgates
    from qibo.models import Circuit

    features = np.asarray(features, dtype=float).reshape(-1)
    if features.shape != (ansatz.n_qubits,):
        raise ValueError(f"features must have {ansatz.n_qubits} values.")
    if angles is None:
        weights = np.asarray(weights, dtype=float)
        angles = [weights[ansatz.gates[i].layer, ansatz.gates[i].param] * ansatz.gates[i].coeff
                  for i in ansatz.occurrences]
    else:
        angles = list(angles)
        if len(angles) != len(ansatz.occurrences):
            raise ValueError(
                f"angles must have {len(ansatz.occurrences)} values, got {len(angles)}."
            )
    circuit = Circuit(ansatz.n_qubits)
    for qubit, value in enumerate(features):
        if encoding == "angle":
            circuit.add(qgates.RY(qubit, theta=float(np.pi * value)))
        elif encoding == "phase":
            circuit.add(qgates.H(qubit))
            circuit.add(qgates.RZ(qubit, theta=float(2.0 * np.pi * value)))
        else:
            raise ValueError("encoding must be 'angle' or 'phase'.")
    column = 0
    for gate in ansatz.gates:
        if gate.kind == "cnot":
            circuit.add(qgates.CNOT(*gate.qubits))
            continue
        theta = float(angles[column])
        column += 1
        constructor = getattr(qgates, gate.kind.upper())
        circuit.add(constructor(*gate.qubits, theta=theta))
    return circuit


def qibo_expectation(ansatz: Ansatz, weights, features, encoding: str = "angle", angles=None) -> float:
    """Exact noiseless ⟨Z⟩ of the readout qubit through Qibo's state vector."""
    circuit = build_qibo_circuit(ansatz, weights, features, encoding, angles)
    state = np.asarray(circuit.execute().state()).reshape(-1)
    probabilities = np.abs(state) ** 2
    index = np.arange(len(probabilities))
    signs = 1.0 - 2.0 * ((index >> (ansatz.n_qubits - 1 - ansatz.readout_qubit)) & 1)
    return float(probabilities @ signs)


__all__ = [
    "ARCHITECTURES",
    "Ansatz",
    "AnsatzQCNN",
    "Gate",
    "build_ansatz",
    "build_qibo_circuit",
    "initialize_ansatz_weights",
    "qibo_expectation",
]
