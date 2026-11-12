"""Vectorized NumPy simulator of the project QCNN, validated against Qibo.

Qibo builds and executes one circuit object per sample and per parameter-shift
term; for a 4-qubit circuit that Python overhead dominates the run time. This
module simulates exactly the same gate sequence as ``qcnn_engine`` for many
circuits at once (one row per circuit execution):

* encoding: angle ``RY(pi*x)`` or phase ``H + RZ(2*pi*x)``;
* convolution per adjacent active pair: ``RY(l,w0) RZ(l,w1) CNOT(l,r)
  RY(r,w2) RZ(r,w3) CNOT(r,l)``, weights shared by all pairs of a layer;
* logical pooling ``CNOT(discard, keep)``;
* Pauli-Z expectations of the surviving qubits.

Noise model (optional): an independent single-qubit depolarizing channel
``rho -> (1-p) rho + p I/2`` on every qubit touched by each RY, RZ or CNOT
(the same convention as ``qcnn_engine`` with ``noise_simulation=
"density_matrix"``), simulated exactly with density matrices, plus an optional
symmetric readout bit-flip probability. Finite shots are drawn as independent
binomial samples of each measured qubit, i.e. noise and sampling are
independent per shot. This is a simulator for experiments; the Qibo engine
remains the reference implementation and the path towards a hardware adapter.

Qubit 0 is the most significant bit, as in Qibo.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SHIFT = np.pi / 2.0


@dataclass(frozen=True)
class FastQCNNConfig:
    n_qubits: int
    n_outputs: int = 1
    encoding: str = "angle"
    depolarizing_probability: float = 0.0
    readout_error: float = 0.0

    def __post_init__(self):
        if not isinstance(self.n_qubits, int) or self.n_qubits < 2:
            raise ValueError("n_qubits must be an integer of at least 2.")
        if self.n_qubits & (self.n_qubits - 1):
            raise ValueError("n_qubits must be a power of two.")
        if (
            not isinstance(self.n_outputs, int)
            or self.n_outputs < 1
            or self.n_outputs >= self.n_qubits
            or self.n_outputs & (self.n_outputs - 1)
        ):
            raise ValueError("n_outputs must be a power of two smaller than n_qubits.")
        if self.encoding not in ("angle", "phase"):
            raise ValueError("encoding must be 'angle' or 'phase'.")
        for name in ("depolarizing_probability", "readout_error"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.readout_error > 0.5:
            raise ValueError("readout_error above 0.5 inverts the readout.")
        if self.depolarizing_probability > 0 and self.n_qubits > 8:
            raise ValueError("density-matrix noise is limited to n_qubits <= 8.")

    @property
    def n_layers(self) -> int:
        return int(math.log2(self.n_qubits // self.n_outputs))

    @property
    def noisy(self) -> bool:
        return self.depolarizing_probability > 0.0


def qcnn_layout(n_qubits: int, n_outputs: int):
    """Return per-layer ``(pairs, pooling)`` and the final active qubits."""
    active = list(range(n_qubits))
    layers = []
    for _ in range(int(math.log2(n_qubits // n_outputs))):
        pairs = list(zip(active, active[1:]))
        pooling = []
        if len(active) > n_outputs:
            keep, discard = active[::2], active[1::2]
            pooling = list(zip(discard, keep))  # CNOT(control=discard, target=keep)
            active = keep
        layers.append((pairs, pooling))
    return layers, active


def weight_occurrences(n_qubits: int, n_outputs: int = 1) -> list[int]:
    """Number of gate occurrences of each shared weight, per layer."""
    layers, _ = qcnn_layout(n_qubits, n_outputs)
    return [len(pairs) for pairs, _ in layers]


class FastQCNN:
    """Batched exact simulator. Every row of an input batch is one circuit."""

    def __init__(self, config: FastQCNNConfig):
        if not isinstance(config, FastQCNNConfig):
            raise TypeError("config must be a FastQCNNConfig instance.")
        self.config = config
        self.n = config.n_qubits
        self.dim = 1 << self.n
        self.layers, self.final_active = qcnn_layout(config.n_qubits, config.n_outputs)
        self.max_pairs = max(len(pairs) for pairs, _ in self.layers)
        self._perms: dict[tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------ helpers
    def _cnot_perm(self, control: int, target: int) -> np.ndarray:
        key = (control, target)
        if key not in self._perms:
            index = np.arange(self.dim)
            cbit = 1 << (self.n - 1 - control)
            tbit = 1 << (self.n - 1 - target)
            self._perms[key] = np.where(index & cbit, index ^ tbit, index)
        return self._perms[key]

    @staticmethod
    def _ry(theta):
        c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
        m = np.empty(theta.shape + (2, 2), dtype=complex)
        m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1] = c, -s, s, c
        return m

    @staticmethod
    def _rz(theta):
        m = np.zeros(theta.shape + (2, 2), dtype=complex)
        m[..., 0, 0] = np.exp(-0.5j * theta)
        m[..., 1, 1] = np.exp(0.5j * theta)
        return m

    def _apply_1q(self, state, matrices, qubit, density):
        """Apply per-row 2x2 matrices to ``qubit`` (ket, and bra if density)."""
        # Explicit 2x2 combinations are much faster than einsum for tiny factors.
        rows = state.shape[0]
        a, b = 1 << qubit, 1 << (self.n - 1 - qubit)

        def combine(view, coefficients):
            """``view`` has shape (rows, a, 2, rest); apply each row's 2x2 matrix."""
            m00, m01, m10, m11 = (
                coefficients[:, i, j].reshape(-1, 1, 1)
                for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))
            )
            zero, one = view[:, :, 0], view[:, :, 1]
            out = np.empty_like(view)
            out[:, :, 0] = m00 * zero + m01 * one
            out[:, :, 1] = m10 * zero + m11 * one
            return out

        if not density:
            return combine(state.reshape(rows, a, 2, b), matrices).reshape(rows, self.dim)
        # Ket side: U rho.
        state = combine(state.reshape(rows, a, 2, b * self.dim), matrices)
        # Bra side: (U rho) U^dagger acts on the column index with conj(U);
        # treat every (row, ket index) pair as an independent row.
        bra_view = state.reshape(rows * self.dim, a, 2, b)
        conjugate = np.repeat(matrices.conj(), self.dim, axis=0)
        return combine(bra_view, conjugate).reshape(rows, self.dim, self.dim)

    def _apply_cnot(self, state, control, target, density):
        perm = self._cnot_perm(control, target)
        if not density:
            return state[:, perm]
        return state[:, perm][:, :, perm]

    def _depolarize(self, state, qubit):
        p = self.config.depolarizing_probability
        rows = state.shape[0]
        a, b = 1 << qubit, 1 << (self.n - 1 - qubit)
        view = state.reshape(rows, a, 2, b, a, 2, b)
        traced = 0.5 * (view[:, :, 0, :, :, 0, :] + view[:, :, 1, :, :, 1, :])
        out = (1.0 - p) * view
        out[:, :, 0, :, :, 0, :] += p * traced
        out[:, :, 1, :, :, 1, :] += p * traced
        return out.reshape(rows, self.dim, self.dim)

    def _noise(self, state, qubits, density):
        if density:
            for qubit in qubits:
                state = self._depolarize(state, qubit)
        return state

    # --------------------------------------------------------------- simulation
    def expectations_from_angles(self, angles, features) -> np.ndarray:
        """Exact (noisy if configured) Z expectations, shape ``(rows, n_outputs)``.

        ``angles`` has shape ``(rows, n_layers, max_pairs, 4)``: the angle of
        each convolution gate occurrence. ``features`` has shape ``(rows, n)``.
        """
        angles = np.asarray(angles, dtype=float)
        features = np.asarray(features, dtype=float)
        rows = features.shape[0]
        if features.shape != (rows, self.n):
            raise ValueError(f"features must have shape (rows, {self.n}).")
        if angles.shape != (rows, len(self.layers), self.max_pairs, 4):
            raise ValueError("angles have an unexpected shape.")
        density = self.config.noisy
        if density:
            state = np.zeros((rows, self.dim, self.dim), dtype=complex)
            state[:, 0, 0] = 1.0
        else:
            state = np.zeros((rows, self.dim), dtype=complex)
            state[:, 0] = 1.0

        hadamard = np.broadcast_to(
            np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0), (rows, 2, 2)
        )
        for qubit in range(self.n):
            if self.config.encoding == "angle":
                state = self._apply_1q(state, self._ry(np.pi * features[:, qubit]), qubit, density)
            else:
                state = self._apply_1q(state, hadamard, qubit, density)  # H is noiseless
                state = self._apply_1q(
                    state, self._rz(2.0 * np.pi * features[:, qubit]), qubit, density
                )
            state = self._noise(state, (qubit,), density)

        for layer_index, (pairs, pooling) in enumerate(self.layers):
            for pair_index, (left, right) in enumerate(pairs):
                theta = angles[:, layer_index, pair_index]
                state = self._noise(self._apply_1q(state, self._ry(theta[:, 0]), left, density), (left,), density)
                state = self._noise(self._apply_1q(state, self._rz(theta[:, 1]), left, density), (left,), density)
                state = self._noise(self._apply_cnot(state, left, right, density), (left, right), density)
                state = self._noise(self._apply_1q(state, self._ry(theta[:, 2]), right, density), (right,), density)
                state = self._noise(self._apply_1q(state, self._rz(theta[:, 3]), right, density), (right,), density)
                state = self._noise(self._apply_cnot(state, right, left, density), (right, left), density)
            for control, target in pooling:
                state = self._noise(self._apply_cnot(state, control, target, density), (control, target), density)

        if density:
            probabilities = np.real(np.einsum("rii->ri", state))
        else:
            probabilities = np.abs(state) ** 2
        index = np.arange(self.dim)
        result = np.empty((rows, len(self.final_active)))
        for out, qubit in enumerate(self.final_active):
            signs = 1.0 - 2.0 * ((index >> (self.n - 1 - qubit)) & 1)
            result[:, out] = probabilities @ signs
        return (1.0 - 2.0 * self.config.readout_error) * result

    def angles_from_weights(self, weights) -> np.ndarray:
        """Broadcast shared weights ``(rows, L, 4)`` or ``(L, 4)`` to occurrences."""
        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 2:
            weights = weights[None]
        if weights.shape[1:] != (len(self.layers), 4):
            raise ValueError(f"weights must have shape ({len(self.layers)}, 4).")
        return np.repeat(weights[:, :, None, :], self.max_pairs, axis=2)

    def expectations(self, weights, features) -> np.ndarray:
        """Expectations of one weight matrix for every sample (rows = samples)."""
        features = np.asarray(features, dtype=float)
        angles = np.broadcast_to(
            self.angles_from_weights(weights), (len(features), len(self.layers), self.max_pairs, 4)
        )
        return self.expectations_from_angles(angles, features)

    def shift_rows(self, weights, features):
        """Angles/features for every parameter-shift circuit of every sample.

        Returns ``(angles, features, index)`` where ``index`` lists
        ``(sample, layer, pair, weight, sign)`` for each row.
        """
        features = np.asarray(features, dtype=float)
        base = self.angles_from_weights(weights)[0]
        rows_angles, rows_features, index = [], [], []
        for layer_index, (pairs, _) in enumerate(self.layers):
            for pair_index in range(len(pairs)):
                for weight_index in range(4):
                    for sign in (1, -1):
                        shifted = base.copy()
                        shifted[layer_index, pair_index, weight_index] += sign * SHIFT
                        rows_angles.append(shifted)
                        index.append((layer_index, pair_index, weight_index, sign))
        n_shift = len(rows_angles)
        angles = np.tile(np.stack(rows_angles), (len(features), 1, 1, 1))
        feats = np.repeat(features, n_shift, axis=0)
        return angles, feats, index, n_shift

    @property
    def occurrences_per_sample(self) -> int:
        return 4 * sum(len(pairs) for pairs, _ in self.layers)


def sample_expectations(expectations, shots, rng) -> np.ndarray:
    """Estimate Z expectations from ``shots`` independent measurements per row."""
    expectations = np.asarray(expectations, dtype=float)
    if shots is None:
        return expectations
    p_zero = np.clip((1.0 + expectations) / 2.0, 0.0, 1.0)
    zeros = rng.binomial(int(shots), p_zero)
    return 2.0 * zeros / shots - 1.0


__all__ = [
    "FastQCNN",
    "FastQCNNConfig",
    "qcnn_layout",
    "sample_expectations",
    "weight_occurrences",
]
