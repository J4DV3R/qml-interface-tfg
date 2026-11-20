"""Batched hardware-style execution, a noisy hardware mock and readout mitigation.

* :class:`BatchCircuitExecutor` extends the vendor-neutral ``CircuitExecutor``
  contract (``src/circuit_execution.py``) with ``execute_batch``: real QPUs
  are dominated by per-connection overhead, so many circuits must be sent per
  submission. :func:`execute_batch` falls back to a loop for executors that
  only implement ``execute_counts``.
* :class:`HardwareMockExecutor` is a **simulator** that behaves like a device
  for the purpose of dry runs: it only accepts native gates on coupled qubits,
  applies local depolarizing noise after one- and two-qubit gates (exact
  density matrix), samples finite shots and flips bits with asymmetric readout
  errors. Its parameters are sensitivity assumptions, not a calibration of any
  BSC chip.
* :func:`calibrate_readout` / :func:`mitigate_z_expectation` implement the
  standard single-qubit confusion-matrix inversion.
* :class:`QililabExecutor` is the integration point for the BSC chips. It is
  intentionally a skeleton: the Qililab API (runcard, connection, execution
  arguments and result format) must be taken from the BSC/Qilimanjaro
  documentation once access is granted, and nothing here has been run on
  hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from qibo import gates
from qibo.models import Circuit

from .circuit_execution import CircuitExecutor, ResourceProfile, validate_counts
from .transpilation import NATIVE_GATES


@runtime_checkable
class BatchCircuitExecutor(CircuitExecutor, Protocol):
    def execute_batch(
        self,
        circuits: Sequence[Any],
        *,
        shots: int,
        seed: int | None = None,
    ) -> list[dict[str, int]]: ...


def execute_batch(executor, circuits: Sequence[Any], *, shots: int, seed: int | None = None) -> list[dict[str, int]]:
    """Run circuits through ``execute_batch`` when available, else one by one.

    Per-circuit seeds are derived from ``seed`` with ``SeedSequence`` so that
    results do not depend on how a batch is split.
    """
    if hasattr(executor, "execute_batch"):
        return list(executor.execute_batch(circuits, shots=shots, seed=seed))
    children = np.random.SeedSequence(seed).spawn(len(circuits)) if seed is not None else [None] * len(circuits)
    return [
        dict(executor.execute_counts(c, shots=shots, seed=None if s is None else int(s.generate_state(1)[0])))
        for c, s in zip(circuits, children)
    ]


# --------------------------------------------------------------------------- noisy mock
@dataclass(frozen=True)
class MockNoise:
    """Sensitivity parameters of the hardware mock (assumptions, not calibrations)."""

    one_qubit_depolarizing: float = 0.0
    two_qubit_depolarizing: float = 0.0
    readout_p01: float = 0.0  # P(read 1 | prepared 0)
    readout_p10: float = 0.0  # P(read 0 | prepared 1)

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


class HardwareMockExecutor:
    """Device-like simulator for transpiled circuits (dry runs only)."""

    def __init__(self, coupling_edges: Sequence[tuple[int, int]], n_qubits: int, noise: MockNoise | None = None,
                 *, max_shots: int | None = 100_000):
        self.noise = noise or MockNoise()
        self.coupling = {tuple(sorted(e)) for e in coupling_edges}
        self.capabilities = ResourceProfile(
            max_qubits=n_qubits,
            max_shots=max_shots,
            basis_gates=NATIVE_GATES,
            connectivity=tuple(self.coupling),
            metadata={"adapter": "hardware-mock (simulator)", "noise": self.noise.to_dict()},
            validated_backend=None,
            transpiled_backend="src.transpilation (generic native set, hypothetical topology)",
        )

    def _check(self, circuit) -> list[int]:
        measured = [q for gate in circuit.measurements for q in gate.target_qubits]
        if not measured:
            raise ValueError("circuit must contain a measurement.")
        for gate in circuit.queue:
            name = gate.__class__.__name__
            if name == "M":
                continue
            if name not in NATIVE_GATES:
                raise ValueError(f"non-native gate {name}: transpile the circuit first.")
            if len(gate.qubits) == 2 and tuple(sorted(gate.qubits)) not in self.coupling:
                raise ValueError(f"two-qubit gate on uncoupled qubits {gate.qubits}.")
        return measured

    def probabilities(self, circuit) -> np.ndarray:
        """Exact noisy distribution of the measured qubits (before readout error)."""
        measured = self._check(circuit)
        n = circuit.nqubits
        p1, p2 = self.noise.one_qubit_depolarizing, self.noise.two_qubit_depolarizing
        noisy = Circuit(n, density_matrix=True)
        for gate in circuit.queue:
            if gate.__class__.__name__ == "M":
                continue
            noisy.add(gate)
            p = p2 if len(gate.qubits) == 2 else p1
            if p > 0:
                for q in gate.qubits:
                    noisy.add(gates.PauliNoiseChannel(q, px=p / 4, py=p / 4, pz=p / 4))
        rho = np.asarray(noisy.execute().state())
        probabilities = np.clip(np.real(np.diag(rho)), 0.0, None).reshape([2] * n)
        other = tuple(q for q in range(n) if q not in measured)
        marginal = probabilities.sum(axis=other) if other else probabilities
        order = [q for q in range(n) if q in measured]
        marginal = np.moveaxis(marginal, [order.index(q) for q in measured], range(len(measured)))
        marginal = marginal.reshape(-1)
        return marginal / marginal.sum()

    def execute_counts(self, circuit, *, shots: int, seed: int | None = None) -> dict[str, int]:
        self.capabilities.validate_request(circuit.nqubits, shots)
        rng = np.random.default_rng(seed)
        probabilities = self.probabilities(circuit)
        n_bits = int(np.log2(probabilities.size))
        outcomes = rng.choice(probabilities.size, size=shots, p=probabilities)
        bits = ((outcomes[:, None] >> np.arange(n_bits - 1, -1, -1)) & 1).astype(np.int8)
        flip = np.where(bits == 0, rng.random(bits.shape) < self.noise.readout_p01,
                        rng.random(bits.shape) < self.noise.readout_p10)
        bits = bits ^ flip.astype(np.int8)
        keys, counts = np.unique(bits, axis=0, return_counts=True)
        result = {"".join(str(int(b)) for b in key): int(c) for key, c in zip(keys, counts)}
        return validate_counts(result, n_bits=n_bits, shots=shots)

    def execute_batch(self, circuits, *, shots: int, seed: int | None = None) -> list[dict[str, int]]:
        children = np.random.SeedSequence(seed).spawn(len(circuits))
        return [self.execute_counts(c, shots=shots, seed=int(s.generate_state(1)[0])) for c, s in zip(circuits, children)]


# --------------------------------------------------------------------------- readout mitigation
@dataclass
class ReadoutCalibration:
    """Per-qubit confusion probabilities estimated from |0> and |1> preparations."""

    p01: dict[int, float]  # P(1 | 0)
    p10: dict[int, float]  # P(0 | 1)
    shots: int
    circuits: int
    raw_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"p01": {str(k): v for k, v in self.p01.items()},
                "p10": {str(k): v for k, v in self.p10.items()},
                "shots": self.shots, "circuits": self.circuits, "raw_counts": self.raw_counts}


def calibrate_readout(executor, n_qubits: int, measured: Sequence[int], *, shots: int,
                      seed: int | None = None) -> ReadoutCalibration:
    """Prepare |0> and |1> on each measured qubit (native RX(pi)) and read it out."""
    circuits, labels = [], []
    for qubit in measured:
        for prepared in (0, 1):
            c = Circuit(n_qubits)
            if prepared:
                c.add(gates.RX(qubit, theta=float(np.pi)))
            else:
                # An explicit identity-equivalent native gate keeps the circuit non-empty.
                c.add(gates.RZ(qubit, theta=0.0))
            c.add(gates.M(qubit))
            circuits.append(c)
            labels.append((qubit, prepared))
    results = execute_batch(executor, circuits, shots=shots, seed=seed)
    p01, p10, raw = {}, {}, {}
    for (qubit, prepared), counts in zip(labels, results):
        ones = counts.get("1", 0) / shots
        raw[f"q{qubit}-prep{prepared}"] = dict(counts)
        if prepared == 0:
            p01[qubit] = ones
        else:
            p10[qubit] = 1.0 - ones
    return ReadoutCalibration(p01=p01, p10=p10, shots=shots, circuits=len(circuits), raw_counts=raw)


def z_expectation(counts: Mapping[str, int], bit: int = 0) -> float:
    """<Z> of one measured bit from counts: P(0) - P(1)."""
    total = sum(counts.values())
    ones = sum(c for key, c in counts.items() if key[bit] == "1")
    return 1.0 - 2.0 * ones / total


def mitigate_z_expectation(raw_z: float, p01: float, p10: float) -> float:
    """Invert the single-qubit confusion matrix and clip to [-1, 1].

    With p1_raw = p01 (1 - p1) + (1 - p10) p1, the true p1 is
    (p1_raw - p01) / (1 - p01 - p10).
    """
    denominator = 1.0 - p01 - p10
    if denominator <= 0.05:
        raise ValueError("readout errors too large to invert reliably.")
    p1_raw = (1.0 - raw_z) / 2.0
    p1 = min(1.0, max(0.0, (p1_raw - p01) / denominator))
    return 1.0 - 2.0 * p1


# --------------------------------------------------------------------------- BSC / Qililab integration point
class QililabExecutor:
    """Skeleton adapter for Qilimanjaro chips through Qililab (NOT functional).

    Expected responsibilities, to be implemented with the BSC documentation:

    1. build/connect the platform from the runcard provided by the BSC;
    2. map compact circuit indices to chip qubits (``physical_qubits`` from
       :class:`~src.transpilation.TranspiledCircuit`) and to the chip's native
       gate names;
    3. submit a whole batch per connection and return canonical bitstring
       counts that pass :func:`~src.circuit_execution.validate_counts`;
    4. record chip id, runcard hash and calibration timestamp in
       ``capabilities.metadata`` for provenance.
    """

    def __init__(self, runcard: str, physical_qubits: Sequence[int], *, max_shots: int | None = None):
        self.runcard = runcard
        self.physical_qubits = tuple(physical_qubits)
        self.capabilities = ResourceProfile(
            max_shots=max_shots,
            basis_gates=NATIVE_GATES,
            metadata={"adapter": "qililab (skeleton, not validated)", "runcard": runcard},
        )

    @staticmethod
    def available() -> bool:
        try:
            import qililab  # noqa: F401
        except ImportError:
            return False
        return True

    def execute_counts(self, circuit, *, shots: int, seed: int | None = None) -> dict[str, int]:
        return self.execute_batch([circuit], shots=shots, seed=seed)[0]

    def execute_batch(self, circuits, *, shots: int, seed: int | None = None) -> list[dict[str, int]]:
        raise NotImplementedError(
            "QililabExecutor is a placeholder: implement it with the Qililab API documented by the BSC "
            "(platform build/connect from the runcard, batched execution, result-to-counts conversion)."
        )


__all__ = [
    "BatchCircuitExecutor",
    "HardwareMockExecutor",
    "MockNoise",
    "QililabExecutor",
    "ReadoutCalibration",
    "calibrate_readout",
    "execute_batch",
    "mitigate_z_expectation",
    "z_expectation",
]
