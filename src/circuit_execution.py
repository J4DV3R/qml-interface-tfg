"""Measured-circuit execution boundary shared by simulators and adapters.

An adapter accepts a measured Qibo circuit and returns canonical binary
bitstrings with counts. A hardware-specific adapter is intentionally not
provided: it must translate/submit circuits for a chosen backend and report
its declared resource profile. The Qibo adapter below is the local reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Collection, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ResourceProfile:
    """Declared backend limits and separately tracked validation status.

    ``max_qubits``, ``max_shots``, ``basis_gates`` and ``connectivity`` are
    declarations used for request validation. They do not imply that a circuit
    has been transpiled or that the declarations have been independently
    validated. Those states are represented separately by
    ``validated_backend`` and ``transpiled_backend``.
    """

    max_qubits: int | None = None
    max_shots: int | None = None
    basis_gates: tuple[str, ...] | None = None
    connectivity: tuple[tuple[int, int], ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validated_backend: str | None = None
    transpiled_backend: str | None = None

    def __post_init__(self):
        for name in ("max_qubits", "max_shots"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be None or a positive integer.")

        if self.basis_gates is not None:
            try:
                basis_gates = tuple(self.basis_gates)
            except TypeError as exc:
                raise ValueError("basis_gates must be a sequence of names or None.") from exc
            if any(not isinstance(gate, str) or not gate.strip() for gate in basis_gates):
                raise ValueError("basis_gates entries must be non-empty strings.")
            object.__setattr__(self, "basis_gates", basis_gates)

        if self.connectivity is not None:
            try:
                edges = tuple(tuple(edge) for edge in self.connectivity)
            except TypeError as exc:
                raise ValueError("connectivity must be a sequence of qubit pairs or None.") from exc
            normalized_edges = []
            for edge in edges:
                if len(edge) != 2:
                    raise ValueError("each connectivity entry must contain two qubits.")
                left, right = edge
                if (
                    not isinstance(left, int)
                    or isinstance(left, bool)
                    or not isinstance(right, int)
                    or isinstance(right, bool)
                    or left < 0
                    or right < 0
                    or left == right
                ):
                    raise ValueError("connectivity entries must be distinct non-negative qubits.")
                if self.max_qubits is not None and max(left, right) >= self.max_qubits:
                    raise ValueError("connectivity references a qubit above max_qubits.")
                normalized_edges.append(tuple(sorted((left, right))))
            object.__setattr__(self, "connectivity", tuple(sorted(set(normalized_edges))))

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        for name in ("validated_backend", "transpiled_backend"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None.")

    def validate_request(self, n_qubits: int, shots: int) -> None:
        """Check a request against declared maxima, not basis/topology support."""
        if (
            not isinstance(n_qubits, int)
            or isinstance(n_qubits, bool)
            or n_qubits < 1
        ):
            raise ValueError("n_qubits must be a positive integer.")
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
            raise ValueError("shots must be a positive integer for measured execution.")
        if self.max_qubits is not None and n_qubits > self.max_qubits:
            raise ValueError(
                f"circuit requires {n_qubits} qubits; declared maximum is {self.max_qubits}."
            )
        if self.max_shots is not None and shots > self.max_shots:
            raise ValueError(
                f"request requires {shots} shots; declared maximum is {self.max_shots}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly metadata with declaration and validation split."""
        return {
            "declared": {
                "max_qubits": self.max_qubits,
                "max_shots": self.max_shots,
                "basis_gates": list(self.basis_gates) if self.basis_gates is not None else None,
                "connectivity": [list(edge) for edge in self.connectivity]
                if self.connectivity is not None
                else None,
                "metadata": dict(self.metadata),
            },
            "validation": {
                "performed": self.validated_backend is not None,
                "backend": self.validated_backend,
            },
            "transpilation": {
                "performed": self.transpiled_backend is not None,
                "backend": self.transpiled_backend,
            },
        }


# Backwards-friendly name for callers that primarily think in capabilities.
BackendCapabilities = ResourceProfile


@runtime_checkable
class CircuitExecutor(Protocol):
    """Vendor-neutral measured-circuit adapter contract.

    Returned keys must contain only ``0``/``1`` and follow the order of the
    measured qubits in the circuit. Hardware adapters must enforce finite shots
    and return counts; exact state-vector access is deliberately not part of
    this interface.
    """

    capabilities: ResourceProfile

    def execute_counts(
        self,
        circuit: Any,
        *,
        shots: int,
        seed: int | None = None,
    ) -> Mapping[str, int]: ...


def validate_counts(
    counts: Mapping[str, int], *, n_bits: int, shots: int
) -> dict[str, int]:
    """Validate and normalize canonical bitstring counts from any adapter."""
    if not isinstance(counts, Mapping):
        raise TypeError("executor counts must be a mapping of bitstrings to counts.")
    if (
        not isinstance(n_bits, int)
        or isinstance(n_bits, bool)
        or n_bits < 1
    ):
        raise ValueError("n_bits must be a positive integer.")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer.")

    normalized: dict[str, int] = {}
    for bitstring, count in counts.items():
        if (
            not isinstance(bitstring, str)
            or len(bitstring) != n_bits
            or any(bit not in "01" for bit in bitstring)
        ):
            raise ValueError(f"invalid {n_bits}-bit measurement key: {bitstring!r}.")
        if (
            not isinstance(count, (int, np.integer))
            or isinstance(count, (bool, np.bool_))
            or count < 1
        ):
            raise ValueError("measurement counts must be positive integers.")
        normalized[bitstring] = int(count)
    if not normalized:
        raise ValueError("executor returned no measurement counts.")
    observed_shots = sum(normalized.values())
    if observed_shots != shots:
        raise ValueError(
            f"executor returned {observed_shots} counts for a {shots}-shot request."
        )
    return normalized


def qibo_local_depolarizing_sensitivity_transform(
    probability: float,
    *,
    gate_names: Collection[str] | None = None,
) -> Callable[[Any], Any]:
    """Create a simulator-only local depolarizing sensitivity transform.

    By default, after each non-measurement logical gate, an independent
    one-qubit Qibo depolarizing channel is added to every qubit touched by that
    gate. This legacy-compatible default also covers controlled-X instructions
    without asking Qibo to construct a high-arity depolarizing channel. When
    ``gate_names`` is supplied, only gates whose Qibo ``name`` is in that set
    receive channels (for example, ``{"RY", "RZ", "CNOT"}`` for the shared
    QCNN sensitivity convention). Measurements and existing channels are never
    transformed. This is a simulator sensitivity model, not a calibration or
    physically complete hardware noise model.
    """
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float, np.number))
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("probability must be finite and between 0 and 1.")

    selected_gate_names = None
    if gate_names is not None:
        if isinstance(gate_names, str):
            raise TypeError("gate_names must be a collection of gate names, not a string.")
        try:
            names = tuple(gate_names)
        except TypeError as exc:
            raise TypeError("gate_names must be a collection of gate names or None.") from exc
        if not names or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError("gate_names must contain non-empty strings.")
        selected_gate_names = frozenset(
            "cx" if name.strip().lower() == "cnot" else name.strip().lower()
            for name in names
        )

    from qibo import gates
    from qibo.models import Circuit

    probability = float(probability)

    def transform(circuit):
        n_qubits = getattr(circuit, "nqubits", None)
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise TypeError("sensitivity transform requires a Qibo circuit.")
        noisy_circuit = Circuit(n_qubits)
        for gate in circuit.queue:
            noisy_circuit.add(gate)
            gate_name = str(getattr(gate, "name", "")).lower()
            if gate_name == "measure" or gate.__class__.__name__.endswith("Channel"):
                continue
            if gate_name == "cnot":
                gate_name = "cx"
            if selected_gate_names is not None and gate_name not in selected_gate_names:
                continue
            for qubit in dict.fromkeys(getattr(gate, "qubits", ())):
                noisy_circuit.add(
                    gates.DepolarizingChannel((qubit,), lam=probability)
                )
        return noisy_circuit

    return transform


def _measured_qubit_count(circuit) -> int:
    measurements = getattr(circuit, "measurements", None)
    if not measurements:
        raise ValueError("circuit must contain at least one measurement gate.")
    return sum(len(gate.target_qubits) for gate in measurements)


class QiboSimulatorExecutor:
    """Reference Qibo shot executor; not a hardware or transpilation adapter.

    ``noise_transform`` is an optional injected simulator/noise layer receiving
    the complete measured circuit. It can be used for QCNN and arithmetic-update
    circuits, but its coverage/physical meaning is the caller's responsibility.
    In particular, the existing QCNN depolarizing model does not automatically
    cover the update circuit's multi-controlled-X operations. The separately
    provided :func:`qibo_local_depolarizing_sensitivity_transform` applies
    one-qubit channels to all qubits touched by each logical gate.
    """

    def __init__(
        self,
        *,
        capabilities: ResourceProfile | None = None,
        noise_transform: Callable[[Any], Any] | None = None,
    ):
        if capabilities is not None and not isinstance(capabilities, ResourceProfile):
            raise TypeError("capabilities must be a ResourceProfile instance.")
        if noise_transform is not None and not callable(noise_transform):
            raise TypeError("noise_transform must be callable or None.")
        self.capabilities = capabilities or ResourceProfile(
            metadata={"adapter": "qibo-local-simulator"}
        )
        self.noise_transform = noise_transform

    def execute_counts(
        self,
        circuit,
        *,
        shots: int,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Execute a measured Qibo circuit without exposing its state vector."""
        if seed is not None and (
            not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_))
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None.")
        n_qubits = getattr(circuit, "nqubits", None)
        if n_qubits is None:
            raise TypeError("QiboSimulatorExecutor requires a Qibo circuit.")
        self.capabilities.validate_request(n_qubits, shots)
        measured_bits = _measured_qubit_count(circuit)

        executable = circuit
        if self.noise_transform is not None:
            executable = self.noise_transform(circuit)
            if executable is None:
                raise ValueError("noise_transform must return an executable circuit.")
            if (
                getattr(executable, "nqubits", None) != n_qubits
                or _measured_qubit_count(executable) != measured_bits
            ):
                raise ValueError(
                    "noise_transform must preserve the circuit register and measured qubits."
                )
        if seed is None:
            result = executable.execute(nshots=shots)
            counts = result.frequencies(binary=True)
        else:
            # Qibo's sampling uses NumPy's global RNG. Preserve caller state so a
            # seeded execution is reproducible without perturbing other code.
            random_state = np.random.get_state()
            try:
                np.random.seed(int(seed))
                result = executable.execute(nshots=shots)
                # Qibo may defer sampling until frequencies() is requested.
                counts = result.frequencies(binary=True)
            finally:
                np.random.set_state(random_state)
        return validate_counts(
            counts, n_bits=measured_bits, shots=shots
        )


__all__ = [
    "BackendCapabilities",
    "CircuitExecutor",
    "QiboSimulatorExecutor",
    "ResourceProfile",
    "qibo_local_depolarizing_sensitivity_transform",
    "validate_counts",
]
