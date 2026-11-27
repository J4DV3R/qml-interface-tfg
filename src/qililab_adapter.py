"""BSC/Qililab adapter for executing Qibo circuits.

This module is optional.  It imports Qililab only when an execution is
requested and supports the public ``ql.execute(circuit, runcard, nshots=...)``
interface documented for MareNostrum Ona.

No credentials or runcards are stored here.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from numbers import Integral
from typing import Any


def _normalize_counts(raw: Any, shots: int) -> dict[str, int]:
    if callable(raw):
        raw = raw()

    if hasattr(raw, "counts") and callable(raw.counts):
        raw = raw.counts()

    if not isinstance(raw, dict):
        raise TypeError("Qililab returned counts in an unsupported format")

    normalized: dict[str, int] = {}
    for key, value in raw.items():
        text = str(key).strip().lower()
        if text.startswith("0b"):
            text = text[2:]
        text = text.replace(" ", "")
        if not text or any(bit not in "01" for bit in text):
            raise ValueError(f"invalid Qililab bitstring: {key!r}")
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"count for {key!r} must be an integer")
        count = int(value)
        if count < 0:
            raise ValueError(f"count for {key!r} must be non-negative")
        if count == 0:
            continue
        normalized[text] = normalized.get(text, 0) + count

    if sum(normalized.values()) != shots:
        raise ValueError(
            f"Qililab counts sum to {sum(normalized.values())}, expected {shots}"
        )
    return normalized


def _circuit_used_qubits(circuit: Any) -> set[int]:
    used: set[int] = set()
    for gate in getattr(circuit, "queue", ()) or ():
        for qubit in getattr(gate, "qubits", ()) or ():
            used.add(int(qubit))
        for qubit in getattr(gate, "target_qubits", ()) or ():
            used.add(int(qubit))
    return used


def execute_batch(
    circuits: Sequence[Any],
    shots: int,
    seed: int | None,
    runcard: str,
    physical_qubits: Sequence[int],
) -> list[dict[str, int]]:
    """Execute Qibo circuits one at a time through Qililab.

    Qililab does not expose a batch contract confirmed for BSC Ona, so this
    adapter uses the documented single-circuit API for each circuit.
    Each circuit must act only on qubits declared in ``physical_qubits``.
    """
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if not isinstance(runcard, str) or not runcard.strip():
        raise ValueError("runcard must be a non-empty path or identifier")
    if not isinstance(circuits, Sequence) or isinstance(circuits, (str, bytes)):
        raise TypeError("circuits must be a sequence")
    if not circuits:
        raise ValueError("circuits must be non-empty")

    physical_list: list[int] = []
    for qubit in physical_qubits:
        if isinstance(qubit, bool) or not isinstance(qubit, Integral):
            raise ValueError("physical_qubits must contain non-negative integers")
        parsed = int(qubit)
        if parsed < 0:
            raise ValueError("physical_qubits must contain non-negative integers")
        physical_list.append(parsed)
    physical = tuple(physical_list)
    if not physical or len(set(physical)) != len(physical):
        raise ValueError("physical_qubits must be a non-empty sequence without duplicates")
    physical_set = set(physical)
    max_physical_index = max(physical)

    try:
        qililab = importlib.import_module("qililab")
    except ImportError as exc:
        raise RuntimeError(
            "Qililab is not installed in this environment; load the "
            "quantum_red or quantum_blue module provided by MareNostrum5-Ona"
        ) from exc

    execute = getattr(qililab, "execute", None)
    if not callable(execute):
        raise RuntimeError("the installed Qililab package does not expose execute")

    del seed  # The public Qililab execute API does not provide a seed parameter.
    outputs: list[dict[str, int]] = []
    for circuit in circuits:
        n_qubits = getattr(circuit, "nqubits", None)
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("circuit must declare a positive integer nqubits")
        used_qubits = _circuit_used_qubits(circuit)
        if used_qubits:
            outside = sorted(used_qubits - physical_set)
            if outside:
                raise ValueError(
                    f"circuit uses qubits {outside} outside physical_qubits {physical}"
                )
        elif n_qubits > max_physical_index + 1:
            raise ValueError("circuit requires more qubits than physical_qubits provides")
        result = execute(circuit, runcard.strip(), nshots=shots)
        outputs.append(_normalize_counts(result, shots))
    return outputs
