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
        if value < 0:
            raise ValueError(f"count for {key!r} must be non-negative")
        normalized[text] = normalized.get(text, 0) + value

    if sum(normalized.values()) != shots:
        raise ValueError(
            f"Qililab counts sum to {sum(normalized.values())}, expected {shots}"
        )
    return normalized


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
    ``physical_qubits`` is validated here and preserved by the caller; routing
    and placement remain under Qililab/runcard control.
    """
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if not isinstance(runcard, str) or not runcard.strip():
        raise ValueError("runcard must be a non-empty path or identifier")
    if not isinstance(circuits, Sequence) or isinstance(circuits, (str, bytes)):
        raise TypeError("circuits must be a sequence")
    if not circuits:
        raise ValueError("circuits must be non-empty")

    physical = tuple(int(qubit) for qubit in physical_qubits)
    if not physical or len(set(physical)) != len(physical):
        raise ValueError("physical_qubits must be a non-empty sequence without duplicates")

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
        if not isinstance(n_qubits, int) or n_qubits > len(physical):
            raise ValueError("circuit requires more qubits than physical_qubits provides")
        result = execute(circuit, runcard.strip(), nshots=shots)
        if hasattr(result, "counts") and callable(result.counts):
            counts = result.counts()
        elif isinstance(result, dict):
            counts = result
        else:
            counts = result
        outputs.append(_normalize_counts(counts, shots))
    return outputs
