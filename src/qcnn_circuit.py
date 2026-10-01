"""Convolutional circuit layers for a one-dimensional QCNN."""

from collections.abc import Sequence

import numpy as np
from qibo import gates
from qibo.models import Circuit


def add_conv_layer(
    circuit: Circuit,
    active_qubits: int | Sequence[int],
    weights: Sequence[float],
    shifted_occurrence: tuple[int, int, float] | None = None,
) -> Circuit:
    """Append a translation-shared two-qubit convolution over adjacent qubits.

    ``weights`` contains four angles shared by every adjacent pair. A shift can
    be applied to one gate occurrence as required by the parameter-shift rule.
    """
    qubits = (
        list(range(active_qubits))
        if isinstance(active_qubits, int)
        else list(active_qubits)
    )
    if len(qubits) < 2:
        raise ValueError("a convolution layer needs at least two active qubits.")
    if len(set(qubits)) != len(qubits):
        raise ValueError("active qubits must be unique.")
    if any(not isinstance(qubit, int) or not 0 <= qubit < circuit.nqubits for qubit in qubits):
        raise ValueError("active qubits must be valid indices in the circuit.")
    if len(weights) != 4:
        raise ValueError("a convolution layer requires exactly four weights.")
    values = np.asarray([float(weight) for weight in weights], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("convolution weights must be finite.")

    if shifted_occurrence is not None:
        pair_index, weight_index, shift = shifted_occurrence
        if not 0 <= pair_index < len(qubits) - 1 or not 0 <= weight_index < 4:
            raise ValueError("shifted convolution occurrence is out of range.")
        if not np.isfinite(shift):
            raise ValueError("convolution shift must be finite.")
    else:
        pair_index, weight_index, shift = -1, -1, 0.0

    for pair_index_current, (left, right) in enumerate(zip(qubits, qubits[1:])):
        angles = values.copy()
        if pair_index_current == pair_index:
            angles[weight_index] += shift
        circuit.add(gates.RY(left, theta=float(angles[0])))
        circuit.add(gates.RZ(left, theta=float(angles[1])))
        circuit.add(gates.CNOT(left, right))
        circuit.add(gates.RY(right, theta=float(angles[2])))
        circuit.add(gates.RZ(right, theta=float(angles[3])))
        circuit.add(gates.CNOT(right, left))
    return circuit


if __name__ == "__main__":
    circuit = Circuit(4)
    add_conv_layer(circuit, [0, 1, 2, 3], [0.12, 0.45, 0.23, 0.88])
    print(circuit.draw())
