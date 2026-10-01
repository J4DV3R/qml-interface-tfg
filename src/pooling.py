"""Pooling operations for QCNN circuits."""

from collections.abc import Sequence

from qibo import gates
from qibo.models import Circuit


def add_pooling_layer(
    circuit: Circuit,
    qubits_to_keep: Sequence[int],
    qubits_to_discard: Sequence[int],
) -> Circuit:
    """Entangle each discarded qubit with its survivor using a CNOT.

    Qibo circuits have a fixed register size: this function does not physically
    remove qubits or reduce state-vector memory. Callers should exclude discarded
    qubits from subsequent layers and measurements, which implements the partial
    trace at the level of the remaining observables.
    """
    keep = list(qubits_to_keep)
    discard = list(qubits_to_discard)
    if not keep or len(keep) != len(discard):
        raise ValueError("pooling requires equally sized, non-empty qubit groups.")
    if len(set(keep + discard)) != len(keep) + len(discard):
        raise ValueError("pooling qubit groups must be disjoint and contain no duplicates.")
    if any(
        not isinstance(qubit, int) or not 0 <= qubit < circuit.nqubits
        for qubit in keep + discard
    ):
        raise ValueError("pooling qubits must be valid indices in the circuit.")

    for survivor, removed in zip(keep, discard):
        circuit.add(gates.CNOT(removed, survivor))
    return circuit


if __name__ == "__main__":
    circuit = Circuit(4)
    add_pooling_layer(circuit, [0, 1], [2, 3])
    print(circuit.draw())
