"""Minimal transpiler for the QCNN: layout, SWAP routing and native gates.

Qibo 0.1.11 (the pinned version) has no transpiler, so this module implements
the three steps needed to run the small QCNN on a device with restricted
connectivity:

1. **Layout**: an injective map from logical to physical qubits that minimizes
   the summed coupling distance of the circuit's two-qubit interactions (a
   SWAP-free embedding is found whenever one exists).
2. **Routing**: before every two-qubit gate on non-adjacent physical qubits,
   SWAPs are inserted along a shortest path and the mapping is updated.
3. **Native decomposition** into ``{RX, RY, RZ, CZ}``; ``CNOT`` becomes
   ``RY(-pi/2) . CZ . RY(pi/2)`` on the target, ``H`` becomes
   ``RZ(pi)`` then ``RY(pi/2)`` (up to global phase) and ``SWAP`` three CNOTs.
   Consecutive rotations about the same axis on a qubit are merged.

The native set is an **assumption** modelled on superconducting devices
(arbitrary XY rotations, virtual RZ, CZ). The real native set, topology and
qubit quality of the BSC chips must be taken from the provider once access is
granted. Topology presets are generic, not descriptions of a specific chip.

The output circuit is compacted to the physical qubits actually used:
``TranspiledCircuit.physical_qubits[i]`` is the device qubit that compact
index ``i`` stands for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import math
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
from qibo import gates
from qibo.models import Circuit

NATIVE_GATES = ("RX", "RY", "RZ", "CZ")
_SUPPORTED_INPUT = {"RX", "RY", "RZ", "H", "X", "CNOT", "CZ", "SWAP"}
_EXHAUSTIVE_LIMIT = 200_000


# --------------------------------------------------------------------------- topologies
def topology_edges(name: str, n_qubits: int) -> tuple[tuple[int, int], ...]:
    """Generic coupling maps (hypothetical presets, not a specific chip).

    ``line``, ``ring``, ``star`` (qubit 0 is the hub), ``grid`` (as square as
    possible, row-major) and ``full``.
    """
    if not isinstance(n_qubits, int) or n_qubits < 2:
        raise ValueError("n_qubits must be an integer of at least 2.")
    if name == "line":
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
    elif name == "ring":
        edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)] if n_qubits > 2 else [(0, 1)]
    elif name == "star":
        edges = [(0, i) for i in range(1, n_qubits)]
    elif name == "grid":
        cols = math.ceil(math.sqrt(n_qubits))
        edges = []
        for q in range(n_qubits):
            r, c = divmod(q, cols)
            if c + 1 < cols and q + 1 < n_qubits:
                edges.append((q, q + 1))
            if q + cols < n_qubits:
                edges.append((q, q + cols))
    elif name == "full":
        edges = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    else:
        raise ValueError("topology must be one of: line, ring, star, grid, full.")
    return tuple(sorted({tuple(sorted(e)) for e in edges}))


def _graph(edges: Iterable[tuple[int, int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        raise ValueError("the coupling map must be a non-empty connected graph.")
    return graph


# --------------------------------------------------------------------------- layout
def interaction_pairs(circuit) -> list[tuple[int, int]]:
    """Ordered list of logical two-qubit interactions (one entry per gate)."""
    pairs = []
    for gate in circuit.queue:
        if gate.__class__.__name__ == "M":
            continue
        if len(gate.qubits) == 2:
            pairs.append(tuple(gate.qubits))
        elif len(gate.qubits) > 2:
            raise ValueError("gates on more than two qubits are not supported.")
    return pairs


def choose_layout(
    circuit,
    edges: Sequence[tuple[int, int]],
) -> dict[int, int]:
    """Logical -> physical map minimizing the summed interaction distance."""
    graph = _graph(edges)
    n_logical = circuit.nqubits
    physical = sorted(graph.nodes)
    if n_logical > len(physical):
        raise ValueError(f"circuit needs {n_logical} qubits; the coupling map has {len(physical)}.")
    distance = dict(nx.all_pairs_shortest_path_length(graph))
    weights: dict[tuple[int, int], int] = {}
    for a, b in interaction_pairs(circuit):
        key = tuple(sorted((a, b)))
        weights[key] = weights.get(key, 0) + 1

    def cost(mapping: Sequence[int]) -> int:
        return sum(w * (distance[mapping[a]][mapping[b]] - 1) for (a, b), w in weights.items())

    def search(candidates: Sequence[int]):
        best, best_cost = None, None
        for mapping in permutations(candidates, n_logical):
            c = cost(mapping)
            if best_cost is None or c < best_cost:
                best, best_cost = mapping, c
                if c == 0:
                    break
        return best, best_cost

    if math.perm(len(physical), n_logical) <= _EXHAUSTIVE_LIMIT:
        best, _ = search(physical)
    else:
        # Large devices: exhaustive search inside BFS neighbourhoods of every node.
        best, best_cost = None, None
        size = min(len(physical), n_logical + 2)
        for root in physical:
            region = list(nx.bfs_tree(graph, root).nodes)[:size]
            mapping, c = search(region)
            if mapping is not None and (best_cost is None or c < best_cost):
                best, best_cost = mapping, c
                if c == 0:
                    break
    return {logical: int(best[logical]) for logical in range(n_logical)}


# --------------------------------------------------------------------------- native decomposition
def _native_ops(name: str, qubits: tuple[int, ...], theta: float | None) -> list[tuple[str, tuple[int, ...], float | None]]:
    if name in ("RX", "RY", "RZ"):
        return [(name, qubits, theta)]
    if name == "CZ":
        return [("CZ", qubits, None)]
    if name == "H":
        (q,) = qubits
        return [("RZ", (q,), math.pi), ("RY", (q,), math.pi / 2)]
    if name == "X":
        (q,) = qubits
        return [("RX", (q,), math.pi)]
    if name == "CNOT":
        control, target = qubits
        return [("RY", (target,), -math.pi / 2), ("CZ", (control, target), None), ("RY", (target,), math.pi / 2)]
    if name == "SWAP":
        a, b = qubits
        return _native_ops("CNOT", (a, b), None) + _native_ops("CNOT", (b, a), None) + _native_ops("CNOT", (a, b), None)
    raise ValueError(f"gate {name} is not supported by the transpiler.")


def _merge_rotations(ops: list[tuple[str, tuple[int, ...], float | None]]):
    """Merge consecutive same-axis rotations on a qubit; drop identities (mod 2*pi up to phase)."""
    merged: list[list[Any]] = []
    last_on_qubit: dict[int, int] = {}
    for name, qubits, theta in ops:
        if name in ("RX", "RY", "RZ"):
            (q,) = qubits
            index = last_on_qubit.get(q)
            if index is not None and merged[index] is not None and merged[index][0] == name:
                merged[index][2] += theta
                continue
            merged.append([name, qubits, theta])
            last_on_qubit[q] = len(merged) - 1
        else:
            merged.append([name, qubits, theta])
            for q in qubits:
                last_on_qubit[q] = len(merged) - 1
    out = []
    for name, qubits, theta in merged:
        if theta is not None:
            # RX/RY/RZ(2*pi) = -I: a global phase on single-qubit gates is harmless.
            reduced = math.remainder(theta, 2 * math.pi)
            if abs(reduced) < 1e-12:
                continue
        out.append((name, qubits, theta))
    return out


@dataclass
class TranspiledCircuit:
    circuit: Any
    physical_qubits: tuple[int, ...]
    initial_layout: dict[int, int]
    final_layout: dict[int, int]
    measured_logical: tuple[int, ...]
    measured_physical: tuple[int, ...]
    coupling_edges: tuple[tuple[int, int], ...]
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_qubits": list(self.physical_qubits),
            "initial_layout": {str(k): v for k, v in self.initial_layout.items()},
            "final_layout": {str(k): v for k, v in self.final_layout.items()},
            "measured_logical": list(self.measured_logical),
            "measured_physical": list(self.measured_physical),
            "native_gates": list(NATIVE_GATES),
            "stats": dict(self.stats),
        }


def transpile(
    circuit,
    edges: Sequence[tuple[int, int]],
    *,
    measured: Sequence[int] = (0,),
    layout: dict[int, int] | None = None,
    compact: bool = True,
) -> TranspiledCircuit:
    """Route and decompose a measurement-free logical Qibo circuit.

    ``measured`` are the logical qubits to measure at the end; measurements are
    appended on their final physical positions (SWAPs move logical qubits).
    When ``compact=True`` (default, intended for local simulators), gate indices
    are compacted to ``0..len(used)-1``; when ``compact=False`` (intended for
    hardware adapters such as Qililab), gates are emitted directly on the real
    physical qubit indices ``0..max(used)``.
    """
    edges = tuple(sorted({tuple(sorted(e)) for e in edges}))
    graph = _graph(edges)
    if any(g.__class__.__name__ == "M" for g in circuit.queue):
        raise ValueError("pass a circuit without measurements; use `measured` instead.")
    for gate in circuit.queue:
        if gate.__class__.__name__ not in _SUPPORTED_INPUT and gate.name.upper() not in _SUPPORTED_INPUT:
            raise ValueError(f"gate {gate.__class__.__name__} is not supported by the transpiler.")
    layout = dict(layout) if layout is not None else choose_layout(circuit, edges)
    if sorted(layout) != list(range(circuit.nqubits)) or len(set(layout.values())) != circuit.nqubits:
        raise ValueError("layout must map every logical qubit to a distinct physical qubit.")
    if any(p not in graph for p in layout.values()):
        raise ValueError("layout references a qubit outside the coupling map.")

    current = dict(layout)  # logical -> physical
    ops: list[tuple[str, tuple[int, ...], float | None]] = []
    swaps = 0
    logical_two_qubit = 0
    for gate in circuit.queue:
        name = gate.__class__.__name__.upper()
        if name == "CNOT" or gate.name.lower() in ("cx", "cnot"):
            name = "CNOT"
        theta = float(gate.parameters[0]) if gate.parameters else None
        logical = tuple(gate.qubits)
        if len(logical) == 2:
            logical_two_qubit += 1
            # Move the first operand along a shortest path until it neighbours the second.
            while not graph.has_edge(current[logical[0]], current[logical[1]]):
                here = current[logical[0]]
                there = nx.shortest_path(graph, here, current[logical[1]])[1]
                ops.extend(_native_ops("SWAP", (here, there), None))
                swaps += 1
                occupant = next((l for l, p in current.items() if p == there), None)
                current[logical[0]] = there
                if occupant is not None:
                    current[occupant] = here
            physical = tuple(current[q] for q in logical)
        else:
            physical = tuple(current[q] for q in logical)
        ops.extend(_native_ops(name, physical, theta))
    ops = _merge_rotations(ops)

    used = sorted({q for _, qubits, _ in ops for q in qubits} | set(current.values()))
    if compact:
        qubit_map = {p: i for i, p in enumerate(used)}
        out = Circuit(len(used))
    else:
        qubit_map = {p: p for p in used}
        out = Circuit(max(used) + 1)
    for name, qubits, theta in ops:
        q = tuple(qubit_map[p] for p in qubits)
        if name == "CZ":
            out.add(gates.CZ(*q))
        else:
            out.add(getattr(gates, name)(q[0], theta=float(theta)))
    measured_logical = tuple(int(q) for q in measured)
    measured_physical = tuple(current[q] for q in measured_logical)
    out.add(gates.M(*(qubit_map[p] for p in measured_physical)))

    one_qubit = sum(1 for name, _, _ in ops if name != "CZ")
    two_qubit = sum(1 for name, _, _ in ops if name == "CZ")
    stats = {
        "logical_depth": int(circuit.depth),
        "logical_gates": int(circuit.ngates),
        "logical_two_qubit_gates": logical_two_qubit,
        "native_depth": int(out.depth),
        "native_one_qubit_gates": one_qubit,
        "native_two_qubit_gates": two_qubit,
        "swaps": swaps,
        "physical_qubits_used": len(used),
    }
    mapped_edges = tuple(
        (qubit_map[a], qubit_map[b]) for a, b in edges if a in qubit_map and b in qubit_map
    )
    return TranspiledCircuit(
        circuit=out,
        physical_qubits=tuple(used),
        initial_layout=layout,
        final_layout=dict(current),
        measured_logical=measured_logical,
        measured_physical=measured_physical,
        coupling_edges=mapped_edges,
        stats=stats,
    )


# --------------------------------------------------------------------------- verification helpers
def unitary(circuit) -> np.ndarray:
    """Dense unitary of a small measurement-free circuit (for tests)."""
    n = circuit.nqubits
    dim = 2 ** n
    columns = []
    for index in range(dim):
        state = np.zeros(dim, dtype=complex)
        state[index] = 1.0
        clean = Circuit(n)
        for gate in circuit.queue:
            if gate.__class__.__name__ != "M":
                clean.add(gate)
        columns.append(np.asarray(clean.execute(initial_state=state.copy()).state()).reshape(-1))
    return np.stack(columns, axis=1)


def measured_marginal(circuit, measured: Sequence[int]) -> np.ndarray:
    """Exact probabilities of the measured qubits (Qibo order: qubit 0 = MSB)."""
    n = circuit.nqubits
    clean = Circuit(n)
    for gate in circuit.queue:
        if gate.__class__.__name__ != "M":
            clean.add(gate)
    amplitudes = np.asarray(clean.execute().state()).reshape(-1)
    probabilities = (np.abs(amplitudes) ** 2).reshape([2] * n)
    keep = list(measured)
    other = tuple(q for q in range(n) if q not in keep)
    marginal = probabilities.sum(axis=other) if other else probabilities
    # Reorder axes to the requested measurement order.
    current_order = [q for q in range(n) if q in keep]
    marginal = np.moveaxis(marginal, [current_order.index(q) for q in keep], range(len(keep)))
    return marginal.reshape(-1)


__all__ = [
    "NATIVE_GATES",
    "TranspiledCircuit",
    "choose_layout",
    "interaction_pairs",
    "measured_marginal",
    "topology_edges",
    "transpile",
    "unitary",
]
