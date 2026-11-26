"""Run the smallest MareNostrum5-Ona Qililab smoke test.

The default circuit applies X and measures one supplied physical qubit. It
uses 100 shots and does not use Iris or any trained QCNN checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from qibo import gates
from qibo.models import Circuit

from src.hardware_execution import QililabExecutor


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_physical_qubits(value: str) -> tuple[int, ...]:
    qubits: list[int] = []
    for item in value.split(","):
        try:
            qubit = int(item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "physical-qubits must be comma-separated integers"
            ) from exc
        if qubit < 0:
            raise argparse.ArgumentTypeError("physical-qubits must be non-negative")
        qubits.append(qubit)
    if len(set(qubits)) != len(qubits):
        raise argparse.ArgumentTypeError("physical-qubits must not contain duplicates")
    return tuple(qubits)


def build_smoke_circuit(operation: str, physical_qubits: tuple[int, ...]) -> Circuit:
    if operation == "x":
        if len(physical_qubits) != 1:
            raise ValueError("operation x requires exactly one physical qubit")
        qubit = physical_qubits[0]
        circuit = Circuit(qubit + 1)
        circuit.add(gates.X(qubit))
        circuit.add(gates.M(qubit))
        return circuit

    if operation == "bell":
        if len(physical_qubits) != 2:
            raise ValueError("operation bell requires exactly two physical qubits")
        a, b = physical_qubits
        circuit = Circuit(max(a, b) + 1)
        circuit.add(gates.H(a))
        circuit.add(gates.CNOT(a, b))
        circuit.add(gates.M(a, b))
        return circuit

    raise ValueError(f"unknown operation: {operation}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runcard",
        default=None,
        help="Qililab runcard path or identifier; alternatively RUNCARD_BLUE or RUNCARD_RED",
    )
    parser.add_argument(
        "--physical-qubits",
        type=parse_physical_qubits,
        default=(0,),
        help="Comma-separated physical qubits, for example 0 or 4,5",
    )
    parser.add_argument("--shots", type=positive_int, default=100)
    parser.add_argument("--chip-id", default=None)
    parser.add_argument(
        "--operation",
        choices=("x", "bell"),
        default="x",
        help="x: X + measurement on one qubit; bell: H-CNOT-measurement on two qubits",
    )
    parser.add_argument(
        "--qililab-adapter-module",
        default="src.qililab_adapter",
        help="Python module exposing execute_batch",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    runcard = args.runcard or os.getenv("RUNCARD_BLUE") or os.getenv("RUNCARD_RED")
    if not runcard:
        raise SystemExit(
            "--runcard is required or RUNCARD_BLUE/RUNCARD_RED must be set"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    circuit = build_smoke_circuit(args.operation, args.physical_qubits)
    executor = QililabExecutor(
        runcard,
        args.physical_qubits,
        adapter_module=args.qililab_adapter_module,
        chip_id=args.chip_id,
        max_qubits=max(args.physical_qubits) + 1,
        max_shots=args.shots,
        basis_gates=("DRAG", "RZ", "CZ", "M"),
    )
    counts = executor.execute_batch([circuit], shots=args.shots, seed=None)

    record: dict[str, Any] = {
        "kind": "qililab-smoke",
        "status": "real-hardware-request",
        "operation": args.operation,
        "shots": args.shots,
        "chip_id": args.chip_id,
        "physical_qubits": list(args.physical_qubits),
        "counts": counts,
        "counts_total": [sum(mapping.values()) for mapping in counts],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return record


if __name__ == "__main__":
    main()
