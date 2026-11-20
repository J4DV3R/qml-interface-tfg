"""Technical figures for the RES quantum application (BSC plan, phases 0-1).

Reports, for both encodings, the logical and transpiled circuit size on
generic topologies (qubits, depth, gate counts, SWAPs) and the circuit/shot
budget and estimated hours of the planned scenarios. Times come from the
assumption-labelled models in ``src.hardware_cost.TIME_MODELS``.

    qml_env/Scripts/python.exe -B -m experiments.estimate_hardware_cost --output docs/bsc/cost-estimates.md --json docs/bsc/cost-estimates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.hardware_cost import TIME_MODELS, default_scenarios
from src.qcnn_engine import QCNNConfig, build_qcnn_circuit
from src.transpilation import topology_edges, transpile

TOPOLOGIES = (("full", 4), ("line", 5), ("ring", 5), ("star", 5), ("grid", 20), ("grid", 35))


def circuit_table(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for encoding in ("angle", "phase"):
        for topology, n in TOPOLOGIES:
            depths, stats = [], None
            for _ in range(20):  # generic weights/features: depth can vary with merged rotations
                circuit = build_qcnn_circuit(rng.uniform(0, 1, 4), rng.uniform(-np.pi, np.pi, (2, 4)),
                                             QCNNConfig(4, encoding=encoding))
                result = transpile(circuit, topology_edges(topology, n))
                depths.append(result.stats["native_depth"])
                stats = result.stats
            rows.append({"encoding": encoding, "topology": f"{topology}-{n}", **stats,
                         "native_depth": max(depths)})
    return rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    circuits = circuit_table()
    scenarios = [s.to_dict() for s in default_scenarios(args.shots)]
    lines = [
        "# Estimaciones técnicas para la solicitud RES (generado)",
        "",
        "Generado por `experiments/estimate_hardware_cost.py`. Topologías **genéricas e hipotéticas**; puertas nativas "
        "supuestas `{RX, RY, RZ, CZ}`. Profundidad nativa: máximo sobre 20 circuitos con pesos y datos aleatorios.",
        "",
        "## Tamaño del circuito (4 qubits lógicos, 1 qubit medido)",
        "",
        "| Cod. | Topología | Prof. lógica | Puertas 2q lógicas | SWAPs | Prof. nativa | CZ | Puertas 1q | Qubits físicos |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in circuits:
        lines.append(f"| {r['encoding']} | {r['topology']} | {r['logical_depth']} | {r['logical_two_qubit_gates']} | "
                     f"{r['swaps']} | {r['native_depth']} | {r['native_two_qubit_gates']} | "
                     f"{r['native_one_qubit_gates']} | {r['physical_qubits_used']} |")
    lines += ["", f"## Escenarios ({args.shots} shots por circuito)", "",
              "| Escenario | Circuitos | Shots totales | Horas (cota sin lotes) | Horas (supuesto con lotes) |",
              "|---|---|---|---|---|"]
    for s in scenarios:
        lines.append(f"| {s['scenario']} | {s['circuits']:,} | {s['total_shots']:,} | "
                     f"{s['hours']['unbatched-upper-bound']:.1f} | {s['hours']['batched-assumption']:.2f} |")
    lines += ["", "## Modelos de tiempo (supuestos)", ""]
    for model in TIME_MODELS.values():
        lines.append(f"- **{model.name}**: {model.seconds_per_job:g} s por trabajo, "
                     f"{model.circuits_per_job} circuitos por trabajo, {model.seconds_per_shot * 1e6:g} µs por shot. "
                     f"{model.source}")
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    payload = {"shots": args.shots, "circuits": circuits, "scenarios": scenarios,
               "time_models": {k: vars(v) for k, v in TIME_MODELS.items()}}
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} and {args.json}")
    return payload


if __name__ == "__main__":
    main()
