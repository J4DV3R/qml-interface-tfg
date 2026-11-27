"""Hardware-style inference of trained QCNN checkpoints (BSC plan, phase 3).

Takes run JSON files produced by ``experiments/run_variational_benchmark.py``
(their ``training.best_weights`` were selected on validation), rebuilds the
same partitions, transpiles one measured circuit per sample to a coupling map
with a generic native gate set, executes them in batches with finite shots,
calibrates the readout before and after, and reports exact, raw and
readout-mitigated metrics. Nothing is selected with test: the checkpoints are
fixed beforehand.

Executors:

* ``mock``: :class:`src.hardware_execution.HardwareMockExecutor`, a simulator
  with local depolarizing noise and readout error (dry runs; parameters are
  assumptions, not a calibration);
* ``qililab``: :class:`src.hardware_execution.QililabExecutor` (defaults to
  :mod:`src.qililab_adapter`; not validated on real QPU hardware yet).

Example (dry run on a hypothetical 5-qubit line):
    qml_env/Scripts/python.exe -B -m experiments.run_hardware_inference \
        results/variational/C-partitions/iris/ds1/sequential-angle/run-angle-sequential-init19.json \
        --executor mock --topology line --device-qubits 5 --shots 1000 \
        --noise-1q 0.001 --noise-2q 0.01 --readout-p01 0.02 --readout-p10 0.04 \
        --output results/hardware/dry-run/example.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from experiments.run_variational_benchmark import collect_provenance, load_partitions
from src.fast_qcnn import FastQCNN, FastQCNNConfig
from src.hardware_cost import TIME_MODELS
from src.hardware_execution import (
    HardwareMockExecutor,
    MockNoise,
    QililabExecutor,
    calibrate_readout,
    execute_batch,
    mitigate_z_expectation,
    z_expectation,
)
from src.qcnn_engine import QCNNConfig, build_qcnn_circuit
from src.transpilation import choose_layout, topology_edges, transpile
from src.variational_training import binary_metrics, mse

SCHEMA_VERSION = "1"


def parse_edges(text: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(text, str) or not text.strip():
        raise SystemExit("--edges must provide a real, non-empty coupling map")
    edges = []
    for item in text.split(","):
        try:
            a, b = item.strip().split("-")
            edge = (int(a), int(b))
        except (ValueError, AttributeError) as exc:
            raise SystemExit(f"invalid --edges entry {item!r}; use '0-1,1-2'") from exc
        if edge[0] < 0 or edge[1] < 0 or edge[0] == edge[1]:
            raise SystemExit(f"invalid --edges entry {item!r}")
        edges.append(tuple(sorted(edge)))
    if len(set(edges)) != len(edges):
        raise SystemExit("--edges contains duplicate coupling entries")
    return tuple(edges)


def parse_basis_gates(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    gates = tuple(item.strip() for item in text.split(",") if item.strip())
    if not gates:
        raise SystemExit("--basis-gates must contain at least one gate name")
    return gates


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_run(path: Path) -> dict[str, Any]:
    run = json.loads(path.read_text(encoding="utf-8"))
    condition = run["condition"]
    if condition.get("shots") is not None or condition.get("noise") or condition.get("readout_error"):
        note = "checkpoint trained with shots/noise"
    else:
        note = "checkpoint trained in exact simulation"
    return {
        "path": str(path),
        "dataset": condition["dataset"],
        "encoding": condition["encoding"],
        "optimizer": condition["optimizer"],
        "dataset_seed": int(run["seeds"]["dataset"]),
        "initialization_seed": int(run["seeds"]["initialization"]),
        "weights": np.asarray(run["training"]["best_weights"], dtype=float).reshape(2, 4),
        "checkpoint_step": run["training"]["best_step"],
        # Compare exact predictions with the stored *exact* test accuracy: for runs
        # trained with shots/noise, test.metrics is a measured estimate and the exact
        # value lives in test.exact_noiseless_diagnostic.
        "stored_test_accuracy": (
            run.get("test", {}).get("exact_noiseless_diagnostic", {}).get("metrics", {}).get("accuracy")
            if run.get("test", {}).get("exact_noiseless_diagnostic")
            else run.get("test", {}).get("metrics", {}).get("accuracy")
        ),
        "training_note": note,
    }


def summarize(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    return {"metrics": binary_metrics(predictions, targets), "mse": mse(predictions, targets)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run-*.json files with best_weights")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=["validation", "test"])
    parser.add_argument("--executor", choices=["mock", "qililab"], default="mock")
    parser.add_argument("--topology", choices=["line", "ring", "star", "grid", "full"], default="line")
    parser.add_argument("--device-qubits", type=int, default=5)
    parser.add_argument("--edges", type=str, default=None, help="explicit coupling map, e.g. '0-1,1-2,2-3'")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--calibration-shots", type=int, default=None)
    parser.add_argument("--noise-1q", type=float, default=0.0)
    parser.add_argument("--noise-2q", type=float, default=0.0)
    parser.add_argument("--readout-p01", type=float, default=0.0)
    parser.add_argument("--readout-p10", type=float, default=0.0)
    parser.add_argument("--runcard", type=str, default=None, help="Qililab runcard (qililab executor)")
    parser.add_argument("--qililab-adapter-module", type=str, default=None,
                        help="Optional external adapter; defaults to src.qililab_adapter")
    parser.add_argument("--chip-id", type=str, default=None)
    parser.add_argument("--basis-gates", type=str, default=None,
                        help="comma-separated declared basis gates, e.g. 'RX,RY,RZ,CZ'")
    parser.add_argument("--max-qubits", type=positive_int, default=None)
    parser.add_argument("--max-shots", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=2026, help="execution seed (shots/noise sampling)")
    parser.add_argument("--max-samples", type=int, default=None, help="limit samples per split (smoke tests)")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    calibration_shots = args.calibration_shots or args.shots
    if args.executor == "qililab":
        if not args.runcard:
            import os
            args.runcard = os.getenv("RUNCARD_BLUE") or os.getenv("RUNCARD_RED")
            if not args.runcard:
                raise SystemExit(
                    "--runcard is required or RUNCARD_BLUE/RUNCARD_RED must be set "
                    "for the qililab executor"
                )
        if not args.edges:
            raise SystemExit("--edges is required for the qililab executor; generic topology is not allowed")
        edges = parse_edges(args.edges)
        basis_gates = parse_basis_gates(args.basis_gates)
    else:
        edges = parse_edges(args.edges) if args.edges else topology_edges(args.topology, args.device_qubits)
        basis_gates = parse_basis_gates(args.basis_gates)
    runs = [load_run(path) for path in args.runs]
    seeds = np.random.SeedSequence(args.seed).spawn(2 + len(runs))
    started = time.time()

    # Layout is fixed once from the circuit structure (identical for every sample).
    compact_circuits = args.executor == "mock"
    probe = build_qcnn_circuit(np.full(4, 0.5), np.zeros((2, 4)), QCNNConfig(4, encoding=runs[0]["encoding"]))
    layout = choose_layout(probe, edges)
    probe_t = transpile(probe, edges, layout=layout, compact=compact_circuits)

    if args.executor == "mock":
        noise = MockNoise(args.noise_1q, args.noise_2q, args.readout_p01, args.readout_p10)
        executor = HardwareMockExecutor(probe_t.coupling_edges, probe_t.circuit.nqubits, noise)
        measured_bit = probe_t.physical_qubits.index(probe_t.measured_physical[0])
    else:
        executor = QililabExecutor(
            args.runcard,
            probe_t.physical_qubits,
            adapter_module=args.qililab_adapter_module,
            chip_id=args.chip_id,
            max_qubits=args.max_qubits,
            max_shots=args.max_shots,
            basis_gates=basis_gates,
            connectivity=edges,
        )
        measured_bit = probe_t.measured_physical[0]

    def calibrate(seed_seq):
        return calibrate_readout(executor, probe_t.circuit.nqubits, [measured_bit], shots=calibration_shots,
                                 seed=int(seed_seq.generate_state(1)[0]))

    calibration_start = calibrate(seeds[0])
    p01, p10 = calibration_start.p01[measured_bit], calibration_start.p10[measured_bit]

    results, total_circuits, max_native_depth = [], 2 * calibration_start.circuits, 0
    for run, seed_seq in zip(runs, seeds[2:]):
        data, partition_summary = load_partitions(run["dataset"], run["dataset_seed"], 4)
        fast = FastQCNN(FastQCNNConfig(4, encoding=run["encoding"]))
        config = QCNNConfig(4, encoding=run["encoding"])
        run_result = {k: v for k, v in run.items() if k != "weights"}
        run_result["weights"] = run["weights"].tolist()
        run_result["splits"] = {}
        for split, split_seed in zip(args.splits, seed_seq.spawn(len(args.splits))):
            features, targets = data[split]
            if args.max_samples is not None:
                features, targets = features[: args.max_samples], targets[: args.max_samples]
            exact = fast.expectations(run["weights"], features)[:, 0]
            transpiled = [
                transpile(build_qcnn_circuit(x, run["weights"], config), edges, layout=layout, compact=compact_circuits)
                for x in features
            ]
            if any(t.measured_physical != probe_t.measured_physical or t.physical_qubits != probe_t.physical_qubits
                   for t in transpiled):
                raise RuntimeError("transpiled circuits do not share the probe layout")
            max_native_depth = max(max_native_depth, max(t.stats["native_depth"] for t in transpiled))
            counts = execute_batch(executor, [t.circuit for t in transpiled], shots=args.shots,
                                   seed=int(split_seed.generate_state(1)[0]))
            total_circuits += len(counts)
            raw = np.array([z_expectation(c, 0) for c in counts])
            mitigated = np.array([mitigate_z_expectation(z, p01, p10) for z in raw])
            run_result["splits"][split] = {
                "samples": int(len(targets)),
                "partition_fingerprint": partition_summary[split],
                "exact": summarize(exact, targets),
                "raw": summarize(raw, targets),
                "mitigated": summarize(mitigated, targets),
                "mean_abs_error_raw_vs_exact": float(np.mean(np.abs(raw - exact))),
                "mean_abs_error_mitigated_vs_exact": float(np.mean(np.abs(mitigated - exact))),
                "per_sample": {"target": targets.tolist(), "exact": exact.tolist(), "raw": raw.tolist(),
                               "mitigated": mitigated.tolist()},
            }
        stored = run["stored_test_accuracy"]
        if "test" in run_result["splits"] and stored is not None and args.max_samples is None:
            run_result["exact_matches_stored_test_accuracy"] = bool(
                abs(run_result["splits"]["test"]["exact"]["metrics"]["accuracy"] - stored) < 1e-12)
        results.append(run_result)

    calibration_end = calibrate(seeds[1])
    elapsed = time.time() - started
    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": "hardware-inference",
        "status": ("dry-run (simulator mock)" if args.executor == "mock"
                   else "adapter-configured (real hardware compatibility not validated)"),
        "created_unix": started,
        "elapsed_seconds": elapsed,
        "provenance": collect_provenance(sys.argv if argv is None else ["run_hardware_inference", *argv]),
        "arguments": {k: (str(v) if isinstance(v, Path) else ([str(p) for p in v] if k == "runs" else v))
                      for k, v in vars(args).items()},
        "selection_note": "checkpoints were selected on validation during training; this script selects nothing.",
        "executor": executor.capabilities.to_dict(),
        "coupling_edges": [list(e) for e in edges],
        "layout": probe_t.to_dict(),
        "max_native_depth": max_native_depth,
        "readout_calibration": {"start": calibration_start.to_dict(), "end": calibration_end.to_dict(),
                                "mitigation_uses": "start"},
        "cost": {
            "circuits": total_circuits,
            "shots_per_circuit": args.shots,
            "estimated_hours": {name: model.hours(total_circuits, args.shots) for name, model in TIME_MODELS.items()},
        },
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(results)} checkpoint(s), {total_circuits} circuits, {elapsed:.1f} s)")
    return output


if __name__ == "__main__":
    main()
