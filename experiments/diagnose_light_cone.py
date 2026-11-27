"""Which inputs, weights and gate occurrences can influence the QCNN output?

The prediction is <Z> of qubit 0 after the last pooling. A gate can only change
it if it lies in the backward *light cone* (causal cone) of that measurement.
This diagnostic checks it numerically for the current 4-qubit circuit:

* sensitivity of the output to each input feature and to each shared weight
  (random resampling, max absolute change);
* per-occurrence parameter-shift contributions (which of the 16 occurrences of
  the 8 shared weights ever contribute to the gradient);
* the effective trigonometric degree of the output in each weight (how many
  evaluations an exact sequential reconstruction really needs);
* a cross-check of the fast simulator against the Qibo circuit builder.

It is a property of the circuit, independent of data and training. Nothing is
trained or evaluated on test data. ``--architecture v1`` (default) is the
original analysis through the Qibo engine (``results/diagnostics/light-cone/``);
``v1-param-pool`` and ``qibo`` run the same checks on the architectures of
``src/qcnn_architectures.py``, cross-checked against their Qibo circuits.

    qml_env/Scripts/python.exe -B -m experiments.diagnose_light_cone --output-dir results/diagnostics/light-cone
    qml_env/Scripts/python.exe -B -m experiments.diagnose_light_cone --architecture qibo --output-dir results/diagnostics/light-cone-qibo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from experiments.run_variational_benchmark import collect_provenance
from src.fast_qcnn import FastQCNN, FastQCNNConfig, qcnn_layout
from src.qcnn_architectures import ARCHITECTURES, AnsatzQCNN, build_ansatz, qibo_expectation
from src.qcnn_engine import QCNNConfig, build_qcnn_circuit
from src.transpilation import measured_marginal

TOLERANCE = 1e-9


def effective_degree(function, max_degree: int, grid: np.ndarray) -> int:
    """Smallest R such that 2R+1 equispaced samples reconstruct the function on ``grid``."""
    truth = np.array([function(v) for v in grid])
    if np.ptp(truth) < TOLERANCE:
        return 0
    for degree in range(1, max_degree + 1):
        points = np.linspace(-np.pi, np.pi, 2 * degree + 1, endpoint=False)
        values = np.array([function(v) for v in points])

        def basis(t):
            return np.column_stack([np.ones_like(t)] + [f(k * t) for k in range(1, degree + 1) for f in (np.cos, np.sin)])

        coefficients = np.linalg.solve(basis(points), values)
        if np.max(np.abs(basis(grid) @ coefficients - truth)) < TOLERANCE:
            return degree
    return -1


def analyse(encoding: str, samples: int, rng: np.random.Generator) -> dict:
    fast = FastQCNN(FastQCNNConfig(4, encoding=encoding))
    config = QCNNConfig(4, encoding=encoding)
    layers, _ = qcnn_layout(4, 1)
    feature_change = np.zeros(4)
    weight_change = np.zeros((2, 4))
    occurrence = {}
    degrees: dict[str, set[int]] = {}
    max_cross_check = 0.0
    grid = np.linspace(-np.pi, np.pi, 120)

    def z_qibo(x, w, shifted=None):
        p = measured_marginal(build_qcnn_circuit(x, w, config, shifted_weight=shifted), [0])
        return float(p[0] - p[1])

    for _ in range(samples):
        w = rng.uniform(-np.pi, np.pi, (2, 4))
        x = rng.uniform(0.0, 1.0, (1, 4))
        base = float(fast.expectations(w, x)[0, 0])
        max_cross_check = max(max_cross_check, abs(base - z_qibo(x[0], w)))
        for i in range(4):
            x2 = x.copy()
            x2[0, i] = rng.uniform(0.0, 1.0)
            feature_change[i] = max(feature_change[i], abs(float(fast.expectations(w, x2)[0, 0]) - base))
        for layer in range(2):
            for k in range(4):
                w2 = w.copy()
                w2[layer, k] = rng.uniform(-np.pi, np.pi)
                weight_change[layer, k] = max(weight_change[layer, k], abs(float(fast.expectations(w2, x)[0, 0]) - base))

                def output(value, layer=layer, k=k):
                    w3 = w.copy()
                    w3[layer, k] = value
                    return float(fast.expectations(w3, x)[0, 0])

                degrees.setdefault(f"{layer},{k}", set()).add(effective_degree(output, len(layers[layer][0]), grid))
        for layer, (pairs, _) in enumerate(layers):
            for pair_index, pair in enumerate(pairs):
                for k in range(4):
                    contribution = (z_qibo(x[0], w, (layer, pair_index, k, np.pi / 2))
                                    - z_qibo(x[0], w, (layer, pair_index, k, -np.pi / 2))) / 2
                    key = f"layer{layer}-pair{pair[0]}{pair[1]}-w{k}"
                    occurrence[key] = max(occurrence.get(key, 0.0), abs(contribution))

    live_occurrences = sum(v > TOLERANCE for v in occurrence.values())
    degree_summary = {key: sorted(values) for key, values in degrees.items()}
    max_degree = {key: max(values) for key, values in degrees.items()}
    return {
        "encoding": encoding,
        "samples": samples,
        "max_abs_output_change_per_feature": feature_change.tolist(),
        "features_without_influence": [i for i in range(4) if feature_change[i] < TOLERANCE],
        "max_abs_output_change_per_weight": weight_change.tolist(),
        "weights_without_influence": [[l, k] for l in range(2) for k in range(4) if weight_change[l, k] < TOLERANCE],
        "max_abs_parameter_shift_contribution_per_occurrence": occurrence,
        "live_occurrences": live_occurrences,
        "total_occurrences": len(occurrence),
        "effective_trigonometric_degree_per_weight": degree_summary,
        "circuits_per_sample": {
            "parameter_shift_current": 1 + 2 * len(occurrence),
            "parameter_shift_live_occurrences_only": 1 + 2 * live_occurrences,
            "sequential_current": sum(4 * (2 * len(pairs) + 1) for pairs, _ in layers),
            "sequential_effective_degree": sum(2 * d + 1 for d in max_degree.values() if d > 0),
        },
        "fast_vs_qibo_max_abs_difference": max_cross_check,
    }


def analyse_architecture(architecture: str, encoding: str, samples: int, rng: np.random.Generator) -> dict:
    """Same checks as :func:`analyse` for any architecture of ``qcnn_architectures``."""
    ansatz = build_ansatz(architecture, 4)
    sim = AnsatzQCNN(ansatz, encoding)
    layers, params = ansatz.weight_shape
    structural = ansatz.weight_degrees
    feature_change = np.zeros(4)
    weight_change = np.zeros(ansatz.weight_shape)
    occurrence: dict[str, float] = {}
    degrees: dict[str, set[int]] = {}
    max_cross_check = 0.0
    grid = np.linspace(-np.pi, np.pi, 120)
    for _ in range(samples):
        w = rng.uniform(-np.pi, np.pi, ansatz.weight_shape)
        x = rng.uniform(0.0, 1.0, (1, 4))
        base = float(sim.expectations(w, x)[0, 0])
        max_cross_check = max(max_cross_check, abs(base - qibo_expectation(ansatz, w, x[0], encoding)))
        for i in range(4):
            x2 = x.copy()
            x2[0, i] = rng.uniform(0.0, 1.0)
            feature_change[i] = max(feature_change[i], abs(float(sim.expectations(w, x2)[0, 0]) - base))
        for layer in range(layers):
            for k in range(params):
                w2 = w.copy()
                w2[layer, k] = rng.uniform(-np.pi, np.pi)
                weight_change[layer, k] = max(weight_change[layer, k],
                                              abs(float(sim.expectations(w2, x)[0, 0]) - base))

                def output(value, layer=layer, k=k):
                    w3 = w.copy()
                    w3[layer, k] = value
                    return float(sim.expectations(w3, x)[0, 0])

                degrees.setdefault(f"{layer},{ansatz.param_labels[k]}", set()).add(
                    effective_degree(output, int(structural[layer, k]), grid))
        angles, feats, _index, _n_shift = sim.shift_rows(w, x)
        shifted = sim.expectations_from_angles(angles, feats)[:, 0]
        for column, gate_index in enumerate(ansatz.occurrences):
            key = f"{column:03d}-{ansatz.occurrence_label(gate_index)}"
            contribution = abs(shifted[2 * column] - shifted[2 * column + 1]) / 2
            occurrence[key] = max(occurrence.get(key, 0.0), float(contribution))

    live = sum(v > TOLERANCE for v in occurrence.values())
    max_degree = {key: max(values) for key, values in degrees.items()}
    return {
        "architecture": ansatz.summary(),
        "encoding": encoding,
        "samples": samples,
        "max_abs_output_change_per_feature": feature_change.tolist(),
        "features_without_influence": [i for i in range(4) if feature_change[i] < TOLERANCE],
        "max_abs_output_change_per_weight": weight_change.tolist(),
        "weights_without_influence": [[l, ansatz.param_labels[k]] for l in range(layers) for k in range(params)
                                      if weight_change[l, k] < TOLERANCE],
        "max_abs_parameter_shift_contribution_per_occurrence": occurrence,
        "live_occurrences": live,
        "total_occurrences": len(occurrence),
        "structural_degree_per_weight": {f"{l},{ansatz.param_labels[k]}": int(structural[l, k])
                                         for l in range(layers) for k in range(params)},
        "effective_trigonometric_degree_per_weight": {key: sorted(v) for key, v in degrees.items()},
        "circuits_per_sample": {
            "parameter_shift_current": 1 + 2 * len(occurrence),
            "parameter_shift_live_occurrences_only": 1 + 2 * live,
            "sequential_current": int(sum(2 * d + 1 for d in structural.reshape(-1) if d > 0)),
            "sequential_effective_degree": sum(2 * d + 1 for d in max_degree.values() if d > 0),
        },
        "fast_vs_qibo_max_abs_difference": max_cross_check,
    }


def render(result: dict) -> str:
    architecture = result.get("arguments", {}).get("architecture", "v1")
    lines = [f"# Diagnóstico del cono de luz de la QCNN, arquitectura `{architecture}` (generado)", "",
             "Generado por `experiments/diagnose_light_cone.py`. Propiedad del circuito, independiente de los datos.", ""]
    if "architecture_summary" in result:
        summary = result["architecture_summary"]
        lines += [f"Pesos entrenables: {summary['trainable_weights']} (forma {summary['weight_shape']}); "
                  f"apariciones parametrizadas: {summary['parametrized_occurrences']}; "
                  f"CNOT: {summary['cnot_count']}; qubit de lectura: {summary['readout_qubit']}.", ""]
    for r in result["encodings"]:
        c = r["circuits_per_sample"]
        lines += [
            f"## Codificación {r['encoding']} ({r['samples']} entradas y pesos aleatorios)", "",
            f"- Características sin influencia en la salida: {r['features_without_influence']} "
            f"(cambio máximo por característica: {[round(v, 4) for v in r['max_abs_output_change_per_feature']]}).",
            f"- Pesos `[capa, k]` sin influencia: {r['weights_without_influence']}.",
            f"- Apariciones con contribución al gradiente: {r['live_occurrences']} de {r['total_occurrences']}.",
            f"- Grado trigonométrico efectivo por peso `capa,k`: {r['effective_trigonometric_degree_per_weight']}.",
            f"- Circuitos por muestra: parameter-shift {c['parameter_shift_current']} (actual) frente a "
            f"{c['parameter_shift_live_occurrences_only']} (solo apariciones vivas); secuencial "
            f"{c['sequential_current']} (actual) frente a {c['sequential_effective_degree']} (grado efectivo).",
            f"- Diferencia máxima simulador rápido frente a Qibo: {r['fast_vs_qibo_max_abs_difference']:.1e}.", "",
            "| Aparición | Contribución máxima |", "|---|---|",
        ]
        lines += [f"| {k} | {v:.4f} |" for k, v in r["max_abs_parameter_shift_contribution_per_occurrence"].items()]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.output_dir / "analysis.json").exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir / 'analysis.json'}")
    rng = np.random.default_rng(args.seed)
    result = {
        "kind": "light-cone-diagnostic",
        "provenance": collect_provenance(sys.argv if argv is None else ["diagnose_light_cone", *argv]),
        "arguments": {"samples": args.samples, "seed": args.seed},
        "tolerance": TOLERANCE,
    }
    if args.architecture == "v1":
        result["encodings"] = [analyse(encoding, args.samples, rng) for encoding in ("angle", "phase")]
    else:
        result["arguments"]["architecture"] = args.architecture
        result["architecture_summary"] = build_ansatz(args.architecture, 4).summary()
        result["encodings"] = [analyse_architecture(args.architecture, encoding, args.samples, rng)
                               for encoding in ("angle", "phase")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "analysis.md").write_text(render(result), encoding="utf-8")
    print(f"Wrote {args.output_dir}/analysis.json and analysis.md")
    return result


if __name__ == "__main__":
    main()
