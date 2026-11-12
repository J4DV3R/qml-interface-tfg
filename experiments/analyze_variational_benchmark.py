"""Statistical summary of ``run_variational_benchmark`` outputs.

Reads the per-run JSON files of one or more benchmark directories (never
modifies them) and writes a Markdown report plus optional CSV files:

* per condition (dataset, encoding, optimizer, shots, noise, readout error):
  number of seeds, mean, standard deviation, median, range and a percentile
  bootstrap 95% CI over initialization seeds for test accuracy, balanced
  accuracy, F1 and MSE (or validation metrics for screening runs), plus the
  exact noise-free diagnostic, best step and circuit cost to the checkpoint;
* paired differences against a reference optimizer on matching seeds, with a
  bootstrap CI and, when n >= 6, a two-sided Wilcoxon signed-rank p-value;
* majority-class and logistic-regression baselines from the manifests;
* learning curves (mean/sd validation loss per step) as CSV.

With few seeds the intervals are wide; the report flags conditions with
n < 5 as not conclusive. No numbers are written by hand: everything is
derived from the JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 12345
CONDITION_KEYS = ("dataset", "encoding", "optimizer", "shots", "noise", "readout_error", "backend")


def bootstrap_ci(values, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def describe(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "median": float(np.median(values)) if values.size else None,
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "ci95": bootstrap_ci(values),
    }


def load_runs(directories):
    runs, manifests = [], []
    for directory in directories:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifests.append((directory, json.loads(manifest_path.read_text(encoding="utf-8"))))
        for path in sorted(directory.glob("run-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_path"] = str(path)
            runs.append(payload)
    return runs, manifests


def run_values(run) -> dict[str, float]:
    training = run["training"]
    best = training["history"][training["best_step"]]
    values = {
        "best_step": training["best_step"],
        "steps_completed": training["steps_completed"],
        "best_validation_loss": training["best_validation_loss"],
        "validation_accuracy_at_best": best["validation_metrics"]["accuracy"],
        "optimization_circuits_to_best": best["optimization_circuits"],
        "optimization_circuits_total": training["optimization_circuits"],
        "elapsed_seconds": training["elapsed_seconds"],
    }
    test = run.get("test", {})
    if test.get("evaluated_once_after_checkpoint_selection"):
        values.update(
            test_accuracy=test["metrics"]["accuracy"],
            test_balanced_accuracy=test["metrics"]["balanced_accuracy"],
            test_f1=test["metrics"]["f1"],
            test_mse=test["mse_loss"],
            test_accuracy_exact_diagnostic=test["exact_noiseless_diagnostic"]["metrics"]["accuracy"],
        )
    return values


def hyperparameter_label(run) -> str:
    items = run.get("hyperparameters", {})
    return ", ".join(f"{k}={v}" for k, v in sorted(items.items()) if v is not None) or "—"


def condition_of(run) -> tuple:
    return tuple(run["condition"].get(key) for key in CONDITION_KEYS) + (hyperparameter_label(run),)


def _base_key(key: tuple) -> tuple:
    """Condition without optimizer and hyperparameters (for pairing)."""
    optimizer_index = CONDITION_KEYS.index("optimizer")
    return tuple(v for i, v in enumerate(key[: len(CONDITION_KEYS)]) if i != optimizer_index)


def fmt(stat, digits=4):
    if stat is None or stat.get("mean") is None:
        return "—"
    text = f"{stat['mean']:.{digits}f}"
    if stat.get("sd") is not None:
        text += f" ± {stat['sd']:.{digits}f}"
    if stat.get("ci95"):
        text += f" [{stat['ci95'][0]:.{digits}f}, {stat['ci95'][1]:.{digits}f}]"
    return text


def analyze(runs, reference: str) -> dict[str, Any]:
    grouped: dict[tuple, list] = {}
    for run in runs:
        grouped.setdefault(condition_of(run), []).append(run)
    conditions = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        per_run = [run_values(run) for run in members]
        metrics = sorted({name for values in per_run for name in values})
        conditions.append(
            {
                "condition": {**dict(zip(CONDITION_KEYS, key)), "hyperparameters": key[-1]},
                "seeds": [run["seeds"]["initialization"] for run in members],
                "stats": {name: describe([v[name] for v in per_run if name in v]) for name in metrics},
            }
        )

    paired = []
    for key, members in grouped.items():
        condition = {**dict(zip(CONDITION_KEYS, key)), "hyperparameters": key[-1]}
        if condition["optimizer"] == reference:
            continue
        candidates = [
            other for other in grouped
            if other[CONDITION_KEYS.index("optimizer")] == reference and _base_key(other) == _base_key(key)
        ]
        if len(candidates) != 1:  # none, or ambiguous (several reference settings)
            continue
        reference_key = candidates[0]
        ref_by_seed = {run["seeds"]["initialization"]: run_values(run) for run in grouped[reference_key]}
        for metric in ("test_accuracy", "test_balanced_accuracy", "best_validation_loss"):
            diffs = [
                run_values(run)[metric] - ref_by_seed[run["seeds"]["initialization"]][metric]
                for run in members
                if run["seeds"]["initialization"] in ref_by_seed
                and metric in run_values(run)
                and metric in ref_by_seed[run["seeds"]["initialization"]]
            ]
            if not diffs:
                continue
            p_value = None
            if len(diffs) >= 6 and np.any(np.asarray(diffs) != 0):
                p_value = float(stats.wilcoxon(diffs, zero_method="zsplit").pvalue)
            paired.append(
                {"condition": condition, "reference": reference, "metric": metric,
                 "difference": describe(diffs), "wilcoxon_p": p_value}
            )
    return {"conditions": conditions, "paired": paired}


def curves(runs) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple, list] = {}
    for run in runs:
        grouped.setdefault(condition_of(run), []).append(run)
    for key, members in grouped.items():
        max_steps = max(len(run["training"]["history"]) for run in members)
        for step in range(max_steps):
            losses = [run["training"]["history"][step]["validation_loss"]
                      for run in members if step < len(run["training"]["history"])]
            circuits = [run["training"]["history"][step]["optimization_circuits"]
                        for run in members if step < len(run["training"]["history"])]
            rows.append({
                **dict(zip(CONDITION_KEYS, key)),
                "hyperparameters": key[-1],
                "step": step,
                "n": len(losses),
                "validation_loss_mean": float(np.mean(losses)),
                "validation_loss_sd": float(np.std(losses, ddof=1)) if len(losses) > 1 else None,
                "optimization_circuits_mean": float(np.mean(circuits)),
            })
    return rows


def render_markdown(result, manifests, inputs) -> str:
    lines = ["# Análisis del benchmark variacional QCNN", ""]
    lines.append("Generado automáticamente a partir de: " + ", ".join(f"`{p}`" for p in inputs) + ".")
    lines.append("Media ± desviación típica [IC95% bootstrap percentil sobre semillas de inicialización].")
    lines.append("")
    for directory, manifest in manifests:
        prov = manifest.get("provenance", {})
        lines.append(f"## Manifiesto `{directory}`")
        lines.append(f"- Estado: {manifest.get('status')}; screening: {manifest.get('screening')}")
        lines.append(f"- Revisión: `{prov.get('git_revision')}`; árbol sucio: {prov.get('working_tree_dirty')}; "
                     f"hash del diff: `{prov.get('uncommitted_diff_sha256')}`")
        parts = manifest.get("dataset", {}).get("partitions", {})
        for name, part in parts.items():
            lines.append(f"- {name}: {part['samples']} muestras ({part['positives']} +1 / {part['negatives']} −1)")
        for baseline, content in manifest.get("baselines", {}).items():
            for partition in ("validation", "test"):
                if partition in content:
                    m = content[partition]
                    lines.append(f"- Baseline {baseline} ({partition}): accuracy {m['accuracy']:.4f}, "
                                 f"balanced {m['balanced_accuracy']:.4f}, F1 {m['f1']:.4f}")
        lines.append("")
    lines.append("## Resultados por condición")
    lines.append("")
    has_test = any("test_accuracy" in item["stats"] for item in result["conditions"])
    columns = ["dataset", "cod.", "optimizador", "hiperparámetros", "shots", "ruido", "readout", "n",
               "mejor val. loss", "acc. val. en el óptimo", "paso óptimo", "circuitos hasta el óptimo"]
    if has_test:
        columns += ["acc. test", "acc. bal. test", "F1 test", "acc. test exacta (diag.)"]
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for item in result["conditions"]:
        c, s = item["condition"], item["stats"]
        n = s["best_validation_loss"]["n"]
        flag = "" if n >= 5 else " ⚠"
        cells = [
            c["dataset"], c["encoding"], c["optimizer"], c["hyperparameters"], str(c["shots"] or "exacto"),
            str(c["noise"]), str(c["readout_error"]), f"{n}{flag}", fmt(s.get("best_validation_loss")),
            fmt(s.get("validation_accuracy_at_best")), fmt(s.get("best_step"), 1),
            fmt(s.get("optimization_circuits_to_best"), 0),
        ]
        if has_test:
            cells += [fmt(s.get("test_accuracy")), fmt(s.get("test_balanced_accuracy")),
                      fmt(s.get("test_f1")), fmt(s.get("test_accuracy_exact_diagnostic"))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("⚠ = menos de 5 semillas: no concluyente.")
    lines.append("")
    if result["paired"]:
        lines.append(f"## Diferencias pareadas frente a `{result['paired'][0]['reference']}` (mismas semillas)")
        lines.append("")
        lines += ["| condición | métrica | n | diferencia media ± sd [IC95%] | Wilcoxon p |", "|---|---|---|---|---|"]
        for item in result["paired"]:
            c = item["condition"]
            label = f"{c['dataset']}/{c['encoding']}/{c['optimizer']}/shots={c['shots'] or 'exacto'}/p={c['noise']}"
            p = "—" if item["wilcoxon_p"] is None else f"{item['wilcoxon_p']:.4f}"
            lines.append(f"| {label} | {item['metric']} | {item['difference']['n']} | {fmt(item['difference'])} | {p} |")
        lines.append("")
        lines.append("Wilcoxon solo se calcula con n ≥ 6. Un IC que cruza 0 no permite afirmar una diferencia.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize variational benchmark directories.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new Markdown report path")
    parser.add_argument("--json", type=Path, default=None, help="optional new JSON summary path")
    parser.add_argument("--curves-csv", type=Path, default=None, help="optional new learning-curve CSV path")
    parser.add_argument("--reference", default="adam")
    args = parser.parse_args(argv)
    for path in (args.output, args.json, args.curves_csv):
        if path is not None and path.exists():
            parser.error(f"refusing to overwrite {path}")
    runs, manifests = load_runs(args.inputs)
    if not runs:
        parser.error("no run-*.json files found")
    result = analyze(runs, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(result, manifests, args.inputs), encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.curves_csv:
        rows = curves(runs)
        with args.curves_csv.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {args.output} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
