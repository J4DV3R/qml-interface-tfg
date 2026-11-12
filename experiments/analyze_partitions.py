"""Partition-level analysis of the variational benchmark (phase C).

Pre-registered in ``results/variational/C-partitions/README.md``. The unit of
analysis is the data partition (dataset seed): metrics are first averaged over
initialization seeds inside each partition, then summarized across partitions.

* per-condition summaries over partitions (mean, sd, bootstrap CI);
* paired differences against Adam per partition, Wilcoxon over partitions and
  Holm correction over the comparisons of each dataset (primary: test accuracy);
* baselines per partition and the QCNN - logistic-regression gap;
* the pre-registered checks O1-O4 with Holm correction over the block;
* early stopping (patience) reconstructed exactly from the stored histories.

Patience reconstruction: in exact simulation the optimizer trajectory does not
depend on the stopping rule (stopping only truncates the loop and SPSA draws the
same random numbers up to that point), so training with ``patience=P`` equals
truncating the full history at the first step with ``P`` consecutive
non-improving validation losses. ``tests/test_variational_training.py`` checks
this against ``train_variational_qcnn(patience=P)``. Test metrics of a patience
checkpoint are computed once per rule from the stored weights with the exact
simulator; they are never used to choose ``P``.

Inputs are never modified; outputs must be new files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from experiments.analyze_variational_benchmark import describe, hyperparameter_label, run_values
from experiments.run_variational_benchmark import load_partitions
from src.fast_qcnn import FastQCNNConfig
from src.variational_training import FastEvaluator, binary_metrics, mse

REFERENCE = "adam"
PATIENCES = (5, 10, 20)
PRIMARY = "test_accuracy"
SUMMARY_METRICS = (
    "test_accuracy",
    "test_balanced_accuracy",
    "test_f1",
    "test_mse",
    "best_validation_loss",
    "optimization_circuits_to_best",
)
PAIRED_METRICS = ("test_accuracy", "test_balanced_accuracy", "test_mse", "optimization_circuits_to_best")


def patience_truncation(history: list[dict[str, Any]], patience: int) -> dict[str, Any]:
    """Reproduce ``train_variational_qcnn``'s patience rule on a full history."""
    if patience < 1:
        raise ValueError("patience must be positive.")
    best_index, best_loss, stale = 0, history[0]["validation_loss"], 0
    stop_index = len(history) - 1
    stopped = False
    for index in range(1, len(history)):
        loss = history[index]["validation_loss"]
        if loss < best_loss:
            best_index, best_loss, stale = index, loss, 0
        else:
            stale += 1
        if stale >= patience:
            stop_index, stopped = index, True
            break
    return {
        "patience": patience,
        "stopped_early": stopped,
        "stop_step": history[stop_index]["step"],
        "best_step": history[best_index]["step"],
        "best_index": best_index,
        "best_validation_loss": float(best_loss),
        "optimization_circuits_at_stop": history[stop_index]["optimization_circuits"],
    }


def holm(p_values: list[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values (``None`` entries are ignored)."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    m = len(indexed)
    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def wilcoxon(values) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size < 6 or np.allclose(values, 0.0):
        return None if values.size < 6 else 1.0
    return float(stats.wilcoxon(values, zero_method="zsplit").pvalue)


def find_run_directories(roots) -> list[Path]:
    directories = set()
    for root in roots:
        root = Path(root)
        for path in root.rglob("run-*.json"):
            directories.add(path.parent)
    return sorted(directories)


class Collection:
    """Runs keyed by (dataset, dataset seed, encoding, optimizer, init seed)."""

    def __init__(self, directories):
        self.runs: dict[tuple, dict[str, Any]] = {}
        self.baselines: dict[tuple, dict[str, Any]] = {}
        self.hyperparameters: dict[str, set[str]] = defaultdict(set)
        self.directories = [str(d) for d in directories]
        for directory in directories:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            for path in sorted(directory.glob("run-*.json")):
                run = json.loads(path.read_text(encoding="utf-8"))
                condition, seeds = run["condition"], run["seeds"]
                if condition.get("shots") is not None or condition.get("noise") or condition.get("readout_error"):
                    raise ValueError(f"{path}: phase C analysis expects exact simulation only.")
                if not run.get("test", {}).get("evaluated_once_after_checkpoint_selection"):
                    raise ValueError(f"{path}: test was not evaluated.")
                key = (condition["dataset"], seeds["dataset"], condition["encoding"],
                       condition["optimizer"], seeds["initialization"])
                if key in self.runs:
                    raise ValueError(f"duplicate run {key}: {path}")
                run["_path"] = str(path)
                self.runs[key] = run
                self.hyperparameters[condition["optimizer"]].add(hyperparameter_label(run))
                base_key = (condition["dataset"], seeds["dataset"])
                baseline = manifest["baselines"]
                previous = self.baselines.setdefault(base_key, baseline)
                if previous["logistic_regression"]["test"] != baseline["logistic_regression"]["test"]:
                    raise ValueError(f"inconsistent baselines for partition {base_key}.")
        for optimizer, labels in self.hyperparameters.items():
            if len(labels) != 1:
                raise ValueError(f"mixed hyperparameters for {optimizer}: {sorted(labels)}")

    def datasets(self):
        return sorted({k[0] for k in self.runs})

    def partitions(self, dataset):
        return sorted({k[1] for k in self.runs if k[0] == dataset})

    def encodings(self, dataset):
        return sorted({k[2] for k in self.runs if k[0] == dataset})

    def optimizers(self, dataset):
        return sorted({k[3] for k in self.runs if k[0] == dataset})

    def seeds(self, dataset, partition, encoding, optimizer):
        return sorted(k[4] for k in self.runs if k[:4] == (dataset, partition, encoding, optimizer))


def partition_means(collection: Collection, values: dict[tuple, dict[str, float]], dataset, encoding, optimizer, metric):
    out = {}
    for partition in collection.partitions(dataset):
        seeds = collection.seeds(dataset, partition, encoding, optimizer)
        if seeds:
            out[partition] = float(np.mean([values[(dataset, partition, encoding, optimizer, s)][metric] for s in seeds]))
    return out


def paired_partition_differences(collection, values, dataset, left, right, metric):
    """Per-partition mean of seed-matched differences ``left - right``.

    ``left``/``right`` are (encoding, optimizer) pairs.
    """
    out = {}
    for partition in collection.partitions(dataset):
        seeds = sorted(set(collection.seeds(dataset, partition, *left)) & set(collection.seeds(dataset, partition, *right)))
        if seeds:
            out[partition] = float(np.mean([
                values[(dataset, partition, *left, s)][metric] - values[(dataset, partition, *right, s)][metric]
                for s in seeds
            ]))
    return out


def evaluate_test(data_cache, dataset, partition, encoding, weights):
    key = (dataset, partition)
    if key not in data_cache:
        data, _ = load_partitions(dataset, partition, 4)
        data_cache[key] = data["test"]
    test_x, test_y = data_cache[key]
    evaluator = FastEvaluator(FastQCNNConfig(4, 1, encoding))
    predictions = evaluator.predict(weights, test_x, shots=None, rng=None)
    return {"accuracy": binary_metrics(predictions, test_y)["accuracy"], "mse": mse(predictions, test_y)}


def patience_analysis(collection: Collection) -> list[dict[str, Any]]:
    data_cache: dict[tuple, Any] = {}
    per_run: dict[tuple, dict[int, dict[str, Any]]] = {}
    max_reproduction_error = 0.0
    for key, run in collection.runs.items():
        dataset, partition, encoding, _optimizer, _seed = key
        training = run["training"]
        history = training["history"]
        shape = np.asarray(training["best_weights"], dtype=float).shape
        # Reproduce the stored test metrics first: validates the re-evaluation path.
        stored = evaluate_test(data_cache, dataset, partition, encoding, np.asarray(training["best_weights"], dtype=float))
        error = max(abs(stored["mse"] - run["test"]["mse_loss"]),
                    abs(stored["accuracy"] - run["test"]["metrics"]["accuracy"]))
        max_reproduction_error = max(max_reproduction_error, error)
        if error > 1e-9:
            raise RuntimeError(f"cannot reproduce stored test metrics for {run['_path']} (error {error}).")
        per_run[key] = {}
        for patience in PATIENCES:
            result = patience_truncation(history, patience)
            if not result["stopped_early"] and result["best_step"] != training["best_step"]:
                raise RuntimeError(f"patience reconstruction disagrees with {run['_path']}.")
            same = result["best_step"] == training["best_step"]
            if same:
                test = {"accuracy": run["test"]["metrics"]["accuracy"], "mse": run["test"]["mse_loss"]}
            else:
                weights = np.asarray(history[result["best_index"]]["weights"], dtype=float).reshape(shape)
                test = evaluate_test(data_cache, dataset, partition, encoding, weights)
            per_run[key][patience] = {
                **result,
                "same_checkpoint": same,
                "validation_loss_increase": result["best_validation_loss"] - training["best_validation_loss"],
                "circuit_fraction": result["optimization_circuits_at_stop"] / training["optimization_circuits"],
                "test_accuracy": test["accuracy"],
                "test_mse": test["mse"],
                "test_accuracy_change": test["accuracy"] - run["test"]["metrics"]["accuracy"],
            }
    rows = []
    for dataset in collection.datasets():
        for encoding in collection.encodings(dataset):
            for optimizer in collection.optimizers(dataset):
                for patience in PATIENCES:
                    keys = [k for k in per_run if k[0] == dataset and k[2] == encoding and k[3] == optimizer]
                    entries = [per_run[k][patience] for k in keys]

                    def by_partition(field):
                        grouped = defaultdict(list)
                        for k in keys:
                            grouped[k[1]].append(per_run[k][patience][field])
                        return [float(np.mean(v)) for _, v in sorted(grouped.items())]

                    rows.append({
                        "dataset": dataset, "encoding": encoding, "optimizer": optimizer, "patience": patience,
                        "runs": len(entries),
                        "stopped_early_fraction": float(np.mean([e["stopped_early"] for e in entries])),
                        "same_checkpoint_fraction": float(np.mean([e["same_checkpoint"] for e in entries])),
                        "stop_step": describe(by_partition("stop_step")),
                        "optimization_circuits_at_stop": describe(by_partition("optimization_circuits_at_stop")),
                        "circuit_fraction": describe(by_partition("circuit_fraction")),
                        "validation_loss_increase": describe(by_partition("validation_loss_increase")),
                        "test_accuracy": describe(by_partition("test_accuracy")),
                        "test_accuracy_change": describe(by_partition("test_accuracy_change")),
                    })
    return rows, max_reproduction_error


def analyze(collection: Collection, with_patience: bool = True) -> dict[str, Any]:
    values = {key: run_values(run) for key, run in collection.runs.items()}
    result: dict[str, Any] = {"datasets": {}, "observations": []}
    for dataset in collection.datasets():
        partitions = collection.partitions(dataset)
        block: dict[str, Any] = {"partitions": partitions, "conditions": [], "paired": [], "baselines": {}, "gap_to_logistic": []}
        majority = [collection.baselines[(dataset, p)]["majority_class"]["test"]["accuracy"] for p in partitions]
        logistic = [collection.baselines[(dataset, p)]["logistic_regression"]["test"]["accuracy"] for p in partitions]
        block["baselines"] = {
            "majority_class_test_accuracy": describe(majority),
            "logistic_regression_test_accuracy": describe(logistic),
            "per_partition": {str(p): {"majority": m, "logistic": l} for p, m, l in zip(partitions, majority, logistic)},
        }
        for encoding in collection.encodings(dataset):
            for optimizer in collection.optimizers(dataset):
                seeds_per_partition = {p: len(collection.seeds(dataset, p, encoding, optimizer)) for p in partitions}
                stats_by_metric, per_partition = {}, {}
                for metric in SUMMARY_METRICS:
                    means = partition_means(collection, values, dataset, encoding, optimizer, metric)
                    stats_by_metric[metric] = describe(list(means.values()))
                    per_partition[metric] = {str(p): v for p, v in means.items()}
                block["conditions"].append({
                    "encoding": encoding, "optimizer": optimizer,
                    "hyperparameters": sorted(collection.hyperparameters[optimizer])[0],
                    "seeds_per_partition": {str(p): n for p, n in seeds_per_partition.items()},
                    "stats": stats_by_metric, "per_partition": per_partition,
                })
                accuracy = partition_means(collection, values, dataset, encoding, optimizer, PRIMARY)
                gap = [accuracy[p] - collection.baselines[(dataset, p)]["logistic_regression"]["test"]["accuracy"]
                       for p in partitions if p in accuracy]
                block["gap_to_logistic"].append({
                    "encoding": encoding, "optimizer": optimizer, "difference": describe(gap),
                    "partitions_at_or_above_logistic": int(np.sum(np.asarray(gap) >= -1e-12)),
                    "wilcoxon_p": wilcoxon(gap),
                })
                if optimizer == REFERENCE:
                    continue
                for metric in PAIRED_METRICS:
                    diffs = paired_partition_differences(
                        collection, values, dataset, (encoding, optimizer), (encoding, REFERENCE), metric)
                    block["paired"].append({
                        "encoding": encoding, "optimizer": optimizer, "metric": metric, "reference": REFERENCE,
                        "difference": describe(list(diffs.values())),
                        "per_partition": {str(p): v for p, v in diffs.items()},
                        "positive_partitions": int(sum(v > 0 for v in diffs.values())),
                        "negative_partitions": int(sum(v < 0 for v in diffs.values())),
                        "wilcoxon_p": wilcoxon(list(diffs.values())),
                    })
        primary = [item for item in block["paired"] if item["metric"] == PRIMARY]
        for item, adjusted in zip(primary, holm([item["wilcoxon_p"] for item in primary])):
            item["holm_p"] = adjusted
        result["datasets"][dataset] = block

    observations = []

    def observe(label, description, dataset, left, right, metric):
        if dataset not in result["datasets"]:
            return
        diffs = paired_partition_differences(collection, values, dataset, left, right, metric)
        if not diffs:
            return
        observations.append({
            "id": label, "description": description, "dataset": dataset, "metric": metric,
            "left": list(left), "right": list(right),
            "difference": describe(list(diffs.values())),
            "per_partition": {str(p): v for p, v in diffs.items()},
            "positive_partitions": int(sum(v > 0 for v in diffs.values())),
            "negative_partitions": int(sum(v < 0 for v in diffs.values())),
            "wilcoxon_p": wilcoxon(list(diffs.values())),
        })

    observe("O1", "Iris: GD angular - Adam angular", "iris", ("angle", "gd"), ("angle", "adam"), PRIMARY)
    observe("O2", "Breast Cancer: SPSA fase - Adam fase (100 pasos)", "breast-cancer",
            ("phase", "spsa"), ("phase", "adam"), PRIMARY)
    observe("O3", "Breast Cancer: Adam angular - Adam fase", "breast-cancer",
            ("angle", "adam"), ("phase", "adam"), PRIMARY)
    for dataset in ("breast-cancer", "iris"):
        for encoding in ("angle", "phase"):
            observe(f"O4-{dataset}-{encoding}", f"{dataset}: secuencial - Adam, circuitos hasta el checkpoint ({encoding})",
                    dataset, (encoding, "sequential"), (encoding, "adam"), "optimization_circuits_to_best")
    for item, adjusted in zip(observations, holm([o["wilcoxon_p"] for o in observations])):
        item["holm_p"] = adjusted
    result["observations"] = observations
    if with_patience:
        result["patience"], result["patience_test_reproduction_max_error"] = patience_analysis(collection)
    return result


def _fmt(stat, digits=4, scale=1.0):
    if not stat or stat.get("mean") is None:
        return "—"
    text = f"{stat['mean'] * scale:.{digits}f}"
    if stat.get("sd") is not None:
        text += f" ± {stat['sd'] * scale:.{digits}f}"
    if stat.get("ci95"):
        low, high = stat["ci95"]
        text += f" [{low * scale:.{digits}f}, {high * scale:.{digits}f}]"
    return text


def _p(value):
    return "—" if value is None else f"{value:.4f}"


def render_markdown(result, collection: Collection) -> str:
    lines = [
        "# Análisis por partición (fase C)",
        "",
        "Generado automáticamente por `experiments/analyze_partitions.py`. Unidad de análisis: la partición "
        "(semilla de dataset); dentro de cada partición se promedian primero las semillas de inicialización. "
        "Formato: media ± sd [IC95 % bootstrap sobre particiones].",
        "",
        f"Directorios de entrada: {len(collection.directories)}; entrenamientos: {len(collection.runs)}.",
        "",
    ]
    for dataset, block in result["datasets"].items():
        lines += [f"## {dataset}", "", f"Particiones: {', '.join(map(str, block['partitions']))} (n = {len(block['partitions'])}).", ""]
        b = block["baselines"]
        lines += [
            "| Baseline (test) | Accuracy |",
            "|---|---|",
            f"| Clase mayoritaria | {_fmt(b['majority_class_test_accuracy'])} |",
            f"| Regresión logística | {_fmt(b['logistic_regression_test_accuracy'])} |",
            "",
            "| cod. | optimizador | acc. test | acc. bal. test | MSE test | mejor val. loss | circuitos hasta el checkpoint |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in block["conditions"]:
            s = c["stats"]
            lines.append(
                f"| {c['encoding']} | {c['optimizer']} | {_fmt(s['test_accuracy'])} | {_fmt(s['test_balanced_accuracy'])} | "
                f"{_fmt(s['test_mse'])} | {_fmt(s['best_validation_loss'])} | {_fmt(s['optimization_circuits_to_best'], 0)} |"
            )
        lines += ["", f"### Diferencias frente a `{REFERENCE}` por partición", "",
                  "| cod. | optimizador | métrica | diferencia | particiones +/− | Wilcoxon p | Holm p |", "|---|---|---|---|---|---|---|"]
        for item in block["paired"]:
            digits = 0 if item["metric"] == "optimization_circuits_to_best" else 4
            lines.append(
                f"| {item['encoding']} | {item['optimizer']} | {item['metric']} | {_fmt(item['difference'], digits)} | "
                f"{item['positive_partitions']}/{item['negative_partitions']} | {_p(item['wilcoxon_p'])} | {_p(item.get('holm_p'))} |"
            )
        lines += ["", "Holm se aplica solo a la métrica principal (accuracy de test), sobre las comparaciones de este dataset.", "",
                  "### QCNN − regresión logística (accuracy de test por partición)", "",
                  "| cod. | optimizador | diferencia | particiones ≥ logística | Wilcoxon p |", "|---|---|---|---|---|"]
        for item in block["gap_to_logistic"]:
            lines.append(f"| {item['encoding']} | {item['optimizer']} | {_fmt(item['difference'])} | "
                         f"{item['partitions_at_or_above_logistic']}/{item['difference']['n']} | {_p(item['wilcoxon_p'])} |")
        lines.append("")
    lines += ["## Observaciones previas (pre-registradas, Holm sobre el bloque)", "",
              "| id | contraste | métrica | diferencia | particiones +/− | Wilcoxon p | Holm p |", "|---|---|---|---|---|---|---|"]
    for o in result["observations"]:
        digits = 0 if o["metric"] == "optimization_circuits_to_best" else 4
        lines.append(f"| {o['id']} | {o['description']} | {o['metric']} | {_fmt(o['difference'], digits)} | "
                     f"{o['positive_partitions']}/{o['negative_partitions']} | {_p(o['wilcoxon_p'])} | {_p(o.get('holm_p'))} |")
    if "patience" in result:
        lines += ["", "## Patience (reconstruida del historial; secundario)", "",
                  f"Reproducción de las métricas de test almacenadas: error máximo {result['patience_test_reproduction_max_error']:.2e}.", "",
                  "| dataset | cod. | optimizador | P | parada temprana | mismo checkpoint | paso de parada | circuitos al parar | fracción de circuitos | Δ val. loss | acc. test | Δ acc. test |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in result["patience"]:
            lines.append(
                f"| {row['dataset']} | {row['encoding']} | {row['optimizer']} | {row['patience']} | "
                f"{row['stopped_early_fraction']:.2f} | {row['same_checkpoint_fraction']:.2f} | {_fmt(row['stop_step'], 1)} | "
                f"{_fmt(row['optimization_circuits_at_stop'], 0)} | {_fmt(row['circuit_fraction'], 3)} | "
                f"{_fmt(row['validation_loss_increase'])} | {_fmt(row['test_accuracy'])} | {_fmt(row['test_accuracy_change'])} |"
            )
    lines += ["", "Wilcoxon (zsplit) solo con n ≥ 6 particiones. Un IC que cruza 0 no permite afirmar una diferencia. "
              "Las particiones comparten muestras: la dispersión mide sensibilidad a la partición."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", help="directories searched recursively for run-*.json")
    parser.add_argument("--output", required=True, help="new Markdown report path")
    parser.add_argument("--json", required=True, help="new JSON summary path")
    parser.add_argument("--no-patience", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.output, args.json):
        if Path(path).exists():
            parser.error(f"refusing to overwrite {path}")
    collection = Collection(find_run_directories(args.roots))
    if not collection.runs:
        parser.error("no run-*.json files found")
    result = analyze(collection, with_patience=not args.no_patience)
    result["inputs"] = collection.directories
    Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output).write_text(render_markdown(result, collection), encoding="utf-8")
    print(f"Wrote {args.output} and {args.json} ({len(collection.runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
