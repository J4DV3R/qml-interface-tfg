"""Figures for the variational benchmark report, derived only from persisted JSON.

Reads the analysis JSON files of phases A, B and C plus the per-run JSON
histories of the exact phase-B runs (partition 42) and writes vector PDF
figures. Nothing here touches test data beyond what the analyses already
evaluated; no figure is used for any selection.

Usage:
    qml_env/Scripts/python.exe -B -m experiments.plot_variational_report --output-dir docs/reports/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "variational"

OPTIMIZERS = ("adam", "gd", "sequential", "spsa")
OPT_LABEL = {"adam": "Adam", "gd": "GD", "sequential": "Secuencial", "spsa": "SPSA"}
# Reference categorical palette, fixed order (slots 1-4); never cycled.
SHORT_OPT = {"adam": "Adam", "gd": "GD", "sequential": "Sec.", "spsa": "SPSA"}
OPT_COLOR = {"adam": "#2a78d6", "gd": "#eb6834", "sequential": "#1baf7a", "spsa": "#eda100"}
ENCODINGS = ("angle", "phase")
ENC_LABEL = {"angle": "angular", "phase": "fase"}
ENC_MARKER = {"angle": "o", "phase": "s"}
ENC_LINESTYLE = {"angle": "-", "phase": "--"}
DATASETS = ("breast-cancer", "iris")
DS_LABEL = {"breast-cancer": "Breast Cancer (PCA 4)", "iris": "Iris setosa/versicolor"}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
LOGISTIC_COLOR = "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "text.color": INK,
    "lines.linewidth": 2,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})


class _CommaFormatter(ScalarFormatter):
    """Default tick precision, Spanish decimal comma."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def comma(_decimals: int | None = None):
    formatter = _CommaFormatter(useOffset=False)
    formatter.set_scientific(False)
    return formatter


def thousands(v, _=None):
    if v >= 1e6:
        return f"{v / 1e6:g}".replace(".", ",") + " M"
    if v >= 1e3:
        return f"{v / 1e3:g}".replace(".", ",") + " k"
    return f"{v:g}"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def marker_kwargs(optimizer: str, encoding: str, size: float = 6.5):
    color = OPT_COLOR[optimizer]
    face = color if encoding == "angle" else "white"
    return dict(marker=ENC_MARKER[encoding], markersize=size, markerfacecolor=face,
                markeredgecolor=color, markeredgewidth=1.4, color=color)


def legend_handles(include_encoding: bool = True):
    handles = [plt.Line2D([], [], linestyle="none", **marker_kwargs(o, "angle")) for o in OPTIMIZERS]
    labels = [OPT_LABEL[o] for o in OPTIMIZERS]
    if include_encoding:
        for enc in ENCODINGS:
            handles.append(plt.Line2D([], [], linestyle="none", marker=ENC_MARKER[enc], markersize=6.5,
                                      markerfacecolor=INK_2 if enc == "angle" else "white",
                                      markeredgecolor=INK_2, markeredgewidth=1.4))
            labels.append(f"Cod. {ENC_LABEL[enc]}")
    return handles, labels


def save(fig, out: Path, name: str) -> Path:
    path = out / f"{name}.pdf"
    # No creation date in the metadata: regenerating from the same JSON yields identical files.
    fig.savefig(path, metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", dpi=160, metadata={"Software": None})
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- phase A
def hyper_value(text: str, key: str) -> float:
    match = re.search(rf"(?:^|, ){re.escape(key)}=([0-9.eE+-]+)", text)
    return float(match.group(1)) if match else float("nan")


def fig_phase_a(out: Path) -> Path:
    data = load(RESULTS / "A-tuning-bc-exact" / "analysis-v2.json")["conditions"]
    panels = (("gd", "learning_rate", "Tasa de aprendizaje", 0.5),
              ("adam", "learning_rate", "Tasa de aprendizaje", 0.1),
              ("spsa", "a", "Ganancia a", 1.6))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (opt, key, xlabel, chosen) in zip(axes, panels):
        ax.axvline(chosen, color=GRID, linewidth=7, zorder=0)
        for enc in ENCODINGS:
            rows = sorted(
                (hyper_value(c["condition"]["hyperparameters"], key), c["stats"]["best_validation_loss"])
                for c in data if c["condition"]["optimizer"] == opt and c["condition"]["encoding"] == enc)
            x = np.array([r[0] for r in rows])
            mean = np.array([r[1]["mean"] for r in rows])
            low = mean - np.array([r[1]["min"] for r in rows])
            high = np.array([r[1]["max"] for r in rows]) - mean
            ax.errorbar(x, mean, yerr=[low, high], linestyle=ENC_LINESTYLE[enc], capsize=0, elinewidth=1,
                        linewidth=1.6, **marker_kwargs(opt, enc, 5.5))
            seq = next(c for c in data if c["condition"]["optimizer"] == "sequential"
                       and c["condition"]["encoding"] == enc)["stats"]["best_validation_loss"]["mean"]
            ax.axhline(seq, color=OPT_COLOR["sequential"], linewidth=1, linestyle=ENC_LINESTYLE[enc])
        ax.set_xscale("log")
        ax.set_xticks(sorted({hyper_value(c["condition"]["hyperparameters"], key) for c in data
                              if c["condition"]["optimizer"] == opt}))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlabel(xlabel)
        ax.set_title(OPT_LABEL[opt], loc="left", fontweight="bold")
    axes[0].set_ylabel("Mejor pérdida de validation")
    axes[0].yaxis.set_major_formatter(comma(2))
    handles = [plt.Line2D([], [], color=INK_2, linestyle=ENC_LINESTYLE[e], marker=ENC_MARKER[e],
                          markerfacecolor=INK_2 if e == "angle" else "white", markeredgecolor=INK_2)
               for e in ENCODINGS]
    handles.append(plt.Line2D([], [], color=OPT_COLOR["sequential"], linewidth=1))
    fig.legend(handles, ["Cod. angular", "Cod. fase", "Secuencial (sin hiperparámetro)"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return save(fig, out, "fig01_phaseA_tuning")


# --------------------------------------------------------------------------- phase B
def run_histories(directory: Path):
    histories = []
    for path in sorted(directory.glob("run-*.json")):
        training = load(path)["training"]
        histories.append((training["history"], training["best_step"]))
    return histories


def fig_curves(out: Path) -> Path:
    roots = {"breast-cancer": RESULTS / "B-confirm-bc" / "B1-exact", "iris": RESULTS / "B-confirm-iris" / "I1-exact"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for row, ds in enumerate(DATASETS):
        for col, enc in enumerate(ENCODINGS):
            ax = axes[row, col]
            for opt in OPTIMIZERS:
                runs = run_histories(roots[ds] / f"{opt}-{enc}")
                steps = min(len(h) for h, _ in runs)
                loss = np.array([[e["validation_loss"] for e in h[:steps]] for h, _ in runs])
                circuits = np.median(np.array([[e["optimization_circuits"] for e in h[:steps]] for h, _ in runs]), 0)
                x = circuits[1:]
                median = np.median(loss, 0)[1:]
                q1, q3 = np.percentile(loss, [25, 75], axis=0)
                ax.fill_between(x, q1[1:], q3[1:], color=OPT_COLOR[opt], alpha=0.15, linewidth=0)
                ax.plot(x, median, color=OPT_COLOR[opt], linewidth=1.6, label=OPT_LABEL[opt])
                best = int(np.median([b for _, b in runs]))
                if best >= 1:
                    ax.plot(circuits[best], np.median(loss, 0)[best], linestyle="none",
                            **marker_kwargs(opt, "angle", 6))
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(thousands))
            ax.yaxis.set_major_formatter(comma(2))
            ax.set_title(f"{DS_LABEL[ds]} · cod. {ENC_LABEL[enc]}", loc="left", fontweight="bold")
            if row == 1:
                ax.set_xlabel("Circuitos de optimización acumulados (escala log)")
            if col == 0:
                ax.set_ylabel("Pérdida de validation (mediana)")
    handles = [plt.Line2D([], [], color=OPT_COLOR[o], linewidth=1.6) for o in OPTIMIZERS]
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=6, color=INK_2))
    fig.legend(handles, [OPT_LABEL[o] for o in OPTIMIZERS] + ["Paso del checkpoint (mediana)"],
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return save(fig, out, "fig02_phaseB_curves")


PHASE_B_FILES = {
    "breast-cancer": ("B-confirm-bc", ("B1-exact", "B2-shots1000", "B3-shots1000-noise", "B4-equal-budget")),
    "iris": ("B-confirm-iris", ("I1-exact", "I2-shots1000", "I3-shots1000-noise", "I4-equal-budget")),
}
PHASE_B_LABEL = ("Exacto", "1000\nshots", "Shots +\nruido", "Mismo\ncoste")


def phase_b_baselines(ds: str):
    folder, blocks = PHASE_B_FILES[ds]
    manifest = load(RESULTS / folder / blocks[0] / "adam-angle" / "manifest.json")["baselines"]
    return manifest["majority_class"]["test"]["accuracy"], manifest["logistic_regression"]["test"]["accuracy"]


def fig_phase_b(out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    offsets = dict(zip(OPTIMIZERS, (-0.27, -0.09, 0.09, 0.27)))
    for row, ds in enumerate(DATASETS):
        folder, blocks = PHASE_B_FILES[ds]
        majority, logistic = phase_b_baselines(ds)
        for col, enc in enumerate(ENCODINGS):
            ax = axes[row, col]
            ax.axhline(logistic, color=LOGISTIC_COLOR, linestyle="--", linewidth=1)
            ax.text(3.45, logistic, "logística", color=INK_2, fontsize=7.5, va="bottom", ha="right")
            for b, block in enumerate(blocks):
                conditions = load(RESULTS / folder / f"{block}-analysis.json")["conditions"]
                for c in conditions:
                    cond = c["condition"]
                    if cond["encoding"] != enc:
                        continue
                    stats = c["stats"]["test_accuracy"]
                    x = b + offsets[cond["optimizer"]]
                    ax.errorbar(x, stats["mean"], yerr=[[stats["mean"] - stats["ci95"][0]], [stats["ci95"][1] - stats["mean"]]],
                                elinewidth=1.2, capsize=0, linestyle="none", **marker_kwargs(cond["optimizer"], enc, 5.5))
            ax.set_xticks(range(4))
            ax.set_xticklabels(PHASE_B_LABEL)
            ax.yaxis.set_major_formatter(comma(2))
            ax.set_title(f"{DS_LABEL[ds]} · cod. {ENC_LABEL[enc]}", loc="left", fontweight="bold")
            ax.grid(axis="x", visible=False)
            if col == 0:
                ax.set_ylabel("Accuracy de test (media, IC95 %)")
    handles, labels = legend_handles(include_encoding=False)
    handles.append(plt.Line2D([], [], color=LOGISTIC_COLOR, linestyle="--", linewidth=1))
    labels.append("Regresión logística")
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return save(fig, out, "fig03_phaseB_conditions")


# --------------------------------------------------------------------------- phase C
def phase_c():
    return load(RESULTS / "C-partitions" / "analysis.json")


def condition_order():
    return [(enc, opt) for enc in ENCODINGS for opt in OPTIMIZERS]


def fig_phase_c_accuracy(out: Path) -> Path:
    data = phase_c()["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    rng = np.random.default_rng(0)  # jitter only
    for ax, ds in zip(axes, DATASETS):
        d = data[ds]
        conditions = {(c["encoding"], c["optimizer"]): c for c in d["conditions"]}
        logistic = d["baselines"]["logistic_regression_test_accuracy"]
        ax.axhspan(logistic["ci95"][0], logistic["ci95"][1], color=GRID, alpha=0.8, linewidth=0)
        ax.axhline(logistic["mean"], color=LOGISTIC_COLOR, linestyle="--", linewidth=1)
        for i, (enc, opt) in enumerate(condition_order()):
            x = i + (0.5 if enc == "phase" else 0)
            c = conditions[(enc, opt)]
            values = np.array(list(c["per_partition"]["test_accuracy"].values()))
            ax.scatter(x + rng.uniform(-0.18, 0.18, values.size), values, s=9, color=OPT_COLOR[opt],
                       alpha=0.35, linewidths=0, zorder=2)
            stats = c["stats"]["test_accuracy"]
            ax.errorbar(x, stats["mean"], yerr=[[stats["mean"] - stats["ci95"][0]], [stats["ci95"][1] - stats["mean"]]],
                        elinewidth=1.6, capsize=0, linestyle="none", zorder=3, **marker_kwargs(opt, enc, 6.5))
        ax.set_xticks([i + (0.5 if i >= 4 else 0) for i in range(8)])
        ax.set_xticklabels([SHORT_OPT[o] for _, o in condition_order()])
        ax.text(1.5, -0.17, "cod. angular", transform=ax.get_xaxis_transform(), ha="center", color=INK_2)
        ax.text(6.0, -0.17, "cod. fase", transform=ax.get_xaxis_transform(), ha="center", color=INK_2)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(comma(2))
        ax.set_title(DS_LABEL[ds], loc="left", fontweight="bold")
        ax.text(7.7, logistic["mean"], "logística", color=INK_2, fontsize=7.5, va="bottom", ha="right")
    axes[0].set_ylabel("Accuracy de test")
    handles, labels = legend_handles()
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=4, color=MUTED, alpha=0.6))
    labels.append("Una partición")
    fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.06), handletextpad=0.2,
               columnspacing=1.0)
    fig.tight_layout()
    return save(fig, out, "fig04_phaseC_accuracy")


def fig_partition_heatmap(out: Path) -> Path:
    data = phase_c()["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, ds in zip(axes, DATASETS):
        d = data[ds]
        conditions = {(c["encoding"], c["optimizer"]): c for c in d["conditions"]}
        partitions = [str(p) for p in d["partitions"]]
        columns = condition_order()
        matrix = np.array([[conditions[k]["per_partition"]["test_accuracy"][p] for k in columns]
                           + [d["baselines"]["per_partition"][p]["logistic"]] for p in partitions])
        lo = np.floor(matrix.min() * 20) / 20
        image = ax.imshow(matrix, cmap="Blues", vmin=lo, vmax=1.0, aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.2f}".replace(".", ","), ha="center", va="center", fontsize=6.3,
                    color="white" if value > lo + 0.62 * (1 - lo) else INK)
        ax.axvline(3.5, color="white", linewidth=2)
        ax.axvline(7.5, color="white", linewidth=2)
        ax.set_xticks(range(9))
        ax.set_xticklabels([SHORT_OPT[o] for _, o in columns] + ["Log."], fontsize=6.4)
        ax.xaxis.tick_top()
        ax.text(1.5, 10.0, "cod. angular", ha="center", va="top", color=INK_2, fontsize=7.5)
        ax.text(5.5, 10.0, "cod. fase", ha="center", va="top", color=INK_2, fontsize=7.5)
        ax.text(8.0, 10.0, "logística", ha="center", va="top", color=INK_2, fontsize=7.5)
        ax.set_yticks(range(len(partitions)))
        ax.set_yticklabels(partitions)
        ax.set_ylabel("Semilla de partición")
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(DS_LABEL[ds], loc="left", fontweight="bold", pad=16)
        bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
        bar.ax.yaxis.set_major_formatter(comma(2))
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return save(fig, out, "fig05_phaseC_partition_heatmap")


def fig_forest(out: Path) -> Path:
    data = phase_c()["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    rows = [(enc, opt) for enc in ENCODINGS for opt in ("gd", "sequential", "spsa")]
    for ax, ds in zip(axes, DATASETS):
        paired = {(p["encoding"], p["optimizer"]): p for p in data[ds]["paired"] if p["metric"] == "test_accuracy"}
        ax.axvline(0, color=INK_2, linewidth=0.8)
        for y, (enc, opt) in enumerate(rows):
            p = paired[(enc, opt)]
            diff = p["difference"]
            ax.errorbar(diff["mean"], y, xerr=[[diff["mean"] - diff["ci95"][0]], [diff["ci95"][1] - diff["mean"]]],
                        elinewidth=1.6, capsize=0, linestyle="none", **marker_kwargs(opt, enc, 6.5))
            ax.text(1.02, y, f"Holm p = {p['holm_p']:.2f}".replace(".", ","), transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.3, color=INK_2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{OPT_LABEL[o]} ({ENC_LABEL[e]})" for e, o in rows])
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(comma(2))
        ax.set_xlabel("Diferencia de accuracy de test frente a Adam")
        ax.set_title(DS_LABEL[ds], loc="left", fontweight="bold")
        ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=7)
    return save(fig, out, "fig06_phaseC_forest_vs_adam")


def fig_cost_accuracy(out: Path) -> Path:
    data = phase_c()["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, ds in zip(axes, DATASETS):
        for c in data[ds]["conditions"]:
            acc = c["stats"]["test_accuracy"]
            cost = c["stats"]["optimization_circuits_to_best"]
            ax.errorbar(cost["mean"], acc["mean"],
                        xerr=[[cost["mean"] - cost["ci95"][0]], [cost["ci95"][1] - cost["mean"]]],
                        yerr=[[acc["mean"] - acc["ci95"][0]], [acc["ci95"][1] - acc["mean"]]],
                        elinewidth=1.1, capsize=0, linestyle="none", **marker_kwargs(c["optimizer"], c["encoding"], 6.5))
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.yaxis.set_major_formatter(comma(2))
        ax.set_xlabel("Circuitos hasta el checkpoint (escala log)")
        ax.set_title(DS_LABEL[ds], loc="left", fontweight="bold")
    axes[0].set_ylabel("Accuracy de test")
    handles, labels = legend_handles()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return save(fig, out, "fig07_phaseC_cost_accuracy")


def fig_patience(out: Path) -> Path:
    records = phase_c()["patience"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for row, ds in enumerate(DATASETS):
        for col, enc in enumerate(ENCODINGS):
            ax = axes[row, col]
            ax.axhline(0, color=INK_2, linewidth=0.8)
            for opt in OPTIMIZERS:
                rows = sorted((r for r in records if r["dataset"] == ds and r["encoding"] == enc and r["optimizer"] == opt),
                              key=lambda r: r["patience"])
                x = [r["circuit_fraction"]["mean"] for r in rows]
                y = [r["test_accuracy_change"]["mean"] for r in rows]
                low = [r["test_accuracy_change"]["mean"] - r["test_accuracy_change"]["ci95"][0] for r in rows]
                high = [r["test_accuracy_change"]["ci95"][1] - r["test_accuracy_change"]["mean"] for r in rows]
                ax.plot(x, y, color=OPT_COLOR[opt], linewidth=1.2)
                ax.errorbar(x, y, yerr=[low, high], elinewidth=1, capsize=0, linestyle="none",
                            **marker_kwargs(opt, enc, 5.5))
                # Lines run P=5 -> 10 -> 20 from left to right; label only the ends
                # (or the cluster, when the three points nearly coincide).
                if x[-1] - x[0] < 0.08:
                    ax.annotate("P5–20", (x[-1], y[-1]), xytext=(4, 4), textcoords="offset points",
                                fontsize=6.3, color=INK_2)
                else:
                    ax.annotate("P5", (x[0], y[0]), xytext=(-4, 4), textcoords="offset points",
                                fontsize=6.3, color=INK_2, ha="right")
                    ax.annotate("P20", (x[-1], y[-1]), xytext=(4, 4), textcoords="offset points",
                                fontsize=6.3, color=INK_2)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}".replace("%", " %")))
            ax.yaxis.set_major_formatter(comma(3))
            ax.set_xlim(0, 1.02)
            ax.set_title(f"{DS_LABEL[ds]} · cod. {ENC_LABEL[enc]}", loc="left", fontweight="bold")
            if row == 1:
                ax.set_xlabel("Circuitos consumidos al parar (fracción de 100 pasos)")
            if col == 0:
                ax.set_ylabel("Δ accuracy de test")
    handles, labels = legend_handles(include_encoding=False)
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return save(fig, out, "fig08_phaseC_patience")


FIGURES = (fig_phase_a, fig_curves, fig_phase_b, fig_phase_c_accuracy, fig_partition_heatmap,
           fig_forest, fig_cost_accuracy, fig_patience)


# --------------------------------------------------------------------------- LaTeX tables
def num(value: float, decimals: int = 3, sign: bool = False) -> str:
    text = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return "$" + text.replace(".", "{,}") + "$"


def ci(stats: dict, decimals: int = 3) -> str:
    return f"[{num(stats['ci95'][0], decimals, True)}, {num(stats['ci95'][1], decimals, True)}]"


def integer(value: float) -> str:
    return f"{value:,.0f}".replace(",", "\\,")


def pvalue(p) -> str:
    return "---" if p is None else num(p, 3)


def table_phase_b() -> str:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Dataset y cod. & Optimizador & Exacto & 1000 shots & Shots + ruido & Mismo coste \\", r"\midrule"]
    for ds in DATASETS:
        folder, blocks = PHASE_B_FILES[ds]
        tables = [load(RESULTS / folder / f"{block}-analysis.json")["conditions"] for block in blocks]
        for enc in ENCODINGS:
            for opt in OPTIMIZERS:
                cells = []
                for conditions in tables:
                    c = next(c for c in conditions if c["condition"]["encoding"] == enc
                             and c["condition"]["optimizer"] == opt)
                    cells.append(num(c["stats"]["test_accuracy"]["mean"]))
                label = f"{DS_LABEL[ds].split(' (')[0].split(' setosa')[0]}, {ENC_LABEL[enc]}" if opt == "adam" else ""
                lines.append(f"{label} & {OPT_LABEL[opt]} & " + " & ".join(cells) + r" \\")
            lines.append(r"\addlinespace")
        majority, logistic = phase_b_baselines(ds)
        lines.append(f"\\multicolumn{{2}}{{l}}{{\\emph{{Baselines {DS_LABEL[ds].split(' (')[0].split(' setosa')[0]}}}}} & "
                     f"\\multicolumn{{4}}{{l}}{{mayoritaria {num(majority)}; logística {num(logistic)}}} \\\\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_phase_c() -> str:
    data = phase_c()["datasets"]
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"Dataset & Cod. & Optimizador & Accuracy test & Circuitos al checkpoint & $\Delta$ frente a Adam [IC95\,\%] & Holm $p$ \\",
             r"\midrule"]
    for ds in DATASETS:
        d = data[ds]
        conditions = {(c["encoding"], c["optimizer"]): c for c in d["conditions"]}
        paired = {(p["encoding"], p["optimizer"]): p for p in d["paired"] if p["metric"] == "test_accuracy"}
        for enc in ENCODINGS:
            for opt in OPTIMIZERS:
                c = conditions[(enc, opt)]
                acc = c["stats"]["test_accuracy"]
                cost = c["stats"]["optimization_circuits_to_best"]["mean"]
                if opt == "adam":
                    delta, holm = "referencia", "---"
                else:
                    p = paired[(enc, opt)]
                    delta = f"{num(p['difference']['mean'], 3, True)} {ci(p['difference'])}"
                    holm = pvalue(p["holm_p"])
                label = DS_LABEL[ds].split(" (")[0].split(" setosa")[0] if (enc, opt) == ("angle", "adam") else ""
                lines.append(f"{label} & {ENC_LABEL[enc] if opt == 'adam' else ''} & {OPT_LABEL[opt]} & "
                             f"{num(acc['mean'])} $\\pm$ {num(acc['sd'])} & {integer(cost)} & {delta} & {holm} \\\\")
            lines.append(r"\addlinespace")
        base = d["baselines"]
        lines.append(f" & \\multicolumn{{2}}{{l}}{{\\emph{{Logística}}}} & {num(base['logistic_regression_test_accuracy']['mean'])} "
                     f"$\\pm$ {num(base['logistic_regression_test_accuracy']['sd'])} & & & \\\\")
        lines.append(f" & \\multicolumn{{2}}{{l}}{{\\emph{{Mayoritaria}}}} & {num(base['majority_class_test_accuracy']['mean'])} "
                     f"$\\pm$ {num(base['majority_class_test_accuracy']['sd'])} & & & \\\\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_gap() -> str:
    data = phase_c()["datasets"]
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"Dataset & Cod. & Optimizador & QCNN $-$ logística [IC95\,\%] & Particiones $\geq$ logística & Wilcoxon $p$ \\",
             r"\midrule"]
    for ds in DATASETS:
        for g in data[ds]["gap_to_logistic"]:
            label = DS_LABEL[ds].split(" (")[0].split(" setosa")[0] if (g["encoding"], g["optimizer"]) == ("angle", "adam") else ""
            lines.append(f"{label} & {ENC_LABEL[g['encoding']]} & {OPT_LABEL[g['optimizer']]} & "
                         f"{num(g['difference']['mean'], 3, True)} {ci(g['difference'])} & "
                         f"{g['partitions_at_or_above_logistic']}/{g['difference']['n']} & {pvalue(g['wilcoxon_p'])} \\\\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_observations() -> str:
    observations = phase_c()["observations"]
    lines = [r"\begin{tabular}{lp{5.6cm}rrrr}", r"\toprule",
             r"Id & Contraste & Diferencia [IC95\,\%] & $+/-$ & $p$ & Holm $p$ \\", r"\midrule"]
    for o in observations:
        diff = o["difference"]
        decimals = 0 if o["metric"] == "optimization_circuits_to_best" else 3
        if decimals:
            value = f"{num(diff['mean'], 3, True)} {ci(diff)}"
        else:
            value = f"${integer(diff['mean'])}$ [${integer(diff['ci95'][0])}$, ${integer(diff['ci95'][1])}$]".replace("$-", "$-")
        description = o["description"].replace("breast-cancer", "Breast Cancer").replace("iris", "Iris") \
            .replace("(angle)", "(angular)").replace("(phase)", "(fase)")
        lines.append(f"{o['id'].split('-')[0]} & {description} & {value} & "
                     f"{o['positive_partitions']}/{o['negative_partitions']} & {pvalue(o['wilcoxon_p'])} & {pvalue(o['holm_p'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_patience(patience: int = 10) -> str:
    records = [r for r in phase_c()["patience"] if r["patience"] == patience]
    lines = [r"\begin{tabular}{lllrrrrr}", r"\toprule",
             r"Dataset & Cod. & Optimizador & Para antes & Mismo checkpoint & Fracción de circuitos & Circuitos al parar & $\Delta$ accuracy test \\",
             r"\midrule"]
    for ds in DATASETS:
        for enc in ENCODINGS:
            for opt in OPTIMIZERS:
                r = next(r for r in records if r["dataset"] == ds and r["encoding"] == enc and r["optimizer"] == opt)
                label = DS_LABEL[ds].split(" (")[0].split(" setosa")[0] if (enc, opt) == ("angle", "adam") else ""
                lines.append(f"{label} & {ENC_LABEL[enc] if opt == 'adam' else ''} & {OPT_LABEL[opt]} & "
                             f"{num(r['stopped_early_fraction'], 2)} & {num(r['same_checkpoint_fraction'], 2)} & "
                             f"{num(r['circuit_fraction']['mean'], 2)} & {integer(r['optimization_circuits_at_stop']['mean'])} & "
                             f"{num(r['test_accuracy_change']['mean'], 3, True)} \\\\")
            lines.append(r"\addlinespace")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


TABLES = {"tab_phaseB_accuracy": table_phase_b, "tab_phaseC_summary": table_phase_c,
          "tab_phaseC_gap_logistic": table_gap, "tab_phaseC_observations": table_observations,
          "tab_phaseC_patience10": table_patience}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "reports" / "figures")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "docs" / "reports" / "tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for figure in FIGURES:
        print("Wrote", figure(args.output_dir))
    for name, builder in TABLES.items():
        path = args.tables_dir / f"{name}.tex"
        path.write_text("% Generado por experiments/plot_variational_report.py; no editar a mano.\n"
                        + builder() + "\n", encoding="utf-8")
        print("Wrote", path)


if __name__ == "__main__":
    main()
