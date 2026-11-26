"""Results window: metrics, baselines, curves, confusion matrices and configuration."""

from __future__ import annotations

import json

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QDialog, QHeaderView, QLabel, QPlainTextEdit, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from ..model.data import EvaluationReport

SPLIT_LABELS = {"train": "Entrenamiento", "validation": "Validación", "test": "Test"}
METRIC_LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Accuracy balanceada", "precision": "Precisión",
                 "recall": "Recall", "f1": "F1", "mse": "MSE", "support": "Muestras"}


def _table(rows: list[list[str]], header: list[str]) -> QTableWidget:
    table = QTableWidget(len(rows), len(header))
    table.setHorizontalHeaderLabels(header)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.setItem(i, j, QTableWidgetItem(value))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    return table


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".replace(".", ",")
    return str(value)


class ResultsWindow(QDialog):
    def __init__(self, report: EvaluationReport, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Resultados — {report.title}")
        self.resize(900, 640)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        tabs.addTab(self._summary(report), "Resumen")
        tabs.addTab(self._curves(report), "Curvas")
        tabs.addTab(self._confusion(report), "Matriz de confusión")
        config = QPlainTextEdit(json.dumps(report.configuration, indent=2, ensure_ascii=False, default=str))
        config.setReadOnly(True)
        tabs.addTab(config, "Configuración")

    def _summary(self, report: EvaluationReport) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        splits = list(report.metrics)
        names = list(next(iter(report.metrics.values())).keys())
        rows = [[METRIC_LABELS.get(n, n), *[_fmt(report.metrics[s][n]) for s in splits]] for n in names]
        layout.addWidget(QLabel("<b>Métricas del checkpoint (elegido con validation; test evaluado una vez)</b>"))
        layout.addWidget(_table(rows, ["Métrica", *[SPLIT_LABELS.get(s, s) for s in splits]]))
        if report.baselines:
            layout.addWidget(QLabel("<b>Baselines clásicas (accuracy)</b>"))
            rows = [[name, *[_fmt(values[s]) for s in splits]] for name, values in report.baselines.items()]
            rows.insert(0, ["Modelo QML", *[_fmt(report.metrics[s]["accuracy"]) for s in splits]])
            layout.addWidget(_table(rows, ["Modelo", *[SPLIT_LABELS.get(s, s) for s in splits]]))
        checkpoint = report.configuration.get("checkpoint", {})
        cost = report.configuration.get("cost", {})
        info = (f"Checkpoint en el paso {checkpoint.get('step')} de {checkpoint.get('steps_completed')} "
                f"({checkpoint.get('stop_reason')}). Circuitos de optimización: "
                f"{cost.get('optimization_circuits', '—')}. Tiempo: {cost.get('elapsed_seconds', 0):.1f} s.")
        label = QLabel(info)
        label.setWordWrap(True)
        layout.addWidget(label)
        for note in report.notes:
            note_label = QLabel(f"⚠ {note}")
            note_label.setWordWrap(True)
            note_label.setStyleSheet("color: #c98500;")
            layout.addWidget(note_label)
        return widget

    def _curves(self, report: EvaluationReport) -> QWidget:
        figure = Figure(figsize=(8, 5), layout="tight")
        canvas = FigureCanvasQTAgg(figure)
        steps = report.curves.get("step", [])
        ax1 = figure.add_subplot(1, 2, 1)
        if any(v is not None for v in report.curves.get("train_loss", [])):
            ax1.plot(steps, report.curves["train_loss"], label="Train", color="#2a78d6", linewidth=1.8)
        ax1.plot(steps, report.curves.get("validation_loss", []), label="Validation", color="#eb6834", linewidth=1.8)
        checkpoint = report.configuration.get("checkpoint", {}).get("step")
        if checkpoint is not None:
            ax1.axvline(checkpoint, color="#52514e", linestyle="--", linewidth=1, label="Checkpoint")
        ax1.set_xlabel("Paso")
        ax1.set_ylabel("Pérdida (MSE)")
        ax1.legend(frameon=False)
        ax1.grid(alpha=0.3)
        ax2 = figure.add_subplot(1, 2, 2)
        ax2.plot(steps, report.curves.get("validation_accuracy", []), color="#1baf7a", linewidth=1.8)
        ax2.set_xlabel("Paso")
        ax2.set_ylabel("Accuracy de validation")
        ax2.grid(alpha=0.3)
        return canvas

    def _confusion(self, report: EvaluationReport) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        for split, c in report.confusion.items():
            layout.addWidget(QLabel(f"<b>{SPLIT_LABELS.get(split, split)}</b>"))
            rows = [["Real +1", str(c["tp"]), str(c["fn"])], ["Real −1", str(c["fp"]), str(c["tn"])]]
            table = _table(rows, ["", "Predicho +1", "Predicho −1"])
            table.setMaximumHeight(100)
            layout.addWidget(table)
        layout.addStretch(1)
        return widget
