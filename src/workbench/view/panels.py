"""Panels of the composer-style layout: headers, operation tiles and result views."""

from __future__ import annotations

import json

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPlainTextEdit, QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from ..controller import WorkflowController
from ..model.data import EvaluationReport
from . import style
from .style import CATEGORY_COLORS, mono_font, node_code

NODE_MIME = "application/x-qml-workbench-node"
SPLIT_LABELS = {"train": "Entrenamiento", "validation": "Validación", "test": "Test"}
METRIC_LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Accuracy balanceada", "precision": "Precisión",
                 "recall": "Recall", "f1": "F1", "mse": "MSE", "support": "Muestras"}


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".replace(".", ",")
    return str(value)


class PanelHeader(QWidget):
    """Title (or selector) on the left, small tool buttons on the right."""

    def __init__(self, title: str, choices: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("PanelHeader")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(40)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 4, 0)
        if choices:
            self.selector = QComboBox()
            self.selector.addItems(choices)
            layout.addWidget(self.selector)
        else:
            self.selector = None
            layout.addWidget(QLabel(title))
        layout.addStretch(1)
        self._tools = layout

    def add_tool(self, text: str, tooltip: str, callback) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        self._tools.addWidget(button)
        return button


class Tile(QToolButton):
    """Coloured operation tile; drag it onto the canvas or double-click it."""

    def __init__(self, entry: dict, parent=None):
        super().__init__(parent)
        self.type_id = entry["type_id"]
        self.setObjectName("Tile")
        self.setText(node_code(self.type_id))
        self.setFixedSize(48, 40)
        self.setToolTip(f"<b>{entry['title']}</b><br>{entry['description']}")
        color = CATEGORY_COLORS.get(entry["category"])
        if color is not None:
            self.setStyleSheet(f"QToolButton#Tile {{ background: {color.name()}; border-radius: 4px; }}"
                               f"QToolButton#Tile:hover {{ background: {color.lighter(115).name()}; border-radius: 4px; }}")
        self._press = None

    def mousePressEvent(self, event) -> None:
        self._press = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._press is not None and (event.position() - self._press).manhattanLength() > 6:
            mime = QMimeData()
            mime.setData(NODE_MIME, self.type_id.encode("utf-8"))
            drag = QDrag(self)
            drag.setMimeData(mime)
            drag.setPixmap(self.grab())
            self._press = None
            drag.exec(Qt.CopyAction)
            self.setDown(False)
            return
        super().mouseMoveEvent(event)


class OperationsPanel(QWidget):
    """Equivalent of the composer's «Operations»: node tiles grouped by category."""

    add_requested = Signal(str)

    def __init__(self, controller: WorkflowController, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = PanelHeader("Nodos")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Buscar nodo…")
        self.search.setVisible(False)
        self.search.textChanged.connect(self._filter)
        header.add_tool("Buscar", "Buscar un nodo", lambda: (self.search.setVisible(not self.search.isVisible()),
                                                 self.search.setFocus()))
        layout.addWidget(header)
        layout.addWidget(self.search)
        body = QWidget()
        self.grid = QGridLayout(body)
        self.grid.setContentsMargins(12, 6, 12, 12)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)
        self.tiles: list[tuple[Tile, dict, QLabel]] = []
        row = 0
        categories: dict[str, list[dict]] = {}
        for entry in controller.palette():
            categories.setdefault(entry["category"], []).append(entry)
        for category, entries in categories.items():
            label = QLabel(category)
            label.setObjectName("CategoryLabel")
            self.grid.addWidget(label, row, 0, 1, 4)
            row += 1
            for entry in entries:
                tile = Tile(entry)
                tile.mouseDoubleClickEvent = lambda event, t=entry["type_id"]: self.add_requested.emit(t)
                name = QLabel(f"<b>{entry['title']}</b><br><span style='font-size:8pt'>"
                              f"{_short(entry['description'])}</span>")
                name.setWordWrap(True)
                name.setToolTip(entry["description"])
                self.grid.addWidget(tile, row, 0, Qt.AlignTop)
                self.grid.addWidget(name, row, 1, 1, 3)
                self.tiles.append((tile, entry, name))
                row += 1
        self.grid.setRowStretch(row, 1)
        self.grid.setColumnStretch(1, 1)
        layout.addWidget(body, 1)
        hint = QLabel("Arrastra una ficha al lienzo o haz doble clic.")
        hint.setObjectName("CategoryLabel")
        hint.setWordWrap(True)
        hint.setContentsMargins(12, 0, 12, 10)
        layout.addWidget(hint)

    def _filter(self, text: str) -> None:
        text = text.lower().strip()
        for tile, entry, name in self.tiles:
            visible = not text or text in entry["title"].lower() or text in entry["description"].lower()
            tile.setVisible(visible)
            name.setVisible(visible)


def _short(text: str, limit: int = 70) -> str:
    first = text.split(". ")[0].rstrip(".")
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


class _ClearableMixin:
    """Right-click anywhere in the panel offers to clear the results views.

    Subclasses (QWidgets) declare the ``clear_requested`` signal.
    """

    def _install_context_menu(self, *widgets) -> None:
        for widget in (self, *widgets):
            widget.setContextMenuPolicy(Qt.CustomContextMenu)
            widget.customContextMenuRequested.connect(
                lambda pos, w=widget: self._context_menu(w.mapToGlobal(pos)))

    def _context_menu(self, global_pos) -> None:
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        clear = menu.addAction("Limpiar métricas y gráficas")
        extra = self._extra_actions(menu)
        chosen = menu.exec(global_pos)
        if chosen == clear:
            self.clear_requested.emit()
        elif chosen is not None and chosen in extra:
            extra[chosen]()

    def _extra_actions(self, menu) -> dict:
        return {}


class ChartPanel(_ClearableMixin, QWidget):
    """Bottom-left panel (like «Probabilities»): accuracy chart or metrics table."""

    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = PanelHeader("", ["Accuracy por partición", "Tabla de métricas"])
        layout.addWidget(self.header)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.table)
        self.header.selector.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.header.add_tool("Limpiar", "Limpiar métricas y gráficas (también con clic derecho)", lambda: self.clear_requested.emit())
        self._install_context_menu(self.canvas, self.table)
        self.report: EvaluationReport | None = None
        self.show_report(None)

    def show_report(self, report: EvaluationReport | None) -> None:
        self.report = report
        self._draw_chart()
        self._fill_table()

    def _draw_chart(self) -> None:
        t = style.theme()
        self.figure.clear()
        self.figure.set_facecolor(t.background)
        ax = self.figure.add_subplot(1, 1, 1)
        _style_axes(ax, t)
        if self.report is None:
            ax.text(0.5, 0.5, "Ejecuta el flujo para ver resultados", ha="center", va="center",
                    color=t.text_secondary, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            splits = list(self.report.metrics)
            series = [("Modelo QML", [self.report.metrics[s]["accuracy"] for s in splits], "#B388FF")]
            colors = ["#9AA3AF", "#FFB74D", "#81C784"]
            for (name, values), color in zip(self.report.baselines.items(), colors):
                series.append((name, [values[s] for s in splits], color))
            width = 0.8 / len(series)
            for i, (name, values, color) in enumerate(series):
                xs = [j + (i - (len(series) - 1) / 2) * width for j in range(len(splits))]
                ax.bar(xs, values, width * 0.92, label=name, color=color)
            ax.set_xticks(range(len(splits)))
            ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy")
            legend = ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                               ncol=len(series))
            for text in legend.get_texts():
                text.set_color(t.text)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _fill_table(self) -> None:
        self.table.clear()
        if self.report is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        splits = list(self.report.metrics)
        names = list(next(iter(self.report.metrics.values())).keys())
        self.table.setColumnCount(len(splits) + 1)
        self.table.setRowCount(len(names) + len(self.report.baselines))
        self.table.setHorizontalHeaderLabels(["Métrica", *[SPLIT_LABELS.get(s, s) for s in splits]])
        rows = [[METRIC_LABELS.get(n, n), *[fmt(self.report.metrics[s][n]) for s in splits]] for n in names]
        rows += [[f"Baseline: {n}", *[fmt(v[s]) for s in splits]] for n, v in self.report.baselines.items()]
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(value))


class DetailPanel(_ClearableMixin, QWidget):
    """Bottom-right panel (like «Q-sphere»): curves, confusion matrices or the log."""

    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = PanelHeader("", ["Curvas de entrenamiento", "Matriz de confusión", "Registro"])
        layout.addWidget(self.header)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.confusion = QTableWidget()
        self.confusion.verticalHeader().setVisible(False)
        self.confusion.setEditTriggers(QTableWidget.NoEditTriggers)
        self.confusion.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(mono_font(9))
        for widget in (self.canvas, self.confusion, self.log):
            self.stack.addWidget(widget)
        self.header.selector.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.header.add_tool("Limpiar", "Limpiar métricas y gráficas (también con clic derecho)", lambda: self.clear_requested.emit())
        self._install_context_menu(self.canvas, self.confusion, self.log)
        self.report: EvaluationReport | None = None
        self.show_report(None)

    def _extra_actions(self, menu) -> dict:
        clear_log = menu.addAction("Limpiar el registro")
        copy_log = menu.addAction("Copiar el registro")
        return {clear_log: self.log.clear, copy_log: lambda: (self.log.selectAll(), self.log.copy())}

    def show_report(self, report: EvaluationReport | None) -> None:
        self.report = report
        t = style.theme()
        self.figure.clear()
        self.figure.set_facecolor(t.background)
        ax = self.figure.add_subplot(1, 1, 1)
        _style_axes(ax, t)
        if report is None:
            ax.text(0.5, 0.5, "Sin resultados todavía", ha="center", va="center", color=t.text_secondary,
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            steps = report.curves.get("step", [])
            if any(v is not None for v in report.curves.get("train_loss", [])):
                ax.plot(steps, report.curves["train_loss"], label="Train", color="#4FC3F7", linewidth=1.8)
            ax.plot(steps, report.curves.get("validation_loss", []), label="Validation", color="#B388FF",
                    linewidth=1.8)
            checkpoint = report.configuration.get("checkpoint", {}).get("step")
            if checkpoint is not None:
                ax.axvline(checkpoint, color=t.text_secondary, linestyle="--", linewidth=1, label="Checkpoint")
            ax.set_xlabel("Paso")
            ax.set_ylabel("Pérdida (MSE)")
            legend = ax.legend(frameon=False, fontsize=8)
            for text in legend.get_texts():
                text.set_color(t.text)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.confusion.clear()
        if report is None:
            self.confusion.setRowCount(0)
            self.confusion.setColumnCount(0)
            return
        self.confusion.setColumnCount(5)
        self.confusion.setHorizontalHeaderLabels(["Partición", "VP", "FN", "FP", "VN"])
        self.confusion.setRowCount(len(report.confusion))
        for i, (split, c) in enumerate(report.confusion.items()):
            for j, value in enumerate([SPLIT_LABELS.get(split, split), c["tp"], c["fn"], c["fp"], c["tn"]]):
                self.confusion.setItem(i, j, QTableWidgetItem(str(value)))


class FlowSourcePanel(QPlainTextEdit):
    """Read-only view of the project file (like the composer's OpenQASM panel)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(mono_font(9))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

    def show_workflow(self, workflow) -> None:
        self.setPlainText(json.dumps(workflow.to_dict(), indent=2, ensure_ascii=False))


def _style_axes(ax, t) -> None:
    ax.set_facecolor(t.background)
    for spine in ax.spines.values():
        spine.set_color(t.border)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=t.text_secondary, labelsize=8)
    ax.yaxis.label.set_color(t.text_secondary)
    ax.xaxis.label.set_color(t.text_secondary)
    ax.grid(axis="y", color=t.border, alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
