"""Settings form generated from a node's parameter schema."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
                               QWidget)

from ..controller import WorkflowController
from ..model.parameters import BoolParam, ChoiceParam, FloatParam, IntParam, Param, ParameterError, StringParam

INT_LIMIT = 2_000_000_000


class _Field:
    """Widget wrapper with uniform get/set for one parameter."""

    def __init__(self, param: Param, on_change):
        self.param = param
        if isinstance(param, ChoiceParam):
            widget = QComboBox()
            for value, label in param.choices:
                widget.addItem(label, value)
            widget.currentIndexChanged.connect(on_change)
        elif isinstance(param, BoolParam):
            widget = QCheckBox()
            widget.toggled.connect(on_change)
        elif isinstance(param, IntParam):
            widget = QSpinBox()
            low = param.minimum if param.minimum is not None else -INT_LIMIT
            if param.optional:
                low -= 1
                widget.setSpecialValueText("—")
            widget.setRange(low, param.maximum if param.maximum is not None else INT_LIMIT)
            widget.valueChanged.connect(on_change)
        elif isinstance(param, FloatParam):
            widget = QDoubleSpinBox()
            widget.setDecimals(param.decimals)
            widget.setSingleStep(param.step)
            widget.setRange(param.minimum if param.minimum is not None else -1e12,
                            param.maximum if param.maximum is not None else 1e12)
            widget.valueChanged.connect(on_change)
        else:
            widget = QLineEdit()
            widget.textChanged.connect(on_change)
        self.input = widget
        if isinstance(param, StringParam) and param.is_path:
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            button = QPushButton("…")
            button.setFixedWidth(28)
            button.clicked.connect(self._browse)
            layout.addWidget(widget)
            layout.addWidget(button)
            self.widget = container
        else:
            self.widget = widget
        if param.help:
            self.widget.setToolTip(param.help)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self.widget, self.param.label, self.input.text())
        if path:
            self.input.setText(path)

    def get(self) -> Any:
        w = self.input
        if isinstance(self.param, ChoiceParam):
            return w.currentData()
        if isinstance(self.param, BoolParam):
            return w.isChecked()
        if isinstance(self.param, IntParam):
            if self.param.optional and w.value() == w.minimum():
                return None
            return w.value()
        if isinstance(self.param, FloatParam):
            return w.value()
        return w.text()

    def set(self, value: Any) -> None:
        w = self.input
        w.blockSignals(True)
        try:
            if isinstance(self.param, ChoiceParam):
                index = w.findData(value)
                w.setCurrentIndex(max(index, 0))
            elif isinstance(self.param, BoolParam):
                w.setChecked(bool(value))
            elif isinstance(self.param, IntParam):
                w.setValue(w.minimum() if value is None else int(value))
            elif isinstance(self.param, FloatParam):
                w.setValue(float(value))
            else:
                w.setText("" if value is None else str(value))
        finally:
            w.blockSignals(False)


class SettingsPanel(QWidget):
    """Shows the parameters of one node; Apply sends them through the controller."""

    applied = Signal(str)

    def __init__(self, controller: WorkflowController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.node_id: str | None = None
        self._fields: dict[str, _Field] = {}
        self._rows: dict[str, tuple[QFormLayout, QWidget]] = {}
        outer = QVBoxLayout(self)
        self.title = QLabel("Selecciona un nodo para ver sus parámetros.")
        self.title.setWordWrap(True)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.description = QLabel("")
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color: #52514e;")
        outer.addWidget(self.title)
        outer.addWidget(self.description)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)
        self.error = QLabel("")
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #e34948;")
        outer.addWidget(self.error)
        buttons = QHBoxLayout()
        hint = QLabel("Se aplica al momento")
        hint.setToolTip("Los cambios válidos se aplican sin pulsar ningún botón.")
        hint.setObjectName("CategoryLabel")
        self.reset_button = QPushButton("Valores por defecto")
        self.apply_button = QPushButton("Aplicar")
        self.apply_button.setVisible(False)  # kept for keyboard/tests; changes apply automatically
        self.reset_button.clicked.connect(self.reset_defaults)
        self.apply_button.clicked.connect(self.apply)
        buttons.addWidget(hint)
        buttons.addStretch(1)
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.apply_button)
        outer.addLayout(buttons)
        self.show_node(None)

    # ------------------------------------------------------------------ building
    def show_node(self, node_id: str | None) -> None:
        self.node_id = node_id if node_id in self.controller.workflow.nodes else None
        enabled = self.node_id is not None
        self.apply_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        if not enabled:
            self.title.setText("Selecciona un nodo para ver sus parámetros.")
            self.description.setText("")
            self.scroll.setWidget(QWidget())
            self._fields = {}
            self.error.setText("")
            return
        node = self.controller.workflow.nodes[self.node_id]
        self.title.setText(node.display_title)
        self.description.setText(self.controller.node_description(self.node_id))
        self._build(dict(node.params))

    def reload(self) -> None:
        self.show_node(self.node_id)

    def reset_defaults(self) -> None:
        if self.node_id is None:
            return
        node = self.controller.workflow.nodes[self.node_id]
        try:
            self.controller.set_params(self.node_id, node.node_type.default_parameters())
        except Exception as exc:  # noqa: BLE001
            self.error.setText(str(exc))
            return
        self.reload()

    def _build(self, values: dict[str, Any]) -> None:
        schema = self.controller.schema(self.node_id, values)
        container = QWidget()
        layout = QVBoxLayout(container)
        groups: dict[str, QFormLayout] = {}
        self._fields, self._rows = {}, {}
        for param in schema:
            if param.group not in groups:
                box = QGroupBox(param.group)
                groups[param.group] = QFormLayout(box)
                groups[param.group].setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
                layout.addWidget(box)
            field = _Field(param, self._on_change)
            field.set(values.get(param.name, param.default))
            groups[param.group].addRow(param.label, field.widget)
            self._fields[param.name] = field
            self._rows[param.name] = (groups[param.group], field.widget)
        if not schema:
            layout.addWidget(QLabel("Este nodo no tiene parámetros."))
        layout.addStretch(1)
        self.scroll.setWidget(container)
        self._schema_names = [p.name for p in schema]
        self._update_visibility()
        self.error.setText("")

    def values(self) -> dict[str, Any]:
        return {name: field.get() for name, field in self._fields.items()}

    def _on_change(self, *_):
        if self.node_id is None:
            return
        values = self.values()
        # A choice may change the schema (e.g. another algorithm): rebuild keeping values.
        new_names = [p.name for p in self.controller.schema(self.node_id, values)]
        if new_names != self._schema_names:
            self._build(values)
        else:
            self._update_visibility()
        problem = self.controller.validate_params(self.node_id, self.values())
        self.error.setText(problem or "")
        if problem is None and self.controller.workflow.nodes[self.node_id].params != self.values():
            # Valid changes apply immediately (no "Aplicar" needed before running).
            self.apply()

    def _update_visibility(self) -> None:
        values = self.values()
        for param_name, field in self._fields.items():
            visible = field.param.is_visible(values)
            layout, widget = self._rows[param_name]
            layout.setRowVisible(widget, visible)

    def _visible_values(self) -> dict[str, Any]:
        values = self.values()
        return {n: v for n, v in values.items() if self._fields[n].param.is_visible(values)}

    def apply(self) -> None:
        if self.node_id is None:
            return
        try:
            self.controller.set_params(self.node_id, self.values())
        except ParameterError as exc:
            self.error.setText(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.error.setText(str(exc))
            return
        self.error.setText("")
        self.applied.emit(self.node_id)


__all__ = ["SettingsPanel"]
