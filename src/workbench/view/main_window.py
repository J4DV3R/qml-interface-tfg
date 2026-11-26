"""Main window, laid out like a circuit composer.

* Top bar: application and project name, menus (Archivo, Editar, Ver, Ayuda),
  «Guardar» and the primary «Ejecutar flujo» button.
* Left: operations panel with coloured node tiles.
* Centre: canvas toolbar and the node canvas.
* Right: «Parámetros» of the selected node, or the project file (.qmlflow).
* Bottom: two visualisation panels (accuracy chart / metrics table, and
  curves / confusion matrices / log), updated when a run finishes.
* Status bar: messages, progress and the light/dark theme switch.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QSplitter,
                               QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from ..controller import WorkflowController
from . import style
from .canvas import WorkflowScene, WorkflowView
from .panels import ChartPanel, DetailPanel, FlowSourcePanel, OperationsPanel, PanelHeader
from .results_view import ResultsWindow
from .settings import SettingsPanel
from .style import STATE_LABELS, ui_font


class GuiDispatcher(QObject):
    """Runs controller callbacks in the GUI thread (queued across threads)."""

    _call = Signal(object)

    def __init__(self):
        super().__init__()
        self._call.connect(lambda fn: fn())

    def __call__(self, fn) -> None:
        self._call.emit(fn)


class MainWindow(QMainWindow):
    def __init__(self, template: str | None = "clasificacion-qml", theme: str = "dark"):
        super().__init__()
        self.dispatcher = GuiDispatcher()
        self.controller = WorkflowController(dispatcher=self.dispatcher)
        self._results_windows: list[ResultsWindow] = []
        self.current_report = None
        self.apply_theme(theme, refresh=False)
        self._build_ui()
        self.controller.subscribe(self._on_event)
        if template:
            self.controller.new_from_template(template)
        else:
            self.controller.new_project()
        self.resize(1440, 900)

    # ------------------------------------------------------------------ theme
    def apply_theme(self, name: str, refresh: bool = True) -> None:
        theme = style.set_theme(name)
        app = QApplication.instance()
        if app is not None:
            app.setFont(ui_font(10))
            app.setStyleSheet(style.stylesheet(theme))
        style.configure_matplotlib()
        if refresh:
            if name in getattr(self, "theme_buttons", {}):
                self.theme_buttons[name].setChecked(True)
            self.scene.update()
            self.view.viewport().update()
            self.chart_panel.show_report(self.current_report)
            self.detail_panel.show_report(self.current_report)

    # ------------------------------------------------------------------ layout
    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.scene = WorkflowScene(self.controller, self)
        self.view = WorkflowView(self.scene)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.scene.node_activated.connect(self._activate_node)
        self.scene.message.connect(lambda text: self.statusBar().showMessage(text, 8000))
        outer.addWidget(self._build_top_bar())

        self.palette_widget = OperationsPanel(self.controller)
        self.palette_widget.add_requested.connect(self._add_at_center)

        canvas_area = QWidget()
        canvas_layout = QVBoxLayout(canvas_area)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        canvas_layout.addWidget(self._build_canvas_toolbar())
        canvas_layout.addWidget(self.view, 1)

        self.chart_panel = ChartPanel()
        self.detail_panel = DetailPanel()
        self.chart_panel.clear_requested.connect(self.clear_results)
        self.detail_panel.clear_requested.connect(self.clear_results)
        self.log = self.detail_panel.log
        self.bottom = QSplitter(Qt.Horizontal)
        self.bottom.addWidget(self.chart_panel)
        self.bottom.addWidget(self.detail_panel)
        self.bottom.setSizes([1, 1])

        self.centre = QSplitter(Qt.Vertical)
        self.centre.addWidget(canvas_area)
        self.centre.addWidget(self.bottom)
        self.centre.setStretchFactor(0, 3)
        self.centre.setStretchFactor(1, 2)

        self.right_panel = self._build_right_panel()
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.palette_widget)
        self.main_splitter.addWidget(self.centre)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([280, 790, 370])
        outer.addWidget(self.main_splitter, 1)
        self.setCentralWidget(root)
        self._build_status_bar()
        self._build_view_menu_actions()

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        bar.setFixedHeight(48)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(0)
        title = QLabel("QML Workbench")
        title.setObjectName("AppTitle")
        layout.addWidget(title)
        layout.addSpacing(16)
        self.project_label = QLabel("")
        self.project_label.setObjectName("ProjectName")
        layout.addWidget(self.project_label)
        layout.addSpacing(4)
        for name, builder in (("Archivo", self._file_menu), ("Editar", self._edit_menu),
                              ("Ver", self._view_menu), ("Ayuda", self._help_menu)):
            button = QToolButton()
            button.setText(name)
            button.setPopupMode(QToolButton.InstantPopup)
            menu = QMenu(button)
            builder(menu)
            button.setMenu(menu)
            layout.addWidget(button)
        layout.addStretch(1)
        self.save_button = QPushButton("⤓  Guardar")
        self.save_button.setObjectName("GhostButton")
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        self.run_button = QPushButton("▶   Ejecutar flujo")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.setMinimumWidth(190)
        self.run_button.clicked.connect(self._run_or_cancel)
        layout.addWidget(self.run_button)
        return bar

    def _build_canvas_toolbar(self) -> QWidget:
        bar = QWidget()
        self.canvas_toolbar = bar
        bar.setObjectName("CanvasToolbar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        bar.setFixedHeight(40)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 8, 0)

        def tool(text, tooltip, callback):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            layout.addWidget(button)
            return button

        tool("⤢  Ajustar vista", "Ajustar el flujo a la vista (Ctrl+0)", self.view.fit_all)
        tool("−  Alejar", "Alejar (Ctrl + rueda)", lambda: self.view.zoom(1 / 1.2))
        tool("+  Acercar", "Acercar (Ctrl + rueda)", lambda: self.view.zoom(1.2))
        layout.addSpacing(12)
        tool("▷  Ejecutar hasta el nodo", "Ejecuta el nodo seleccionado y lo que necesita", self._run_selected)
        tool("✕  Eliminar", "Elimina los nodos o conexiones seleccionados (Supr)", self.scene.delete_selection).setObjectName("DangerButton")
        layout.addStretch(1)
        self.snap_button = tool("▦  Cuadrícula", "Alinea los nodos a la cuadrícula al moverlos",
                                self._toggle_snap)
        self.snap_button.setCheckable(True)
        self.snap_button.setChecked(True)
        return bar

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = PanelHeader("", ["Parámetros", "Proyecto (.qmlflow)"])
        self.right_selector = header.selector
        layout.addWidget(header)
        stack = QStackedWidget()
        self.settings = SettingsPanel(self.controller)
        self.flow_source = FlowSourcePanel()
        stack.addWidget(self.settings)
        stack.addWidget(self.flow_source)
        header.selector.currentIndexChanged.connect(stack.setCurrentIndex)
        layout.addWidget(stack, 1)
        return panel

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.status_label = QLabel("")
        bar.addPermanentWidget(self.status_label)
        bar.addPermanentWidget(self.progress)
        group = QButtonGroup(self)
        self.theme_buttons: dict[str, QToolButton] = {}
        for name, label, tooltip in (("light", "☀  Claro", "Tema claro"), ("dark", "☾  Oscuro", "Tema oscuro")):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tooltip)
            button.setCheckable(True)
            button.setChecked(style.theme().name == name)
            button.clicked.connect(lambda _=False, n=name: self.apply_theme(n))
            group.addButton(button)
            self.theme_buttons[name] = button
            bar.addPermanentWidget(button)

    # ------------------------------------------------------------------ menus
    def _file_menu(self, menu: QMenu) -> None:
        new_menu = menu.addMenu("Nuevo desde plantilla")
        for template in self.controller.templates():
            action = new_menu.addAction(template["name"])
            action.setToolTip(template["description"])
            action.triggered.connect(lambda _=False, t=template["id"]: self._new_from_template(t))
        menu.addAction("Proyecto vacío", self._new_empty)
        menu.addSeparator()
        self._shortcut(menu.addAction("Abrir…", self._open), QKeySequence.Open)
        self._shortcut(menu.addAction("Guardar", self._save), QKeySequence.Save)
        self._shortcut(menu.addAction("Guardar como…", self._save_as), QKeySequence.SaveAs)
        menu.addSeparator()
        self._shortcut(menu.addAction("Salir", self.close), QKeySequence.Quit)

    def _edit_menu(self, menu: QMenu) -> None:
        menu.addAction("Eliminar selección", lambda: self.scene.delete_selection())
        menu.addAction("Seleccionar todo", lambda: [n.setSelected(True) for n in self.scene.nodes.values()])
        menu.addAction("Renombrar nodo…", self._rename_selected)

    def _view_menu(self, menu: QMenu) -> None:
        # Filled once the panels exist (see _build_view_menu_actions).
        self.view_menu = menu

    def _build_view_menu_actions(self) -> None:
        menu = self.view_menu
        self._shortcut(menu.addAction("Ajustar el flujo a la vista", lambda: self.view.fit_all()), "Ctrl+0")
        menu.addSeparator()
        menu.addSection("Paneles")
        self.panel_actions = {}
        for key, label, widget, shortcut in (
                ("nodes", "Nodos", self.palette_widget, "Ctrl+1"),
                ("right", "Parámetros / Proyecto", self.right_panel, "Ctrl+2"),
                ("chart", "Resultados: accuracy y métricas", self.chart_panel, "Ctrl+3"),
                ("detail", "Resultados: curvas, matriz y registro", self.detail_panel, "Ctrl+4"),
                ("toolbar", "Barra de herramientas del lienzo", self.canvas_toolbar, None)):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(lambda visible, w=widget: self._set_panel_visible(w, visible))
            if shortcut:
                self._shortcut(action, shortcut)
            self.panel_actions[key] = action
        menu.addSeparator()
        content = menu.addMenu("Contenido de los paneles")
        for panel, title in ((self.chart_panel, "Abajo izquierda"), (self.detail_panel, "Abajo derecha"),
                             (self.right_panel, "Derecha")):
            selector = self.right_selector if panel is self.right_panel else panel.header.selector
            sub = content.addMenu(title)
            for index in range(selector.count()):
                sub.addAction(selector.itemText(index), lambda s=selector, i=index: s.setCurrentIndex(i))
        menu.addAction("Restablecer la disposición", self.reset_layout)
        menu.addSeparator()
        menu.addSection("Tema")
        menu.addAction("Tema oscuro", lambda: self.apply_theme("dark"))
        menu.addAction("Tema claro", lambda: self.apply_theme("light"))
        menu.addSeparator()
        menu.addAction("Limpiar métricas y gráficas", self.clear_results)

    def _set_panel_visible(self, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        # Hide the whole bottom area when both result panels are hidden.
        self.bottom.setVisible(not (self.chart_panel.isHidden() and self.detail_panel.isHidden()))

    def reset_layout(self) -> None:
        for action in self.panel_actions.values():
            action.setChecked(True)
        self.main_splitter.setSizes([280, 790, 370])
        self.centre.setSizes([3, 2])
        self.bottom.setSizes([1, 1])
        self.chart_panel.header.selector.setCurrentIndex(0)
        self.detail_panel.header.selector.setCurrentIndex(0)
        self.right_selector.setCurrentIndex(0)

    def clear_results(self) -> None:
        """Clear the metrics and charts of the bottom panels (results stay cached in the engine)."""
        self._set_report(None)
        self.statusBar().showMessage("Métricas y gráficas limpiadas.", 4000)

    def _help_menu(self, menu: QMenu) -> None:
        menu.addAction("Cómo se usa", self._show_help)

    def _shortcut(self, action: QAction, key) -> None:
        action.setShortcut(key)
        self.addAction(action)

    # ------------------------------------------------------------------ controller events
    def _on_event(self, event: str, payload: dict) -> None:
        if event == "workflow_reset":
            self.scene.rebuild()
            self.view.fit_all()
            self.settings.show_node(None)
            self.flow_source.show_workflow(self.controller.workflow)
            self._set_report(None)
            self._update_title()
        elif event == "workflow_changed":
            if payload.get("kind") != "node_moved":
                self.scene.rebuild()
                if self.settings.node_id not in self.controller.workflow.nodes:
                    self.settings.show_node(None)
            self.flow_source.show_workflow(self.controller.workflow)
        elif event == "node_state":
            item = self.scene.nodes.get(payload["node_id"])
            if item is not None:
                item.set_state(payload["state"])
            if payload.get("error"):
                self._append_log(payload["node_id"], f"ERROR: {payload['error']}")
        elif event == "node_progress":
            item = self.scene.nodes.get(payload["node_id"])
            if item is not None:
                item.set_progress(payload["fraction"], payload["message"])
            self.progress.setValue(int(payload["fraction"] * 100))
        elif event == "log":
            self._append_log(payload.get("node_id"), payload["message"])
        elif event == "run_started":
            self._set_running(True)
            self._append_log(None, "— Ejecución iniciada —")
        elif event == "run_finished":
            self._set_running(False)
            states = payload.get("states", {})
            summary = ", ".join(f"{self._title(n)}: {STATE_LABELS.get(s, s)}" for n, s in states.items())
            self._append_log(None, f"— Ejecución terminada — {summary}")
            for node_id, state in states.items():
                node = self.controller.workflow.nodes.get(node_id)
                if state == "done" and node is not None and node.type_id == "results":
                    self._set_report(self.controller.outputs(node_id)["report"])
        elif event == "dirty":
            self._update_title()

    def _set_report(self, report) -> None:
        self.current_report = report
        self.chart_panel.show_report(report)
        self.detail_panel.show_report(report)

    def _title(self, node_id: str | None) -> str:
        node = self.controller.workflow.nodes.get(node_id) if node_id else None
        return node.display_title if node else ""

    def _append_log(self, node_id: str | None, message: str) -> None:
        prefix = f"[{self._title(node_id)}] " if node_id else ""
        self.log.appendPlainText(prefix + message)

    def _set_running(self, running: bool) -> None:
        self.run_button.setText("■   Cancelar" if running else "▶   Ejecutar flujo")
        self.progress.setVisible(running)
        self.progress.setValue(0)
        self.status_label.setText("Ejecutando…" if running else "")
        self.settings.apply_button.setEnabled(not running and self.settings.node_id is not None)

    def _update_title(self) -> None:
        name = self.controller.path.stem if self.controller.path else self.controller.workflow.name
        star = "*" if self.controller.dirty else ""
        self.project_label.setText(f"{name}{star}")
        self.setWindowTitle(f"{star}{name} — QML Workbench")

    # ------------------------------------------------------------------ actions
    def _run_or_cancel(self) -> None:
        if self.controller.is_running:
            self.controller.cancel()
        else:
            self.controller.run()

    def _toggle_snap(self) -> None:
        self.scene.snap = self.snap_button.isChecked()

    def _selection_changed(self) -> None:
        selected = self.scene.selected_node_ids()
        self.settings.show_node(selected[0] if len(selected) == 1 else None)
        if len(selected) == 1:
            outputs = self.controller.outputs(selected[0]) or {}
            if outputs.get("report") is not None:
                self._set_report(outputs["report"])

    def _activate_node(self, node_id: str) -> None:
        outputs = self.controller.outputs(node_id) or {}
        if outputs.get("report") is not None:
            window = ResultsWindow(outputs["report"], self)
            window.setModal(False)
            window.show()
            self._results_windows.append(window)
        else:
            self.scene.clearSelection()
            self.scene.nodes[node_id].setSelected(True)

    def _add_at_center(self, type_id: str) -> None:
        center = self.view.mapToScene(self.view.viewport().rect().center())
        try:
            self.controller.add_node(type_id, (center.x() - 100, center.y() - 40))
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(str(exc), 8000)

    def _run_selected(self) -> None:
        selected = self.scene.selected_node_ids()
        if selected:
            self.controller.run(selected)
        else:
            self.statusBar().showMessage("Selecciona un nodo.", 5000)

    def _rename_selected(self) -> None:
        selected = self.scene.selected_node_ids()
        if len(selected) != 1:
            self.statusBar().showMessage("Selecciona un nodo.", 5000)
            return
        node = self.controller.workflow.nodes[selected[0]]
        title, ok = QInputDialog.getText(self, "Renombrar nodo", "Nombre:", text=node.display_title)
        if ok:
            self.controller.rename_node(selected[0], title.strip() or None)

    def _show_help(self) -> None:
        QMessageBox.information(self, "Cómo se usa", (
            "• Arrastra una ficha del panel «Nodos» al lienzo o haz doble clic en ella.\n"
            "• Conecta arrastrando de una salida (derecha) a una entrada (izquierda); "
            "solo se aceptan tipos compatibles.\n"
            "• Selecciona un nodo para editar sus parámetros a la derecha y pulsa «Aplicar».\n"
            "• «Ejecutar flujo» (F5) ejecuta lo que ha cambiado; los resultados aparecen abajo.\n"
            "• Doble clic en «Resultados» abre el informe completo."))

    def _confirm_discard(self) -> bool:
        if not self.controller.dirty:
            return True
        answer = QMessageBox.question(self, "Cambios sin guardar", "Hay cambios sin guardar. ¿Descartarlos?",
                                      QMessageBox.Discard | QMessageBox.Cancel)
        return answer == QMessageBox.Discard

    def _new_from_template(self, template_id: str) -> None:
        if self._confirm_discard():
            self.controller.new_from_template(template_id)

    def _new_empty(self) -> None:
        if self._confirm_discard():
            self.controller.new_project()
            name, ok = QInputDialog.getText(self, "Proyecto vacío", "Nombre del proyecto:")
            if ok and name:
                self.controller.workflow.name = name
                self._update_title()

    def _open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Abrir proyecto", "", "Flujos QML (*.qmlflow)")
        if path:
            try:
                self.controller.open(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "No se pudo abrir", str(exc))

    def _save(self) -> None:
        if self.controller.path is None:
            self._save_as()
        else:
            self.controller.save()

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Guardar proyecto", "proyecto.qmlflow", "Flujos QML (*.qmlflow)")
        if path:
            saved = self.controller.save(Path(path))
            self.statusBar().showMessage(f"Guardado en {saved}", 5000)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_F5:
            self._run_or_cancel()
        elif event.key() == Qt.Key_Escape and self.controller.is_running:
            self.controller.cancel()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.controller.is_running:
            self.controller.cancel()
            self.controller.wait(5)
        if self._confirm_discard():
            # Items are destroyed with the scene; stop reacting to their selection changes.
            self.scene.selectionChanged.disconnect(self._selection_changed)
            event.accept()
        else:
            event.ignore()
