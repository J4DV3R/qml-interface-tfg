"""Controller: the only entry point the view uses to change or run the workflow.

It is independent of Qt. The view subscribes to events with
:meth:`WorkflowController.subscribe`; executions run in a background thread
(``run_in_background=True``) and every event is delivered through
``dispatcher``, which the Qt view replaces with a function that marshals the
call to the GUI thread.

Events (name, payload):

* ``workflow_reset``: a new workflow was loaded (template, file or new).
* ``workflow_changed``: structure, parameters or title changed (``kind``, ids).
* ``node_state``: ``node_id``, ``state``, ``error``.
* ``node_progress``: ``node_id``, ``fraction``, ``message``.
* ``log``: ``node_id`` (or None), ``message``.
* ``run_started`` / ``run_finished``: ``states`` on finish.
* ``dirty``: unsaved-changes flag (``dirty``).
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable

from ..model import (Engine, EngineListener, ParameterError, Workflow, WorkflowError, get_node_type,
                     list_templates, load_template, node_types)

Observer = Callable[[str, dict[str, Any]], None]


class WorkflowController:
    def __init__(self, workflow: Workflow | None = None, *, run_in_background: bool = True,
                 dispatcher: Callable[[Callable[[], None]], None] | None = None):
        self.engine = Engine()
        self.run_in_background = run_in_background
        self.dispatcher = dispatcher or (lambda fn: fn())
        self._observers: list[Observer] = []
        self._thread: threading.Thread | None = None
        self.path: Path | None = None
        self.dirty = False
        self.workflow = workflow or Workflow()
        self.workflow.subscribe(self._on_model_event)

    # ------------------------------------------------------------------ observers
    def subscribe(self, observer: Observer) -> None:
        self._observers.append(observer)

    def _emit(self, event: str, **payload: Any) -> None:
        def deliver():
            for observer in list(self._observers):
                observer(event, payload)
        self.dispatcher(deliver)

    def _on_model_event(self, event: str, payload: dict[str, Any]) -> None:
        self._set_dirty(True)
        self._emit("workflow_changed", kind=event, **payload)
        # Structural or parameter changes may make cached results outdated.
        for node_id in self.workflow.nodes:
            self._emit_state(node_id)

    def _set_dirty(self, value: bool) -> None:
        if self.dirty != value:
            self.dirty = value
            self._emit("dirty", dirty=value)

    def _emit_state(self, node_id: str) -> None:
        self._emit("node_state", node_id=node_id, state=self.engine.state(self.workflow, node_id),
                   error=self.engine.error(node_id))

    def _replace_workflow(self, workflow: Workflow, path: Path | None) -> None:
        self.cancel()
        self.workflow = workflow
        self.workflow.subscribe(self._on_model_event)
        self.engine = Engine()
        self.path = path
        self._emit("workflow_reset")
        for node_id in workflow.nodes:
            self._emit_state(node_id)
        self.dirty = path is not None
        self._set_dirty(path is None)

    # ------------------------------------------------------------------ queries
    @staticmethod
    def templates() -> list[dict[str, str]]:
        return list_templates()

    @staticmethod
    def palette() -> list[dict[str, Any]]:
        return [{"type_id": n.type_id, "title": n.title, "category": n.category, "description": n.description}
                for n in node_types()]

    def schema(self, node_id: str, values: dict[str, Any] | None = None):
        node = self.workflow.nodes[node_id]
        return node.node_type.parameter_schema(values if values is not None else node.params)

    def state(self, node_id: str) -> str:
        return self.engine.state(self.workflow, node_id)

    def outputs(self, node_id: str) -> dict[str, Any] | None:
        return self.engine.outputs(self.workflow, node_id)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ project commands
    def new_project(self) -> None:
        self._replace_workflow(Workflow(), None)

    def new_from_template(self, template_id: str) -> None:
        self._replace_workflow(load_template(template_id), None)

    def open(self, path: str | Path) -> None:
        self._replace_workflow(Workflow.load(path), Path(path))
        self._set_dirty(False)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("indica dónde guardar el proyecto")
        if target.suffix != ".qmlflow":
            target = target.with_suffix(".qmlflow")
        self.workflow.save(target)
        self.path = target
        self._set_dirty(False)
        return target

    # ------------------------------------------------------------------ editing commands
    def add_node(self, type_id: str, position: tuple[float, float] = (0, 0)) -> str:
        self._require_idle()
        return self.workflow.add_node(type_id, position)

    def remove_nodes(self, node_ids: list[str]) -> None:
        self._require_idle()
        for node_id in node_ids:
            if node_id in self.workflow.nodes:
                self.workflow.remove_node(node_id)
                self.engine.forget(node_id)

    def connect(self, source_node: str, source_port: str, target_node: str, target_port: str) -> str:
        self._require_idle()
        return self.workflow.connect(source_node, source_port, target_node, target_port)

    def can_connect(self, source_node: str, source_port: str, target_node: str, target_port: str) -> str | None:
        return self.workflow.can_connect(source_node, source_port, target_node, target_port)

    def disconnect(self, link_id: str) -> None:
        self._require_idle()
        self.workflow.disconnect(link_id)

    def set_params(self, node_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Validate and apply; raises ParameterError with every problem found."""
        self._require_idle()
        node = self.workflow.nodes[node_id]
        validated = node.node_type.validate_parameters(values)
        return self.workflow.set_params(node_id, validated)

    def validate_params(self, node_id: str, values: dict[str, Any]) -> str | None:
        node = self.workflow.nodes[node_id]
        try:
            node.node_type.validate_parameters(values)
        except ParameterError as exc:
            return str(exc)
        return None

    def move_node(self, node_id: str, position: tuple[float, float]) -> None:
        self.workflow.move_node(node_id, position)

    def rename_node(self, node_id: str, title: str | None) -> None:
        self.workflow.rename_node(node_id, title)

    def _require_idle(self) -> None:
        if self.is_running:
            raise WorkflowError("espera a que termine la ejecución o cancélala")

    # ------------------------------------------------------------------ execution
    def run(self, targets: list[str] | None = None) -> None:
        if self.is_running:
            return
        listener = EngineListener(
            node_started=lambda node_id: self._emit("node_state", node_id=node_id, state="running", error=None),
            node_progress=lambda node_id, fraction, message: self._emit(
                "node_progress", node_id=node_id, fraction=fraction, message=message),
            node_log=lambda node_id, message: self._emit("log", node_id=node_id, message=message),
            node_finished=lambda node_id, state, error: self._emit(
                "node_state", node_id=node_id, state=state, error=error),
        )

        def work():
            self._emit("run_started")
            try:
                states = self.engine.run(self.workflow, targets, listener)
            except WorkflowError as exc:
                self._emit("log", node_id=None, message=f"Error: {exc}")
                states = {}
            self._emit("run_finished", states=states)

        if self.run_in_background:
            self._thread = threading.Thread(target=work, name="workbench-run", daemon=True)
            self._thread.start()
        else:
            work()

    def cancel(self) -> None:
        if self.is_running:
            self.engine.cancel()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def node_description(self, node_id: str) -> str:
        node = self.workflow.nodes[node_id]
        return get_node_type(node.type_id).description
