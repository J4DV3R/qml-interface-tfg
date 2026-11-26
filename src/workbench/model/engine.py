"""Execution engine: runs a workflow in topological order with output caching.

Each node output is cached under a *signature* that hashes its type, its
parameters and the signatures of everything upstream. Changing a parameter
therefore invalidates exactly that node and its descendants; unchanged nodes
are not recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import threading
import time
import traceback
from typing import Any, Callable

from .node import Cancelled, RunContext
from .workflow import Workflow

# Node states reported to the controller/view.
PENDING = "pending"        # can run, not computed (or outdated)
INCOMPLETE = "incomplete"  # a required input is not connected
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"


@dataclass
class EngineListener:
    """Callbacks used by the engine; every field is optional."""

    node_started: Callable[[str], None] = lambda node_id: None
    node_progress: Callable[[str, float, str], None] = lambda node_id, fraction, message: None
    node_log: Callable[[str, str], None] = lambda node_id, message: None
    node_finished: Callable[[str, str, str | None], None] = lambda node_id, state, error: None


@dataclass
class _CacheEntry:
    signature: str
    outputs: dict[str, Any]
    seconds: float


@dataclass
class Engine:
    _cache: dict[str, _CacheEntry] = field(default_factory=dict)
    _errors: dict[str, str] = field(default_factory=dict)
    _cancel: threading.Event = field(default_factory=threading.Event)

    # ------------------------------------------------------------------ signatures and state
    def signature(self, workflow: Workflow, node_id: str, _memo: dict[str, str] | None = None) -> str:
        memo = _memo if _memo is not None else {}
        if node_id in memo:
            return memo[node_id]
        node = workflow.nodes[node_id]
        upstream = sorted(
            (link.target_port, link.source_port, self.signature(workflow, link.source_node, memo))
            for link in workflow.incoming(node_id)
        )
        payload = json.dumps({"type": node.type_id, "params": node.params, "inputs": upstream},
                             sort_keys=True, default=str)
        memo[node_id] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return memo[node_id]

    def state(self, workflow: Workflow, node_id: str) -> str:
        if self._is_incomplete(workflow, node_id):
            return INCOMPLETE
        entry = self._cache.get(node_id)
        if entry is not None and entry.signature == self.signature(workflow, node_id):
            return DONE
        if node_id in self._errors:
            return ERROR
        return PENDING

    def error(self, node_id: str) -> str | None:
        return self._errors.get(node_id)

    def outputs(self, workflow: Workflow, node_id: str) -> dict[str, Any] | None:
        """Cached outputs if they are up to date, else None."""
        entry = self._cache.get(node_id)
        if entry is not None and entry.signature == self.signature(workflow, node_id):
            return entry.outputs
        return None

    def forget(self, node_id: str) -> None:
        self._cache.pop(node_id, None)
        self._errors.pop(node_id, None)

    def _is_incomplete(self, workflow: Workflow, node_id: str) -> bool:
        if workflow.missing_inputs(node_id):
            return True
        return any(self._is_incomplete(workflow, l.source_node) for l in workflow.incoming(node_id))

    # ------------------------------------------------------------------ execution
    def cancel(self) -> None:
        self._cancel.set()

    def run(self, workflow: Workflow, targets: list[str] | None = None,
            listener: EngineListener | None = None) -> dict[str, str]:
        """Run ``targets`` (default: every node) and what they need. Returns final states."""
        listener = listener or EngineListener()
        self._cancel.clear()
        if targets is None:
            wanted = set(workflow.nodes)
        else:
            wanted = set(targets)
            for target in targets:
                wanted.update(workflow.upstream(target))
        states: dict[str, str] = {}
        for node_id in workflow.topological_order(wanted):
            if self._cancel.is_set():
                states[node_id] = CANCELLED
                continue
            if self._is_incomplete(workflow, node_id):
                states[node_id] = INCOMPLETE
                listener.node_finished(node_id, INCOMPLETE, None)
                continue
            if any(states.get(l.source_node) in (ERROR, CANCELLED) for l in workflow.incoming(node_id)):
                states[node_id] = CANCELLED
                listener.node_finished(node_id, CANCELLED, None)
                continue
            if self.state(workflow, node_id) == DONE:
                states[node_id] = DONE
                listener.node_finished(node_id, DONE, None)
                continue
            states[node_id] = self._run_node(workflow, node_id, listener)
        return states

    def _run_node(self, workflow: Workflow, node_id: str, listener: EngineListener) -> str:
        node = workflow.nodes[node_id]
        inputs = {}
        for link in workflow.incoming(node_id):
            inputs[link.target_port] = self._cache[link.source_node].outputs[link.source_port]
        context = RunContext(
            progress=lambda fraction, message: listener.node_progress(node_id, fraction, message),
            log=lambda message: listener.node_log(node_id, message),
            cancelled=self._cancel.is_set,
        )
        listener.node_started(node_id)
        started = time.perf_counter()
        try:
            outputs = node.node_type.run(inputs, node.params, context)
            context.check_cancelled()
        except Cancelled:
            self._errors.pop(node_id, None)
            listener.node_finished(node_id, CANCELLED, None)
            return CANCELLED
        except Exception as exc:  # noqa: BLE001 - reported to the user, never swallowed silently
            message = f"{type(exc).__name__}: {exc}"
            self._errors[node_id] = message
            listener.node_log(node_id, traceback.format_exc())
            listener.node_finished(node_id, ERROR, message)
            return ERROR
        missing = [p.name for p in node.node_type.outputs if p.name not in outputs]
        if missing:
            message = f"el nodo no produjo las salidas {missing}"
            self._errors[node_id] = message
            listener.node_finished(node_id, ERROR, message)
            return ERROR
        self._errors.pop(node_id, None)
        self._cache[node_id] = _CacheEntry(self.signature(workflow, node_id), outputs,
                                           time.perf_counter() - started)
        listener.node_finished(node_id, DONE, None)
        return DONE
