"""The workflow graph: node instances, typed links, validation and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

from .node import get_node_type
from .parameters import ParameterError

FORMAT = "qml-workbench-flow"
FORMAT_VERSION = 1


class WorkflowError(ValueError):
    """Invalid structural change of the workflow."""


@dataclass
class NodeInstance:
    node_id: str
    type_id: str
    params: dict[str, Any]
    position: tuple[float, float] = (0.0, 0.0)
    title: str | None = None

    @property
    def node_type(self):
        return get_node_type(self.type_id)

    @property
    def display_title(self) -> str:
        return self.title or self.node_type.title


@dataclass(frozen=True)
class Link:
    link_id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str


@dataclass
class Workflow:
    """Directed acyclic graph of nodes. Emits events to subscribed observers."""

    name: str = "Proyecto sin título"
    nodes: dict[str, NodeInstance] = field(default_factory=dict)
    links: dict[str, Link] = field(default_factory=dict)
    description: str = ""
    _observers: list[Callable[[str, dict[str, Any]], None]] = field(default_factory=list, repr=False)
    _counter: int = 0

    # ------------------------------------------------------------------ observers
    def subscribe(self, observer: Callable[[str, dict[str, Any]], None]) -> None:
        self._observers.append(observer)

    def _emit(self, event: str, **payload: Any) -> None:
        for observer in list(self._observers):
            observer(event, payload)

    def _new_id(self, prefix: str) -> str:
        existing = set(self.nodes) | set(self.links)
        while True:
            self._counter += 1
            candidate = f"{prefix}{self._counter}"
            if candidate not in existing:
                return candidate

    # ------------------------------------------------------------------ nodes
    def add_node(self, type_id: str, position: tuple[float, float] = (0.0, 0.0),
                 params: dict[str, Any] | None = None, node_id: str | None = None,
                 title: str | None = None) -> str:
        node_type = get_node_type(type_id)
        values = node_type.default_parameters()
        values.update(params or {})
        validated = node_type.validate_parameters(values)
        node_id = node_id or self._new_id("n")
        if node_id in self.nodes:
            raise WorkflowError(f"duplicate node id {node_id}")
        self.nodes[node_id] = NodeInstance(node_id, type_id, validated, tuple(position), title)
        self._emit("node_added", node_id=node_id)
        return node_id

    def remove_node(self, node_id: str) -> None:
        self._node(node_id)
        for link_id in [l.link_id for l in self.links.values() if node_id in (l.source_node, l.target_node)]:
            self.disconnect(link_id)
        del self.nodes[node_id]
        self._emit("node_removed", node_id=node_id)

    def set_params(self, node_id: str, params: dict[str, Any]) -> dict[str, Any]:
        node = self._node(node_id)
        validated = node.node_type.validate_parameters({**node.params, **params})
        node.params = validated
        self._emit("params_changed", node_id=node_id)
        return validated

    def move_node(self, node_id: str, position: tuple[float, float]) -> None:
        self._node(node_id).position = (float(position[0]), float(position[1]))
        self._emit("node_moved", node_id=node_id)

    def rename_node(self, node_id: str, title: str | None) -> None:
        self._node(node_id).title = title or None
        self._emit("node_renamed", node_id=node_id)

    def _node(self, node_id: str) -> NodeInstance:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise WorkflowError(f"unknown node {node_id}") from exc

    # ------------------------------------------------------------------ links
    def can_connect(self, source_node: str, source_port: str, target_node: str, target_port: str) -> str | None:
        """Return None if the link is valid, otherwise a human-readable reason."""
        if source_node == target_node:
            return "Un nodo no puede conectarse consigo mismo."
        source, target = self._node(source_node), self._node(target_node)
        try:
            out_spec = source.node_type.output_port(source_port)
        except StopIteration:
            return f"El nodo «{source.display_title}» no tiene la salida {source_port}."
        try:
            in_spec = target.node_type.input_port(target_port)
        except StopIteration:
            return f"El nodo «{target.display_title}» no tiene la entrada {target_port}."
        if out_spec.data_type != in_spec.data_type:
            return (f"Tipos incompatibles: «{out_spec.label}» ({out_spec.data_type}) no puede ir a "
                    f"«{in_spec.label}» ({in_spec.data_type}).")
        if any(l.target_node == target_node and l.target_port == target_port for l in self.links.values()):
            return f"La entrada «{in_spec.label}» ya está conectada."
        if self._reaches(target_node, source_node):
            return "La conexión crearía un ciclo."
        return None

    def connect(self, source_node: str, source_port: str, target_node: str, target_port: str,
                link_id: str | None = None) -> str:
        reason = self.can_connect(source_node, source_port, target_node, target_port)
        if reason:
            raise WorkflowError(reason)
        link_id = link_id or self._new_id("l")
        self.links[link_id] = Link(link_id, source_node, source_port, target_node, target_port)
        self._emit("link_added", link_id=link_id)
        return link_id

    def disconnect(self, link_id: str) -> None:
        if link_id not in self.links:
            raise WorkflowError(f"unknown link {link_id}")
        link = self.links.pop(link_id)
        self._emit("link_removed", link_id=link_id, target_node=link.target_node)

    def _reaches(self, start: str, goal: str) -> bool:
        stack, seen = [start], set()
        while stack:
            current = stack.pop()
            if current == goal:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(l.target_node for l in self.links.values() if l.source_node == current)
        return False

    # ------------------------------------------------------------------ queries
    def incoming(self, node_id: str) -> list[Link]:
        return [l for l in self.links.values() if l.target_node == node_id]

    def outgoing(self, node_id: str) -> list[Link]:
        return [l for l in self.links.values() if l.source_node == node_id]

    def upstream(self, node_id: str) -> list[str]:
        """All ancestors of a node."""
        result, stack = [], [node_id]
        while stack:
            current = stack.pop()
            for link in self.incoming(current):
                if link.source_node not in result:
                    result.append(link.source_node)
                    stack.append(link.source_node)
        return result

    def downstream(self, node_id: str) -> list[str]:
        result, stack = [], [node_id]
        while stack:
            current = stack.pop()
            for link in self.outgoing(current):
                if link.target_node not in result:
                    result.append(link.target_node)
                    stack.append(link.target_node)
        return result

    def topological_order(self, subset: set[str] | None = None) -> list[str]:
        nodes = [n for n in self.nodes if subset is None or n in subset]
        indegree = {n: 0 for n in nodes}
        for link in self.links.values():
            if link.source_node in indegree and link.target_node in indegree:
                indegree[link.target_node] += 1
        ready = sorted(n for n, d in indegree.items() if d == 0)
        order = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for link in sorted(self.outgoing(current), key=lambda l: l.target_node):
                if link.target_node in indegree:
                    indegree[link.target_node] -= 1
                    if indegree[link.target_node] == 0:
                        ready.append(link.target_node)
        if len(order) != len(nodes):
            raise WorkflowError("the workflow contains a cycle")
        return order

    def missing_inputs(self, node_id: str) -> list[str]:
        node = self._node(node_id)
        connected = {l.target_port for l in self.incoming(node_id)}
        return [p.label for p in node.node_type.inputs if p.name not in connected]

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "name": self.name,
            "description": self.description,
            "nodes": [{"id": n.node_id, "type": n.type_id, "title": n.title,
                       "position": list(n.position), "params": n.params} for n in self.nodes.values()],
            "links": [{"id": l.link_id, "source": [l.source_node, l.source_port],
                       "target": [l.target_node, l.target_port]} for l in self.links.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Workflow":
        if data.get("format") != FORMAT:
            raise WorkflowError("not a QML Workbench workflow file")
        if int(data.get("version", 0)) > FORMAT_VERSION:
            raise WorkflowError("workflow file written by a newer version")
        workflow = cls(name=data.get("name", "Proyecto"), description=data.get("description", ""))
        try:
            for node in data.get("nodes", []):
                workflow.add_node(node["type"], tuple(node.get("position", (0, 0))), node.get("params", {}),
                                  node_id=node["id"], title=node.get("title"))
            for link in data.get("links", []):
                workflow.connect(link["source"][0], link["source"][1], link["target"][0], link["target"][1],
                                 link_id=link["id"])
        except (KeyError, ParameterError) as exc:
            raise WorkflowError(f"invalid workflow file: {exc}") from exc
        return workflow

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Workflow":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
