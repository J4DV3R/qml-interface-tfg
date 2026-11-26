"""Node canvas: nodes, typed ports and links drawn with QGraphicsView.

The canvas never changes the model directly: every edit goes through the
controller, and the canvas redraws itself from controller events. Nodes are
drawn like the operation tiles of a circuit composer: a coloured square with
the node code, the title and the typed ports.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView, QMenu,
                               QStyleOptionGraphicsItem, QWidget)

from ..controller import WorkflowController
from ..model.data import PORT_TYPE_LABELS
from . import style
from .style import (CATEGORY_COLORS, HEADER_HEIGHT, NODE_WIDTH, PORT_COLORS, PORT_RADIUS, PORT_SPACING,
                    STATE_COLORS, STATE_LABELS, TILE_SIZE, node_code, ui_font)

NODE_MIME = "application/x-qml-workbench-node"
GRID = 24


class PortItem(QGraphicsItem):
    def __init__(self, node_item: "NodeItem", name: str, label: str, data_type: str, is_output: bool):
        super().__init__(node_item)
        self.node_item, self.name, self.label = node_item, name, label
        self.data_type, self.is_output = data_type, is_output
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{label} · {PORT_TYPE_LABELS.get(data_type, data_type)}")
        self._hover = False

    def boundingRect(self) -> QRectF:
        r = PORT_RADIUS + 4
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        color = PORT_COLORS.get(self.data_type, QColor("#A2A9B0"))
        painter.setPen(QPen(QColor(style.theme().canvas), 2))
        painter.setBrush(QBrush(color.lighter(125) if self._hover else color))
        r = PORT_RADIUS + (2 if self._hover else 0)
        painter.drawEllipse(QPointF(0, 0), r, r)

    def hoverEnterEvent(self, event) -> None:
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event) -> None:
        self._hover = False
        self.update()

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))


class NodeItem(QGraphicsItem):
    def __init__(self, scene: "WorkflowScene", node_id: str):
        super().__init__()
        self.workflow_scene = scene
        self.node_id = node_id
        self.state = "pending"
        self.progress: float | None = None
        self.message = ""
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        node = scene.controller.workflow.nodes[node_id]
        node_type = node.node_type
        self.title = node.display_title
        self.type_id = node.type_id
        self.category = node_type.category
        self.setToolTip(node_type.description)
        self.inputs = [PortItem(self, p.name, p.label, p.data_type, False) for p in node_type.inputs]
        self.outputs = [PortItem(self, p.name, p.label, p.data_type, True) for p in node_type.outputs]
        rows = max(len(self.inputs), len(self.outputs), 1)
        self.height = HEADER_HEIGHT + rows * PORT_SPACING + 22
        for i, port in enumerate(self.inputs):
            port.setPos(0, HEADER_HEIGHT + PORT_SPACING * (i + 0.5))
        for i, port in enumerate(self.outputs):
            port.setPos(NODE_WIDTH, HEADER_HEIGHT + PORT_SPACING * (i + 0.5))
        self.setPos(*node.position)

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_RADIUS - 6, -6, NODE_WIDTH + 2 * PORT_RADIUS + 12, self.height + 12)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        t = style.theme()
        painter.setRenderHint(QPainter.Antialiasing)
        accent = CATEGORY_COLORS.get(self.category, QColor("#A2A9B0"))
        state_color = STATE_COLORS.get(self.state, QColor("#A2A9B0"))
        body = QRectF(0, 0, NODE_WIDTH, self.height)
        # Card (square corners, Carbon style) with a selection outline.
        painter.setPen(QPen(QColor(t.selection), 2) if self.isSelected() else QPen(QColor(t.border), 1))
        painter.setBrush(QColor(t.layer))
        painter.drawRect(body)
        # Operation tile, like a gate in a circuit composer.
        tile = QRectF(8, (HEADER_HEIGHT - TILE_SIZE) / 2, TILE_SIZE + 8, TILE_SIZE)
        painter.setPen(Qt.NoPen)
        painter.setBrush(accent)
        painter.drawRect(tile)
        painter.setPen(QColor("#121619"))
        painter.setFont(ui_font(8, bold=True))
        painter.drawText(tile, Qt.AlignCenter, node_code(self.type_id))
        painter.setPen(QColor(t.text))
        painter.setFont(ui_font(10.5, bold=True))
        title_rect = QRectF(tile.right() + 8, 0, NODE_WIDTH - tile.right() - 14, HEADER_HEIGHT)
        painter.drawText(title_rect, Qt.AlignVCenter | Qt.AlignLeft,
                         painter.fontMetrics().elidedText(self.title, Qt.ElideRight, int(title_rect.width())))
        painter.setPen(QPen(QColor(t.border), 1))
        painter.drawLine(QPointF(0, HEADER_HEIGHT), QPointF(NODE_WIDTH, HEADER_HEIGHT))
        painter.setFont(ui_font(9))
        painter.setPen(QColor(t.text_secondary))
        for port in self.inputs:
            painter.drawText(QRectF(12, port.y() - 9, NODE_WIDTH / 2, 18), Qt.AlignVCenter | Qt.AlignLeft, port.label)
        for port in self.outputs:
            painter.drawText(QRectF(NODE_WIDTH / 2 - 12, port.y() - 9, NODE_WIDTH / 2, 18),
                             Qt.AlignVCenter | Qt.AlignRight, port.label)
        # Status line: dot + text, and a progress bar while running.
        status = STATE_LABELS.get(self.state, self.state)
        if self.state == "running" and self.progress is not None:
            status = f"{int(self.progress * 100)} % · {self.message}" if self.message else f"{int(self.progress * 100)} %"
        painter.setPen(Qt.NoPen)
        painter.setBrush(state_color)
        painter.drawEllipse(QPointF(14, self.height - 12), 3.5, 3.5)
        painter.setPen(QColor(t.text_secondary))
        painter.drawText(QRectF(24, self.height - 21, NODE_WIDTH - 32, 18), Qt.AlignVCenter | Qt.AlignLeft,
                         painter.fontMetrics().elidedText(status, Qt.ElideRight, NODE_WIDTH - 32))
        if self.state == "running" and self.progress is not None:
            painter.fillRect(QRectF(0, self.height - 3, NODE_WIDTH * self.progress, 3), QColor(t.primary))
        elif self.state in ("done", "error"):
            painter.fillRect(QRectF(0, self.height - 2, NODE_WIDTH, 2), state_color)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.workflow_scene.snap:
            return QPointF(round(value.x() / GRID) * GRID, round(value.y() / GRID) * GRID)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.workflow_scene.update_links_for(self.node_id)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        node = self.workflow_scene.controller.workflow.nodes.get(self.node_id)
        if node is not None and (abs(node.position[0] - self.x()) > 0.5 or abs(node.position[1] - self.y()) > 0.5):
            self.workflow_scene.controller.move_node(self.node_id, (self.x(), self.y()))

    def mouseDoubleClickEvent(self, event) -> None:
        self.workflow_scene.node_activated.emit(self.node_id)

    def set_state(self, state: str) -> None:
        self.state = state
        if state != "running":
            self.progress, self.message = None, ""
        self.update()

    def set_progress(self, fraction: float, message: str) -> None:
        self.progress, self.message = fraction, message
        self.update()

    def port(self, name: str, output: bool) -> PortItem | None:
        return next((p for p in (self.outputs if output else self.inputs) if p.name == name), None)


class LinkItem(QGraphicsPathItem):
    def __init__(self, link_id: str | None, source: PortItem, target: PortItem | None = None):
        super().__init__()
        self.link_id, self.source, self.target = link_id, source, target
        self.color = PORT_COLORS.get(source.data_type, QColor("#A2A9B0"))
        self.setPen(QPen(self.color, 2))
        self.setZValue(-1)
        if link_id is not None:
            self.setFlag(QGraphicsItem.ItemIsSelectable)
            self.setToolTip(PORT_TYPE_LABELS.get(source.data_type, source.data_type))
        self.update_path()

    def update_path(self, end: QPointF | None = None) -> None:
        start = self.source.scene_center()
        if end is not None:
            stop = end
        elif self.target is not None:
            stop = self.target.scene_center()
        else:  # temporary link being dragged, before the first mouse move
            stop = start
        dx = max(60.0, abs(stop.x() - start.x()) * 0.5)
        path = QPainterPath(start)
        path.cubicTo(start + QPointF(dx, 0), stop - QPointF(dx, 0), stop)
        self.setPath(path)

    def paint(self, painter, option, widget=None):
        pen = QPen(self.pen())
        if self.isSelected():
            pen.setColor(QColor(style.theme().selection))
            pen.setWidthF(3.5)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pen)
        painter.drawPath(self.path())


class WorkflowScene(QGraphicsScene):
    node_activated = Signal(str)       # double click
    message = Signal(str)              # status-bar message

    def __init__(self, controller: WorkflowController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.nodes: dict[str, NodeItem] = {}
        self.links: dict[str, LinkItem] = {}
        self._drag: LinkItem | None = None
        self.snap = True
        self.setSceneRect(-2000, -2000, 6000, 4000)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        t = style.theme()
        painter.fillRect(rect, QColor(t.canvas))
        painter.setPen(QPen(QColor(t.grid), 1.6))
        left = int(rect.left()) - int(rect.left()) % GRID
        top = int(rect.top()) - int(rect.top()) % GRID
        for x in range(left, int(rect.right()) + GRID, GRID):
            for y in range(top, int(rect.bottom()) + GRID, GRID):
                painter.drawPoint(x, y)

    # ------------------------------------------------------------------ building
    def rebuild(self) -> None:
        selected = {n.node_id for n in self.nodes.values() if n.isSelected()}
        states = {node_id: item.state for node_id, item in self.nodes.items()}
        # Selection is restored below; do not notify transient changes (keeps the
        # settings form, and its focus, intact while editing).
        blocked = self.blockSignals(True)
        try:
            self._rebuild(selected, states)
        finally:
            self.blockSignals(blocked)

    def _rebuild(self, selected: set[str], states: dict[str, str]) -> None:
        self.clear()
        self.nodes, self.links, self._drag = {}, {}, None
        workflow = self.controller.workflow
        for node_id in workflow.nodes:
            item = NodeItem(self, node_id)
            item.set_state(states.get(node_id, self.controller.state(node_id)))
            self.addItem(item)
            self.nodes[node_id] = item
            item.setSelected(node_id in selected)
        for link in workflow.links.values():
            source = self.nodes[link.source_node].port(link.source_port, True)
            target = self.nodes[link.target_node].port(link.target_port, False)
            item = LinkItem(link.link_id, source, target)
            self.addItem(item)
            self.links[link.link_id] = item

    def update_links_for(self, node_id: str) -> None:
        for item in self.links.values():
            if node_id in (item.source.node_item.node_id, item.target.node_item.node_id):
                item.update_path()

    def selected_node_ids(self) -> list[str]:
        return [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]

    def delete_selection(self) -> None:
        links = [i.link_id for i in self.selectedItems() if isinstance(i, LinkItem) and i.link_id]
        nodes = self.selected_node_ids()
        try:
            for link_id in links:
                if link_id in self.controller.workflow.links:
                    self.controller.disconnect(link_id)
            if nodes:
                self.controller.remove_nodes(nodes)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.message.emit(str(exc))

    # ------------------------------------------------------------------ link dragging
    def _port_at(self, pos: QPointF) -> PortItem | None:
        for item in self.items(pos):
            if isinstance(item, PortItem):
                return item
        return None

    def mousePressEvent(self, event) -> None:
        port = self._port_at(event.scenePos())
        if port is not None and event.button() == Qt.LeftButton:
            if port.is_output:
                self._drag = LinkItem(None, port)
                self._drag.setPen(QPen(PORT_COLORS.get(port.data_type, QColor("#A2A9B0")), 2, Qt.DashLine))
                self.addItem(self._drag)
                self._drag.update_path(event.scenePos())
                return
            # Dragging from a connected input detaches the link and lets the user move it.
            for link in list(self.controller.workflow.incoming(port.node_item.node_id)):
                if link.target_port == port.name:
                    source_node, source_port = link.source_node, link.source_port
                    try:
                        self.controller.disconnect(link.link_id)  # the scene is rebuilt here
                    except Exception as exc:  # noqa: BLE001
                        self.message.emit(str(exc))
                        return
                    source = self.nodes[source_node].port(source_port, True)
                    self._drag = LinkItem(None, source)
                    self.addItem(self._drag)
                    self._drag.update_path(event.scenePos())
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            self._drag.update_path(event.scenePos())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        source = self._drag.source
        self.removeItem(self._drag)
        self._drag = None
        target = self._port_at(event.scenePos())
        if target is None or target.is_output:
            return
        args = (source.node_item.node_id, source.name, target.node_item.node_id, target.name)
        reason = self.controller.can_connect(*args)
        if reason:
            self.message.emit(reason)
            return
        try:
            self.controller.connect(*args)
        except Exception as exc:  # noqa: BLE001
            self.message.emit(str(exc))

    # ------------------------------------------------------------------ context menu
    def contextMenuEvent(self, event) -> None:
        item = next((i for i in self.items(event.scenePos()) if isinstance(i, (NodeItem, LinkItem))), None)
        menu = QMenu()
        if isinstance(item, NodeItem):
            if not item.isSelected():
                self.clearSelection()
                item.setSelected(True)
            open_action = menu.addAction("Ver resultados")
            run_action = menu.addAction("Ejecutar hasta aquí")
            menu.addSeparator()
            delete_action = menu.addAction("Eliminar")
            chosen = menu.exec(event.screenPos())
            if chosen == open_action:
                self.node_activated.emit(item.node_id)
            elif chosen == run_action:
                self.controller.run([item.node_id])
            elif chosen == delete_action:
                self.delete_selection()
        elif isinstance(item, LinkItem) and item.link_id:
            delete_action = menu.addAction("Eliminar conexión")
            if menu.exec(event.screenPos()) == delete_action:
                self.controller.disconnect(item.link_id)


class WorkflowView(QGraphicsView):
    def __init__(self, scene: WorkflowScene, parent=None):
        super().__init__(scene, parent)
        self.workflow_scene = scene
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setAcceptDrops(True)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        delete = QAction(self)
        delete.setShortcuts([Qt.Key_Delete, Qt.Key_Backspace])
        delete.triggered.connect(scene.delete_selection)
        self.addAction(delete)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
        else:
            super().wheelEvent(event)

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    def fit_all(self) -> None:
        rect = self.workflow_scene.itemsBoundingRect().adjusted(-60, -60, 60, 60)
        if rect.isValid():
            self.fitInView(rect, Qt.KeepAspectRatio)
            if self.transform().m11() > 1.1:
                self.resetTransform()
                self.centerOn(rect.center())

    # Drops from the operations panel create nodes.
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(NODE_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(NODE_MIME):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasFormat(NODE_MIME):
            return super().dropEvent(event)
        type_id = bytes(event.mimeData().data(NODE_MIME)).decode("utf-8")
        pos = self.mapToScene(event.position().toPoint())
        try:
            self.workflow_scene.controller.add_node(type_id, (pos.x() - NODE_WIDTH / 2, pos.y() - 20))
        except Exception as exc:  # noqa: BLE001
            self.workflow_scene.message.emit(str(exc))
        event.acceptProposedAction()


__all__ = ["NODE_MIME", "NodeItem", "WorkflowScene", "WorkflowView"]
