"""Visual theme of the Workbench.

The look follows the IBM Carbon design language used by the IBM Quantum
Composer (layout and colour roles only, not its content): dark grey panels
separated by thin borders, a blue primary action, coloured square tiles for the
operations and IBM Plex Sans (Segoe UI as fallback). A light variant is
included and the user can switch at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont


@dataclass(frozen=True)
class Theme:
    name: str
    background: str      # panels and window
    canvas: str          # node canvas
    layer: str           # cards, inputs
    layer_hover: str
    border: str
    text: str
    text_secondary: str
    primary: str
    primary_hover: str
    grid: str
    selection: str
    code_background: str


DARK = Theme(
    name="dark",
    background="#1B1E24",
    canvas="#15171C",
    layer="#252932",
    layer_hover="#30353F",
    border="#343A46",
    text="#E6E8EB",
    text_secondary="#9AA3AF",
    primary="#5B8CFF",
    primary_hover="#7AA2FF",
    grid="#2A2F38",
    selection="#8FB2FF",
    code_background="#15171C",
)

LIGHT = Theme(
    name="light",
    background="#F5F6F8",
    canvas="#FFFFFF",
    layer="#FFFFFF",
    layer_hover="#EBEEF3",
    border="#D5D9E0",
    text="#1B1E24",
    text_secondary="#5F6B7A",
    primary="#3D6BFF",
    primary_hover="#2952E0",
    grid="#E4E7EC",
    selection="#3D6BFF",
    code_background="#FAFBFC",
)

_current = DARK


def theme() -> Theme:
    return _current


def set_theme(name: str) -> Theme:
    global _current
    _current = LIGHT if name == "light" else DARK
    return _current


# Bahnschrift (DIN-style, ships with Windows 10/11) gives the interface a technical,
# distinctive look; Cascadia Code is used for code and logs.
UI_FAMILIES = ["Bahnschrift", "IBM Plex Sans", "Segoe UI", "Helvetica Neue", "Arial"]
MONO_FAMILIES = ["Cascadia Code", "Cascadia Mono", "IBM Plex Mono", "Consolas", "Courier New"]


def ui_font(point_size: float = 10, bold: bool = False) -> QFont:
    font = QFont()
    font.setFamilies(UI_FAMILIES)
    font.setPointSizeF(point_size)
    font.setWeight(QFont.DemiBold if bold else QFont.Normal)
    return font


def mono_font(point_size: float = 9.5) -> QFont:
    font = QFont()
    font.setFamilies(MONO_FAMILIES)
    font.setPointSizeF(point_size)
    return font


def configure_matplotlib() -> None:
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in UI_FAMILIES if f in available] + ["DejaVu Sans"]
    matplotlib.rcParams["font.family"] = families


# Operation tiles: colour and short code per category (like gate tiles).
CATEGORY_COLORS = {
    "Datos": QColor("#4FC3F7"),
    "Modelos": QColor("#B388FF"),
    "Evaluación": QColor("#FFB74D"),
    "Salida": QColor("#81C784"),
}

NODE_CODES = {"dataset": "DS", "qml": "QML", "evaluation": "EV", "results": "RES"}

STATE_COLORS = {
    "pending": QColor("#A2A9B0"),
    "incomplete": QColor("#F1C21B"),
    "running": QColor("#4589FF"),
    "done": QColor("#42BE65"),
    "error": QColor("#FA4D56"),
    "cancelled": QColor("#878D96"),
}

STATE_LABELS = {
    "pending": "Pendiente",
    "incomplete": "Faltan entradas",
    "running": "Ejecutando…",
    "done": "Hecho",
    "error": "Error",
    "cancelled": "Cancelado",
}

PORT_COLORS = {
    "Dataset": QColor("#4FC3F7"),
    "TrainingResult": QColor("#B388FF"),
    "EvaluationReport": QColor("#FFB74D"),
}

NODE_WIDTH = 236
HEADER_HEIGHT = 46
PORT_SPACING = 26
PORT_RADIUS = 6
TILE_SIZE = 30


def node_code(type_id: str) -> str:
    return NODE_CODES.get(type_id, type_id[:3].upper())


def stylesheet(t: Theme) -> str:
    """Qt style sheet for the whole application."""
    return f"""
    QWidget {{ background: {t.background}; color: {t.text}; font-family: "Bahnschrift", "Segoe UI"; }}
    QPlainTextEdit, QTextEdit {{ font-family: "Cascadia Code", "Consolas"; }}
    #TopBar QLabel#AppTitle {{ font-size: 12pt; letter-spacing: 0.5px; }}
    QMainWindow::separator {{ background: {t.border}; width: 1px; height: 1px; }}
    QSplitter::handle {{ background: {t.border}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}
    #TopBar {{ background: {t.background}; border-bottom: 1px solid {t.border}; }}
    #TopBar QLabel#AppTitle {{ font-weight: 600; }}
    #TopBar QLabel#ProjectName {{ font-size: 11pt; padding: 0 12px; border-right: 1px solid {t.border}; }}
    #TopBar QToolButton {{ background: transparent; border: none; padding: 6px 12px; color: {t.text}; }}
    #TopBar QToolButton:hover {{ background: {t.layer}; }}
    #TopBar QToolButton::menu-indicator {{ image: none; }}
    QPushButton#PrimaryButton {{ background: {t.primary}; color: #FFFFFF; border: none; border-radius: 0; padding: 0 28px; font-weight: 600;
        min-height: 47px; text-align: left; font-size: 10pt; }}
    QPushButton#PrimaryButton:hover {{ background: {t.primary_hover}; }}
    QPushButton#PrimaryButton:disabled {{ background: {t.layer}; color: {t.text_secondary}; }}
    QPushButton#GhostButton {{ border-radius: 0; background: transparent; color: {"#78A9FF" if t.name == "dark" else t.primary};
        border: none; padding: 0 14px; min-height: 47px; }}
    QPushButton#GhostButton:hover {{ background: {t.layer}; }}
    #PanelHeader {{ border-bottom: 1px solid {t.border}; }}
    #PanelHeader QLabel {{ font-weight: 600; background: transparent; }}
    #PanelHeader QComboBox {{ border: none; font-weight: 600; background: transparent; padding: 2px 6px; }}
    #PanelHeader QComboBox QAbstractItemView {{ background: {t.layer}; selection-background-color: {t.layer_hover}; }}
    #PanelHeader QToolButton, #CanvasToolbar QToolButton {{ background: {t.layer}; border: 1px solid {t.border};
        border-radius: 4px; padding: 5px 12px; margin: 0 2px; color: {t.text}; }}
    #PanelHeader QToolButton:hover, #CanvasToolbar QToolButton:hover {{ background: {t.layer_hover};
        border-color: {t.primary}; }}
    #PanelHeader QToolButton:pressed, #CanvasToolbar QToolButton:pressed {{ background: {t.primary}; color: #FFFFFF; }}
    #CanvasToolbar QToolButton:checked {{ background: {t.primary}; border-color: {t.primary}; color: #FFFFFF; }}
    #CanvasToolbar QToolButton#DangerButton:hover {{ border-color: #F06A6A; color: #F06A6A; }}
    #CanvasToolbar {{ border-bottom: 1px solid {t.border}; }}
    #CanvasToolbar QLabel {{ color: {t.text_secondary}; background: transparent; }}
    QToolButton#Tile {{ border: none; color: #121619; font-weight: 600; font-size: 8pt; padding: 0; }}
    QLabel#CategoryLabel {{ color: {t.text_secondary}; font-size: 8pt; padding-top: 6px; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {t.layer}; border: none;
        border-bottom: 1px solid {t.text_secondary}; padding: 4px 6px; min-height: 20px;
        selection-background-color: {t.primary}; }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 2px solid {"#FFFFFF" if t.name == "dark" else t.primary}; }}
    QComboBox QAbstractItemView {{ background: {t.layer}; border: 1px solid {t.border};
        selection-background-color: {t.layer_hover}; selection-color: {t.text}; }}
    QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {t.text}; background: {t.background}; }}
    QCheckBox::indicator:checked {{ background: {t.text}; }}
    QGroupBox {{ border: none; border-top: 1px solid {t.border}; margin-top: 18px; padding-top: 8px;
        font-weight: 600; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 0px; padding: 0 0 2px 0; color: {t.text_secondary}; }}
    QPushButton {{ background: {t.layer}; border: 1px solid {t.border}; border-radius: 4px; padding: 7px 16px; }}
    QPushButton:hover {{ background: {t.layer_hover}; border-color: {t.primary}; }}
    QPushButton:default {{ background: {t.primary}; color: #FFFFFF; }}
    QPushButton:default:hover {{ background: {t.primary_hover}; }}
    QPushButton:disabled {{ color: {t.text_secondary}; }}
    QPlainTextEdit, QTextEdit {{ background: {t.code_background}; border: none; }}
    QTableWidget {{ background: {t.background}; gridline-color: {t.border}; border: none; }}
    QHeaderView::section {{ background: {t.layer}; color: {t.text}; border: none; padding: 6px; font-weight: 600; }}
    QScrollArea {{ border: none; }}
    QScrollBar:vertical {{ background: {t.background}; width: 10px; }}
    QScrollBar::handle:vertical {{ background: {t.border}; min-height: 30px; }}
    QScrollBar:horizontal {{ background: {t.background}; height: 10px; }}
    QScrollBar::handle:horizontal {{ background: {t.border}; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QMenu {{ background: {t.layer}; border: 1px solid {t.border}; padding: 4px 0; }}
    QMenu::item {{ padding: 6px 28px 6px 16px; }}
    QMenu::item:selected {{ background: {t.layer_hover}; }}
    QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 0; }}
    QStatusBar {{ border-top: 1px solid {t.border}; }}
    QStatusBar QToolButton {{ background: transparent; border: 1px solid {t.border}; border-radius: 4px;
        padding: 3px 12px; margin: 2px; }}
    QStatusBar QToolButton:hover {{ border-color: {t.primary}; }}
    QStatusBar QToolButton:checked {{ background: {t.primary}; border-color: {t.primary}; color: #FFFFFF; }}
    QPushButton#TileRow {{ background: transparent; border: 1px solid transparent; border-radius: 4px;
        text-align: left; padding: 4px 8px 4px 4px; }}
    QPushButton#TileRow:hover {{ background: {t.layer}; border-color: {t.border}; }}
    QProgressBar {{ background: {t.layer}; border: none; max-height: 6px; }}
    QProgressBar::chunk {{ background: {t.primary}; }}
    QTabWidget::pane {{ border: none; border-top: 1px solid {t.border}; }}
    QTabBar::tab {{ background: transparent; padding: 8px 16px; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ border-bottom: 2px solid {t.primary}; font-weight: 600; }}
    QToolTip {{ background: {t.layer}; color: {t.text}; border: 1px solid {t.border}; padding: 4px; }}
    """
