"""Project templates: ordinary, editable workflows stored as ``*.qmlflow`` files."""

from __future__ import annotations

import json
from pathlib import Path

from .workflow import Workflow

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


def list_templates() -> list[dict[str, str]]:
    templates = []
    for path in sorted(TEMPLATE_DIR.glob("*.qmlflow")):
        data = json.loads(path.read_text(encoding="utf-8"))
        templates.append({"id": path.stem, "name": data.get("name", path.stem),
                          "description": data.get("description", ""), "path": str(path)})
    return templates


def load_template(template_id: str) -> Workflow:
    path = TEMPLATE_DIR / f"{template_id}.qmlflow"
    if not path.is_file():
        raise FileNotFoundError(f"no existe la plantilla {template_id!r}")
    return Workflow.load(path)
