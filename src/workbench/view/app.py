"""Desktop entry point: ``python -m src.workbench``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QML Workbench")
    parser.add_argument("flow", nargs="?", help="proyecto .qmlflow que abrir")
    parser.add_argument("--template", default="clasificacion-qml", help="plantilla inicial")
    args = parser.parse_args(argv)

    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("QML Workbench")
    window = MainWindow(template=None if args.flow else args.template)
    if args.flow:
        window.controller.open(args.flow)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
