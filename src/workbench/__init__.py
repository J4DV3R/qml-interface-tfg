"""QML Workbench: node-based workflow platform for quantum machine learning.

Architecture (Model-View-Controller):

* ``src.workbench.model``: workflow graph, typed ports, node and algorithm
  registries, execution engine and persistence. It never imports Qt, so it can
  be used and tested headless (``python -m src.workbench.run <flow.qmlflow>``).
* ``src.workbench.controller``: translates user actions into model commands and
  model events into notifications for the view. Also Qt-free.
* ``src.workbench.view``: PySide6 desktop interface (canvas, palette, settings
  forms generated from parameter schemas, results windows).

Projects start from templates (``src/workbench/templates/*.qmlflow``) that are
ordinary, editable workflows.
"""
