"""Run a workflow headless (no interface).

    qml_env/Scripts/python.exe -B -m src.workbench.run path/to/flow.qmlflow
    qml_env/Scripts/python.exe -B -m src.workbench.run --template clasificacion-qml
"""

from __future__ import annotations

import argparse
import sys

from .model import DONE, Engine, EngineListener, Workflow, load_template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("flow", nargs="?", help="workflow file (.qmlflow)")
    group.add_argument("--template", help="template id")
    args = parser.parse_args(argv)
    workflow = load_template(args.template) if args.template else Workflow.load(args.flow)
    engine = Engine()

    def finished(node_id, state, error):
        title = workflow.nodes[node_id].display_title
        print(f"[{state}] {title}" + (f": {error}" if error else ""))

    listener = EngineListener(node_log=lambda node_id, message: print(f"    {message}"),
                              node_finished=finished)
    states = engine.run(workflow, listener=listener)
    for node_id in workflow.nodes:
        outputs = engine.outputs(workflow, node_id) or {}
        report = outputs.get("report")
        if report is not None and workflow.nodes[node_id].type_id == "results":
            print(f"\n{report.title}")
            for split, metrics in report.metrics.items():
                print(f"  {split:<10} accuracy={metrics['accuracy']:.4f}  mse={metrics['mse']:.4f}")
            for name, values in report.baselines.items():
                print(f"  baseline {name}: test accuracy={values['test']:.4f}")
    return 0 if all(state == DONE for state in states.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
