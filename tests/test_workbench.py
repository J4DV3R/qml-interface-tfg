"""QML Workbench: model, phase-C parity, controller and a headless view smoke test."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.workbench.controller import WorkflowController
from src.workbench.model import (DONE, INCOMPLETE, PENDING, Engine, ParameterError, Workflow, WorkflowError,
                                 get_node_type, list_templates, load_template)
from src.workbench.model.algorithms import algorithms
from src.workbench.model.nodes.dataset import load_builtin
from src.workbench.model.preprocessing import split_and_preprocess

ROOT = Path(__file__).resolve().parents[1]


def small_flow(steps=3, optimizer="adam") -> Workflow:
    workflow = Workflow()
    data = workflow.add_node("dataset", params={"source": "iris-binary"}, node_id="d")
    qml = workflow.add_node("qml", params={"optimizer": optimizer, "max_steps": steps}, node_id="q")
    evaluation = workflow.add_node("evaluation", node_id="e")
    results = workflow.add_node("results", node_id="r")
    workflow.connect(data, "data", qml, "data")
    workflow.connect(qml, "result", evaluation, "result")
    workflow.connect(evaluation, "report", results, "report")
    return workflow


class ParameterAndGraphTests(unittest.TestCase):
    def test_split_percentages_must_sum_100(self):
        node = get_node_type("qml")
        values = node.default_parameters()
        values["test_percent"] = 20
        with self.assertRaisesRegex(ParameterError, "sumar 100"):
            node.validate_parameters(values)

    def test_schema_depends_on_selected_optimizer(self):
        node = get_node_type("qml")
        values = node.default_parameters()
        lr = next(p for p in node.parameter_schema(values) if p.name == "learning_rate")
        self.assertTrue(lr.is_visible({**values, "optimizer": "adam"}))
        self.assertFalse(lr.is_visible({**values, "optimizer": "spsa"}))
        self.assertGreaterEqual(len(algorithms()), 1)

    def test_connections_are_type_checked_and_acyclic(self):
        workflow = Workflow()
        d = workflow.add_node("dataset")
        e = workflow.add_node("evaluation")
        q = workflow.add_node("qml")
        self.assertIn("incompatibles", workflow.can_connect(d, "data", e, "result"))
        workflow.connect(d, "data", q, "data")
        self.assertIn("ya está conectada", workflow.can_connect(d, "data", q, "data"))
        with self.assertRaises(WorkflowError):
            workflow.connect(q, "result", q, "data")

    def test_branching_graph_runs_two_models_from_one_dataset(self):
        workflow = small_flow(steps=2)
        q2 = workflow.add_node("qml", params={"optimizer": "spsa", "max_steps": 2})
        e2 = workflow.add_node("evaluation")
        workflow.connect("d", "data", q2, "data")
        workflow.connect(q2, "result", e2, "result")
        states = Engine().run(workflow)
        self.assertTrue(all(state == DONE for state in states.values()), states)

    def test_save_and_load_round_trip(self):
        workflow = small_flow()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flow.qmlflow"
            workflow.save(path)
            loaded = Workflow.load(path)
        self.assertEqual(loaded.to_dict(), workflow.to_dict())

    def test_template_is_an_editable_workflow(self):
        self.assertIn("clasificacion-qml", [t["id"] for t in list_templates()])
        workflow = load_template("clasificacion-qml")
        self.assertEqual(len(workflow.nodes), 4)
        self.assertEqual(len(workflow.links), 3)
        workflow.remove_node("evaluation")
        self.assertEqual(len(workflow.links), 1)


class EngineTests(unittest.TestCase):
    def test_states_caching_and_invalidation(self):
        workflow = small_flow(steps=2)
        engine = Engine()
        self.assertEqual(engine.state(workflow, "q"), PENDING)
        engine.run(workflow)
        self.assertEqual(engine.state(workflow, "r"), DONE)
        first = engine.outputs(workflow, "q")["result"]
        # Re-running reuses the cache (same object).
        engine.run(workflow)
        self.assertIs(engine.outputs(workflow, "q")["result"], first)
        # Changing a parameter invalidates that node and its descendants only.
        workflow.set_params("q", {"max_steps": 3})
        self.assertEqual(engine.state(workflow, "d"), DONE)
        self.assertEqual(engine.state(workflow, "q"), PENDING)
        self.assertEqual(engine.state(workflow, "r"), PENDING)
        workflow.disconnect(next(l for l in workflow.links.values() if l.target_node == "q").link_id)
        self.assertEqual(engine.state(workflow, "r"), INCOMPLETE)

    def test_errors_are_reported_not_raised(self):
        workflow = Workflow()
        workflow.add_node("dataset", params={"source": "csv", "csv_path": "no-existe.csv"}, node_id="d")
        states = Engine().run(workflow)
        self.assertEqual(states["d"], "error")


class PhaseCParityTests(unittest.TestCase):
    """A workflow configured like phase C reproduces the stored runs."""

    def run_case(self, source, dataset, seed, optimizer, encoding, extra):
        path = ROOT / f"results/variational/C-partitions/{dataset}/ds{seed}/{optimizer}-{encoding}/run-{encoding}-{optimizer}-init19.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        workflow = Workflow()
        d = workflow.add_node("dataset", params={"source": source})
        q = workflow.add_node("qml", params={"split_seed": seed, "optimizer": optimizer, "encoding": encoding,
                                             "max_steps": 100, "init_seed": 19, "execution_seed": 101, **extra})
        e = workflow.add_node("evaluation")
        workflow.connect(d, "data", q, "data")
        workflow.connect(q, "result", e, "result")
        engine = Engine()
        engine.run(workflow)
        result = engine.outputs(workflow, q)["result"]
        report = engine.outputs(workflow, e)["report"]
        self.assertEqual(result.checkpoint["step"], stored["training"]["best_step"])
        self.assertAlmostEqual(result.checkpoint["validation_loss"], stored["training"]["best_validation_loss"],
                               places=12)
        self.assertEqual(report.metrics["test"]["accuracy"], stored["test"]["metrics"]["accuracy"])
        # Layer-2 weights w1..w3 do not influence the output (log section 37); compare the live ones.
        live = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        ours, theirs = np.array(result.model_state["best_weights"]), np.array(stored["training"]["best_weights"])
        for index in live:
            self.assertAlmostEqual(ours[index], theirs[index], places=9)

    def test_iris_adam_angle(self):
        self.run_case("iris-binary", "iris", 1, "adam", "angle", {"learning_rate": 0.1})

    def test_breast_cancer_sequential_phase_with_pca(self):
        self.run_case("breast-cancer", "breast-cancer", 3, "sequential", "phase", {})

    def test_partitions_match_the_historical_loader(self):
        from experiments.run_variational_benchmark import load_partitions

        for source, dataset in (("iris-binary", "iris"), ("breast-cancer", "breast-cancer")):
            data, _ = load_partitions(dataset, 5, 4)
            splits = split_and_preprocess(load_builtin(source), train_percent=70, validation_percent=15, seed=5,
                                          n_features=4, reduction="auto")
            for name in ("train", "validation", "test"):
                np.testing.assert_array_equal(splits.get(name)[0], data[name][0])
                np.testing.assert_array_equal(splits.get(name)[1], data[name][1])


class ControllerTests(unittest.TestCase):
    def test_events_run_and_cancel(self):
        controller = WorkflowController(run_in_background=False)
        events = []
        controller.subscribe(lambda name, payload: events.append((name, payload)))
        controller.new_from_template("clasificacion-qml")
        controller.set_params("qml", {"max_steps": 2})
        controller.run()
        finished = [p for n, p in events if n == "run_finished"][-1]
        self.assertTrue(all(s == DONE for s in finished["states"].values()))
        self.assertIsNotNone(controller.outputs("results")["report"])
        with self.assertRaises(ParameterError):
            controller.set_params("qml", {"train_percent": 90})
        with tempfile.TemporaryDirectory() as tmp:
            saved = controller.save(Path(tmp) / "p")
            self.assertEqual(saved.suffix, ".qmlflow")
            self.assertFalse(controller.dirty)

    def test_cancellation_stops_training(self):
        controller = WorkflowController(run_in_background=True)
        controller.new_from_template("clasificacion-qml")
        controller.set_params("qml", {"optimizer": "adam", "max_steps": 5000})
        controller.run()
        controller.cancel()
        controller.wait(60)
        self.assertFalse(controller.is_running)
        self.assertNotEqual(controller.state("qml"), DONE)

    def test_training_callback_does_not_change_results(self):
        from src.fast_qcnn import FastQCNNConfig
        from src.variational_training import FastEvaluator, VariationalTrainingConfig, train_variational_qcnn

        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 1, (10, 4)), np.where(rng.random(10) > 0.5, 1.0, -1.0)
        config = VariationalTrainingConfig(optimizer="spsa", max_steps=6, shots=50)
        args = (x, y, x[:4], y[:4], np.zeros((2, 4)), FastEvaluator(FastQCNNConfig(4)), config)
        plain = train_variational_qcnn(*args, np.random.default_rng(3))
        seen = []
        with_cb = train_variational_qcnn(*args, np.random.default_rng(3), callback=lambda r: seen.append(r["step"]))
        np.testing.assert_array_equal(plain.best_weights, with_cb.best_weights)
        self.assertEqual(seen, list(range(7)))
        stopped = train_variational_qcnn(*args, np.random.default_rng(3), callback=lambda r: r["step"] < 2)
        self.assertEqual((stopped.stop_reason, stopped.steps_completed), ("cancelled", 2))


@unittest.skipUnless(__import__("importlib").util.find_spec("PySide6"), "PySide6 not installed")
class ViewSmokeTests(unittest.TestCase):
    def test_main_window_builds_runs_and_shows_results(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import time

        from PySide6.QtWidgets import QApplication

        from src.workbench.view.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        self.assertEqual(set(window.scene.nodes), {"dataset", "qml", "evaluation", "results"})
        self.assertEqual(len(window.scene.links), 3)
        window.scene.nodes["qml"].setSelected(True)
        app.processEvents()
        self.assertEqual(window.settings.node_id, "qml")
        window.settings._fields["max_steps"].input.setValue(2)
        window.settings.apply()
        self.assertEqual(window.controller.workflow.nodes["qml"].params["max_steps"], 2)
        window.controller.run()
        deadline = time.time() + 120
        while (window.controller.is_running or window.current_report is None) and time.time() < deadline:
            app.processEvents()
            time.sleep(0.02)
        self.assertEqual({i.state for i in window.scene.nodes.values()}, {"done"})
        # Results are shown in the bottom panels; double-click opens the full report.
        self.assertIs(window.chart_panel.report, window.current_report)
        self.assertGreater(window.chart_panel.table.rowCount(), 0)
        window._activate_node("results")
        self.assertEqual(len(window._results_windows), 1)
        window.apply_theme("light")
        window.apply_theme("dark")
        # View menu toggles panels; clearing empties the bottom panels.
        window.panel_actions["nodes"].setChecked(False)
        self.assertTrue(window.palette_widget.isHidden())
        window.panel_actions["chart"].setChecked(False)
        window.panel_actions["detail"].setChecked(False)
        self.assertTrue(window.bottom.isHidden())
        window.reset_layout()
        self.assertFalse(window.bottom.isHidden())
        window.clear_results()
        self.assertIsNone(window.chart_panel.report)
        self.assertEqual(window.chart_panel.table.rowCount(), 0)

    def test_form_changes_apply_without_button_and_export(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import time

        from PySide6.QtWidgets import QApplication

        from src.workbench.view.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.controller.set_params("qml", {"max_steps": 2})
        window.scene.nodes["results"].setSelected(True)
        app.processEvents()
        with tempfile.TemporaryDirectory() as tmp:
            window.settings._fields["export_json"].input.setChecked(True)
            window.settings._fields["json_path"].input.setText(str(Path(tmp) / "informe.json"))
            window.settings._fields["export_csv"].input.setChecked(True)
            window.settings._fields["csv_path"].input.setText(str(Path(tmp) / "metricas.csv"))
            app.processEvents()
            window.controller.run()
            deadline = time.time() + 120
            while (window.controller.is_running or window.current_report is None) and time.time() < deadline:
                app.processEvents()
                time.sleep(0.02)
            report = json.loads((Path(tmp) / "informe.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["predictions"]), {"train", "validation", "test"})
            header = (Path(tmp) / "metricas.csv").read_text(encoding="utf-8-sig").splitlines()
            self.assertTrue(header[0].startswith("modelo;particion;accuracy"))
            self.assertIn(",", header[1].split(";")[2])  # decimal comma
        window.controller.dirty = False
        window.close()
        window.controller.dirty = False
        window.close()

    def test_canvas_drag_connects_detaches_and_rejects_incompatible_ports(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication

        from src.workbench.view.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.resize(1400, 820)
        window.show()
        window.view.resetTransform()
        window.view.centerOn(390, 40)
        app.processEvents()
        viewport = window.view.viewport()

        def point(node, port, output):
            return window.view.mapFromScene(window.scene.nodes[node].port(port, output).scene_center())

        def drag(start, end):
            QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, start)
            QTest.mouseMove(viewport, end)
            QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, end)
            app.processEvents()

        def links():
            return sorted((l.source_node, l.target_node) for l in window.controller.workflow.links.values())

        start = point("evaluation", "result", False)
        drag(start, start + QPoint(0, 150))  # pull the link off the input
        self.assertNotIn(("qml", "evaluation"), links())
        drag(point("qml", "result", True), point("evaluation", "result", False))
        self.assertIn(("qml", "evaluation"), links())
        messages = []
        window.scene.message.connect(messages.append)
        drag(point("dataset", "data", True), point("results", "report", False))
        self.assertNotIn(("dataset", "results"), links())
        self.assertTrue(any("incompatibles" in m for m in messages))
        window.controller.dirty = False
        window.close()


if __name__ == "__main__":
    unittest.main()
