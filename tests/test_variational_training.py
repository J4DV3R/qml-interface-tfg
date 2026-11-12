import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.fast_qcnn import FastQCNN, FastQCNNConfig
from src.variational_training import (
    FastEvaluator,
    VariationalTrainingConfig,
    _sequential_sweep,
    binary_metrics,
    mse,
    train_variational_qcnn,
)


def teacher_data(samples=16, seed=3):
    """Labels produced by a teacher QCNN: a learnable 4-qubit problem."""
    rng = np.random.default_rng(seed)
    teacher = rng.uniform(-np.pi, np.pi, (2, 4))
    features = rng.uniform(0.0, 1.0, (samples, 4))
    labels = np.where(FastQCNN(FastQCNNConfig(4)).expectations(teacher, features)[:, 0] >= 0, 1.0, -1.0)
    return features, labels


class FixedGradientEvaluator:
    """Fake evaluator: scripted gradients and validation predictions."""

    n_qubits = 4
    occurrences = [3, 1]

    def __init__(self, gradients, validation_predictions=None):
        self.gradients = list(gradients)
        self.validation_predictions = list(validation_predictions or [])

    def predict(self, weights, features, *, shots, rng):
        if len(features) == 3 and self.validation_predictions:
            return self.validation_predictions.pop(0)
        return np.zeros(len(features))

    def predict_variants(self, weight_variants, features, *, shots, rng):
        raise NotImplementedError

    def gradient(self, weights, features, targets, *, shots, rng):
        return self.gradients.pop(0), 7


class VariationalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = teacher_data(16)
        self.xv, self.yv = teacher_data(8, seed=4)
        self.w0 = np.random.default_rng(19).uniform(-0.25, 0.25, (2, 4))
        self.evaluator = FastEvaluator(FastQCNNConfig(4))

    def _train(self, optimizer, **kwargs):
        config = VariationalTrainingConfig(optimizer=optimizer, **kwargs)
        return train_variational_qcnn(
            self.x, self.y, self.xv, self.yv, self.w0, self.evaluator, config, np.random.default_rng(0)
        )

    def test_every_optimizer_reduces_the_exact_training_loss(self):
        for optimizer, kwargs in (
            ("gd", {"learning_rate": 0.5}),
            ("adam", {"learning_rate": 0.1}),
            ("spsa", {}),
            ("sequential", {}),
        ):
            result = self._train(optimizer, max_steps=15, **kwargs)
            with self.subTest(optimizer=optimizer):
                self.assertLess(result.history[-1]["train_loss"], result.history[0]["train_loss"])

    def test_sequential_sweep_never_increases_the_exact_loss(self):
        model = self.evaluator.model
        weights = self.w0.copy()
        config = VariationalTrainingConfig(optimizer="sequential")
        previous = mse(model.expectations(weights, self.x)[:, 0], self.y)
        for _ in range(3):
            weights, circuits = _sequential_sweep(weights, self.x, self.y, self.evaluator, config, None)
            current = mse(model.expectations(weights, self.x)[:, 0], self.y)
            self.assertLessEqual(current, previous + 1e-9)
            previous = current
        self.assertEqual(circuits, len(self.x) * (4 * 7 + 4 * 3))

    def test_sequential_reconstruction_is_exact(self):
        # The reconstructed coordinate curve must equal direct evaluation.
        model = self.evaluator.model
        layer, index, degree = 0, 1, 3
        points = 2 * degree + 1
        offsets = 2 * np.pi * np.arange(points) / points
        variants = np.repeat(self.w0[None], points, axis=0)
        variants[:, layer, index] += offsets
        values = self.evaluator.predict_variants(variants, self.x, shots=None, rng=None)
        k = np.arange(-degree, degree + 1)
        coefficients = (np.exp(-1j * np.outer(k, offsets)) / points) @ values
        for phi in (0.3, 1.7, -2.2):
            probe = self.w0.copy()
            probe[layer, index] += phi
            direct = model.expectations(probe, self.x)[:, 0]
            reconstructed = np.real(np.exp(1j * phi * k) @ coefficients)
            np.testing.assert_allclose(reconstructed, direct, atol=1e-10)

    def test_step_zero_is_a_checkpoint_candidate(self):
        xv = self.xv[:3]
        yv = np.array([1.0, -1.0, 1.0])
        evaluator = FixedGradientEvaluator(
            [np.ones((2, 4))] * 2,
            validation_predictions=[yv.copy(), -yv, -yv],  # step 0 is perfect
        )
        result = train_variational_qcnn(
            self.x, self.y, xv, yv, self.w0, evaluator,
            VariationalTrainingConfig(optimizer="gd", max_steps=2, learning_rate=0.1),
            np.random.default_rng(0),
        )
        self.assertEqual(result.best_step, 0)
        np.testing.assert_array_equal(result.best_weights, self.w0)
        self.assertEqual(result.optimization_circuits, 14)

    def test_adam_matches_torch_adam(self):
        rng = np.random.default_rng(2)
        gradients = [rng.normal(size=(2, 4)) for _ in range(5)]
        evaluator = FixedGradientEvaluator([g.copy() for g in gradients])
        result = train_variational_qcnn(
            self.x, self.y, self.xv, self.yv, self.w0, evaluator,
            VariationalTrainingConfig(optimizer="adam", max_steps=5, learning_rate=0.03),
            np.random.default_rng(0),
        )
        parameter = torch.nn.Parameter(torch.tensor(self.w0, dtype=torch.float64))
        optimizer = torch.optim.Adam([parameter], lr=0.03)
        for gradient in gradients:
            optimizer.zero_grad()
            parameter.grad = torch.tensor(gradient, dtype=torch.float64)
            optimizer.step()
        np.testing.assert_allclose(result.final_weights, parameter.detach().numpy(), atol=1e-12)

    def test_circuit_accounting_and_budget(self):
        spsa = self._train("spsa", max_steps=3)
        self.assertEqual(spsa.optimization_circuits, 3 * 2 * len(self.x))
        gd = self._train("gd", max_steps=2, learning_rate=0.1)
        self.assertEqual(gd.optimization_circuits, 2 * len(self.x) * 33)
        budget = self._train("gd", max_steps=50, max_optimization_circuits=len(self.x) * 33 * 3)
        self.assertEqual(budget.steps_completed, 3)
        self.assertEqual(budget.stop_reason, "circuit_budget")

    def test_shots_make_training_stochastic_but_reproducible(self):
        first = self._train("adam", max_steps=2, shots=50)
        second = self._train("adam", max_steps=2, shots=50)
        np.testing.assert_array_equal(first.final_weights, second.final_weights)

    def test_api_has_no_test_partition(self):
        parameters = inspect.signature(train_variational_qcnn).parameters
        self.assertFalse(any("test" in name for name in parameters))

    def test_metrics_include_balanced_accuracy_and_confusion(self):
        metrics = binary_metrics([0.5, -0.2, 0.1, -0.9], [1, 1, -1, -1])
        self.assertEqual(metrics["confusion"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_invalid_configuration(self):
        for kwargs in ({"optimizer": "lbfgs"}, {"optimizer": "gd", "learning_rate": 0},
                       {"optimizer": "adam", "shots": 0}, {"optimizer": "spsa", "max_steps": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VariationalTrainingConfig(**kwargs)


class BenchmarkRunnerTests(unittest.TestCase):
    def _run(self, directory, *extra):
        from experiments.run_variational_benchmark import run

        return run([
            "--dataset", "iris", "--optimizers", "adam", "sequential",
            "--init-seeds", "19", "23", "--max-steps", "2",
            "--output-dir", str(directory), *extra,
        ])

    def test_runner_writes_runs_manifest_and_test_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self._run(Path(tmp) / "bench")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["runs"]), 4)
            self.assertIn("test", manifest["baselines"]["majority_class"])
            self.assertEqual(manifest["dataset"]["partitions"]["test"]["samples"], 15)
            run = json.loads((output / manifest["runs"][0]).read_text(encoding="utf-8"))
            self.assertTrue(run["test"]["evaluated_once_after_checkpoint_selection"])
            self.assertIn("exact_noiseless_diagnostic", run["test"])
            self.assertEqual(run["training"]["history"][0]["step"], 0)

            from experiments.analyze_variational_benchmark import main as analyze

            report = Path(tmp) / "report.md"
            curves = Path(tmp) / "curves.csv"
            summary = Path(tmp) / "summary.json"
            self.assertEqual(
                analyze([str(output), "--output", str(report), "--curves-csv", str(curves),
                         "--json", str(summary)]),
                0,
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("Resultados por condición", text)
            self.assertIn("Diferencias pareadas", text)
            self.assertTrue(curves.exists())
            parsed = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(len(parsed["conditions"]), 2)

    def test_screening_never_evaluates_test_and_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self._run(Path(tmp) / "screen", "--screening")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("test", manifest["baselines"]["majority_class"])
            run = json.loads((output / manifest["runs"][0]).read_text(encoding="utf-8"))
            self.assertEqual(run["test"], {"evaluated": False, "reason": "screening"})
            with self.assertRaises(SystemExit):
                self._run(output)

    def test_frozen_results_are_protected(self):
        with self.assertRaises(SystemExit):
            self._run(Path("results/initial-hybrid-adam/new-benchmark"))


if __name__ == "__main__":
    unittest.main()
