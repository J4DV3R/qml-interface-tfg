"""The vectorized simulator must reproduce the Qibo engine exactly."""

import unittest

import numpy as np

from src.execution import ExecutionConfig
from src.fast_qcnn import FastQCNN, FastQCNNConfig, sample_expectations, weight_occurrences
from src.qcnn_engine import QCNNConfig, parameter_shift_gradient, predict
from src.variational_training import FastEvaluator


class FastQCNNEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def _case(self, n_qubits, n_outputs, samples=5):
        n_layers = int(np.log2(n_qubits // n_outputs))
        weights = self.rng.uniform(-np.pi, np.pi, (n_layers, 4))
        features = self.rng.uniform(0.0, 1.0, (samples, n_qubits))
        return weights, features

    def test_exact_expectations_match_qibo(self):
        for n_qubits, n_outputs in ((2, 1), (4, 1), (4, 2), (8, 1)):
            for encoding in ("angle", "phase"):
                weights, features = self._case(n_qubits, n_outputs)
                qibo = predict(weights, features, QCNNConfig(n_qubits, n_outputs, encoding))
                fast = FastQCNN(FastQCNNConfig(n_qubits, n_outputs, encoding)).expectations(
                    weights, features
                )
                with self.subTest(n=n_qubits, out=n_outputs, encoding=encoding):
                    np.testing.assert_allclose(fast, qibo, atol=1e-10)

    def test_density_matrix_noise_matches_qibo_density_matrix(self):
        for encoding in ("angle", "phase"):
            weights, features = self._case(4, 1)
            qibo = predict(
                weights,
                features,
                QCNNConfig(4, 1, encoding, execution=ExecutionConfig(depolarizing_probability=0.07)),
            )
            fast = FastQCNN(
                FastQCNNConfig(4, 1, encoding, depolarizing_probability=0.07)
            ).expectations(weights, features)
            with self.subTest(encoding=encoding):
                np.testing.assert_allclose(fast, qibo, atol=1e-10)

    def test_parameter_shift_gradient_matches_qibo(self):
        for encoding in ("angle", "phase"):
            weights, features = self._case(4, 1, samples=4)
            targets = np.array([1.0, -1.0, 1.0, -1.0])
            qibo = parameter_shift_gradient(weights, features, targets, QCNNConfig(4, 1, encoding))
            evaluator = FastEvaluator(FastQCNNConfig(4, 1, encoding))
            fast, circuits = evaluator.gradient(weights, features, targets, shots=None, rng=None)
            with self.subTest(encoding=encoding):
                np.testing.assert_allclose(fast, qibo, atol=1e-10)
                self.assertEqual(circuits, 4 * (1 + 2 * 16))

    def test_parameter_shift_matches_finite_differences_under_noise(self):
        weights, features = self._case(4, 1, samples=3)
        targets = np.array([1.0, -1.0, 1.0])
        config = FastQCNNConfig(4, 1, "angle", depolarizing_probability=0.05, readout_error=0.02)
        evaluator = FastEvaluator(config)
        gradient, _ = evaluator.gradient(weights, features, targets, shots=None, rng=None)
        model = FastQCNN(config)

        def loss(w):
            return float(np.mean((model.expectations(w, features)[:, 0] - targets) ** 2))

        numeric = np.zeros_like(weights)
        for index in np.ndindex(weights.shape):
            step = np.zeros_like(weights)
            step[index] = 1e-6
            numeric[index] = (loss(weights + step) - loss(weights - step)) / 2e-6
        np.testing.assert_allclose(gradient, numeric, atol=1e-6)

    def test_readout_error_scales_expectations(self):
        weights, features = self._case(4, 1)
        clean = FastQCNN(FastQCNNConfig(4)).expectations(weights, features)
        readout = FastQCNN(FastQCNNConfig(4, readout_error=0.1)).expectations(weights, features)
        np.testing.assert_allclose(readout, 0.8 * clean, atol=1e-12)

    def test_shot_sampling_is_unbiased_and_bounded(self):
        rng = np.random.default_rng(0)
        exact = np.full((20000, 1), 0.3)
        sampled = sample_expectations(exact, 50, rng)
        self.assertTrue(np.all(np.abs(sampled) <= 1.0))
        self.assertAlmostEqual(float(sampled.mean()), 0.3, delta=0.01)
        np.testing.assert_array_equal(sample_expectations(exact, None, rng), exact)

    def test_occurrences(self):
        self.assertEqual(weight_occurrences(4), [3, 1])
        self.assertEqual(weight_occurrences(8), [7, 3, 1])

    def test_invalid_configuration(self):
        for kwargs in (
            {"n_qubits": 3},
            {"n_qubits": 4, "n_outputs": 3},
            {"n_qubits": 4, "encoding": "amplitude"},
            {"n_qubits": 4, "depolarizing_probability": 1.5},
            {"n_qubits": 4, "readout_error": 0.6},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FastQCNNConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
