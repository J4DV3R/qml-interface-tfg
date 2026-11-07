import os
import subprocess
import sys
import unittest

import numpy as np
import torch

from src.execution import ExecutionConfig
from src.hybrid_model import QCNNModel
from src.qcnn_engine import (
    QCNNConfig,
    build_qcnn_circuit,
    expectation_z,
    initialize_weights,
    mse_loss,
    parameter_shift_gradient,
    predict,
)


def _features(seed: int, batch_size: int, n_qubits: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(0.0, 1.0, size=(batch_size, n_qubits)).astype(
        np.float32
    )


class EngineLegacyParityTests(unittest.TestCase):
    def test_predict_matches_legacy_qcnn_model(self):
        cases = [
            (4, 1, "angle", 3, 101),
            (4, 1, "phase", 3, 102),
            (4, 2, "angle", 3, 103),
            (8, 1, "angle", 1, 104),
        ]
        for n_qubits, n_outputs, encoding, batch_size, seed in cases:
            with self.subTest(
                n_qubits=n_qubits,
                n_outputs=n_outputs,
                encoding=encoding,
            ):
                model = QCNNModel(
                    n_qubits=n_qubits,
                    n_outputs=n_outputs,
                    encoding=encoding,
                )
                weights = model.weights.detach().numpy().astype(np.float64)
                features = _features(seed, batch_size, n_qubits)
                with torch.no_grad():
                    legacy = model(torch.as_tensor(features)).detach().numpy()
                engine = predict(
                    weights,
                    features.astype(np.float64),
                    QCNNConfig(n_qubits, n_outputs, encoding),
                )
                self.assertEqual(engine.shape, (batch_size, n_outputs))
                self.assertEqual(engine.dtype, np.dtype(float))
                self.assertTrue(np.isfinite(engine).all())
                np.testing.assert_allclose(
                    engine,
                    legacy,
                    rtol=0.0,
                    atol=1e-6,
                )

    def test_mse_loss_matches_torch_mean_mse(self):
        cases = [
            (4, 1, "angle", 5, 201),
            (4, 2, "angle", 5, 202),
        ]
        for n_qubits, n_outputs, encoding, batch_size, seed in cases:
            with self.subTest(n_qubits=n_qubits, n_outputs=n_outputs):
                model = QCNNModel(
                    n_qubits=n_qubits,
                    n_outputs=n_outputs,
                    encoding=encoding,
                )
                weights = model.weights.detach().numpy().astype(np.float64)
                features = _features(seed, batch_size, n_qubits)
                generator = np.random.default_rng(seed + 1)
                targets = generator.choice(
                    [-1.0, 1.0], size=(batch_size, n_outputs)
                ).astype(np.float32)
                with torch.no_grad():
                    legacy_loss = torch.nn.functional.mse_loss(
                        model(torch.as_tensor(features)),
                        torch.as_tensor(targets),
                    ).item()
                engine_loss = mse_loss(
                    weights,
                    features.astype(np.float64),
                    targets.astype(np.float64),
                    QCNNConfig(n_qubits, n_outputs, encoding),
                )
                self.assertAlmostEqual(engine_loss, legacy_loss, delta=1e-6)

                if n_outputs == 1:
                    flattened_loss = mse_loss(
                        weights,
                        features.astype(np.float64),
                        targets.reshape(batch_size),
                        QCNNConfig(n_qubits, n_outputs, encoding),
                    )
                    self.assertAlmostEqual(flattened_loss, engine_loss, delta=1e-12)

    def test_parameter_shift_gradient_matches_legacy_autograd(self):
        cases = [
            (4, 1, "angle", 4, 401),
            (4, 2, "angle", 4, 402),
        ]
        for n_qubits, n_outputs, encoding, batch_size, seed in cases:
            with self.subTest(n_qubits=n_qubits, n_outputs=n_outputs):
                model = QCNNModel(
                    n_qubits=n_qubits,
                    n_outputs=n_outputs,
                    encoding=encoding,
                )
                weights = model.weights.detach().numpy().astype(np.float64)
                features = _features(seed, batch_size, n_qubits)
                generator = np.random.default_rng(seed + 1)
                targets = generator.choice(
                    [-1.0, 1.0], size=(batch_size, n_outputs)
                ).astype(np.float32)
                legacy_loss = torch.nn.functional.mse_loss(
                    model(torch.as_tensor(features)),
                    torch.as_tensor(targets),
                )
                legacy_loss.backward()
                legacy_gradient = model.weights.grad.detach().numpy()

                engine_gradient = parameter_shift_gradient(
                    weights,
                    features.astype(np.float64),
                    targets.astype(np.float64),
                    QCNNConfig(n_qubits, n_outputs, encoding),
                )
                np.testing.assert_allclose(
                    engine_gradient,
                    legacy_gradient,
                    rtol=0.0,
                    atol=1e-6,
                )

    def test_parameter_shift_gradient_matches_central_finite_differences(self):
        # Parameter-shift is exact for RY/RZ gates, so the analytic gradient
        # agrees with central differences up to floating-point error. With
        # epsilon=1e-6 the observed error is far below 1e-4; this tolerance is
        # deliberately loose inside the 1e-4..1e-3 range used for
        # finite-difference checks.
        epsilon = 1e-6
        tolerance = 1e-4
        cases = [
            (4, "angle", 301),
            (4, "phase", 302),
            (8, "angle", 303),
        ]
        for n_qubits, encoding, seed in cases:
            with self.subTest(n_qubits=n_qubits, encoding=encoding):
                weights = initialize_weights(n_qubits, seed=seed)
                generator = np.random.default_rng(seed + 1)
                features = generator.uniform(
                    0.05, 0.95, size=(2, n_qubits)
                )
                targets = np.array([1.0, -1.0])
                config = QCNNConfig(n_qubits, encoding=encoding)

                analytic = parameter_shift_gradient(
                    weights, features, targets, config
                )
                self.assertEqual(analytic.shape, (config.n_layers, 4))
                self.assertTrue(np.isfinite(analytic).all())

                numeric = np.zeros_like(weights)
                for layer_index in range(weights.shape[0]):
                    for weight_index in range(weights.shape[1]):
                        positive = weights.copy()
                        negative = weights.copy()
                        positive[layer_index, weight_index] += epsilon
                        negative[layer_index, weight_index] -= epsilon
                        numeric[layer_index, weight_index] = (
                            mse_loss(positive, features, targets, config)
                            - mse_loss(negative, features, targets, config)
                        ) / (2.0 * epsilon)

                np.testing.assert_allclose(
                    analytic,
                    numeric,
                    rtol=0.0,
                    atol=tolerance,
                    err_msg=(
                        f"parameter-shift mismatch for n_qubits={n_qubits}, "
                        f"encoding={encoding}"
                    ),
                )


class EngineValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = QCNNConfig(4)
        self.weights = initialize_weights(4, seed=1)
        self.features = np.array([[0.1, 0.2, 0.3, 0.4]])

    def test_invalid_config_raises(self):
        invalid_kwargs = [
            {"n_qubits": 0},
            {"n_qubits": 3},
            {"n_qubits": 4, "n_outputs": 0},
            {"n_qubits": 4, "n_outputs": 3},
            {"n_qubits": 4, "n_outputs": 4},
            {"n_qubits": 4, "encoding": "amplitude"},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    QCNNConfig(**kwargs)

    def test_invalid_features_and_weights_raise(self):
        invalid_features = [
            np.array([[0.1, 0.2, 0.3, 1.2]]),
            np.array([[0.1, 0.2, 0.3, float("nan")]]),
            np.array([[0.1, 0.2, 0.3]]),
            np.empty((0, 4)),
        ]
        for features in invalid_features:
            with self.subTest(shape=features.shape):
                with self.assertRaises(ValueError):
                    predict(self.weights, features, self.config)
        with self.assertRaises(ValueError):
            predict(np.zeros((3, 4)), self.features, self.config)
        non_finite_weights = self.weights.copy()
        non_finite_weights[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            predict(non_finite_weights, self.features, self.config)

    def test_invalid_targets_raise(self):
        with self.assertRaises(ValueError):
            mse_loss(self.weights, self.features, [1.0, -1.0], self.config)
        with self.assertRaises(ValueError):
            mse_loss(self.weights, self.features, [[1.0, -1.0]], self.config)
        with self.assertRaises(ValueError):
            mse_loss(
                self.weights,
                self.features,
                [[float("nan")]],
                self.config,
            )

        two_output_config = QCNNConfig(4, n_outputs=2)
        two_output_weights = initialize_weights(4, n_outputs=2, seed=2)
        with self.assertRaises(ValueError):
            mse_loss(
                two_output_weights,
                self.features,
                [1.0],
                two_output_config,
            )
        with self.assertRaises(ValueError):
            mse_loss(
                two_output_weights,
                self.features,
                [[1.0]],
                two_output_config,
            )
        with self.assertRaises(ValueError):
            parameter_shift_gradient(
                self.weights,
                self.features,
                [float("nan")],
                self.config,
            )

    def test_noise_without_shots_raises(self):
        config = QCNNConfig(
            4,
            execution=ExecutionConfig(depolarizing_probability=0.1),
        )
        with self.assertRaisesRegex(
            ValueError,
            "noise requires a positive shots value",
        ):
            predict(self.weights, self.features, config)

    def test_sampled_noisy_execution_is_finite_and_bounded(self):
        config = QCNNConfig(
            4,
            execution=ExecutionConfig(
                shots=200,
                depolarizing_probability=0.1,
                seed=11,
            ),
        )
        output = predict(self.weights, self.features, config)
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.all(output >= -1.0))
        self.assertTrue(np.all(output <= 1.0))


class EngineUtilityTests(unittest.TestCase):
    def test_initialize_weights_is_reproducible_and_shaped(self):
        first = initialize_weights(8, n_outputs=2, seed=42)
        second = initialize_weights(8, n_outputs=2, seed=42)
        different = initialize_weights(8, n_outputs=2, seed=43)
        self.assertEqual(first.shape, (2, 4))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertTrue(np.all(first >= -0.25))
        self.assertTrue(np.all(first <= 0.25))
        self.assertEqual(initialize_weights(4, n_outputs=2, seed=1).shape, (1, 4))
        self.assertEqual(initialize_weights(4, seed=1).shape, (2, 4))
        with self.assertRaises(ValueError):
            initialize_weights(4, seed=True)

    def test_public_expectation_z_and_circuit_builder(self):
        zero_state = np.array([1.0, 0.0], dtype=complex)
        one_state = np.array([0.0, 1.0], dtype=complex)
        self.assertAlmostEqual(expectation_z(zero_state, 1, 0), 1.0)
        self.assertAlmostEqual(expectation_z(one_state, 1, 0), -1.0)

        # With Qibo's state-vector convention qubit 0 is the high-order bit,
        # while qubit 3 is the low-order bit in a four-qubit register.
        q0_one = np.zeros(16, dtype=complex)
        q0_one[8] = 1.0
        q3_one = np.zeros(16, dtype=complex)
        q3_one[1] = 1.0
        self.assertAlmostEqual(expectation_z(q0_one, 4, 0), -1.0)
        self.assertAlmostEqual(expectation_z(q0_one, 4, 3), 1.0)
        self.assertAlmostEqual(expectation_z(q3_one, 4, 0), 1.0)
        self.assertAlmostEqual(expectation_z(q3_one, 4, 3), -1.0)

        weights = initialize_weights(4, seed=5)
        circuit = build_qcnn_circuit(
            np.array([0.1, 0.2, 0.3, 0.4]),
            weights,
            QCNNConfig(4),
        )
        self.assertEqual(circuit.nqubits, 4)
        # 4 encoding gates + 3 pairs * 6 gates + 2 pooling CNOTs
        # + 1 pair * 6 gates in the second layer + 1 pooling CNOT.
        self.assertEqual(len(circuit.queue), 4 + 18 + 2 + 6 + 1)

    def test_engine_does_not_import_torch(self):
        repository_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        code = (
            "import sys; "
            "import src.qcnn_engine; "
            "assert 'torch' not in sys.modules, "
            "'src.qcnn_engine imported torch'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_sampled_predict_is_reproducible_and_bounded(self):
        weights = initialize_weights(4, seed=5)
        features = np.array(
            [
                [0.2, 0.7, 0.4, 0.9],
                [0.1, 0.3, 0.8, 0.5],
            ]
        )
        config = QCNNConfig(
            4,
            execution=ExecutionConfig(shots=200, seed=11),
        )
        first = predict(weights, features, config)
        second = predict(weights, features, config)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 1))
        self.assertTrue(np.all(first >= -1.0))
        self.assertTrue(np.all(first <= 1.0))


if __name__ == "__main__":
    unittest.main()
