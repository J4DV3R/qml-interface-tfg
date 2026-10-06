import unittest

import numpy as np
import torch
from qibo.models import Circuit

from src.data_encoding import (
    add_angle_encoding,
    add_phase_encoding,
    angle_encoding,
    phase_encoding,
)
from src.datasets import SplitConfig, load_binary_iris_splits, split_dataset
from src.execution import ExecutionConfig
from src.metrics import classification_metrics
from src.training import TrainingConfig, train_qcnn
from src.hybrid_model import QCNNModel, _expectation_z
from src.pooling import add_pooling_layer
from src.qcnn_circuit import add_conv_layer


class EncodingTests(unittest.TestCase):
    def test_angle_encoding_prepares_expected_basis_states(self):
        zero = angle_encoding([0.0]).execute().state()
        one = angle_encoding([1.0]).execute().state()
        self.assertAlmostEqual(_expectation_z(zero, 1, 0), 1.0, places=6)
        self.assertAlmostEqual(_expectation_z(one, 1, 0), -1.0, places=6)

        two_qubit_state = angle_encoding([0.0, 1.0]).execute().state()
        self.assertAlmostEqual(_expectation_z(two_qubit_state, 2, 0), 1.0, places=6)
        self.assertAlmostEqual(_expectation_z(two_qubit_state, 2, 1), -1.0, places=6)

    def test_phase_encoding_prepares_relative_phase(self):
        circuit = phase_encoding([0.25])
        state = circuit.execute().state()
        self.assertAlmostEqual(abs(state[0]), 1.0 / np.sqrt(2.0), places=6)
        self.assertAlmostEqual(abs(state[1]), 1.0 / np.sqrt(2.0), places=6)
        self.assertFalse(np.allclose(state, angle_encoding([0.25]).execute().state()))

    def test_encodings_reject_out_of_range_features(self):
        with self.assertRaises(ValueError):
            angle_encoding([1.1])
        with self.assertRaises(ValueError):
            phase_encoding([float("nan")])

    def test_encodings_reject_non_finite_parameter_shifts(self):
        with self.assertRaises(ValueError):
            add_angle_encoding(Circuit(1), [0.5], (0, float("nan")))
        with self.assertRaises(ValueError):
            add_phase_encoding(Circuit(1), [0.5], (0, float("inf")))


class DatasetTests(unittest.TestCase):
    def test_iris_splits_are_reproducible_and_normalized(self):
        first = load_binary_iris_splits(seed=17)
        second = load_binary_iris_splits(seed=17)
        self.assertEqual(first.train_features.shape, (70, 4))
        self.assertEqual(first.validation_features.shape, (15, 4))
        self.assertEqual(first.test_features.shape, (15, 4))
        self.assertTrue(torch.equal(first.train_features, second.train_features))
        self.assertTrue(torch.equal(first.test_targets, second.test_targets))
        different = load_binary_iris_splits(seed=18)
        self.assertFalse(torch.equal(first.train_features, different.train_features))
        self.assertTrue(torch.all(first.train_features >= 0.0))
        self.assertTrue(torch.all(first.train_features <= 1.0))
        self.assertEqual(sorted(first.train_targets.unique().tolist()), [-1.0, 1.0])

    def test_binary_metrics_are_model_independent(self):
        predictions = torch.tensor([[-0.2], [0.4], [-0.3], [0.1]])
        targets = torch.tensor([[-1.0], [1.0], [1.0], [-1.0]])
        metrics = classification_metrics(predictions, targets)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_iris_split_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            load_binary_iris_splits(train_fraction=0.9, validation_fraction=0.2)
        with self.assertRaises(ValueError):
            load_binary_iris_splits(seed=True)

    def test_iris_accepts_custom_partition_percentages(self):
        splits = load_binary_iris_splits(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=17,
        )
        self.assertEqual(len(splits.train_features), 60)
        self.assertEqual(len(splits.validation_features), 20)
        self.assertEqual(len(splits.test_features), 20)
        with self.assertRaises(ValueError):
            load_binary_iris_splits(
                train_fraction=0.5,
                validation_fraction=0.2,
                test_fraction=0.1,
            )

    def test_generic_splitter_preserves_pipeline_contract(self):
        features = torch.arange(40, dtype=torch.float32).reshape(20, 2)
        targets = torch.tensor([0, 1] * 10)
        config = SplitConfig(
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
            seed=5,
        )
        first = split_dataset(features, targets, config)
        second = split_dataset(features, targets, config)
        self.assertEqual(first.train_features.shape, (10, 2))
        self.assertEqual(first.validation_features.shape, (5, 2))
        self.assertEqual(first.test_features.shape, (5, 2))
        self.assertTrue(torch.equal(first.train_features, second.train_features))
        self.assertTrue(torch.all(first.train_features >= 0.0))
        self.assertTrue(torch.all(first.train_features <= 1.0))
        self.assertEqual(sorted(first.train_targets.unique().tolist()), [-1.0, 1.0])


class CircuitLayerTests(unittest.TestCase):
    def test_convolution_uses_shared_filter_on_adjacent_pairs(self):
        circuit = Circuit(4)
        add_conv_layer(circuit, [0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(circuit.nqubits, 4)
        self.assertEqual(len(circuit.queue), 18)
        with self.assertRaises(ValueError):
            add_conv_layer(
                Circuit(2),
                [0, 1],
                [0.1, 0.2, 0.3, 0.4],
                shifted_occurrence=(0, 1, float("inf")),
            )

    def test_pooling_validates_pairs_and_keeps_fixed_register(self):
        circuit = Circuit(4)
        returned = add_pooling_layer(circuit, [0, 1], [2, 3])
        self.assertIs(returned, circuit)
        self.assertEqual(circuit.nqubits, 4)
        self.assertEqual(len(circuit.queue), 2)
        with self.assertRaises(ValueError):
            add_pooling_layer(circuit, [0, 1], [1, 2])


class QCNNTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = QCNNModel(n_qubits=2)
        self.features = torch.tensor([[0.2, 0.7]], dtype=torch.float32)

    def test_backward_produces_gradients_and_optimizer_updates_weights(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        original_weights = self.model.weights.detach().clone()
        optimizer.zero_grad()
        loss = self.model(self.features).square().mean()
        loss.backward()

        self.assertIsNotNone(self.model.weights.grad)
        self.assertTrue(torch.isfinite(self.model.weights.grad).all())
        self.assertGreater(self.model.weights.grad.abs().sum().item(), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(original_weights, self.model.weights.detach()))

    def test_phase_encoding_model_produces_parameter_gradients(self):
        model = QCNNModel(n_qubits=2, encoding="phase")
        prediction = model(self.features)
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(model.weights.grad).all())
        self.assertGreater(model.weights.grad.abs().sum().item(), 0.0)

    def test_shared_parameter_shift_matches_finite_difference(self):
        model = QCNNModel(n_qubits=4)
        features = torch.tensor(
            [[0.2, 0.7, 0.4, 0.9]],
            dtype=torch.float32,
            requires_grad=True,
        )
        prediction = model(features)
        prediction.sum().backward()
        analytic_weight_gradient = model.weights.grad[0, 0].item()
        analytic_feature_gradient = features.grad[0, 1].item()

        original = model.weights[0, 0].item()
        epsilon = 1e-3
        with torch.no_grad():
            model.weights[0, 0] = original + epsilon
        positive_weight = model(features.detach()).sum().item()
        with torch.no_grad():
            model.weights[0, 0] = original - epsilon
        negative_weight = model(features.detach()).sum().item()
        with torch.no_grad():
            model.weights[0, 0] = original

        weight_difference = (positive_weight - negative_weight) / (2.0 * epsilon)
        self.assertAlmostEqual(
            analytic_weight_gradient, weight_difference, delta=2e-3
        )

        original_feature = features[0, 1].item()
        with torch.no_grad():
            features[0, 1] = original_feature + epsilon
        positive_feature = model(features).sum().item()
        with torch.no_grad():
            features[0, 1] = original_feature - epsilon
        negative_feature = model(features).sum().item()
        with torch.no_grad():
            features[0, 1] = original_feature
        feature_difference = (positive_feature - negative_feature) / (2.0 * epsilon)
        self.assertAlmostEqual(
            analytic_feature_gradient, feature_difference, delta=2e-3
        )

    def test_pooling_layout_and_input_validation(self):
        self.assertEqual(QCNNModel(n_qubits=4, n_outputs=2).weights.shape, (1, 4))
        self.assertEqual(
            QCNNModel(n_qubits=4, n_outputs=2)(
                torch.zeros((1, 4), dtype=torch.float32)
            ).shape,
            (1, 2),
        )
        for n_qubits in (0, 3):
            with self.assertRaises(ValueError):
                QCNNModel(n_qubits)
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[0.2]], dtype=torch.float32))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[0.2, 1.2]], dtype=torch.float32))

    def test_sampled_execution_returns_expectations(self):
        first_model = QCNNModel(
            n_qubits=2,
            execution_config=ExecutionConfig(shots=200, seed=11),
        )
        second_model = QCNNModel(
            n_qubits=2,
            execution_config=ExecutionConfig(shots=200, seed=11),
        )
        with torch.no_grad():
            second_model.weights.copy_(first_model.weights)
        output = first_model(self.features)
        self.assertTrue(torch.equal(output, second_model(self.features)))
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_noise_requires_shots_and_changes_sampled_execution(self):
        with self.assertRaises(ValueError):
            QCNNModel(
                n_qubits=2,
                execution_config=ExecutionConfig(
                    depolarizing_probability=0.1,
                ),
            )(self.features)

        model = QCNNModel(
            n_qubits=2,
            execution_config=ExecutionConfig(
                shots=200,
                depolarizing_probability=1.0,
                seed=11,
            ),
        )
        output = model(self.features)
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_training_config_and_loop_track_each_epoch(self):
        model = QCNNModel(n_qubits=2)
        targets = torch.tensor([[1.0]], dtype=torch.float32)
        history = train_qcnn(
            model,
            self.features,
            targets,
            TrainingConfig(epochs=3, learning_rate=0.01),
        )
        self.assertEqual(len(history), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(history))))
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0.0)


if __name__ == "__main__":
    unittest.main()
