import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.execution import ExecutionConfig
from src.qcnn_engine import (
    QCNNConfig,
    initialize_weights,
    parameter_shift_gradient,
)
from src.quantum_training import (
    QuantumTrainingConfig,
    derive_execution_seed,
    train_quantum_qcnn,
)
from src.quantum_update import QGDConfig, quantum_update_step


class QuantumTrainingTests(unittest.TestCase):
    def setUp(self):
        self.features = np.array([[0.2, 0.7]], dtype=float)
        self.targets = np.array([[-1.0]], dtype=float)
        self.validation_features = np.array([[0.8, 0.3]], dtype=float)
        self.validation_targets = np.array([[1.0]], dtype=float)
        self.model_config = QCNNConfig(2)
        self.qgd_config = QGDConfig(n_bits=2, learning_rate=0.1)
        self.initial_weights = initialize_weights(2, seed=19)

    def _train(self, *, training_config=None, initial_weights=None):
        return train_quantum_qcnn(
            self.features,
            self.targets,
            self.validation_features,
            self.validation_targets,
            self.initial_weights if initial_weights is None else initial_weights,
            self.model_config,
            self.qgd_config,
            training_config or QuantumTrainingConfig(),
        )

    def test_training_module_does_not_import_torch_or_adam(self):
        repository_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        code = (
            "import sys; import src.quantum_training; "
            "assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('torch.optim') for name in sys.modules); "
            "assert not any(name.endswith('.Adam') for name in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_first_update_matches_exact_initial_gradient_and_qibo_update(self):
        exact_model = QCNNConfig(
            2,
            execution=ExecutionConfig(shots=None, seed=333),
        )
        exact_gradient = parameter_shift_gradient(
            self.initial_weights,
            self.features,
            self.targets,
            exact_model,
        )
        expected = quantum_update_step(
            self.initial_weights.reshape(-1),
            exact_gradient.reshape(-1),
            self.qgd_config,
        ).reshape(self.initial_weights.shape)

        result = self._train(
            training_config=QuantumTrainingConfig(
                max_steps=1,
                shots=50,
                execution_seed=202,
            )
        )

        np.testing.assert_array_equal(result.final_weights, expected)
        np.testing.assert_array_equal(result.best_weights, expected)
        self.assertEqual(result.steps_completed, 1)
        self.assertEqual(result.best_step, 1)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0]["train_loss"].__class__, float)

    def test_initial_gradient_is_exact_and_later_gradient_uses_shots(self):
        # Deliberately set different model shots to verify training_config wins.
        model_config = QCNNConfig(
            2,
            execution=ExecutionConfig(shots=17, seed=9),
        )
        with patch(
            "src.quantum_training.parameter_shift_gradient",
            wraps=parameter_shift_gradient,
        ) as gradient_spy:
            train_quantum_qcnn(
                self.features,
                self.targets,
                self.validation_features,
                self.validation_targets,
                self.initial_weights,
                model_config,
                self.qgd_config,
                QuantumTrainingConfig(
                    max_steps=2,
                    shots=50,
                    execution_seed=303,
                ),
            )

        self.assertEqual(gradient_spy.call_count, 2)
        initial_execution = gradient_spy.call_args_list[0].args[3].execution
        sampled_execution = gradient_spy.call_args_list[1].args[3].execution
        self.assertIsNone(initial_execution.shots)
        self.assertEqual(initial_execution.depolarizing_probability, 0.0)
        self.assertEqual(sampled_execution.shots, 50)
        self.assertEqual(sampled_execution.seed, derive_execution_seed(303, "sampled_gradient", 1))
        self.assertIsNotNone(initial_execution.seed)

    def test_best_checkpoint_uses_validation_only_and_api_has_no_test_arguments(self):
        train_losses = iter([0.9, 0.1])
        validation_losses = iter([0.2, 0.4])

        def fake_predict(weights, features, config):
            del weights, config
            if np.array_equal(features, self.features):
                loss = next(train_losses)
                return self.targets + np.sqrt(loss)
            loss = next(validation_losses)
            return self.validation_targets - np.sqrt(loss)

        def fake_update(weights, gradients, config):
            del gradients, config
            return np.asarray(weights, dtype=float) + 1.0

        with (
            patch(
                "src.quantum_training.parameter_shift_gradient",
                return_value=np.zeros((1, 4)),
            ),
            patch("src.quantum_training.quantum_update_step", side_effect=fake_update),
            patch("src.quantum_training.predict", side_effect=fake_predict),
        ):
            result = self._train(
                training_config=QuantumTrainingConfig(max_steps=2, shots=10)
            )

        self.assertEqual(result.best_step, 1)
        self.assertAlmostEqual(result.history[1]["train_loss"], 0.1)
        self.assertAlmostEqual(result.history[1]["validation_loss"], 0.4)
        np.testing.assert_array_equal(
            result.best_weights,
            self.initial_weights + 1.0,
        )
        np.testing.assert_array_equal(
            result.final_weights,
            self.initial_weights + 2.0,
        )
        self.assertNotIn("test_features", inspect.signature(train_quantum_qcnn).parameters)
        self.assertNotIn("test_targets", inspect.signature(train_quantum_qcnn).parameters)

    def test_reproducibility_and_separate_initialization_execution_seeds(self):
        first_initial = initialize_weights(2, seed=19)
        other_initial = initialize_weights(2, seed=20)
        self.assertFalse(np.array_equal(first_initial, other_initial))

        first_execution = derive_execution_seed(101, "sampled_gradient", 1)
        repeated_execution = derive_execution_seed(101, "sampled_gradient", 1)
        other_execution = derive_execution_seed(102, "sampled_gradient", 1)
        self.assertEqual(first_execution, repeated_execution)
        self.assertNotEqual(first_execution, other_execution)

        config = QuantumTrainingConfig(
            max_steps=2,
            shots=50,
            execution_seed=101,
        )
        first_result = self._train(training_config=config, initial_weights=first_initial)
        second_result = self._train(training_config=config, initial_weights=first_initial)
        np.testing.assert_array_equal(first_result.best_weights, second_result.best_weights)
        np.testing.assert_array_equal(first_result.final_weights, second_result.final_weights)
        self.assertEqual(first_result.history, second_result.history)

    def test_circuit_execution_estimate_for_two_qubit_one_step(self):
        result = self._train(
            training_config=QuantumTrainingConfig(max_steps=1, shots=50)
        )
        estimate = result.circuit_executions_estimate
        components = estimate["components"]
        initial = components["initial_exact_gradient"]
        self.assertEqual(estimate["logical_weight_occurrences_per_sample"], 4)
        self.assertEqual(estimate["shifted_circuits_per_gradient"], 8)
        self.assertEqual(initial["base_prediction_circuit_executions_estimate"], 1)
        self.assertEqual(initial["parameter_shift_circuit_executions_estimate"], 8)
        self.assertEqual(initial["circuit_executions_estimate"], 9)
        self.assertEqual(
            components["qgd_weight_updates"]["circuit_executions_estimate"], 4
        )
        self.assertEqual(
            components["post_update_train_evaluation"]["circuit_executions_estimate"],
            1,
        )
        self.assertEqual(
            components["validation_evaluation"]["circuit_executions_estimate"],
            1,
        )
        self.assertEqual(
            estimate["total_training_circuit_executions_estimate"], 15
        )
        self.assertEqual(
            components["post_update_train_evaluation"]["logical_batch_evaluations"],
            1,
        )
        self.assertEqual(
            set(result.history[0]["validation_metrics"]),
            {"accuracy", "precision", "recall", "f1"},
        )

    def test_invalid_inputs_and_shots_are_rejected(self):
        with self.assertRaises(ValueError):
            QuantumTrainingConfig(shots=0)
        with self.assertRaises(ValueError):
            QuantumTrainingConfig(max_steps=0)
        with self.assertRaises(ValueError):
            QuantumTrainingConfig(execution_seed=-1)
        with self.assertRaises(ValueError):
            self._train(
                training_config=QuantumTrainingConfig(),
                initial_weights=np.zeros((2, 4)),
            )
        with self.assertRaises(ValueError):
            train_quantum_qcnn(
                self.features,
                self.targets,
                self.validation_features,
                [[0.0]],
                self.initial_weights,
                self.model_config,
                self.qgd_config,
                QuantumTrainingConfig(),
            )
        with self.assertRaises(ValueError):
            train_quantum_qcnn(
                [[0.2, 1.2]],
                self.targets,
                self.validation_features,
                self.validation_targets,
                self.initial_weights,
                self.model_config,
                self.qgd_config,
                QuantumTrainingConfig(),
            )

    def test_result_serializes_to_json(self):
        result = self._train(
            training_config=QuantumTrainingConfig(max_steps=1, shots=20)
        )
        encoded = json.dumps(result.to_dict(), allow_nan=False)
        restored = json.loads(encoded)
        self.assertEqual(restored["best_step"], 1)
        self.assertEqual(restored["steps_completed"], 1)
        self.assertEqual(restored["best_weights"], result.best_weights.tolist())
        self.assertIn("components", restored["circuit_executions_estimate"])

    def test_runner_payload_is_json_serializable_and_tracks_final_test_once(self):
        from experiments.run_quantum_training import build_result_payload

        training_config = QuantumTrainingConfig(
            max_steps=1,
            shots=20,
            execution_seed=101,
        )
        result = self._train(training_config=training_config)
        test_seed = derive_execution_seed(101, "test_evaluation", result.best_step)
        payload = build_result_payload(
            dataset="iris",
            dataset_seed=42,
            initialization_seed=19,
            execution_seed=101,
            initial_weights=self.initial_weights,
            model_config=self.model_config,
            qgd_config=self.qgd_config,
            training_config=training_config,
            training_result=result,
            test_predictions=np.array([[0.25]]),
            test_targets=np.array([[1.0]]),
            test_execution_seed=test_seed,
        )
        restored = json.loads(json.dumps(payload, allow_nan=False))
        self.assertTrue(restored["test"]["evaluated_once_after_checkpoint_selection"])
        self.assertEqual(
            restored["circuit_executions_estimate"]["final_test_evaluation"][
                "circuit_executions_estimate"
            ],
            1,
        )
        self.assertIn("not gradient clipping", restored["qgd"]["finite_register_semantics"])

    def test_runner_exposes_optional_depolarizing_noise(self):
        from experiments.run_quantum_training import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            ["--output", "out.json", "--noise", "0.01"]
        )
        self.assertEqual(args.noise, 0.01)

    def test_runner_evaluates_test_once_after_validation_checkpoint(self):
        from experiments import run_quantum_training as runner

        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=float)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        splits = SimpleNamespace(
            train_features=FakeTensor([[0.2, 0.7]]),
            train_targets=FakeTensor([[-1.0]]),
            validation_features=FakeTensor([[0.8, 0.3]]),
            validation_targets=FakeTensor([[1.0]]),
            test_features=FakeTensor([[0.5, 0.5]]),
            test_targets=FakeTensor([[1.0]]),
        )
        training_result = SimpleNamespace(
            best_weights=np.array([[0.1, 0.2, 0.3, 0.4]]),
            final_weights=np.array([[0.1, 0.2, 0.3, 0.4]]),
            best_step=1,
            steps_completed=1,
            history=({"validation_loss": 0.2},),
            elapsed_seconds=0.01,
            circuit_executions_estimate={
                "total_training_circuit_executions_estimate": 0
            },
        )
        events = []

        def fake_train(*args):
            events.append("train")
            self.assertEqual(args[5].execution.depolarizing_probability, 0.01)
            return training_result

        def fake_predict(weights, features, config):
            events.append("test_predict")
            self.assertEqual(config.execution.shots, 10)
            self.assertEqual(config.execution.depolarizing_probability, 0.01)
            np.testing.assert_array_equal(weights, training_result.best_weights)
            np.testing.assert_array_equal(features, [[0.5, 0.5]])
            return np.array([[0.4]])

        with tempfile.TemporaryDirectory() as temp_directory:
            output = Path(temp_directory) / "pilot.json"
            with (
                patch.object(runner, "_load_splits", return_value=splits),
                patch.object(runner, "train_quantum_qcnn", side_effect=fake_train),
                patch.object(runner, "predict", side_effect=fake_predict),
                patch.object(
                    runner,
                    "build_result_payload",
                    side_effect=lambda **kwargs: events.append("payload") or {"ok": True},
                ),
            ):
                runner.main(
                    ["--output", str(output), "--shots", "10", "--noise", "0.01"]
                )

            self.assertEqual(events, ["train", "test_predict", "payload"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})


if __name__ == "__main__":
    unittest.main()
