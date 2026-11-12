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
from src.circuit_execution import QiboSimulatorExecutor
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

    def test_none_shots_keeps_every_simulator_gradient_exact(self):
        with patch(
            "src.quantum_training.parameter_shift_gradient",
            wraps=parameter_shift_gradient,
        ) as gradient_spy:
            result = self._train(
                training_config=QuantumTrainingConfig(max_steps=2, shots=None)
            )

        self.assertEqual(gradient_spy.call_count, 2)
        self.assertTrue(
            all(
                call.args[3].execution.shots is None
                for call in gradient_spy.call_args_list
            )
        )
        self.assertEqual(result.history[1]["gradient_mode"], "parameter_shift_exact")
        updates = result.circuit_executions_estimate["components"]["qgd_weight_updates"]
        self.assertEqual(updates["shots_per_circuit"], 1)
        self.assertEqual(updates["shots_total_estimate"], 8)
        self.assertEqual(result.history[0]["update_mode_probabilities"], [1.0] * 4)
        self.assertEqual(result.history[0]["update_minimum_mode_probability"], 0.5)
        self.assertEqual(
            result.history[0]["update_execution_seed"],
            derive_execution_seed(101, "weight_update", 1),
        )

    def test_exact_mode_is_not_accepted_for_an_injected_executor(self):
        backend_model = QCNNConfig(2, executor=QiboSimulatorExecutor())
        with self.assertRaisesRegex(ValueError, "requires finite shots"):
            train_quantum_qcnn(
                self.features,
                self.targets,
                self.validation_features,
                self.validation_targets,
                self.initial_weights,
                backend_model,
                self.qgd_config,
                QuantumTrainingConfig(shots=None),
            )

    def test_injected_executor_samples_initial_gradient_with_training_shots(self):
        executor = QiboSimulatorExecutor()
        model_config = QCNNConfig(2, executor=executor)
        training_config = QuantumTrainingConfig(max_steps=1, shots=16)
        with patch(
            "src.quantum_training.parameter_shift_gradient",
            wraps=parameter_shift_gradient,
        ) as gradient_spy:
            result = train_quantum_qcnn(
                self.features,
                self.targets,
                self.validation_features,
                self.validation_targets,
                self.initial_weights,
                model_config,
                self.qgd_config,
                training_config,
            )

        self.assertEqual(gradient_spy.call_count, 1)
        initial_execution = gradient_spy.call_args.args[3]
        self.assertEqual(initial_execution.execution.shots, 16)
        self.assertIs(initial_execution.executor, executor)
        self.assertEqual(result.history[0]["gradient_mode"], "parameter_shift_shots")
        self.assertEqual(result.history[0]["gradient_shots"], 16)

    def test_explicit_update_executor_is_separate_from_qcnn_executor(self):
        qcnn_executor = QiboSimulatorExecutor()
        update_executor = QiboSimulatorExecutor()
        model_config = QCNNConfig(
            2,
            execution=ExecutionConfig(shots=12),
            executor=qcnn_executor,
        )
        update_calls = []

        def fake_update(weights, gradients, config, **kwargs):
            del gradients, config
            update_calls.append(kwargs)
            updated = np.asarray(weights, dtype=float)
            return updated, {
                "mode_probabilities": [1.0] * len(updated),
                "minimum_mode_probability": 0.5,
            }

        with (
            patch(
                "src.quantum_training.parameter_shift_gradient",
                return_value=np.zeros((1, 4)),
            ) as gradient,
            patch("src.quantum_training.quantum_update_step", side_effect=fake_update),
            patch(
                "src.quantum_training.predict",
                side_effect=lambda weights, features, config: np.zeros(
                    (len(features), 1)
                ),
            ),
        ):
            train_quantum_qcnn(
                self.features,
                self.targets,
                self.validation_features,
                self.validation_targets,
                self.initial_weights,
                model_config,
                self.qgd_config,
                QuantumTrainingConfig(max_steps=1, shots=12),
                update_executor=update_executor,
            )

        self.assertIs(gradient.call_args.args[3].executor, qcnn_executor)
        self.assertIs(update_calls[0]["executor"], update_executor)
        self.assertEqual(update_calls[0]["shots"], 12)
        self.assertIsNot(gradient.call_args.args[3].executor, update_calls[0]["executor"])

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

        def fake_update(weights, gradients, config, **kwargs):
            del gradients, config
            del kwargs
            updated = np.asarray(weights, dtype=float) + 1.0
            return updated, {
                "mode_probabilities": [1.0] * len(updated),
                "minimum_mode_probability": 0.5,
            }

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

    def test_moving_configuration_updates_weights_through_the_loop(self):
        from src.quantum_update import classical_grid_update

        moving = QGDConfig(n_bits=2, learning_rate=2.0)
        initial = np.array([[0.4, -2.5, 1.3, 2.9]])
        exact_gradient = np.array([[0.9, -0.95, 0.05, 1.5]])
        with patch(
            "src.quantum_training.parameter_shift_gradient",
            return_value=exact_gradient,
        ):
            result = train_quantum_qcnn(
                self.features,
                self.targets,
                self.validation_features,
                self.validation_targets,
                initial,
                self.model_config,
                moving,
                QuantumTrainingConfig(max_steps=1, shots=None),
            )
        expected = classical_grid_update(
            initial.reshape(-1), exact_gradient.reshape(-1), moving
        ).reshape(initial.shape)
        np.testing.assert_array_equal(result.final_weights, expected)
        quantized = classical_grid_update(
            initial.reshape(-1), np.zeros(4), QGDConfig(n_bits=2, learning_rate=1e-9)
        ).reshape(initial.shape)
        self.assertFalse(np.array_equal(result.final_weights, quantized))
        record = result.history[0]
        self.assertEqual(record["gradient"], exact_gradient.reshape(-1).tolist())
        self.assertAlmostEqual(record["gradient_saturated_fraction"], 0.25)

    def test_static_configuration_emits_a_warning(self):
        with self.assertWarnsRegex(RuntimeWarning, "static"):
            self._train(training_config=QuantumTrainingConfig(max_steps=1, shots=None))

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
        initial = components["initial_gradient"]
        self.assertEqual(estimate["logical_weight_occurrences_per_sample"], 4)
        self.assertEqual(estimate["shifted_circuits_per_gradient"], 8)
        self.assertEqual(initial["base_prediction_circuit_executions_estimate"], 1)
        self.assertEqual(initial["parameter_shift_circuit_executions_estimate"], 8)
        self.assertEqual(initial["circuit_executions_estimate"], 9)
        self.assertEqual(
            components["qgd_weight_updates"]["circuit_executions_estimate"], 4
        )
        self.assertEqual(components["qgd_weight_updates"]["shots_per_circuit"], 50)
        self.assertEqual(components["qgd_weight_updates"]["qubits_per_circuit"], 4)
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

    def test_run_provenance_captures_git_state_and_cli_arguments(self):
        from experiments.run_quantum_training import _collect_run_provenance

        with patch(
            "experiments.run_quantum_training.subprocess.run",
            side_effect=(
                SimpleNamespace(stdout="abc123\n"),
                SimpleNamespace(stdout=" M src/quantum_update.py\n?? results/pilot.json\n"),
            ),
        ) as run:
            provenance = _collect_run_provenance(
                ["--dataset", "iris", "--output", "results/pilot.json"]
            )

        self.assertEqual(provenance["git_revision"], "abc123")
        self.assertTrue(provenance["working_tree_dirty"])
        self.assertEqual(
            provenance["working_tree_status"],
            [" M src/quantum_update.py", "?? results/pilot.json"],
        )
        self.assertEqual(
            provenance["command_argv"][1:3],
            ["-m", "experiments.run_quantum_training"],
        )
        self.assertEqual(provenance["command_argv"][-4:], [
            "--dataset", "iris", "--output", "results/pilot.json"
        ])
        self.assertEqual(run.call_count, 2)

    def test_run_provenance_marks_unavailable_git_metadata_unknown(self):
        from experiments.run_quantum_training import _collect_run_provenance

        with patch(
            "experiments.run_quantum_training.subprocess.run",
            side_effect=OSError("git unavailable"),
        ):
            provenance = _collect_run_provenance(["--output", "out.json"])

        self.assertIsNone(provenance["git_revision"])
        self.assertIsNone(provenance["working_tree_dirty"])
        self.assertEqual(provenance["working_tree_status"], [])
        self.assertIn("git unavailable", provenance["git_metadata_error"])

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

    def test_runner_accepts_exact_simulator_shots_mode(self):
        from experiments.run_quantum_training import _build_parser

        args = _build_parser().parse_args(
            ["--output", "out.json", "--shots", "exact"]
        )
        self.assertIsNone(args.shots)

    def test_runner_execution_modes_and_json_are_consistent(self):
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
        scenarios = (
            {
                "name": "default-exact-initial-gradient",
                "options": [],
                "noise": 0.0,
                "requested": False,
                "enabled": False,
                "gradient_mode": "exact_initialization",
                "executor_mode": "default_qibo_simulator",
            },
            {
                "name": "measured-noiseless",
                "options": ["--measured-execution"],
                "noise": 0.0,
                "requested": True,
                "enabled": True,
                "gradient_mode": "parameter_shift_shots",
                "executor_mode": "qibo_simulator_measured",
            },
            {
                "name": "measured-with-sensitivity-noise",
                "options": ["--noise", "0.01"],
                "noise": 0.01,
                "requested": False,
                "enabled": True,
                "gradient_mode": "parameter_shift_shots",
                "executor_mode": "qibo_simulator_measured_with_sensitivity_noise",
            },
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            for scenario in scenarios:
                with self.subTest(mode=scenario["name"]):
                    events = []
                    selected_executor = []

                    def fake_train(*args, **kwargs):
                        events.append("train")
                        model = args[5]
                        training_config = args[7]
                        self.assertEqual(model.execution.depolarizing_probability, 0.0)
                        self.assertEqual(training_config.shots, 10)
                        if scenario["enabled"]:
                            self.assertIsInstance(model.executor, QiboSimulatorExecutor)
                        else:
                            self.assertIsNone(model.executor)
                        self.assertEqual(
                            model.executor is not None, scenario["enabled"]
                        )
                        if model.executor is not None:
                            self.assertEqual(
                                model.executor.noise_transform is not None,
                                scenario["noise"] > 0.0,
                            )
                        selected_executor.append(model.executor)
                        return training_result

                    def fake_predict(weights, features, config):
                        events.append("test_predict")
                        self.assertEqual(config.execution.shots, 10)
                        self.assertEqual(config.execution.depolarizing_probability, 0.0)
                        self.assertIs(config.executor, selected_executor[0])
                        np.testing.assert_array_equal(weights, training_result.best_weights)
                        np.testing.assert_array_equal(features, [[0.5, 0.5]])
                        return np.array([[0.4]])

                    output = Path(temp_directory) / f"{scenario['name']}.json"
                    with (
                        patch.object(runner, "_load_splits", return_value=splits),
                        patch.object(runner, "train_quantum_qcnn", side_effect=fake_train),
                        patch.object(runner, "predict", side_effect=fake_predict),
                    ):
                        runner.main(
                            [
                                "--output",
                                str(output),
                                "--shots",
                                "10",
                                *scenario["options"],
                            ]
                        )

                    self.assertEqual(events, ["train", "test_predict"])
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    model_payload = payload["model"]
                    self.assertEqual(
                        model_payload["depolarizing_probability"], scenario["noise"]
                    )
                    self.assertEqual(
                        model_payload["execution_config_depolarizing_probability"],
                        0.0,
                    )
                    self.assertEqual(
                        model_payload["measured_execution"], scenario["enabled"]
                    )
                    self.assertEqual(
                        model_payload["measured_execution_requested"],
                        scenario["requested"],
                    )
                    self.assertEqual(
                        model_payload["executor_mode"], scenario["executor_mode"]
                    )
                    self.assertEqual(
                        payload["training"]["initial_gradient_mode"],
                        scenario["gradient_mode"],
                    )
                    self.assertEqual(
                        payload["training"]["initial_gradient_shots"],
                        10 if scenario["enabled"] else None,
                    )
                    self.assertEqual(
                        model_payload["simulator_noise"]["requested_probability"],
                        scenario["noise"],
                    )
                    self.assertEqual(
                        model_payload["simulator_noise"]["applied_to"],
                        ["QCNN circuits", "QGD arithmetic update circuits"]
                        if scenario["noise"] > 0
                        else [],
                    )
                    self.assertIn(
                        "finite shots" if scenario["enabled"] else "exact, noiseless",
                        payload["classical_quantum_boundary"]["initial_gradient"],
                    )

    def test_runner_rejects_exact_shots_with_noise_or_measured_mode_before_loading_data(self):
        from experiments import run_quantum_training as runner

        for extra_options in (["--noise", "0.01"], ["--measured-execution"]):
            with self.subTest(options=extra_options):
                with patch.object(runner, "_load_splits") as load_splits:
                    with self.assertRaises(SystemExit):
                        runner.main(
                            [
                                "--output",
                                "unused.json",
                                "--shots",
                                "exact",
                                *extra_options,
                            ]
                        )
                load_splits.assert_not_called()


if __name__ == "__main__":
    unittest.main()
