import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from qibo import gates

from experiments import compare_qgd_adam as comparison
from src.qcnn_engine import build_qcnn_circuit
from src.quantum_update import QGDConfig, build_update_circuit


class CompareQgdAdamTests(unittest.TestCase):
    @staticmethod
    def _adam_result(*, model=None):
        return comparison.AdamTrainingResult(
            best_weights=np.full((1, 4), 0.25),
            final_weights=np.full((1, 4), 0.75),
            best_epoch=1,
            history=(
                {"epoch": 1, "validation_loss": 0.2, "train_loss": 0.3},
                {"epoch": 2, "validation_loss": 0.4, "train_loss": 0.1},
            ),
            elapsed_seconds=0.01,
            model=model,
        )

    @staticmethod
    def _qgd_result():
        return SimpleNamespace(
            best_weights=np.full((1, 4), -0.25),
            final_weights=np.full((1, 4), -0.75),
            best_step=2,
            steps_completed=2,
            history=(
                {"step": 1, "validation_loss": 0.6, "train_loss": 0.5},
                {"step": 2, "validation_loss": 0.1, "train_loss": 0.4},
            ),
            elapsed_seconds=0.02,
            circuit_executions_estimate={"total_training_circuit_executions_estimate": 20},
        )

    @staticmethod
    def _splits(*, forbid_test=False):
        class Splits:
            train_features = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
            train_targets = torch.tensor([[-1.0]], dtype=torch.float32)
            validation_features = torch.tensor([[0.8, 0.3]], dtype=torch.float32)
            validation_targets = torch.tensor([[1.0]], dtype=torch.float32)
            _test_features = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
            test_targets = torch.tensor([[1.0]], dtype=torch.float32)

            @property
            def test_features(self):
                if forbid_test:
                    raise AssertionError("screening must not access test features")
                return self._test_features

        return Splits()

    def _run_main(
        self, output, *, screening=True, qgd_only=False, extra=(), splits=None
    ):
        initial_weights = np.array([[0.11, -0.12, 0.13, -0.14]])
        adam_calls = []
        qgd_calls = []
        events = []

        def fake_adam(*args, **kwargs):
            events.append("adam_training_and_checkpoint")
            adam_calls.append((args, kwargs))
            return self._adam_result(model=object())

        def fake_qgd(*args, **kwargs):
            events.append("qgd_training_and_checkpoint")
            qgd_calls.append((args, kwargs))
            return self._qgd_result()

        patches = [
            patch.object(
                comparison,
                "_load_splits",
                return_value=splits or self._splits(forbid_test=screening),
            ),
            patch.object(comparison, "initialize_weights", return_value=initial_weights),
            patch.object(comparison, "train_adam_with_validation", side_effect=fake_adam),
            patch.object(comparison, "train_quantum_qcnn", side_effect=fake_qgd),
            patch.object(
                comparison,
                "_collect_run_provenance",
                return_value={"git_revision": "test-revision", "working_tree_dirty": True},
            ),
        ]
        if not screening:
            def adam_test(*args, **kwargs):
                events.append("adam_test")
                return {"mse_loss": 0.25, "metrics": {"accuracy": 1.0}}

            def qgd_test(*args, **kwargs):
                events.append("qgd_test")
                return np.array([[0.5]])

            patches.extend(
                [
                    patch.object(comparison, "_evaluate_adam_partition", side_effect=adam_test),
                    patch.object(comparison, "predict", side_effect=qgd_test),
                ]
            )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            if not screening:
                with patches[5], patches[6]:
                    options = ["--output", str(output), *extra]
                    if qgd_only:
                        options.append("--qgd-only")
                    if screening:
                        options.extend(["--screening", "--epochs", "2"])
                    else:
                        options.extend(["--epochs", "2"])
                    comparison.main(options)
            else:
                options = ["--output", str(output), *extra]
                if qgd_only:
                    options.append("--qgd-only")
                if screening:
                    options.extend(["--screening", "--epochs", "2"])
                else:
                    options.extend(["--epochs", "2"])
                comparison.main(options)
        return comparison._prepare_shared_initial_weights(initial_weights), adam_calls, qgd_calls, events

    def test_screening_shares_initial_weights_and_never_accesses_test(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "screen.json"
            initial, adam_calls, qgd_calls, _ = self._run_main(output)

            adam_initial = adam_calls[0][0][4]
            qgd_initial = qgd_calls[0][0][4]
            self.assertIs(adam_initial, qgd_initial)
            np.testing.assert_array_equal(adam_initial, initial)
            np.testing.assert_array_equal(adam_initial, qgd_initial)
            adam_executor = adam_calls[0][1]["executor"]
            qgd_model_config = qgd_calls[0][0][5]
            update_executor = qgd_calls[0][1]["update_executor"]
            self.assertIs(adam_executor, qgd_model_config.executor)
            self.assertIsNotNone(adam_executor)
            self.assertIsNotNone(update_executor)
            self.assertIsNone(adam_executor.noise_transform)
            self.assertIsNone(update_executor.noise_transform)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["methods"]["adam"]["test"]["evaluated"])
            self.assertFalse(payload["methods"]["qgd"]["test"]["evaluated"])
            self.assertEqual(payload["methods"]["adam"]["checkpoint"]["epoch"], 1)
            self.assertEqual(payload["methods"]["qgd"]["checkpoint"]["step"], 2)
            np.testing.assert_array_equal(
                payload["methods"]["adam"]["checkpoint"]["weights"],
                self._adam_result().best_weights,
            )
            np.testing.assert_array_equal(
                payload["methods"]["qgd"]["checkpoint"]["weights"],
                self._qgd_result().best_weights,
            )

    def test_exact_mode_is_allowed_only_without_noise_executors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "exact-screen.json"
            _, adam_calls, qgd_calls, _ = self._run_main(
                output,
                extra=("--shots", "exact"),
            )
            self.assertIsNone(adam_calls[0][1]["executor"])
            self.assertIsNone(qgd_calls[0][0][5].executor)
            self.assertIsNone(qgd_calls[0][1]["update_executor"])
            self.assertIsNone(qgd_calls[0][0][7].shots)

    def test_screening_defaults_to_two_and_confirmation_to_thirty_steps(self):
        parser = comparison._build_parser()
        screening_args = parser.parse_args(
            ["--output", "screen.json", "--screening"]
        )
        confirmation_args = parser.parse_args(["--output", "confirm.json"])
        self.assertEqual(comparison._effective_epochs(screening_args), 2)
        self.assertEqual(comparison._effective_epochs(confirmation_args), 30)
        too_long_screen = parser.parse_args(
            ["--output", "bad-screen.json", "--screening", "--epochs", "30"]
        )
        with self.assertRaisesRegex(ValueError, "fixed to exactly 2"):
            comparison._effective_epochs(too_long_screen)

    def test_adam_effective_parameter_matches_shared_qgd_initial_array(self):
        raw_initial = comparison.initialize_weights(4, seed=29)
        qgd_initial = comparison._prepare_shared_initial_weights(raw_initial)
        adam_model = comparison._build_initialized_adam_model(
            qgd_initial,
            n_qubits=4,
            encoding="angle",
            shots=None,
            execution_seed=101,
            executor=None,
        )
        adam_effective = adam_model.weights.detach().cpu().numpy()
        self.assertEqual(qgd_initial.dtype, np.dtype(np.float64))
        self.assertEqual(adam_effective.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(adam_effective, qgd_initial)

    def test_qgd_only_does_not_train_adam_or_claim_a_paired_comparison(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "u0.json"
            _, adam_calls, qgd_calls, _ = self._run_main(
                output,
                qgd_only=True,
                extra=("--shots", "16", "--p-update", "0.01"),
            )
            self.assertEqual(adam_calls, [])
            self.assertEqual(len(qgd_calls), 1)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_mode"], "qgd_only")
            self.assertEqual(set(payload["methods"]), {"qgd"})
            self.assertNotIn("adam", payload["configuration"]["methods"])
            self.assertEqual(payload["model"]["noise"]["p_model"]["probability"], 0.0)
            self.assertEqual(payload["model"]["noise"]["p_update"]["probability"], 0.01)
            self.assertFalse(payload["methods"]["qgd"]["test"]["evaluated"])

    def test_qgd_only_confirmation_evaluates_only_qgd_test_once(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "u0-confirm.json"
            _, adam_calls, _, events = self._run_main(
                output,
                screening=False,
                qgd_only=True,
                extra=("--shots", "16", "--p-update", "0.01"),
                splits=self._splits(),
            )
            self.assertEqual(adam_calls, [])
            self.assertEqual(events, ["qgd_training_and_checkpoint", "qgd_test"])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["methods"]), {"qgd"})
            self.assertTrue(
                payload["methods"]["qgd"]["test"][
                    "evaluated_once_after_checkpoint_selection"
                ]
            )

    def test_qgd_only_rejects_model_noise_before_loading_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "invalid-u0.json"
            with patch.object(comparison, "_load_splits") as load_splits:
                with self.assertRaises(SystemExit):
                    comparison.main(
                        [
                            "--output",
                            str(output),
                            "--qgd-only",
                            "--p-model",
                            "0.001",
                        ]
                    )
            load_splits.assert_not_called()
            self.assertFalse(output.exists())

    def test_exact_mode_rejects_nonzero_noise_before_loading_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "invalid-exact.json"
            with patch.object(comparison, "_load_splits") as load_splits:
                with self.assertRaises(SystemExit):
                    comparison.main(
                        [
                            "--output",
                            str(output),
                            "--shots",
                            "exact",
                            "--p-model",
                            "0.001",
                        ]
                    )
            load_splits.assert_not_called()
            self.assertFalse(output.exists())

    def test_model_and_update_noise_executors_have_disjoint_scopes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "noise-screen.json"
            _, adam_calls, qgd_calls, _ = self._run_main(
                output,
                extra=(
                    "--shots", "16", "--p-model", "0.02", "--p-update", "0.04"
                ),
            )
            model_executor = adam_calls[0][1]["executor"]
            qgd_model_config = qgd_calls[0][0][5]
            update_executor = qgd_calls[0][1]["update_executor"]
            self.assertIs(model_executor, qgd_model_config.executor)
            self.assertIsNot(model_executor, update_executor)
            self.assertIsNotNone(model_executor.noise_transform)
            self.assertIsNotNone(update_executor.noise_transform)

            model_circuit = build_qcnn_circuit(
                [0.2, 0.4], [[0.1, 0.2, 0.3, 0.4]], qgd_model_config
            )
            model_circuit.add(gates.M(1))
            transformed_model = model_executor.noise_transform(model_circuit)
            model_channels = [
                index
                for index, gate in enumerate(transformed_model.queue)
                if gate.name == "PauliNoiseChannel"
            ]
            self.assertEqual(
                len(model_channels),
                sum(
                    len(gate.qubits)
                    for gate in model_circuit.queue
                    if gate.name in {"ry", "rz", "cx"}
                ),
            )

            config = qgd_calls[0][0][6]
            update_circuit = build_update_circuit(3, 2, config)
            transformed_update = update_executor.noise_transform(update_circuit)
            update_channels = [
                index
                for index, gate in enumerate(transformed_update.queue)
                if gate.name == "PauliNoiseChannel"
            ]
            self.assertEqual(
                len(update_channels),
                sum(
                    len(gate.qubits)
                    for gate in update_circuit.queue
                    if gate.name != "measure"
                    and not gate.__class__.__name__.endswith("Channel")
                ),
            )
            self.assertTrue(
                any(
                    transformed_update.queue[index - 1].name == "x"
                    for index in update_channels
                )
            )
            self.assertEqual(
                transformed_model.queue[-1].name,
                "measure",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            noise = payload["model"]["noise"]
            self.assertEqual(noise["p_model"]["probability"], 0.02)
            self.assertEqual(noise["p_model"]["eligible_gate_names"], ["RY", "RZ", "CNOT"])
            self.assertEqual(noise["p_update"]["probability"], 0.04)
            self.assertIn("QGD arithmetic update circuits only", noise["p_update"]["scope"])

    def test_confirmation_evaluates_test_once_after_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "confirm.json"
            initial, adam_calls, qgd_calls, events = self._run_main(
                output,
                screening=False,
                splits=self._splits(),
            )
            self.assertEqual(
                events,
                [
                    "adam_training_and_checkpoint",
                    "qgd_training_and_checkpoint",
                    "adam_test",
                    "qgd_test",
                ],
            )
            self.assertIs(adam_calls[0][0][4], qgd_calls[0][0][4])
            np.testing.assert_array_equal(adam_calls[0][0][4], initial)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(
                payload["methods"]["adam"]["test"][
                    "evaluated_once_after_checkpoint_selection"
                ]
            )
            self.assertTrue(
                payload["methods"]["qgd"]["test"][
                    "evaluated_once_after_checkpoint_selection"
                ]
            )

    def test_existing_and_historical_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            existing = Path(temporary_directory) / "existing.json"
            existing.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                comparison._resolve_new_output_path(existing)
        historical = comparison.HISTORICAL_ADAM_RESULTS / "would-not-be-created.json"
        with self.assertRaisesRegex(ValueError, "historical results are frozen"):
            comparison._resolve_new_output_path(historical)

    def test_adam_uses_one_optimizer_and_selects_minimum_validation_checkpoint(self):
        optimizer_instances = []

        class FakeModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.n_qubits = kwargs["n_qubits"]
                self.n_outputs = 1
                self.weights = torch.nn.Parameter(torch.zeros((1, 4)))
                self.execution_config = kwargs["execution_config"]

            def forward(self, features):
                parameter = self.weights[0, 0]
                value = 1.5 - parameter if float(features[0, 0]) > 0.5 else parameter
                return value.expand((len(features), 1))

        class FakeAdam:
            def __init__(self, parameters, lr):
                del lr
                self.parameters = list(parameters)
                self.steps = 0
                optimizer_instances.append(self)

            def zero_grad(self):
                for parameter in self.parameters:
                    parameter.grad = None

            def step(self):
                self.steps += 1
                with torch.no_grad():
                    self.parameters[0][0, 0] = float(self.steps)

        with (
            patch.object(comparison, "QCNNModel", side_effect=FakeModel),
            patch.object(comparison.torch.optim, "Adam", FakeAdam),
        ):
            result = comparison.train_adam_with_validation(
                np.array([[0.0, 0.0]]),
                np.array([[-1.0]]),
                np.array([[1.0, 1.0]]),
                np.array([[1.0]]),
                np.zeros((1, 4)),
                n_qubits=2,
                encoding="angle",
                shots=None,
                execution_seed=101,
                epochs=2,
                learning_rate=0.03,
                executor=None,
            )

        self.assertEqual(len(optimizer_instances), 1)
        self.assertEqual(optimizer_instances[0].steps, 2)
        self.assertEqual(result.best_epoch, 1)
        self.assertLess(
            result.history[0]["validation_loss"],
            result.history[1]["validation_loss"],
        )
        self.assertEqual(result.best_weights[0, 0], 1.0)
        self.assertEqual(result.final_weights[0, 0], 2.0)
        self.assertEqual(
            result.circuit_executions_estimate[
                "per_epoch"
            ]["total_circuit_executions_estimate"],
            11,
        )


if __name__ == "__main__":
    unittest.main()
