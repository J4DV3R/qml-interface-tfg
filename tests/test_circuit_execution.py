import unittest

import numpy as np
import torch
from qibo import gates
from qibo.models import Circuit

from src.circuit_execution import (
    QiboSimulatorExecutor,
    ResourceProfile,
    qibo_local_depolarizing_sensitivity_transform,
    validate_counts,
)
from src.execution import ExecutionConfig
from src.hybrid_model import QCNNModel
from src.qcnn_engine import QCNNConfig, build_qcnn_circuit, predict
from src.quantum_update import QGDConfig, build_update_circuit, encode_gradient, encode_weight


class NoiseSamplingSemanticsTests(unittest.TestCase):
    """The corrected modes draw independent noise per shot; legacy does not."""

    @staticmethod
    def _flip_counts(simulation: str, seed: int) -> dict:
        circuit = Circuit(1)
        circuit.add(gates.X(0))
        circuit.add(gates.M(0))
        # Bit-flip probability of a depolarizing channel with p=0.6 is 0.3.
        executor = QiboSimulatorExecutor(
            noise_transform=qibo_local_depolarizing_sensitivity_transform(
                0.6, gate_names=("X",), simulation=simulation
            )
        )
        return executor.execute_counts(circuit, shots=400, seed=seed)

    def test_corrected_modes_sample_noise_independently_per_shot(self):
        for simulation in ("density_matrix", "per_shot_trajectories"):
            counts = self._flip_counts(simulation, seed=3)
            with self.subTest(simulation=simulation):
                self.assertEqual(set(counts), {"0", "1"})
                self.assertAlmostEqual(counts.get("0", 0) / 400, 0.3, delta=0.08)

    def test_legacy_mode_shares_one_trajectory_across_shots(self):
        for seed in range(5):
            counts = self._flip_counts("legacy_shared_trajectory", seed=seed)
            self.assertEqual(len(counts), 1)

    def test_density_matrix_channel_is_local_to_its_target_qubit(self):
        circuit = Circuit(2)
        circuit.add(gates.X(1))
        circuit.add(gates.H(0))
        circuit.add(gates.M(1))
        counts = QiboSimulatorExecutor(
            noise_transform=qibo_local_depolarizing_sensitivity_transform(
                1.0, gate_names=("H",)
            )
        ).execute_counts(circuit, shots=200, seed=1)
        # Full depolarizing on qubit 0 must not touch qubit 1.
        self.assertEqual(counts, {"1": 200})

    def test_invalid_simulation_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            qibo_local_depolarizing_sensitivity_transform(0.1, simulation="bogus")


class CircuitExecutionTests(unittest.TestCase):
    def test_resource_profile_checks_declared_qubit_and_shot_limits(self):
        profile = ResourceProfile(
            max_qubits=4,
            max_shots=100,
            basis_gates=("x", "cx"),
            connectivity=((1, 0), (2, 3)),
            metadata={"target": "unselected"},
        )
        profile.validate_request(4, 100)
        with self.assertRaisesRegex(ValueError, "qubits"):
            profile.validate_request(5, 10)
        with self.assertRaisesRegex(ValueError, "shots"):
            profile.validate_request(2, 101)
        serialized = profile.to_dict()
        self.assertEqual(serialized["declared"]["connectivity"], [[0, 1], [2, 3]])
        self.assertFalse(serialized["validation"]["performed"])
        self.assertFalse(serialized["transpilation"]["performed"])

    def test_resource_profile_rejects_invalid_capabilities(self):
        for kwargs in (
            {"max_qubits": 0},
            {"max_shots": True},
            {"basis_gates": ("x", "")},
            {"connectivity": ((0, 0),)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ResourceProfile(**kwargs)

    def test_counts_validate_bitstring_width_and_requested_shots(self):
        self.assertEqual(
            validate_counts({"00": 2, "11": np.int64(3)}, n_bits=2, shots=5),
            {"00": 2, "11": 3},
        )
        for counts in (
            {"0": 2},
            {"0x": 2},
            {"00": 0},
            {"00": 2},
        ):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                validate_counts(counts, n_bits=2, shots=2 if counts == {"00": 0} else 3)

    def test_qibo_executor_returns_reproducible_measured_counts(self):
        circuit = Circuit(1)
        circuit.add(gates.H(0))
        circuit.add(gates.M(0))
        executor = QiboSimulatorExecutor()
        first = executor.execute_counts(circuit, shots=128, seed=117)
        repeated = executor.execute_counts(circuit, shots=128, seed=117)
        self.assertEqual(first, repeated)
        self.assertEqual(sum(first.values()), 128)
        self.assertEqual(set(first), {"0", "1"})

    def test_noise_transform_receives_complete_measured_circuit(self):
        seen = []

        def transform(circuit):
            seen.append(circuit)
            return circuit

        circuit = Circuit(1)
        circuit.add(gates.X(0))
        circuit.add(gates.M(0))
        result = QiboSimulatorExecutor(noise_transform=transform).execute_counts(
            circuit, shots=3, seed=9
        )
        self.assertEqual(seen, [circuit])
        self.assertEqual(result, {"1": 3})

    def test_qcnn_sampled_prediction_uses_injected_counts_adapter(self):
        class CountsAdapter:
            capabilities = ResourceProfile(max_qubits=2, max_shots=20)

            def execute_counts(self, circuit, *, shots, seed=None):
                del seed
                self.asserted = (circuit.nqubits, shots, bool(circuit.measurements))
                return {"0": shots}

        adapter = CountsAdapter()
        model = QCNNConfig(
            2,
            execution=ExecutionConfig(shots=7),
            executor=adapter,
        )
        predictions = predict(np.zeros((1, 4)), [[0.2, 0.8]], model)
        np.testing.assert_array_equal(predictions, [[1.0]])
        self.assertEqual(adapter.asserted, (2, 7, True))

    def test_local_sensitivity_noise_runs_qcnn_and_controlled_update_counts(self):
        noisy_circuits = []
        local_transform = qibo_local_depolarizing_sensitivity_transform(0.001)

        def tracked_transform(circuit):
            noisy = local_transform(circuit)
            noisy_circuits.append(noisy)
            return noisy

        executor = QiboSimulatorExecutor(noise_transform=tracked_transform)
        model = QCNNConfig(
            2,
            execution=ExecutionConfig(shots=16, seed=23),
            executor=executor,
        )
        predictions = predict(np.zeros((1, 4)), [[0.2, 0.8]], model)
        self.assertEqual(predictions.shape, (1, 1))

        qgd_config = QGDConfig(n_bits=2, learning_rate=2.0)
        update_circuit = build_update_circuit(
            encode_weight(0.4, qgd_config),
            encode_gradient(0.7, qgd_config),
            qgd_config,
        )
        update_counts = executor.execute_counts(
            update_circuit, shots=16, seed=29
        )

        self.assertEqual(len(noisy_circuits), 2)
        self.assertTrue(
            all(
                any(gate.name == "PauliNoiseChannel" for gate in circuit.queue)
                for circuit in noisy_circuits
            )
        )
        arithmetic_noisy = noisy_circuits[1]
        self.assertTrue(
            any(
                len(gate.qubits) >= 3
                and any(
                    following.name == "PauliNoiseChannel"
                    for following in arithmetic_noisy.queue[index + 1 : index + 2]
                )
                for index, gate in enumerate(arithmetic_noisy.queue)
            )
        )
        self.assertEqual(sum(update_counts.values()), 16)

    def test_named_sensitivity_transform_only_noises_qcnn_ry_rz_and_cnot(self):
        circuit = Circuit(3)
        circuit.add(gates.RY(0, theta=0.1))
        circuit.add(gates.RZ(1, theta=0.2))
        circuit.add(gates.CNOT(0, 2))
        circuit.add(gates.H(1))
        circuit.add(gates.X(2))
        circuit.add(gates.M(0, 2))

        transformed = qibo_local_depolarizing_sensitivity_transform(
            0.02, gate_names=("RY", "RZ", "CNOT")
        )(circuit)
        queue = transformed.queue
        names = [gate.name for gate in queue]
        self.assertEqual(
            names,
            [
                "ry",
                "PauliNoiseChannel",
                "rz",
                "PauliNoiseChannel",
                "cx",
                "PauliNoiseChannel",
                "PauliNoiseChannel",
                "h",
                "x",
                "measure",
            ],
        )
        channels = [gate for gate in queue if gate.name == "PauliNoiseChannel"]
        self.assertEqual([gate.qubits for gate in channels], [(0,), (1,), (0,), (2,)])
        self.assertEqual(len(transformed.measurements), 1)

    def test_hybrid_forward_and_parameter_shift_share_injected_executor(self):
        transform_inputs = []
        shared_transform = qibo_local_depolarizing_sensitivity_transform(
            0.015, gate_names=("RY", "RZ", "CNOT")
        )

        def tracked_transform(circuit):
            transform_inputs.append(circuit)
            return shared_transform(circuit)

        executor = QiboSimulatorExecutor(noise_transform=tracked_transform)
        model = QCNNModel(
            2,
            execution_config=ExecutionConfig(shots=24, seed=17),
            executor=executor,
        )
        features = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
        model(features).square().mean().backward()

        self.assertGreater(len(transform_inputs), 1)
        self.assertIs(model.executor, executor)
        for circuit in transform_inputs:
            transformed = shared_transform(circuit)
            transformed_names = [gate.name for gate in transformed.queue]
            self.assertIn("measure", transformed_names)
            transformed_index = 0
            for gate in circuit.queue:
                self.assertEqual(transformed.queue[transformed_index].name, gate.name)
                transformed_index += 1
                if gate.name in {"ry", "rz", "cx"}:
                    for qubit in gate.qubits:
                        channel = transformed.queue[transformed_index]
                        self.assertEqual(channel.name, "PauliNoiseChannel")
                        self.assertEqual(channel.qubits, (qubit,))
                        transformed_index += 1
            self.assertEqual(transformed_index, len(transformed.queue))


if __name__ == "__main__":
    unittest.main()
