"""BSC plan, phase 1: cost model, transpiler, hardware mock, readout mitigation and dry run."""

import json
import math
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
from qibo import gates
from qibo.models import Circuit

from src.hardware_cost import SPLIT_SIZES, circuits_per_sample_step, default_scenarios, training_cost
from src.hardware_execution import (
    HardwareMockExecutor,
    MockNoise,
    QililabExecutor,
    calibrate_readout,
    execute_batch,
    mitigate_z_expectation,
    z_expectation,
)
from src.qcnn_engine import QCNNConfig, build_qcnn_circuit
from src.transpilation import NATIVE_GATES, measured_marginal, topology_edges, transpile, unitary

ROOT = Path(__file__).resolve().parents[1]


def equal_up_to_phase(a, b, tol=1e-9):
    index = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    phase = a[index] / b[index]
    return abs(abs(phase) - 1) < tol and np.allclose(a, phase * b, atol=tol)


class CostModelTests(unittest.TestCase):
    def test_circuits_per_sample_step_match_training_accounting(self):
        self.assertEqual(circuits_per_sample_step("gd"), 33)
        self.assertEqual(circuits_per_sample_step("adam"), 33)
        self.assertEqual(circuits_per_sample_step("spsa"), 2)
        self.assertEqual(circuits_per_sample_step("sequential"), 40)
        # Per-step totals reported in the phase-B READMEs.
        self.assertEqual(SPLIT_SIZES["breast-cancer"]["train"] * 33, 13_134)
        self.assertEqual(SPLIT_SIZES["iris"]["train"] * 40, 2_800)

    def test_training_cost_counts_validation_including_step_zero(self):
        estimate = training_cost("x", optimizer="spsa", n_train=70, n_validation=15, steps=10, shots=100)
        self.assertEqual(estimate.circuits, 10 * 70 * 2 + 11 * 15)
        self.assertEqual(estimate.total_shots, estimate.circuits * 100)
        self.assertTrue(all(len(s.hours) == 2 for s in default_scenarios()))

    def test_estimate_matches_an_actual_training_run(self):
        from src.fast_qcnn import FastQCNNConfig
        from src.variational_training import FastEvaluator, VariationalTrainingConfig, train_variational_qcnn

        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 1, (12, 4)), np.where(rng.random(12) > 0.5, 1.0, -1.0)
        xv, yv = rng.uniform(0, 1, (5, 4)), np.where(rng.random(5) > 0.5, 1.0, -1.0)
        for optimizer in ("gd", "adam", "spsa", "sequential"):
            with self.subTest(optimizer=optimizer):
                config = VariationalTrainingConfig(optimizer=optimizer, max_steps=4, evaluate_train_every=2)
                result = train_variational_qcnn(x, y, xv, yv, np.zeros((2, 4)), FastEvaluator(FastQCNNConfig(4)),
                                                config, np.random.default_rng(1))
                estimate = training_cost("x", optimizer=optimizer, n_train=12, n_validation=5, steps=4, shots=1,
                                         train_evaluation_every=2)
                self.assertEqual(estimate.circuits, result.optimization_circuits + result.evaluation_circuits)


class TranspilerTests(unittest.TestCase):
    def test_native_decompositions_are_exact(self):
        for name, build in (
            ("H", lambda c: c.add(gates.H(0))),
            ("CNOT", lambda c: c.add(gates.CNOT(0, 1))),
            ("SWAP", lambda c: c.add(gates.SWAP(0, 1))),
        ):
            with self.subTest(gate=name):
                logical = Circuit(2)
                build(logical)
                result = transpile(logical, [(0, 1)], measured=(0, 1), layout={0: 0, 1: 1})
                if name == "SWAP":
                    # Routing does not move qubits here, so the unitaries must agree.
                    self.assertEqual(result.final_layout, {0: 0, 1: 1})
                self.assertTrue(equal_up_to_phase(unitary(result.circuit), unitary(logical)))

    def test_qcnn_marginal_preserved_on_generic_topologies(self):
        rng = np.random.default_rng(7)
        for encoding in ("angle", "phase"):
            for topology, n in (("full", 4), ("line", 4), ("ring", 6), ("star", 5), ("grid", 9), ("line", 20)):
                with self.subTest(encoding=encoding, topology=topology, n=n):
                    x, w = rng.uniform(0, 1, 4), rng.uniform(-np.pi, np.pi, (2, 4))
                    logical = build_qcnn_circuit(x, w, QCNNConfig(4, encoding=encoding))
                    edges = topology_edges(topology, n)
                    result = transpile(logical, edges)
                    coupled = {tuple(sorted(e)) for e in result.coupling_edges}
                    for gate in result.circuit.queue:
                        name = gate.__class__.__name__
                        if name == "M":
                            continue
                        self.assertIn(name, NATIVE_GATES)
                        if len(gate.qubits) == 2:
                            self.assertIn(tuple(sorted(gate.qubits)), coupled)
                    bit = result.physical_qubits.index(result.measured_physical[0])
                    np.testing.assert_allclose(measured_marginal(result.circuit, [bit]),
                                               measured_marginal(logical, [0]), atol=1e-10)
                    if topology == "full":
                        self.assertEqual(result.stats["swaps"], 0)
                    if topology == "line":
                        # The (0, 2) interaction of the second layer cannot be embedded in a line.
                        self.assertGreater(result.stats["swaps"], 0)


class MockAndMitigationTests(unittest.TestCase):
    def setUp(self):
        self.logical = build_qcnn_circuit(np.array([0.2, 0.7, 0.4, 0.9]),
                                          np.random.default_rng(1).uniform(-2, 2, (2, 4)), QCNNConfig(4))
        self.result = transpile(self.logical, topology_edges("line", 5))
        self.bit = self.result.physical_qubits.index(self.result.measured_physical[0])

    def mock(self, **noise):
        return HardwareMockExecutor(self.result.coupling_edges, self.result.circuit.nqubits, MockNoise(**noise))

    def test_mock_rejects_untranspiled_circuits(self):
        circuit = Circuit(4)
        circuit.add(gates.CNOT(0, 2))
        circuit.add(gates.M(0))
        with self.assertRaises(ValueError):
            HardwareMockExecutor(topology_edges("line", 4), 4).execute_counts(circuit, shots=10)

    def test_noiseless_mock_matches_exact_distribution_and_is_seeded(self):
        mock = self.mock()
        np.testing.assert_allclose(mock.probabilities(self.result.circuit), measured_marginal(self.logical, [0]),
                                   atol=1e-10)
        a = execute_batch(mock, [self.result.circuit] * 3, shots=200, seed=5)
        b = execute_batch(mock, [self.result.circuit] * 3, shots=200, seed=5)
        self.assertEqual(a, b)
        self.assertTrue(all(sum(c.values()) == 200 for c in a))

    def test_depolarizing_noise_shrinks_the_expectation(self):
        exact = measured_marginal(self.logical, [0])
        noisy = self.mock(one_qubit_depolarizing=0.002, two_qubit_depolarizing=0.02).probabilities(self.result.circuit)
        self.assertLess(abs(noisy[0] - noisy[1]), abs(exact[0] - exact[1]))

    def test_readout_mitigation_recovers_expectation(self):
        mock = self.mock(readout_p01=0.03, readout_p10=0.08)
        probabilities = mock.probabilities(self.result.circuit)
        true_z = probabilities[0] - probabilities[1]
        calibration = calibrate_readout(mock, self.result.circuit.nqubits, [self.bit], shots=100_000, seed=1)
        self.assertAlmostEqual(calibration.p01[self.bit], 0.03, delta=0.004)
        self.assertAlmostEqual(calibration.p10[self.bit], 0.08, delta=0.004)
        raw = z_expectation(mock.execute_counts(self.result.circuit, shots=100_000, seed=2))
        mitigated = mitigate_z_expectation(raw, calibration.p01[self.bit], calibration.p10[self.bit])
        # Analytic readout bias: z_raw = (1 - p01 - p10) z + (p10 - p01).
        expected_raw = (1 - 0.03 - 0.08) * true_z + (0.08 - 0.03)
        self.assertAlmostEqual(raw, expected_raw, delta=0.012)
        self.assertAlmostEqual(mitigated, true_z, delta=0.015)
        # Symmetric errors bias the raw estimate towards zero; mitigation undoes it.
        symmetric = self.mock(readout_p01=0.08, readout_p10=0.08)
        raw_sym = z_expectation(symmetric.execute_counts(self.result.circuit, shots=100_000, seed=3))
        self.assertGreater(abs(raw_sym - true_z), 0.025)
        self.assertAlmostEqual(mitigate_z_expectation(raw_sym, 0.08, 0.08), true_z, delta=0.015)

    def test_qililab_adapter_is_an_explicit_placeholder(self):
        with self.assertRaises(NotImplementedError):
            QililabExecutor("runcard.yml", (0, 1, 2, 3)).execute_batch([self.result.circuit], shots=10)

    def test_qililab_loads_external_adapter_and_validates_counts(self):
        module_name = "mock_qililab_adapter_for_test"
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            module_path = Path(tmp) / f"{module_name}.py"
            module_path.write_text(
                "def execute_batch(circuits, shots, seed, runcard, physical_qubits):\n"
                "    assert runcard == 'runcard.yml'\n"
                "    assert tuple(physical_qubits) == (0, 1, 2, 3)\n"
                "    return [{'0': shots // 2, '1': shots - shots // 2} for _ in circuits]\n",
                encoding="utf-8",
            )
            sys.path.insert(0, tmp)
            try:
                executor = QililabExecutor(
                    "runcard.yml", (0, 1, 2, 3),
                    adapter_module=module_name,
                    chip_id="test-chip",
                    max_qubits=4,
                    max_shots=100,
                    basis_gates=("RX", "RY", "RZ", "CZ"),
                    connectivity=((0, 1), (1, 2), (2, 3)),
                )
                counts = executor.execute_batch([self.result.circuit], shots=10, seed=7)
                self.assertEqual(counts, [{"0": 5, "1": 5}])
                self.assertEqual(executor.capabilities.metadata["chip_id"], "test-chip")
                self.assertEqual(executor.capabilities.metadata["compatibility"], "not validated with any real QPU")
                with self.assertRaises(ValueError):
                    executor.capabilities.validate_request(5, 10)
                with self.assertRaises(ValueError):
                    executor.execute_batch([self.result.circuit], shots=101)
            finally:
                sys.path.remove(tmp)
                sys.modules.pop(module_name, None)

    def test_qililab_rejects_malformed_external_counts(self):
        module_name = "mock_qililab_bad_counts_for_test"
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            module_path = Path(tmp) / f"{module_name}.py"
            module_path.write_text(
                "def execute_batch(circuits, shots, seed, runcard, physical_qubits):\n"
                "    return [{'2': shots} for _ in circuits]\n",
                encoding="utf-8",
            )
            sys.path.insert(0, tmp)
            try:
                executor = QililabExecutor(
                    "runcard.yml", (0, 1, 2, 3), adapter_module=module_name,
                )
                with self.assertRaisesRegex(ValueError, "invalid counts|invalid 1-bit"):
                    executor.execute_batch([self.result.circuit], shots=10)
            finally:
                sys.path.remove(tmp)
                sys.modules.pop(module_name, None)

    def test_qililab_rejects_incomplete_cli_configuration(self):
        from experiments.run_hardware_inference import main
        run = ROOT / "results/variational/B-confirm-iris/I1-exact/sequential-angle/run-angle-sequential-init19.json"
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            output = Path(tmp) / "missing.json"
            with self.assertRaisesRegex(SystemExit, "--runcard"):
                main([str(run), "--executor", "qililab", "--output", str(output)])


class DryRunTests(unittest.TestCase):
    def test_inference_dry_run_reproduces_stored_exact_metrics(self):
        from experiments.run_hardware_inference import main

        run = ROOT / "results/variational/B-confirm-iris/I1-exact/sequential-angle/run-angle-sequential-init19.json"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dry.json"
            result = main([str(run), "--executor", "mock", "--topology", "line", "--device-qubits", "5",
                           "--shots", "4000", "--output", str(output)])
            stored = json.loads(run.read_text(encoding="utf-8"))["test"]["metrics"]["accuracy"]
            record = result["runs"][0]
            self.assertTrue(record["exact_matches_stored_test_accuracy"])
            self.assertTrue(math.isclose(record["splits"]["test"]["exact"]["metrics"]["accuracy"], stored))
            # Noise-free mock: only shot noise separates raw from exact expectations.
            self.assertLess(record["splits"]["test"]["mean_abs_error_raw_vs_exact"], 0.05)
            self.assertEqual(result["cost"]["circuits"], 15 + 15 + 2 * 2)
            self.assertTrue(output.exists())
            with self.assertRaises(SystemExit):
                main([str(run), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
