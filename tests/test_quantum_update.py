import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from qibo.models import Circuit

from src.circuit_execution import QiboSimulatorExecutor, ResourceProfile
from src.quantum_update import (
    QGDConfig,
    build_qgd_update_circuit,
    build_update_circuit,
    build_update_unitary,
    classical_grid_update,
    decode_update_counts,
    decode_gradient,
    decode_weight,
    encode_gradient,
    encode_weight,
    estimate_update_resources,
    quantum_update_step,
    quantum_update_weight,
)


class QuantumUpdateTests(unittest.TestCase):
    def test_grid_round_trip_and_clipping(self):
        for bits in (2, 3, 4, 5):
            config = QGDConfig(n_bits=bits)
            for code in range(config.n_levels):
                self.assertEqual(encode_weight(decode_weight(code, config), config), code)
                self.assertEqual(encode_gradient(decode_gradient(code, config), config), code)
            self.assertEqual(encode_weight(-100.0, config), 0)
            self.assertEqual(encode_weight(100.0, config), config.n_levels - 1)
            self.assertEqual(encode_gradient(-100.0, config), 0)
            self.assertEqual(encode_gradient(100.0, config), config.n_levels - 1)

    def test_unitary_is_permutation(self):
        for bits in (2, 3, 4):
            unitary = build_update_unitary(QGDConfig(n_bits=bits))
            np.testing.assert_array_equal(unitary.sum(axis=0), 1)
            np.testing.assert_array_equal(unitary.sum(axis=1), 1)
            np.testing.assert_array_equal(unitary.real, unitary.real.astype(bool))
            np.testing.assert_allclose(unitary @ unitary.conj().T, np.eye(len(unitary)))

    def test_all_basis_mappings_match_reference_for_small_registers(self):
        for bits in (2, 3):
            config = QGDConfig(n_bits=bits)
            unitary = build_update_unitary(config)
            levels = config.n_levels
            for gradient_code in range(levels):
                for weight_code in range(levels):
                    source = (weight_code << bits) + gradient_code
                    target = int(np.argmax(np.abs(unitary[:, source])))
                    gradient = decode_gradient(gradient_code, config)
                    weight = decode_weight(weight_code, config)
                    expected = classical_grid_update([weight], [gradient], config)[0]
                    expected_code = encode_weight(expected, config)
                    self.assertEqual(target >> bits, expected_code)
                    self.assertEqual(target & (levels - 1), gradient_code)

    def test_circuit_update_matches_discrete_reference(self):
        for bits, count, seed in ((2, 5, 1), (3, 3, 2), (4, 2, 3)):
            config = QGDConfig(n_bits=bits, learning_rate=0.1)
            rng = np.random.default_rng(seed)
            weights = rng.uniform(-np.pi, np.pi, count)
            gradients = rng.uniform(-1.0, 1.0, count)
            expected = classical_grid_update(weights, gradients, config)
            actual = quantum_update_step(weights, gradients, config)
            np.testing.assert_array_equal(actual, expected)
            expected_codes = [encode_weight(value, config) for value in expected]
            actual_codes = [encode_weight(value, config) for value in actual]
            self.assertEqual(actual_codes, expected_codes)

    def test_single_weight_and_all_weights_use_measured_circuits(self):
        config = QGDConfig(n_bits=3, learning_rate=0.2)
        weights = np.array([-2.7, -0.1, 2.8, 0.9])
        gradients = np.array([0.8, -0.4, 1.7, -1.9])
        expected = classical_grid_update(weights, gradients, config)

        one_weight = quantum_update_weight(
            weights[0], gradients[0], config, shots=5, seed=19
        )
        actual, report = quantum_update_step(
            weights, gradients, config, shots=5, seed=19, return_report=True
        )

        self.assertEqual(one_weight, expected[0])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(report["qubits_per_weight_circuit"], 6)
        self.assertEqual(report["shots_per_weight_circuit"], 5)
        self.assertEqual(report["total_shots_estimate"], 20)
        self.assertTrue(all(probability == 1.0 for probability in report["mode_probabilities"]))
        self.assertIn("serial depth upper bound", report["resource_estimate_scope"])

    def test_generic_arithmetic_circuit_matches_every_small_basis_input(self):
        config = QGDConfig(n_bits=2, learning_rate=0.3)
        arithmetic = build_qgd_update_circuit(config, measure=False)
        self.assertFalse(any(gate.name == "Unitary" for gate in arithmetic.queue))
        for gradient_code in range(config.n_levels):
            for weight_code in range(config.n_levels):
                circuit = build_update_circuit(weight_code, gradient_code, config)
                result = circuit.execute(nshots=1)
                counts = result.frequencies(binary=True)
                decoded, confidence = decode_update_counts(
                    counts, config, shots=1
                )
                expected = classical_grid_update(
                    [decode_weight(weight_code, config)],
                    [decode_gradient(gradient_code, config)],
                    config,
                )[0]
                self.assertEqual(decoded, expected)
                self.assertEqual(confidence, 1.0)

    def test_counts_policy_rejects_ties_and_low_confidence(self):
        config = QGDConfig(n_bits=2)
        with self.assertRaisesRegex(RuntimeError, "unique modal"):
            decode_update_counts({"0000": 2, "1100": 2}, config, shots=4)
        with self.assertRaisesRegex(RuntimeError, "below the required"):
            decode_update_counts(
                {"0000": 2, "1100": 1, "1010": 1},
                config,
                shots=4,
                minimum_mode_probability=0.75,
            )
        with self.assertRaises(ValueError):
            decode_update_counts({"00": 1}, config, shots=1)

    def test_qibo_sampled_update_does_not_access_state_vector(self):
        config = QGDConfig(n_bits=2)
        expected_code = encode_weight(
            classical_grid_update([0.4], [0.7], config)[0], config
        )
        expected_bits = format(expected_code, "02b") + format(
            encode_gradient(0.7, config), "02b"
        )

        class SampledResult:
            def frequencies(self, binary=False):
                self.asserted_binary = binary
                return {expected_bits: 8}

            def state(self):
                raise AssertionError("sampled update must not inspect state vector")

        result = SampledResult()
        executor = QiboSimulatorExecutor()
        with patch.object(Circuit, "execute", return_value=result) as execute:
            actual = quantum_update_weight(
                0.4, 0.7, config, shots=8, executor=executor
            )
        self.assertEqual(actual, decode_weight(expected_code, config))
        self.assertTrue(result.asserted_binary)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs["nshots"], 8)

    def test_update_executor_accepts_an_injected_noise_transform(self):
        seen = []

        def transform(circuit):
            seen.append(circuit)
            return circuit

        config = QGDConfig(n_bits=2, learning_rate=2.0)
        quantum_update_weight(
            0.4,
            0.7,
            config,
            shots=2,
            executor=QiboSimulatorExecutor(noise_transform=transform),
        )
        self.assertEqual(len(seen), 1)
        self.assertTrue(
            any(
                len(gate.qubits) >= 3 and gate.name != "measure"
                for gate in seen[0].queue
            )
        )
        self.assertTrue(seen[0].measurements)

    def test_declared_update_resource_limits_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "explicit finite shots"):
            quantum_update_weight(
                0.0,
                0.0,
                QGDConfig(n_bits=2),
                executor=QiboSimulatorExecutor(),
            )
        profile = ResourceProfile(max_qubits=3, max_shots=10)
        executor = QiboSimulatorExecutor(capabilities=profile)
        with self.assertRaisesRegex(ValueError, "declared maximum"):
            quantum_update_weight(
                0.0, 0.0, QGDConfig(n_bits=2), shots=1, executor=executor
            )
        with self.assertRaisesRegex(ValueError, "declared maximum"):
            quantum_update_weight(
                0.0,
                0.0,
                QGDConfig(n_bits=2),
                shots=11,
                executor=QiboSimulatorExecutor(
                    capabilities=ResourceProfile(max_qubits=4, max_shots=10)
                ),
            )
        with self.assertRaises(ValueError):
            ResourceProfile(max_qubits=2, connectivity=((0, 2),))

    def test_reversible_mapping_and_torch_independence(self):
        config = QGDConfig(n_bits=2)
        unitary = build_update_unitary(config)
        self.assertEqual(unitary.shape, (16, 16))
        code = "import sys; import src.quantum_update; assert 'torch' not in sys.modules"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_update_resource_estimates_distinguish_declared_from_transpiled(self):
        profile = ResourceProfile(
            max_qubits=10,
            max_shots=1000,
            basis_gates=("x", "cx"),
            connectivity=((0, 1),),
        )
        estimate = estimate_update_resources(
            QGDConfig(n_bits=2),
            n_weights=4,
            shots=100,
            resource_profile=profile,
        )
        self.assertEqual(estimate["qubits_per_weight_circuit"], 4)
        self.assertEqual(estimate["total_shots_estimate"], 400)
        self.assertFalse(estimate["resource_profile"]["validation"]["performed"])
        self.assertFalse(estimate["resource_profile"]["transpilation"]["performed"])

    def test_invalid_config_and_arrays(self):
        for kwargs in (
            {"n_bits": 1}, {"n_bits": True}, {"weight_min": 1, "weight_max": 1},
            {"gradient_min": np.nan}, {"learning_rate": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QGDConfig(**kwargs)
        config = QGDConfig(n_bits=2)
        with self.assertRaises(ValueError):
            classical_grid_update([0, 1], [0], config)
        with self.assertRaises(ValueError):
            classical_grid_update([np.inf], [0], config)


class MeasuredUpdateCountsTests(unittest.TestCase):
    """Deterministic M5 tests for finite-count decoding and rejection paths."""

    class ControlledCountsExecutor:
        def __init__(self, *responses):
            self.capabilities = ResourceProfile()
            self.responses = list(responses)
            self.requests = []

        def execute_counts(self, circuit, *, shots, seed=None):
            self.requests.append((circuit, shots, seed))
            return self.responses.pop(0)

    def setUp(self):
        self.config = QGDConfig(n_bits=2, learning_rate=2.0)
        self.weight_code = 2
        self.gradient_code = 3
        self.weight = decode_weight(self.weight_code, self.config)
        self.gradient = decode_gradient(self.gradient_code, self.config)
        self.expected_weight = classical_grid_update(
            [self.weight], [self.gradient], self.config
        )[0]
        self.expected_weight_code = encode_weight(self.expected_weight, self.config)

    @staticmethod
    def _bitstring(weight_code, gradient_code, config):
        return (
            f"{weight_code:0{config.n_bits}b}"
            f"{gradient_code:0{config.n_bits}b}"
        )

    def test_step_decodes_correct_mode_at_the_confidence_threshold(self):
        other_weight_code = (self.expected_weight_code + 1) % self.config.n_levels
        counts = {
            self._bitstring(
                self.expected_weight_code, self.gradient_code, self.config
            ): 7,
            self._bitstring(other_weight_code, self.gradient_code, self.config): 3,
        }
        executor = self.ControlledCountsExecutor(counts)

        actual = quantum_update_step(
            [self.weight],
            [self.gradient],
            self.config,
            shots=10,
            executor=executor,
            minimum_mode_probability=0.7,
        )

        np.testing.assert_array_equal(actual, [self.expected_weight])
        self.assertEqual(len(executor.requests), 1)
        self.assertEqual(executor.requests[0][1], 10)

    def test_step_returns_a_confidently_wrong_modal_weight(self):
        wrong_weight_code = (self.expected_weight_code + 1) % self.config.n_levels
        counts = {
            self._bitstring(wrong_weight_code, self.gradient_code, self.config): 8,
            self._bitstring(
                self.expected_weight_code, self.gradient_code, self.config
            ): 2,
        }
        executor = self.ControlledCountsExecutor(counts)

        actual = quantum_update_step(
            [self.weight],
            [self.gradient],
            self.config,
            shots=10,
            executor=executor,
        )

        self.assertEqual(actual[0], decode_weight(wrong_weight_code, self.config))
        self.assertNotEqual(actual[0], self.expected_weight)

    def test_step_rejects_a_corrupted_modal_gradient_register(self):
        corrupted_gradient_code = (self.gradient_code + 1) % self.config.n_levels
        counts = {
            self._bitstring(
                self.expected_weight_code, corrupted_gradient_code, self.config
            ): 8,
            self._bitstring(
                self.expected_weight_code, self.gradient_code, self.config
            ): 2,
        }
        executor = self.ControlledCountsExecutor(counts)

        with self.assertRaisesRegex(RuntimeError, "gradient register differs"):
            quantum_update_step(
                [self.weight],
                [self.gradient],
                self.config,
                shots=10,
                executor=executor,
            )
        self.assertEqual(len(executor.requests), 1)

    def test_step_propagates_tie_and_low_confidence_rejections(self):
        other_weight_code = (self.expected_weight_code + 1) % self.config.n_levels
        another_weight_code = (self.expected_weight_code + 2) % self.config.n_levels
        expected_bits = self._bitstring(
            self.expected_weight_code, self.gradient_code, self.config
        )
        cases = (
            (
                "tie",
                {
                    expected_bits: 5,
                    self._bitstring(
                        other_weight_code, self.gradient_code, self.config
                    ): 5,
                },
                "unique modal",
            ),
            (
                "below threshold",
                {
                    expected_bits: 4,
                    self._bitstring(
                        other_weight_code, self.gradient_code, self.config
                    ): 3,
                    self._bitstring(
                        another_weight_code, self.gradient_code, self.config
                    ): 3,
                },
                "below the required",
            ),
        )

        for label, counts, message in cases:
            with self.subTest(policy_case=label):
                executor = self.ControlledCountsExecutor(counts)
                with self.assertRaisesRegex(RuntimeError, message):
                    quantum_update_step(
                        [self.weight],
                        [self.gradient],
                        self.config,
                        shots=10,
                        executor=executor,
                    )
                self.assertEqual(len(executor.requests), 1)


class NonStaticArithmeticTests(unittest.TestCase):
    """Exercise the subtractor with configurations that really move weights.

    The earlier equivalence tests used configurations whose integer shift is
    zero for every gradient code, so they only compared identity maps.
    """

    MOVING_CONFIGS = ((2, 2.0), (3, 1.0), (3, 0.5), (4, 0.5), (4, 1.3))

    def test_static_configurations_are_detected(self):
        self.assertTrue(QGDConfig(n_bits=5, learning_rate=0.1).is_static)
        self.assertEqual(QGDConfig(n_bits=5, learning_rate=0.1).moving_gradient_codes, 0)
        self.assertEqual(QGDConfig(n_bits=5, learning_rate=0.15).moving_gradient_codes, 12)
        self.assertEqual(QGDConfig(n_bits=5, learning_rate=0.2).moving_gradient_codes, 16)
        self.assertEqual(QGDConfig(n_bits=4, learning_rate=0.5).moving_gradient_codes, 10)

    def test_zero_shift_codes_are_not_synthesized(self):
        static = build_qgd_update_circuit(
            QGDConfig(n_bits=5, learning_rate=0.1), measure=False
        )
        self.assertEqual(len(static.queue), 0)
        report = estimate_update_resources(
            QGDConfig(n_bits=5, learning_rate=0.1), n_weights=1, shots=1
        )
        self.assertTrue(report["static_update"])
        self.assertEqual(report["moving_gradient_codes"], 0)

    def test_every_basis_input_matches_reference_for_moving_configs(self):
        for bits, learning_rate in self.MOVING_CONFIGS:
            config = QGDConfig(n_bits=bits, learning_rate=learning_rate)
            self.assertGreater(config.moving_gradient_codes, 0)
            moved = 0
            for gradient_code in range(config.n_levels):
                for weight_code in range(config.n_levels):
                    counts = (
                        build_update_circuit(weight_code, gradient_code, config)
                        .execute(nshots=1)
                        .frequencies(binary=True)
                    )
                    decoded, _ = decode_update_counts(
                        counts,
                        config,
                        shots=1,
                        expected_gradient_code=gradient_code,
                    )
                    expected = classical_grid_update(
                        [decode_weight(weight_code, config)],
                        [decode_gradient(gradient_code, config)],
                        config,
                    )[0]
                    with self.subTest(bits=bits, lr=learning_rate, w=weight_code, g=gradient_code):
                        self.assertEqual(decoded, expected)
                    moved += decoded != decode_weight(weight_code, config)
            self.assertGreater(moved, 0)

    def test_vector_update_moves_weights_like_reference(self):
        config = QGDConfig(n_bits=4, learning_rate=1.3)
        rng = np.random.default_rng(5)
        weights = rng.uniform(-np.pi, np.pi, 6)
        gradients = rng.uniform(-1.0, 1.0, 6)
        expected = classical_grid_update(weights, gradients, config)
        actual = quantum_update_step(weights, gradients, config)
        np.testing.assert_array_equal(actual, expected)
        quantized = np.array(
            [decode_weight(encode_weight(value, config), config) for value in weights]
        )
        self.assertFalse(np.array_equal(actual, quantized))


if __name__ == "__main__":
    unittest.main()
