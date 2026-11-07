import subprocess
import sys
import unittest

import numpy as np

from src.quantum_update import (
    QGDConfig,
    build_update_unitary,
    classical_grid_update,
    decode_gradient,
    decode_weight,
    encode_gradient,
    encode_weight,
    quantum_update_step,
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

    def test_reversible_mapping_and_torch_independence(self):
        config = QGDConfig(n_bits=2)
        unitary = build_update_unitary(config)
        self.assertEqual(unitary.shape, (16, 16))
        code = "import sys; import src.quantum_update; assert 'torch' not in sys.modules"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

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


if __name__ == "__main__":
    unittest.main()
