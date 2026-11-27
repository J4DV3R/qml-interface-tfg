"""QCNN architectures: structure, Qibo equivalence, light cone and training parity."""

from __future__ import annotations

import unittest

import numpy as np

from src.fast_qcnn import FastQCNN, FastQCNNConfig
from src.qcnn_architectures import (
    ARCHITECTURES,
    AnsatzQCNN,
    build_ansatz,
    initialize_ansatz_weights,
    qibo_expectation,
)
from src.qcnn_engine import initialize_weights
from src.variational_training import (
    AnsatzEvaluator,
    FastEvaluator,
    VariationalTrainingConfig,
    mse,
    train_variational_qcnn,
)


def _data(rng, n=12):
    x = rng.uniform(0.0, 1.0, (n, 4))
    y = np.where(rng.uniform(size=n) > 0.5, 1.0, -1.0)
    return x, y


class StructureTests(unittest.TestCase):
    def test_parameter_counts(self):
        expected = {"v1": ((2, 4), 16, 11, 0), "v1-param-pool": ((2, 6), 25, 11, 0), "qibo": ((2, 21), 102, 3, 3)}
        for name, (shape, occurrences, cnots, readout) in expected.items():
            ansatz = build_ansatz(name, 4)
            self.assertEqual(ansatz.weight_shape, shape)
            self.assertEqual(len(ansatz.occurrences), occurrences)
            self.assertEqual(ansatz.cnot_count, cnots)
            self.assertEqual(ansatz.readout_qubit, readout)

    def test_qibo_layout_follows_quantum_cnn(self):
        ansatz = build_ansatz("qibo", 4)
        two_qubit = [g.qubits for g in ansatz.gates if g.kind == "rzz"]
        self.assertEqual(two_qubit, [(0, 1), (2, 3), (1, 2), (3, 0), (2, 3)])
        cnots = [g.qubits for g in ansatz.gates if g.kind == "cnot"]
        self.assertEqual(cnots, [(0, 2), (1, 3), (2, 3)])
        # The sink basis selector is undone with negated angles in reverse order.
        first_pool = [g for g in ansatz.gates if g.layer == 0 and g.param >= 15][:9]
        self.assertEqual([(g.kind, g.param, g.coeff) for g in first_pool[6:]],
                         [("rz", 17, -1.0), ("ry", 16, -1.0), ("rx", 15, -1.0)])

    def test_v1_initialization_matches_engine(self):
        np.testing.assert_array_equal(initialize_ansatz_weights(build_ansatz("v1"), 19),
                                      initialize_weights(4, seed=19))

    def test_rejects_unknown_architecture(self):
        with self.assertRaises(ValueError):
            build_ansatz("nope")


class SimulationTests(unittest.TestCase):
    def test_v1_matches_fast_simulator_exact_and_noisy(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 1, (6, 4))
        w = rng.uniform(-np.pi, np.pi, (2, 4))
        for encoding in ("angle", "phase"):
            for p, readout in ((0.0, 0.0), (0.01, 0.03)):
                fast = FastQCNN(FastQCNNConfig(4, encoding=encoding, depolarizing_probability=p,
                                               readout_error=readout))
                generic = AnsatzQCNN(build_ansatz("v1"), encoding, p, readout)
                np.testing.assert_allclose(generic.expectations(w, x), fast.expectations(w, x), atol=1e-12)

    def test_every_architecture_matches_qibo(self):
        rng = np.random.default_rng(4)
        for name in ARCHITECTURES:
            ansatz = build_ansatz(name, 4)
            for encoding in ("angle", "phase"):
                sim = AnsatzQCNN(ansatz, encoding)
                for _ in range(3):
                    w = rng.uniform(-np.pi, np.pi, ansatz.weight_shape)
                    x = rng.uniform(0, 1, 4)
                    self.assertAlmostEqual(float(sim.expectations(w, x[None])[0, 0]),
                                           qibo_expectation(ansatz, w, x, encoding), places=10)

    def test_noisy_density_matrix_is_valid_for_two_qubit_rotations(self):
        rng = np.random.default_rng(5)
        ansatz = build_ansatz("qibo", 4)
        w = rng.uniform(-np.pi, np.pi, ansatz.weight_shape)
        x = rng.uniform(0, 1, (3, 4))
        clean = AnsatzQCNN(ansatz).expectations(w, x)
        noisy = AnsatzQCNN(ansatz, depolarizing_probability=0.02).expectations(w, x)
        full = AnsatzQCNN(ansatz, depolarizing_probability=1.0).expectations(w, x)
        self.assertTrue(np.all(np.abs(noisy) <= np.abs(clean) + 1e-9))
        np.testing.assert_allclose(full, 0.0, atol=1e-12)


class LightConeTests(unittest.TestCase):
    def _influence(self, name):
        rng = np.random.default_rng(6)
        ansatz = build_ansatz(name, 4)
        sim = AnsatzQCNN(ansatz)
        features, weights = np.zeros(4), np.zeros(ansatz.weight_shape)
        for _ in range(8):
            w = rng.uniform(-np.pi, np.pi, ansatz.weight_shape)
            x = rng.uniform(0, 1, (1, 4))
            base = sim.expectations(w, x)[0, 0]
            for i in range(4):
                x2 = x.copy()
                x2[0, i] = rng.uniform()
                features[i] = max(features[i], abs(sim.expectations(w, x2)[0, 0] - base))
            for idx in np.ndindex(ansatz.weight_shape):
                w2 = w.copy()
                w2[idx] = rng.uniform(-np.pi, np.pi)
                weights[idx] = max(weights[idx], abs(sim.expectations(w2, x)[0, 0] - base))
        return features > 1e-9, weights > 1e-9

    def test_v1_ignores_x3_and_layer2_weights(self):
        features, weights = self._influence("v1")
        self.assertEqual(features.tolist(), [True, True, True, False])
        self.assertEqual(weights[1].tolist(), [True, False, False, False])

    def test_parametrized_pooling_restores_full_light_cone(self):
        features, weights = self._influence("v1-param-pool")
        self.assertTrue(features.all())
        self.assertTrue(weights.all())

    def test_qibo_only_source_rz_before_pooling_cnot_is_dead(self):
        features, weights = self._influence("qibo")
        self.assertTrue(features.all())
        dead = [idx for idx in np.ndindex(weights.shape) if not weights[idx]]
        self.assertEqual(dead, [(0, 20), (1, 20)])


class TrainingTests(unittest.TestCase):
    def test_v1_ansatz_evaluator_is_identical_to_fast_evaluator(self):
        rng = np.random.default_rng(7)
        x, y = _data(rng)
        w = rng.uniform(-1, 1, (2, 4))
        fast, generic = FastEvaluator(FastQCNNConfig(4)), AnsatzEvaluator("v1")
        for optimizer in ("adam", "spsa", "sequential"):
            config = VariationalTrainingConfig(optimizer, max_steps=2, learning_rate=0.1, shots=50)
            a = train_variational_qcnn(x, y, x, y, w, fast, config, np.random.default_rng(1))
            b = train_variational_qcnn(x, y, x, y, w, generic, config, np.random.default_rng(1))
            np.testing.assert_array_equal(a.best_weights, b.best_weights)
            self.assertEqual(a.optimization_circuits, b.optimization_circuits)

    def test_parameter_shift_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(8)
        x, y = _data(rng, 6)
        for name in ("v1-param-pool", "qibo"):
            evaluator = AnsatzEvaluator(name, encoding="phase")
            w = rng.uniform(-1, 1, evaluator.weight_shape)
            gradient, circuits = evaluator.gradient(w, x, y, shots=None, rng=None)
            self.assertEqual(circuits, len(x) * (1 + 2 * len(evaluator.ansatz.occurrences)))

            def loss(weights):
                return mse(evaluator.predict(weights, x, shots=None, rng=None), y)

            numeric = np.zeros_like(w)
            for idx in np.ndindex(w.shape):
                step = np.zeros_like(w)
                step[idx] = 1e-6
                numeric[idx] = (loss(w + step) - loss(w - step)) / 2e-6
            np.testing.assert_allclose(gradient, numeric, atol=1e-6)

    def test_sequential_sweep_never_increases_exact_train_loss(self):
        rng = np.random.default_rng(9)
        x, y = _data(rng)
        evaluator = AnsatzEvaluator("v1-param-pool")
        w = initialize_ansatz_weights(evaluator.ansatz, 2)
        result = train_variational_qcnn(x, y, x, y, w, evaluator,
                                        VariationalTrainingConfig("sequential", max_steps=3),
                                        np.random.default_rng(0))
        losses = [h["train_loss"] for h in result.history]
        self.assertTrue(all(b <= a + 1e-6 for a, b in zip(losses, losses[1:])))

    def test_rejects_weights_of_another_architecture(self):
        rng = np.random.default_rng(10)
        x, y = _data(rng)
        with self.assertRaises(ValueError):
            train_variational_qcnn(x, y, x, y, np.zeros((2, 4)), AnsatzEvaluator("qibo"),
                                   VariationalTrainingConfig("adam", max_steps=1), rng)


if __name__ == "__main__":
    unittest.main()
