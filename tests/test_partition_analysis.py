import unittest

import numpy as np

from experiments.analyze_partitions import holm, patience_truncation
from src.fast_qcnn import FastQCNN, FastQCNNConfig
from src.variational_training import FastEvaluator, VariationalTrainingConfig, train_variational_qcnn


def teacher_data(samples, seed):
    rng = np.random.default_rng(seed)
    teacher = rng.uniform(-np.pi, np.pi, (2, 4))
    features = rng.uniform(0.0, 1.0, (samples, 4))
    labels = np.where(FastQCNN(FastQCNNConfig(4)).expectations(teacher, features)[:, 0] >= 0, 1.0, -1.0)
    return features, labels


class PatienceReconstructionTests(unittest.TestCase):
    """Post-hoc patience must equal an actual run with ``patience=P``."""

    def test_truncated_history_matches_training_with_patience(self):
        x, y = teacher_data(16, 3)
        xv, yv = teacher_data(8, 4)
        w0 = np.random.default_rng(19).uniform(-0.25, 0.25, (2, 4))
        evaluator = FastEvaluator(FastQCNNConfig(4))
        stopped_early = 0
        for optimizer, kwargs in (
            ("gd", {"learning_rate": 0.5}),
            ("adam", {"learning_rate": 0.1}),
            ("spsa", {"spsa_a": 1.6, "spsa_stability": 6.0}),
            ("sequential", {}),
        ):
            def train(**extra):
                config = VariationalTrainingConfig(optimizer=optimizer, max_steps=40, **kwargs, **extra)
                return train_variational_qcnn(x, y, xv, yv, w0, evaluator, config, np.random.default_rng(101))

            full = train()
            for patience in (2, 3, 5):
                with self.subTest(optimizer=optimizer, patience=patience):
                    actual = train(patience=patience)
                    rebuilt = patience_truncation(full.history, patience)
                    self.assertEqual(rebuilt["stopped_early"], actual.stop_reason == "patience")
                    self.assertEqual(rebuilt["stop_step"], actual.steps_completed)
                    self.assertEqual(rebuilt["best_step"], actual.best_step)
                    self.assertEqual(rebuilt["optimization_circuits_at_stop"], actual.optimization_circuits)
                    np.testing.assert_array_equal(
                        np.asarray(full.history[rebuilt["best_index"]]["weights"]).reshape(w0.shape),
                        actual.best_weights,
                    )
                    stopped_early += rebuilt["stopped_early"]
        self.assertGreater(stopped_early, 0, "no configuration exercised an early stop")

    def test_patience_without_improvement_stops_at_patience(self):
        history = [{"step": s, "validation_loss": loss, "optimization_circuits": 10 * s}
                   for s, loss in enumerate([1.0, 0.5, 0.6, 0.4, 0.7, 0.8, 0.9])]
        result = patience_truncation(history, 2)
        self.assertEqual((result["best_step"], result["stop_step"]), (3, 5))
        self.assertEqual(result["optimization_circuits_at_stop"], 50)
        self.assertFalse(patience_truncation(history, 10)["stopped_early"])

    def test_holm_adjustment(self):
        adjusted = holm([0.01, 0.04, None, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[3], 0.06)
        self.assertAlmostEqual(adjusted[1], 0.06)  # monotone step-down
        self.assertIsNone(adjusted[2])


if __name__ == "__main__":
    unittest.main()
