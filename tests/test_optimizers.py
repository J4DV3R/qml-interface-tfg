import unittest

import numpy as np

from src.optimizers import OptimizerConfig, optimize


class OptimizerTests(unittest.TestCase):
    @staticmethod
    def objective(weights):
        return float(np.sum((weights - np.array([0.3, -0.4])) ** 2))

    def test_all_optimizers_reduce_toy_loss_and_respect_budget(self):
        budgets = {"cobyla": 100, "nelder-mead": 100, "spsa": 200}
        for method, budget in budgets.items():
            with self.subTest(method=method):
                result = optimize(
                    self.objective,
                    np.array([1.0, 1.0]),
                    OptimizerConfig(method=method, max_evaluations=budget, seed=13),
                )
                self.assertLess(result.best_loss, 2.0 / 10.0)
                self.assertLessEqual(result.evaluations, budget)
                self.assertEqual(result.evaluations, len(result.loss_history))
                self.assertEqual(result.best_loss, min(result.loss_history))
                self.assertAlmostEqual(self.objective(result.weights), result.best_loss)
                self.assertTrue(np.all(result.weights >= -np.pi))
                self.assertTrue(np.all(result.weights <= np.pi))

    def test_spsa_is_reproducible(self):
        config = OptimizerConfig(method="spsa", max_evaluations=120, seed=22)
        first = optimize(self.objective, [1.0, 1.0], config)
        second = optimize(self.objective, [1.0, 1.0], config)
        np.testing.assert_array_equal(first.weights, second.weights)
        self.assertEqual(first.loss_history, second.loss_history)

    def test_initial_point_counts_and_odd_spsa_budget(self):
        result = optimize(
            self.objective,
            [0.0, 0.0],
            OptimizerConfig(method="spsa", max_evaluations=4, seed=1),
        )
        self.assertEqual(result.evaluations, 3)  # initial point + one pair
        self.assertEqual(len(result.loss_history), 3)

    def test_invalid_inputs_and_objective_are_rejected(self):
        for kwargs in (
            {"method": "unknown", "max_evaluations": 2},
            {"method": "spsa", "max_evaluations": 0},
            {"method": "spsa", "max_evaluations": 2, "seed": True},
            {"method": "spsa", "max_evaluations": 2, "bounds": (1, -1)},
            {"method": "spsa", "max_evaluations": 2, "rhobeg": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OptimizerConfig(**kwargs)
        config = OptimizerConfig(method="spsa", max_evaluations=2)
        for start in ([], [np.nan], [[1.0]]):
            with self.subTest(start=start), self.assertRaises(ValueError):
                optimize(self.objective, start, config)
        for objective in (lambda _: np.nan, lambda _: [1.0, 2.0]):
            with self.subTest(objective=objective), self.assertRaises(ValueError):
                optimize(objective, [0.0], config)


if __name__ == "__main__":
    unittest.main()
