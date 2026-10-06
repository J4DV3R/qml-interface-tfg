"""Classical reference models used only by experiments."""

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from .metrics import classification_metrics


def run_logistic_baseline(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    seed: int = 42,
) -> dict[str, float]:
    """Fit logistic regression on train and evaluate once on test."""
    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(
        train_features.detach().cpu().numpy(),
        train_targets.detach().cpu().numpy().reshape(-1),
    )
    predictions = torch.from_numpy(
        np.where(model.predict(test_features.detach().cpu().numpy()) > 0, 1.0, -1.0)
    ).reshape(-1, 1)
    return classification_metrics(predictions, test_targets)
