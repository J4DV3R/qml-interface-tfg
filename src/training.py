"""Classical training loop for the QCNN prototype."""

from dataclasses import dataclass

import torch
import math
from torch import nn

from .hybrid_model import QCNNModel


@dataclass(frozen=True)
class TrainingConfig:
    """Controls the classical optimizer loop."""

    epochs: int = 1
    learning_rate: float = 0.01
    optimizer: str = "adam"

    def __post_init__(self):
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'.")


def train_qcnn(
    model: QCNNModel,
    features: torch.Tensor,
    targets: torch.Tensor,
    config: TrainingConfig | None = None,
) -> list[float]:
    """Train a QCNN on a complete tensor batch and return loss per epoch."""
    training_config = config or TrainingConfig()
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape (batch, n_qubits).")
    if targets.ndim != 2 or targets.shape[0] != features.shape[0]:
        raise ValueError("targets must have shape (batch, n_outputs).")
    if targets.shape[1] != model.n_outputs:
        raise ValueError("targets must have one column per model output.")
    if not targets.is_floating_point():
        raise TypeError("targets must use a floating-point dtype.")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("features and targets must contain only finite values.")

    optimizer_type = (
        torch.optim.Adam if training_config.optimizer == "adam" else torch.optim.SGD
    )
    optimizer = optimizer_type(
        model.parameters(),
        lr=training_config.learning_rate,
    )
    losses = []
    for _ in range(training_config.epochs):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(features), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return losses
