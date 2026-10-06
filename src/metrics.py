"""Model-independent metrics for binary experiments."""

import torch


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 for -1/+1 predictions."""
    if predictions.numel() == 0 or predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same non-empty shape.")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("predictions and targets must use floating-point tensors.")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("predictions and targets must contain only finite values.")
    if not torch.all((targets == -1.0) | (targets == 1.0)):
        raise ValueError("targets must use the -1/+1 binary encoding.")
    predicted = torch.where(predictions >= 0.0, 1.0, -1.0)
    expected = targets.reshape_as(predicted)
    positive = expected == 1.0
    predicted_positive = predicted == 1.0
    true_positive = (positive & predicted_positive).sum().item()
    false_positive = ((~positive) & predicted_positive).sum().item()
    false_negative = (positive & (~predicted_positive)).sum().item()
    total = expected.numel()

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": (predicted == expected).float().mean().item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(total),
    }
