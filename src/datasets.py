"""Reproducible dataset loading and preprocessing utilities."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetSplits:
    """Normalized train, validation, and test tensors."""

    train_features: torch.Tensor
    train_targets: torch.Tensor
    validation_features: torch.Tensor
    validation_targets: torch.Tensor
    test_features: torch.Tensor
    test_targets: torch.Tensor


@dataclass(frozen=True)
class SplitConfig:
    """Configuration shared by dataset splitters."""

    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self):
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               for value in fractions):
            raise ValueError("split fractions must be numeric.")
        if any(not math.isfinite(value) or value <= 0.0 or value >= 1.0 for value in fractions):
            raise ValueError("split fractions must be between 0 and 1.")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer.")


def split_dataset(
    features: Any,
    targets: Any,
    config: SplitConfig | None = None,
    *,
    stratify: bool = True,
) -> DatasetSplits:
    """Split, stratify, and min-max normalize any tabular dataset.

    The normalization parameters are fitted on training features only and
    reused for validation and test. Values outside the training range are
    clipped to the interval required by the quantum encoders.
    """
    split_config = config or SplitConfig()
    values = torch.as_tensor(features, dtype=torch.float64).detach().cpu().numpy()
    labels = torch.as_tensor(targets).detach().cpu().numpy()
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values.")
    if labels.ndim not in (1, 2) or len(values) != len(labels):
        raise ValueError("targets must contain one label per feature row.")
    if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
        raise ValueError("targets must contain finite numeric labels.")
    if len(values) < 3:
        raise ValueError("at least three samples are required.")
    labels_for_stratification = labels.reshape(-1) if labels.ndim == 2 else labels
    stratification = labels_for_stratification if stratify else None

    train_count = round(len(values) * split_config.train_fraction)
    validation_count = round(len(values) * split_config.validation_fraction)
    if train_count < 1 or validation_count < 1 or len(values) - train_count - validation_count < 1:
        raise ValueError("split fractions produce an empty dataset partition.")

    train_values, remainder_values, train_labels, remainder_labels = train_test_split(
        values,
        labels,
        train_size=train_count,
        random_state=split_config.seed,
        stratify=stratification,
    )
    remainder_stratification = (
        remainder_labels.reshape(-1) if remainder_labels.ndim == 2 else remainder_labels
    )
    validation_values, test_values, validation_labels, test_labels = train_test_split(
        remainder_values,
        remainder_labels,
        train_size=validation_count,
        random_state=split_config.seed,
        stratify=remainder_stratification if stratify else None,
    )

    minimum = train_values.min(axis=0)
    scale = train_values.max(axis=0) - minimum
    scale[scale == 0.0] = 1.0

    def transform(data):
        normalized = ((data - minimum) / scale).clip(0.0, 1.0)
        return torch.tensor(normalized, dtype=torch.float32)

    unique_targets = torch.unique(torch.as_tensor(labels))
    if unique_targets.numel() != 2:
        raise ValueError("binary datasets must contain exactly two target classes.")
    target_values = unique_targets.sort().values

    def tensor_targets(data):
        encoded = torch.where(
            torch.as_tensor(data) == target_values[0],
            -1.0,
            1.0,
        )
        return encoded.to(dtype=torch.float32).reshape(-1, 1)

    return DatasetSplits(
        transform(train_values),
        tensor_targets(train_labels),
        transform(validation_values),
        tensor_targets(validation_labels),
        transform(test_values),
        tensor_targets(test_labels),
    )


def load_binary_iris_splits(
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float | None = None,
) -> DatasetSplits:
    """Load binary Iris and normalize using training data only.

    The two selected classes are stratified across the three splits. The
    returned labels map setosa to -1 and versicolor to +1.
    """
    if test_fraction is None:
        test_fraction = 1.0 - train_fraction - validation_fraction
    iris = load_iris()
    selected = iris.target < 2
    values = iris.data[selected]
    labels = iris.target[selected]
    labels = 2 * labels - 1
    return split_dataset(
        values,
        labels,
        SplitConfig(
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        ),
    )


def load_binary_iris() -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible loader returning all binary Iris samples.

    Use ``load_binary_iris_splits`` for experiments and model evaluation.
    """
    splits = load_binary_iris_splits()
    return (
        torch.cat(
            [splits.train_features, splits.validation_features, splits.test_features]
        ),
        torch.cat(
            [splits.train_targets, splits.validation_targets, splits.test_targets]
        ),
    )


def load_binary_breast_cancer_splits(
    seed: int = 42,
    n_components: int = 4,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float | None = None,
) -> DatasetSplits:
    """Load Breast Cancer Wisconsin and reduce it to quantum features.

    Standardization and PCA are fitted on training samples only. The resulting
    components are min-max normalized using training statistics before being
    sent to the quantum encoder.
    """
    if test_fraction is None:
        test_fraction = 1.0 - train_fraction - validation_fraction
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    dataset = load_breast_cancer()
    values = dataset.data
    labels = dataset.target
    config = SplitConfig(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_count = round(len(values) * config.train_fraction)
    validation_count = round(len(values) * config.validation_fraction)
    train_values, remainder_values, train_labels, remainder_labels = train_test_split(
        values,
        labels,
        train_size=train_count,
        random_state=seed,
        stratify=labels,
    )
    validation_values, test_values, validation_labels, test_labels = train_test_split(
        remainder_values,
        remainder_labels,
        train_size=validation_count,
        random_state=seed,
        stratify=remainder_labels,
    )
    if n_components > train_values.shape[1] or n_components > len(train_values):
        raise ValueError("n_components exceeds the available training dimensions.")

    scaler = StandardScaler().fit(train_values)
    reducer = PCA(n_components=n_components, random_state=seed).fit(
        scaler.transform(train_values)
    )
    projected = [
        reducer.transform(scaler.transform(part))
        for part in (train_values, validation_values, test_values)
    ]
    minimum = projected[0].min(axis=0)
    scale = projected[0].max(axis=0) - minimum
    scale[scale == 0.0] = 1.0

    def transform(data):
        normalized = ((data - minimum) / scale).clip(0.0, 1.0)
        return torch.tensor(normalized, dtype=torch.float32)

    target_values = np.unique(labels)

    def tensor_targets(data):
        encoded = np.where(np.asarray(data) == target_values[0], -1.0, 1.0)
        return torch.tensor(encoded, dtype=torch.float32).reshape(-1, 1)

    return DatasetSplits(
        transform(projected[0]),
        tensor_targets(train_labels),
        transform(projected[1]),
        tensor_targets(validation_labels),
        transform(projected[2]),
        tensor_targets(test_labels),
    )
