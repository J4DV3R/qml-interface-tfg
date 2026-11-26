"""Split and preprocess a dataset without leaking information from validation/test.

The procedure reproduces ``src.datasets`` exactly (same stratified
``train_test_split`` calls, scaler/PCA/min-max fitted on train only, float32
rounding of the encoded features), so a workflow configured like the phase-C
experiments yields the same partitions.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import Dataset, Splits


def _fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array, dtype=float)).tobytes()).hexdigest()


def split_and_preprocess(dataset: Dataset, *, train_percent: int, validation_percent: int, seed: int,
                         n_features: int | None, reduction: str) -> Splits:
    """Stratified split + (standardization and PCA) + min-max to [0, 1], all fitted on train.

    ``reduction``: ``"auto"`` applies PCA when the dataset has more features than
    ``n_features``; ``"pca"`` always applies it; ``"none"`` never does (then the
    feature count must already match).
    """
    values = np.asarray(dataset.features, dtype=float)
    labels = np.asarray(dataset.labels)
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("solo se admiten problemas binarios (exactamente dos clases).")
    n = len(values)
    train_count = round(n * train_percent / 100)
    validation_count = round(n * validation_percent / 100)
    if train_count < 2 or validation_count < 1 or n - train_count - validation_count < 1:
        raise ValueError("los porcentajes dejan alguna partición vacía.")

    train_v, rest_v, train_l, rest_l = train_test_split(
        values, labels, train_size=train_count, random_state=seed, stratify=labels)
    val_v, test_v, val_l, test_l = train_test_split(
        rest_v, rest_l, train_size=validation_count, random_state=seed, stratify=rest_l)

    use_pca = reduction == "pca" or (reduction == "auto" and n_features is not None and values.shape[1] > n_features)
    steps: list[str] = []
    if use_pca:
        if n_features is None:
            raise ValueError("PCA requiere que el algoritmo declare su número de características.")
        if n_features > train_v.shape[1] or n_features > len(train_v):
            raise ValueError("el número de componentes PCA supera las dimensiones de train.")
        scaler = StandardScaler().fit(train_v)
        reducer = PCA(n_components=n_features, random_state=seed).fit(scaler.transform(train_v))
        parts = [reducer.transform(scaler.transform(p)) for p in (train_v, val_v, test_v)]
        steps.append(f"estandarización + PCA a {n_features} componentes (ajustados en train)")
    else:
        if n_features is not None and values.shape[1] != n_features:
            raise ValueError(
                f"el algoritmo necesita {n_features} características y el dataset tiene {values.shape[1]}; "
                "activa la reducción PCA.")
        parts = [train_v, val_v, test_v]

    minimum = parts[0].min(axis=0)
    scale = parts[0].max(axis=0) - minimum
    scale[scale == 0.0] = 1.0
    # float32 rounding mirrors the historical loaders (torch.float32 tensors).
    encoded = [np.asarray(((p - minimum) / scale).clip(0.0, 1.0), dtype=np.float32).astype(float) for p in parts]
    steps.append("min-max a [0, 1] (ajustado en train, recorte fuera de rango)")

    def to_pm1(part_labels):
        return np.where(np.asarray(part_labels) == classes[0], -1.0, 1.0)

    ys = [to_pm1(l) for l in (train_l, val_l, test_l)]
    description: dict[str, Any] = {
        "percentages": {"train": train_percent, "validation": validation_percent,
                        "test": 100 - train_percent - validation_percent},
        "seed": seed,
        "stratified": True,
        "steps": steps,
        "label_mapping": {str(classes[0]): -1, str(classes[1]): 1},
        "sizes": {"train": len(ys[0]), "validation": len(ys[1]), "test": len(ys[2])},
        "fingerprints": {name: {"features_sha256": _fingerprint(x), "targets_sha256": _fingerprint(y)}
                         for name, x, y in zip(("train", "validation", "test"), encoded, ys)},
    }
    return Splits(encoded[0], ys[0], encoded[1], ys[1], encoded[2], ys[2], description)
