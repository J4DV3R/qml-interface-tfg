"""Data types that travel along workflow links (port types)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DATASET = "Dataset"
TRAINING_RESULT = "TrainingResult"
EVALUATION_REPORT = "EvaluationReport"

PORT_TYPE_LABELS = {
    DATASET: "Conjunto de datos",
    TRAINING_RESULT: "Resultado de entrenamiento",
    EVALUATION_REPORT: "Informe de evaluación",
}


@dataclass
class Dataset:
    """A labelled tabular dataset, before any split or preprocessing."""

    name: str
    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    source: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {
            "name": self.name,
            "samples": int(len(self.labels)),
            "features": int(self.features.shape[1]),
            "feature_names": list(self.feature_names),
            "classes": {str(v): int(c) for v, c in zip(values, counts)},
            "class_names": list(self.class_names),
            "source": dict(self.source),
        }


@dataclass
class Splits:
    """Preprocessed train/validation/test partitions with labels in {-1, +1}."""

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    description: dict[str, Any] = field(default_factory=dict)

    def get(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        return {"train": (self.train_x, self.train_y),
                "validation": (self.validation_x, self.validation_y),
                "test": (self.test_x, self.test_y)}[split]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train_y), "validation": len(self.validation_y), "test": len(self.test_y)}


@dataclass
class TrainingResult:
    """Output of a QML node: the selected checkpoint and its predictions.

    ``predictions`` hold real-valued model outputs (for the QCNN, <Z>) for each
    split, computed once at the checkpoint chosen on validation. Test was not
    used for any selection.
    """

    algorithm_id: str
    algorithm_label: str
    hyperparameters: dict[str, Any]
    splits: Splits
    predictions: dict[str, np.ndarray]
    history: list[dict[str, Any]]
    checkpoint: dict[str, Any]
    model_state: dict[str, Any]
    cost: dict[str, Any]
    dataset_summary: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Metrics, baselines and curves derived from a TrainingResult."""

    title: str
    metrics: dict[str, dict[str, Any]]
    baselines: dict[str, dict[str, Any]]
    curves: dict[str, list[float]]
    confusion: dict[str, dict[str, int]]
    configuration: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    # Per split: {"target": [...], "output": [...]} (model output at the checkpoint).
    predictions: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "metrics": self.metrics, "baselines": self.baselines,
                "curves": self.curves, "confusion": self.confusion,
                "configuration": self.configuration, "predictions": self.predictions,
                "notes": list(self.notes)}
