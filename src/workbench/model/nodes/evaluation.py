"""Results-processing node: metrics per split, baselines and training curves."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.variational_training import binary_metrics, mse

from ..data import EVALUATION_REPORT, TRAINING_RESULT, EvaluationReport
from ..node import NodeType, PortSpec, RunContext, register_node
from ..parameters import BoolParam, Param

SPLITS = ("train", "validation", "test")


@register_node
class EvaluationNode(NodeType):
    type_id = "evaluation"
    title = "Procesado de resultados"
    category = "Evaluación"
    description = ("Calcula accuracy, accuracy balanceada, precisión, recall, F1 y MSE por partición, las "
                   "baselines (clase mayoritaria y regresión logística, ajustadas en train) y las curvas.")
    inputs = (PortSpec("result", TRAINING_RESULT, "Resultado"),)
    outputs = (PortSpec("report", EVALUATION_REPORT, "Informe"),)

    def parameter_schema(self, values: Mapping[str, Any] | None = None) -> list[Param]:
        return [BoolParam("baselines", "Calcular baselines clásicas", True,
                          "Clase mayoritaria y regresión logística sobre las mismas características.")]

    def run(self, inputs, params, context: RunContext) -> dict[str, Any]:
        result = inputs["result"]
        splits = result.splits
        metrics, confusion = {}, {}
        for split in SPLITS:
            _, y = splits.get(split)
            m = binary_metrics(result.predictions[split], y)
            confusion[split] = m.pop("confusion")
            metrics[split] = {**m, "mse": mse(result.predictions[split], y)}

        baselines: dict[str, dict[str, Any]] = {}
        if params["baselines"]:
            majority = 1.0 if np.sum(splits.train_y == 1) >= np.sum(splits.train_y == -1) else -1.0
            baselines["Clase mayoritaria"] = {
                split: binary_metrics(np.full(len(splits.get(split)[1]), majority), splits.get(split)[1])["accuracy"]
                for split in SPLITS}
            seed = splits.description.get("seed", 0)
            logistic = LogisticRegression(random_state=seed, max_iter=1000).fit(splits.train_x, splits.train_y)
            baselines["Regresión logística"] = {
                split: float(np.mean(logistic.predict(splits.get(split)[0]) == splits.get(split)[1]))
                for split in SPLITS}

        curves = {
            "step": [r["step"] for r in result.history],
            "train_loss": [r.get("train_loss") for r in result.history],
            "validation_loss": [r["validation_loss"] for r in result.history],
            "validation_accuracy": [r.get("validation_accuracy") for r in result.history],
            "optimization_circuits": [r.get("optimization_circuits") for r in result.history],
        }
        configuration = {
            "algorithm": result.algorithm_label,
            "hyperparameters": result.hyperparameters,
            "dataset": result.dataset_summary,
            "partition": splits.description,
            "checkpoint": result.checkpoint,
            "cost": result.cost,
            "model_state": result.model_state,
        }
        title = f"{result.algorithm_label} · {result.dataset_summary['name']}"
        context.log(f"Accuracy de test {metrics['test']['accuracy']:.3f} (checkpoint en el paso "
                    f"{result.checkpoint.get('step')})")
        predictions = {split: {"target": [float(v) for v in splits.get(split)[1]],
                               "output": [float(v) for v in result.predictions[split]]} for split in SPLITS}
        return {"report": EvaluationReport(title, metrics, baselines, curves, confusion, configuration,
                                           list(result.notes), predictions)}
