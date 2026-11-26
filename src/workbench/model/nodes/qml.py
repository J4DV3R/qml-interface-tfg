"""QML node: split, preprocess (fitted on train) and train the selected algorithm."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..algorithms import algorithms, get_algorithm
from ..data import DATASET, TRAINING_RESULT, TrainingResult
from ..node import NodeType, PortSpec, RunContext, register_node
from ..parameters import ChoiceParam, IntParam, Param
from ..preprocessing import split_and_preprocess


@register_node
class QMLNode(NodeType):
    type_id = "qml"
    title = "QML"
    category = "Modelos"
    description = ("Parte los datos en train/validation/test, ajusta el preprocesado solo con train y entrena el "
                   "algoritmo elegido. El checkpoint se elige con validation; test se predice una sola vez.")
    inputs = (PortSpec("data", DATASET, "Datos"),)
    outputs = (PortSpec("result", TRAINING_RESULT, "Resultado"),)

    def parameter_schema(self, values: Mapping[str, Any] | None = None) -> list[Param]:
        values = values or {}
        available = algorithms()
        algorithm_id = values.get("algorithm", available[0].algorithm_id)
        common: list[Param] = [
            ChoiceParam("algorithm", "Algoritmo", available[0].algorithm_id,
                        choices=[(a.algorithm_id, a.label) for a in available], group="Algoritmo"),
            IntParam("train_percent", "Entrenamiento (%)", 70, minimum=1, maximum=98, group="Partición"),
            IntParam("validation_percent", "Validación (%)", 15, minimum=1, maximum=98, group="Partición"),
            IntParam("test_percent", "Test (%)", 15, minimum=1, maximum=98, group="Partición"),
            IntParam("split_seed", "Semilla de partición", 42, minimum=0, group="Partición"),
            ChoiceParam("reduction", "Reducción de dimensión", "auto",
                        "PCA ajustado en train si el dataset tiene más características de las que usa el algoritmo.",
                        choices=[("auto", "Automática"), ("pca", "PCA siempre"), ("none", "Ninguna")],
                        group="Partición"),
        ]
        try:
            algorithm = get_algorithm(algorithm_id)
        except KeyError:
            return common
        return common + algorithm.parameter_schema(values)

    def cross_validate(self, values: Mapping[str, Any]) -> list[str]:
        problems = []
        total = values["train_percent"] + values["validation_percent"] + values["test_percent"]
        if total != 100:
            problems.append(f"Los porcentajes de partición deben sumar 100 (suman {total}).")
        problems.extend(get_algorithm(values["algorithm"]).cross_validate(values))
        return problems

    def run(self, inputs, params, context: RunContext) -> dict[str, Any]:
        dataset = inputs["data"]
        algorithm = get_algorithm(params["algorithm"])
        context.progress(0.0, "partición y preprocesado")
        splits = split_and_preprocess(dataset, train_percent=params["train_percent"],
                                      validation_percent=params["validation_percent"], seed=params["split_seed"],
                                      n_features=algorithm.n_features, reduction=params["reduction"])
        context.log(f"Partición {splits.sizes()} · {'; '.join(splits.description['steps'])}")
        outcome = algorithm.fit(splits, params, context)
        # Test first (once), then validation and train: matches the benchmark's random stream.
        predictions = {}
        for split in ("test", "validation", "train"):
            x, _ = splits.get(split)
            predictions[split] = np.asarray(outcome.predict(x), dtype=float)
        context.progress(1.0, "entrenamiento terminado")
        hyper = {k: v for k, v in params.items()
                 if k not in ("train_percent", "validation_percent", "test_percent", "split_seed", "reduction")}
        return {"result": TrainingResult(
            algorithm_id=algorithm.algorithm_id,
            algorithm_label=algorithm.label,
            hyperparameters=hyper,
            splits=splits,
            predictions=predictions,
            history=outcome.history,
            checkpoint=outcome.checkpoint,
            model_state=outcome.model_state,
            cost=outcome.cost,
            dataset_summary=dataset.summary(),
            notes=outcome.notes,
        )}
