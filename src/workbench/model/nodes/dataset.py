"""Dataset node: loads a labelled binary dataset (no split, no preprocessing)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris

from ..data import DATASET, Dataset
from ..node import NodeType, PortSpec, RunContext, register_node
from ..parameters import ChoiceParam, Param, StringParam

SOURCES = [("iris-binary", "Iris (setosa / versicolor)"),
           ("breast-cancer", "Breast Cancer Wisconsin"),
           ("csv", "Archivo CSV")]


def load_builtin(source: str) -> Dataset:
    if source == "iris-binary":
        iris = load_iris()
        selected = iris.target < 2
        return Dataset("Iris binario", iris.data[selected], 2 * iris.target[selected] - 1,
                       list(iris.feature_names), ["setosa", "versicolor"],
                       {"source": "sklearn.datasets.load_iris", "classes": "setosa=-1, versicolor=+1"})
    if source == "breast-cancer":
        data = load_breast_cancer()
        return Dataset("Breast Cancer Wisconsin", data.data, data.target, list(data.feature_names),
                       list(data.target_names), {"source": "sklearn.datasets.load_breast_cancer"})
    raise ValueError(f"fuente desconocida {source!r}")


def load_csv(path: str, label_column: str, delimiter: str) -> Dataset:
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"no existe el archivo {path!r}")
    with file.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle, delimiter=delimiter or ",") if row and any(c.strip() for c in row)]
    if len(rows) < 4:
        raise ValueError("el CSV necesita cabecera y al menos 3 filas")
    header, body = rows[0], rows[1:]
    if label_column not in header:
        raise ValueError(f"la columna de etiqueta {label_column!r} no está en la cabecera")
    index = header.index(label_column)
    if any(len(row) != len(header) for row in body):
        raise ValueError("todas las filas del CSV deben tener el mismo número de columnas que la cabecera")
    labels = [row[index] for row in body]
    try:
        features = np.array([[float(v) for j, v in enumerate(row) if j != index] for row in body])
    except ValueError as exc:
        raise ValueError("todas las columnas salvo la etiqueta deben ser numéricas") from exc
    if not np.all(np.isfinite(features)):
        raise ValueError("las características numéricas del CSV deben ser finitas")
    classes = sorted(set(labels))
    if len(classes) != 2:
        raise ValueError(f"se esperaban 2 clases y hay {len(classes)}")
    return Dataset(file.stem, features, np.array([classes.index(v) for v in labels]),
                   [h for j, h in enumerate(header) if j != index], classes,
                   {"source": str(file), "label_column": label_column})


@register_node
class DatasetNode(NodeType):
    type_id = "dataset"
    title = "Dataset"
    category = "Datos"
    description = "Carga un conjunto de datos binario. La partición y el preprocesado se hacen en el nodo QML."
    inputs = ()
    outputs = (PortSpec("data", DATASET, "Datos"),)

    def parameter_schema(self, values: Mapping[str, Any] | None = None) -> list[Param]:
        return [
            ChoiceParam("source", "Fuente", "iris-binary", choices=SOURCES),
            StringParam("csv_path", "Ruta del CSV", "", is_path=True, visible_when={"source": ("csv",)}),
            StringParam("label_column", "Columna de etiqueta", "label", visible_when={"source": ("csv",)}),
            StringParam("delimiter", "Separador", ",", visible_when={"source": ("csv",)}),
        ]

    def cross_validate(self, values: Mapping[str, Any]) -> list[str]:
        if values["source"] == "csv" and not values.get("csv_path"):
            return ["Indica la ruta del CSV."]
        return []

    def run(self, inputs, params, context: RunContext) -> dict[str, Any]:
        if params["source"] == "csv":
            dataset = load_csv(params["csv_path"], params["label_column"], params["delimiter"])
        else:
            dataset = load_builtin(params["source"])
        summary = dataset.summary()
        context.log(f"{summary['name']}: {summary['samples']} muestras, {summary['features']} características, "
                    f"clases {summary['classes']}")
        return {"data": dataset}
