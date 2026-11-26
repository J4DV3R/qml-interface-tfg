"""Results node: routes an evaluation report to the screen and/or to files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from ..data import EVALUATION_REPORT
from ..node import NodeType, PortSpec, RunContext, register_node
from ..parameters import BoolParam, ChoiceParam, Param, StringParam

CSV_FORMATS = [("excel-es", "Excel en español (; y coma decimal)"), ("standard", "Estándar (, y punto decimal)")]


def _json_default(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(value)


def resolve_path(text: str) -> Path:
    """Absolute path; relative paths are resolved against the current working directory."""
    return Path(text).expanduser().resolve()


def write_csv(report, path: Path, fmt: str) -> None:
    delimiter, decimal = (";", ",") if fmt == "excel-es" else (",", ".")

    def cell(value):
        if isinstance(value, float):
            return repr(value).replace(".", decimal)
        return value

    names = list(next(iter(report.metrics.values())).keys())
    # utf-8-sig so that Excel recognises accents.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["modelo", "particion", *names])
        for split, values in report.metrics.items():
            writer.writerow(["QML", split, *[cell(values[n]) for n in names]])
        for baseline, values in report.baselines.items():
            for split, accuracy in values.items():
                writer.writerow([baseline, split, cell(float(accuracy)), *[""] * (len(names) - 1)])


@register_node
class ResultsNode(NodeType):
    type_id = "results"
    title = "Resultados"
    category = "Salida"
    description = ("Muestra el informe en la aplicación y, opcionalmente, lo exporta: JSON con todo el informe "
                   "(métricas, baselines, curvas, configuración y predicciones) y CSV con las métricas.")
    inputs = (PortSpec("report", EVALUATION_REPORT, "Informe"),)
    outputs = ()

    def parameter_schema(self, values: Mapping[str, Any] | None = None) -> list[Param]:
        return [
            BoolParam("export_json", "Exportar informe JSON", False, group="Destino"),
            StringParam("json_path", "Ruta del JSON", "results/workbench/informe.json", is_path=True,
                        help="Ruta relativa a la carpeta desde la que se abrió la aplicación, o absoluta.",
                        visible_when={"export_json": (True,)}, group="Destino"),
            BoolParam("export_csv", "Exportar métricas CSV", False, group="Destino"),
            StringParam("csv_path", "Ruta del CSV", "results/workbench/metricas.csv", is_path=True,
                        help="Ruta relativa a la carpeta desde la que se abrió la aplicación, o absoluta.",
                        visible_when={"export_csv": (True,)}, group="Destino"),
            ChoiceParam("csv_format", "Formato del CSV", "excel-es", choices=CSV_FORMATS,
                        visible_when={"export_csv": (True,)}, group="Destino"),
        ]

    def cross_validate(self, values: Mapping[str, Any]) -> list[str]:
        problems = []
        if values["export_json"] and not values["json_path"].strip():
            problems.append("Indica la ruta del JSON.")
        if values["export_csv"] and not values["csv_path"].strip():
            problems.append("Indica la ruta del CSV.")
        return problems

    def run(self, inputs, params, context: RunContext) -> dict[str, Any]:
        report = inputs["report"]
        exported = []
        if params["export_json"]:
            path = resolve_path(params["json_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=_json_default),
                            encoding="utf-8")
            exported.append(str(path))
        if params["export_csv"]:
            path = resolve_path(params["csv_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(report, path, params["csv_format"])
            exported.append(str(path))
        for item in exported:
            context.log(f"Exportado: {item}")
        return {"report": report, "exported": exported}
