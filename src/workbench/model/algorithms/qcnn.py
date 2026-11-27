"""Variational QCNN (4 qubits) trained with GD, Adam, SPSA or the sequential optimizer.

Wraps ``src.variational_training`` with the same seeds and random streams as
``experiments/run_variational_benchmark.py``: with the ``v1`` architecture an
exact run configured like phase C reproduces its checkpoint and test metrics.
The ``v1-param-pool`` and ``qibo`` architectures come from
``src.qcnn_architectures`` (corrected pooling and Qibo's ``QuantumCNN`` ansatz).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.fast_qcnn import FastQCNNConfig
from src.qcnn_engine import initialize_weights
from src.qcnn_architectures import build_ansatz, initialize_ansatz_weights
from src.variational_training import (
    AnsatzEvaluator,
    FastEvaluator,
    VariationalTrainingConfig,
    train_variational_qcnn,
)

from ..data import Splits
from ..node import Cancelled, RunContext
from ..parameters import ChoiceParam, FloatParam, IntParam, Param
from . import FitOutcome, QMLAlgorithm, register_algorithm

GRADIENT = ("gd", "adam")
ARCHITECTURE_CHOICES = [
    ("v1", "v1 original (8 pesos; ignora x3)"),
    ("v1-param-pool", "v1 con pooling parametrizado (12 pesos)"),
    ("qibo", "Ansatz QuantumCNN de Qibo (42 pesos)"),
]
ARCHITECTURE_NOTES = {
    "v1": "La salida del circuito v1 no depende de la característica x3 ni de 3 de los 8 pesos "
          "(log técnico, sección 37).",
    "v1-param-pool": "Pooling parametrizado RY(a)·RY(b)·CNOT·RY(−a): las 4 características y los 12 pesos "
                     "influyen en la salida (diagnóstico del cono de luz).",
    "qibo": "Ansatz de qibo.models.QuantumCNN sin sesgo entrenable y con nuestra codificación; 2 de los 42 "
            "pesos (RZ del qubit fuente antes de la CNOT de pooling) no influyen en la salida.",
}


@register_algorithm
class VariationalQCNN(QMLAlgorithm):
    algorithm_id = "qcnn-variational"
    label = "QCNN variacional (4 qubits)"
    description = ("Red convolucional cuántica de 4 qubits con arquitectura seleccionable (v1, pooling "
                   "parametrizado o ansatz de Qibo); el circuito evalúa ⟨Z⟩ y un optimizador clásico actualiza "
                   "los pesos. Checkpoint elegido solo con validation.")
    n_features = 4

    def parameter_schema(self, values: Mapping[str, Any]) -> list[Param]:
        optimizer = values.get("optimizer", "adam")
        default_lr = 0.5 if optimizer == "gd" else 0.1
        return [
            ChoiceParam("architecture", "Arquitectura", "v1",
                        "Circuito de la QCNN. v1 se conserva para reproducir las fases A-C.",
                        choices=ARCHITECTURE_CHOICES, group="Modelo"),
            ChoiceParam("encoding", "Codificación", "angle",
                        "Cómo se cargan las 4 características en los qubits.",
                        choices=[("angle", "Angular  RY(πx)"), ("phase", "Fase  H + RZ(2πx)")], group="Modelo"),
            ChoiceParam("optimizer", "Optimizador", "adam",
                        "Regla de actualización de los pesos.",
                        choices=[("gd", "Descenso de gradiente"), ("adam", "Adam"), ("spsa", "SPSA"),
                                 ("sequential", "Secuencial (Rotosolve/NFT)")], group="Optimización"),
            IntParam("max_steps", "Pasos máximos", 100, "Pasos (o barridos, en el secuencial).",
                     minimum=1, maximum=10_000, group="Optimización"),
            FloatParam("learning_rate", "Tasa de aprendizaje", default_lr,
                       "Fase A: GD 0,5 y Adam 0,1.", minimum=1e-6, maximum=10.0, decimals=4,
                       visible_when={"optimizer": GRADIENT}, group="Optimización"),
            FloatParam("spsa_a", "SPSA: ganancia a", 1.6, "Fase A: 1,6.", minimum=1e-6, maximum=100.0,
                       visible_when={"optimizer": ("spsa",)}, group="Optimización"),
            FloatParam("spsa_c", "SPSA: perturbación c", 0.1, minimum=1e-6, maximum=10.0,
                       visible_when={"optimizer": ("spsa",)}, group="Optimización"),
            FloatParam("spsa_stability", "SPSA: estabilidad A", 6.0, minimum=0.0, maximum=1e4,
                       visible_when={"optimizer": ("spsa",)}, group="Optimización"),
            IntParam("sequential_grid_points", "Secuencial: puntos de rejilla", 2048, minimum=16, maximum=100_000,
                     visible_when={"optimizer": ("sequential",)}, group="Optimización"),
            IntParam("patience", "Patience (vacío = sin parada temprana)", None,
                     "Parar tras P pasos sin mejorar la pérdida de validation.", minimum=1, maximum=10_000,
                     optional=True, group="Optimización"),
            IntParam("init_seed", "Semilla de inicialización", 19, minimum=0, group="Semillas"),
            FloatParam("init_scale", "Escala de inicialización", 0.25,
                       "Pesos iniciales uniformes en [-escala, escala].", minimum=0.0, maximum=3.15,
                       group="Semillas"),
            ChoiceParam("execution", "Ejecución", "exact", "Simulación exacta, con shots o con shots y ruido.",
                        choices=[("exact", "Exacta"), ("shots", "Shots"), ("noisy", "Shots + ruido")],
                        group="Ejecución"),
            IntParam("shots", "Shots", 1000, minimum=1, maximum=1_000_000,
                     visible_when={"execution": ("shots", "noisy")}, group="Ejecución"),
            FloatParam("noise", "Ruido depolarizante p", 0.001, "Sensibilidad, no calibración de hardware.",
                       minimum=0.0, maximum=1.0, decimals=5, step=0.001,
                       visible_when={"execution": ("noisy",)}, group="Ejecución"),
            FloatParam("readout_error", "Error de lectura", 0.02, minimum=0.0, maximum=0.5, decimals=4,
                       visible_when={"execution": ("noisy",)}, group="Ejecución"),
            IntParam("execution_seed", "Semilla de ejecución", 101, minimum=0, group="Semillas"),
        ]

    def fit(self, splits: Splits, params: Mapping[str, Any], context: RunContext) -> FitOutcome:
        optimizer = params["optimizer"]
        execution = params["execution"]
        shots = params["shots"] if execution in ("shots", "noisy") else None
        noise = params["noise"] if execution == "noisy" else 0.0
        readout = params["readout_error"] if execution == "noisy" else 0.0
        config = VariationalTrainingConfig(
            optimizer=optimizer,
            max_steps=params["max_steps"],
            learning_rate=params.get("learning_rate", 0.05) if optimizer in GRADIENT else 0.05,
            shots=shots,
            patience=params.get("patience"),
            spsa_a=params.get("spsa_a", 0.2),
            spsa_c=params.get("spsa_c", 0.1),
            spsa_stability=params.get("spsa_stability"),
            sequential_grid_points=params.get("sequential_grid_points", 2048),
        )
        architecture = params.get("architecture", "v1")
        if architecture == "v1":
            # Same evaluator as the phase A-C benchmark (bit-for-bit reproducible).
            evaluator = FastEvaluator(FastQCNNConfig(4, 1, params["encoding"],
                                                     depolarizing_probability=noise, readout_error=readout))
            initial = initialize_weights(4, seed=params["init_seed"], low=-params["init_scale"],
                                         high=params["init_scale"])
        else:
            evaluator = AnsatzEvaluator(architecture, 4, params["encoding"],
                                        depolarizing_probability=noise, readout_error=readout)
            initial = initialize_ansatz_weights(evaluator.ansatz, params["init_seed"],
                                                low=-params["init_scale"], high=params["init_scale"])
        rng = np.random.default_rng(np.random.SeedSequence([params["execution_seed"], params["init_seed"]]))

        def callback(record: dict[str, Any]) -> bool:
            context.progress(record["step"] / config.max_steps,
                             f"paso {record['step']}/{config.max_steps} · pérdida val. {record['validation_loss']:.4f}")
            return not context.is_cancelled

        result = train_variational_qcnn(splits.train_x, splits.train_y, splits.validation_x, splits.validation_y,
                                        initial, evaluator, config, rng, callback=callback)
        if result.stop_reason == "cancelled":
            raise Cancelled()

        best = result.best_weights

        def predict(features: np.ndarray) -> np.ndarray:
            # Same evaluator and random stream as the benchmark (measured when shots are set).
            return evaluator.predict(best, features, shots=shots, rng=rng)

        history = [{
            "step": r["step"],
            "train_loss": r.get("train_loss"),
            "validation_loss": r["validation_loss"],
            "validation_accuracy": r["validation_metrics"]["accuracy"],
            "optimization_circuits": r["optimization_circuits"],
        } for r in result.history]
        notes = [ARCHITECTURE_NOTES[architecture]]
        return FitOutcome(
            predict=predict,
            history=history,
            checkpoint={"step": result.best_step, "validation_loss": result.best_validation_loss,
                        "steps_completed": result.steps_completed, "stop_reason": result.stop_reason},
            model_state={"architecture": build_ansatz(architecture, 4).summary(),
                         "best_weights": best.tolist(), "initial_weights": result.initial_weights.tolist(),
                         "training_config": {k: v for k, v in vars(config).items()}},
            cost={"optimization_circuits": result.optimization_circuits,
                  "evaluation_circuits": result.evaluation_circuits,
                  "elapsed_seconds": result.elapsed_seconds},
            notes=notes,
        )
