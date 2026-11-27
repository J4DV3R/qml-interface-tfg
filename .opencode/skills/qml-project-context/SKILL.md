---
name: Contexto del proyecto QML
description: "Arquitectura, módulos, decisiones, estado híbrido frente al plan QGD y limitaciones conocidas del TFG. Úsala antes de planificar, implementar, revisar o redactar cambios."
---

# Contexto del proyecto QML

## Objetivo

Plataforma modular para experimentos reproducibles de Quantum Machine Learning
sobre Qibo. La QCNN es el primer algoritmo; la arquitectura debe permitir
incorporar otros modelos QML sin rehacer las capas comunes.

## Flujo de datos

1. `src/datasets.py` carga el dataset, separa train/validation/test de forma
   estratificada y ajusta normalización o PCA solo con train.
2. `src/data_encoding.py` codifica cada característica en un qubit:
   - angular: `RY(pi * x)`;
   - fase: `H` seguida de `RZ(2 * pi * x)`.
3. `src/qcnn_circuit.py` aplica un filtro compartido de cuatro ángulos sobre
   pares adyacentes de qubits.
4. `src/pooling.py` aplica CNOT del qubit descartado al superviviente. El
   registro de Qibo no se reduce físicamente: el pooling es lógico.
5. **Ruta legacy:** `src/hybrid_model.py` calcula expectativas Z y parameter-shift
   dentro de autograd de PyTorch; `src/training.py` actualiza con Adam o SGD.
6. **Ruta QGD M1-M5:** `src/qcnn_engine.py` es el motor NumPy/Qibo sin Torch;
   `src/quantum_training.py` integra el entrenamiento,
   `src/quantum_update.py` ejecuta los updates aritméticos reversibles y
   `src/circuit_execution.py` define el contrato de ejecución medida y recursos.
7. `src/metrics.py` conserva métricas Torch del legado; el entrenamiento QGD
   calcula métricas binarias NumPy sobre train/validation y el runner calcula
   test una sola vez después de seleccionar por validation.
8. `src/classical_baseline.py` entrena una regresión logística externa al motor
   QML, con la misma partición y transformaciones.

## Estado actual del código (27 de noviembre de 2026)

La fuente de verdad viva del estado y el mapa completo de archivos es `AGENTS.md`
junto con `docs/TFG-technical-log.md` (secciones 30-41). Resumen operativo:

- **Suite completa:** **180 pruebas OK** (27-nov-2026, árbol sucio basado en
  `b5abe03`, `QT_QPA_PLATFORM=offscreen`, Python 3.14.3; `compileall` OK).
  Validación en CPython 3.10.13 (`.venv-py310`) pendiente.
- **Línea variacional (evaluada en simulación):** `src/fast_qcnn.py` (simulador
  NumPy vectorizado ~76× más rápido, error < 1e-10 frente a Qibo) y
  `src/variational_training.py` (GD, Adam, SPSA y secuencial tipo Rotosolve/NFT;
  checkpoint solo por validation incluido el paso 0). Evaluada en fases A, B
  (Breast Cancer e Iris, partición 42) y C (pre-registrada, 10 particiones por
  dataset, 1280 entrenamientos, corrección Holm) en `results/variational/`.
  Hallazgos: supera a la mayoritaria pero queda por debajo de la regresión
  logística; ninguna diferencia de accuracy frente a Adam es significativa tras
  Holm; el secuencial reduce los circuitos 3,2–9,4× en las 40 combinaciones.
- **Hallazgo del cono de luz y arquitecturas alternativas:** en `v1` la
  convolución y el pooling repiten la misma CNOT y se anulan (salida solo
  depende de 3 características y 5 pesos; `results/diagnostics/light-cone/`).
  `src/qcnn_architectures.py` implementa y valida unitariamente `v1-param-pool`
  (12 pesos, cono completo) y `qibo` (42 pesos, ansatz `QuantumCNN`, 40 con
  influencia) manteniendo `v1` intacto. Evaluación en fase C pendiente de
  tutoría (decisión 6).
- **QML Workbench (implementado y validado):** aplicación de escritorio PySide6
  con patrón MVC en `src/workbench/` (`model/` y `controller/` sin Qt; solo
  `view/` importa PySide6), paridad verificada con la fase C y soporte para las
  tres arquitecturas (`tests/test_workbench.py`, 18 pruebas;
  `docs/workbench/README.md`).
- **Preparación BSC / MareNostrum5-Ona (implementado y validado en simulación /
  mocks; sin QPU):** `src/hardware_cost.py`, `src/transpilation.py` (con
  `compact=False` para índices físicos reales), `src/hardware_execution.py`
  (`HardwareMockExecutor`, calibración/mitigación de lectura, `QililabExecutor`),
  `src/qililab_adapter.py` (`qililab.execute`), `experiments/run_qililab_smoke.py`,
  `experiments/run_hardware_inference.py`, `slurm/`, `docs/bsc/` y
  `tests/test_hardware_preparation.py` (21 pruebas). Sin validar en hardware real.
- **Ruta QGD M1-M5 (implementada y validada unitariamente; no evaluada):**
  solo existen 5 perfiles exploratorios y screens parciales S0/G0/U0 con
  `b=5/lr=0.1` no-op y ruido legado (`results/quantum-training/`). Desviación
  modular reversible pendiente de aprobación del tutor.
- **Ruido corregido:** desde el 12-nov (§30) el modo por defecto es
  `density_matrix` exacto (`PauliNoiseChannel`) con shots independientes;
  `legacy_shared_trajectory` reproduce el comportamiento histórico de Qibo 0.1.11.
- **Legado híbrido:** congelado en `results/initial-hybrid-adam/`.

## Decisiones metodológicas vigentes

- Etiquetas binarias `-1/+1`.
- Pérdida MSE contra la expectativa Z.
- Partición 70/15/15, estratificada, semilla de dataset `42` por defecto.
- Normalización min-max ajustada solo con train; valores fuera de rango se
  recortan a `[0, 1]`.
- Breast Cancer: estandarización y PCA ajustados solo con train; cuatro
  componentes normalizados con estadísticas de train.
- 30 épocas exactas como configuración de referencia provisional, no como
  óptimo global.
- Baseline logístico externo, nunca parte del motor QML.
- Test se usa una sola vez para la evaluación final; no para seleccionar pesos
  ni hiperparámetros.
- La ruta QGD selecciona el checkpoint por pérdida de validation; su API de
  entrenamiento no acepta test. Los contadores de circuitos son estimaciones
  estructurales, no instrumentación.

## Unidades de comparación

- **Época:** una pasada completa por train con pérdida, gradiente y una
  actualización. Aplica a Adam y QGD.
- **Paso:** actualización de QGD; equivale a una pasada por los datos, pero no
  necesariamente al coste de una época del entrenamiento legacy. Parameter-shift,
  shots y circuitos de actualización deben contabilizarse por separado.
- **Evaluación:** una ejecución de la pérdida sobre train, sin gradiente ni
  actualización. Aplica a COBYLA y SPSA.
- **Iteración:** una actualización del optimizador; SPSA consume dos
  evaluaciones por iteración.

No compares épocas de Adam con evaluaciones de COBYLA como si fueran
equivalentes.

## Limitaciones conocidas

- El parameter-shift es el principal coste: reejecuta el circuito por cada
  aparición de puerta parametrizada.
- El pooling es lógico: no reduce la memoria del vector de estado.
- `v1` ignora `x3` y `w1–w3` de capa 2 (sección 37). `v1-param-pool` y `qibo`
  no tienen ese bloqueo pero aún no están evaluadas ni integradas en
  `transpilation`/`hardware_cost`/`run_hardware_inference`.
- Solo clasificación binaria; no hay multiclase ni minibatches.
- El ruido local es de sensibilidad (depolarizante + error de lectura simétrico
  opcional en la línea variacional / asimétrico en el mock hardware), sin
  amplitude ni phase damping ni calibración real de QPU. En el modo
  `legacy_shared_trajectory` (usado en los JSON M5/screens previos al 12-nov),
  Qibo 0.1.11 muestrea una sola trayectoria compartida por todos los shots.
- Simulación local en CPU; ninguna ejecución real en MareNostrum5-Ona todavía.
  `QililabExecutor`/`src/qililab_adapter.py` solo están probados con mocks en
  memoria; topologías, puertas nativas `{RX,RY,RZ,CZ}` y tiempos con lotes son
  supuestos.
- QGD M1-M5 no tiene ninguna ejecución con pesos que se muevan (`b=5/lr=0.1` es
  estático); su regla modular y la política de U0 siguen pendientes de tutoría.
- `load_binary_iris()` se eliminó intencionadamente; usar
  `load_binary_iris_splits()`.

## Dónde mirar

- Estado vivo y mapa completo: `AGENTS.md`.
- Decisiones técnicas: `docs/TFG-technical-log.md` (secciones 1-41).
- Protocolo: `docs/experimental-protocol.md`.
- Plan histórico QGD: `docs/quantum-training-plan.md` (plan, no estado actual).
- Tutoría e informes PDF: `docs/tutor-briefing-2026-11-12.md` y `docs/reports/`.
- Plan BSC y piloto Ona: `docs/bsc/README.md`, `docs/bsc/pilot-execution.md`,
  `docs/bsc/res-application-draft.md`, `slurm/`.
- Workbench: `docs/workbench/README.md`.
- Resultados: `results/initial-hybrid-adam/`, `results/reproductions/legacy/`,
  `results/quantum-training/`, `results/variational/`, `results/diagnostics/` y
  `results/hardware/`.
- Memoria: `memory/main.tex`, `memory/preamble.tex`, `memory/chapters/` y
  `memory/bibliography.bib` (vacío).

## Entorno y dependencias comprobados

- `requirements.txt` (57 líneas, UTF-8, aprobado el 25-nov-2026) conserva los
  pins de `origin/python3.10-variation` (objetivo CPython 3.10.13,
  `scikit-learn==1.6.1`) y añade `PySide6==6.12.0` para Workbench.
- Entorno local verificado: `qml_env\Scripts\python.exe` (Python 3.14.3,
  Qibo 0.1.11, NumPy 2.5.3, SciPy 1.18.1, Torch 2.14.0, scikit-learn 1.9.1,
  PySide6, Matplotlib, NetworkX). No valida CPython 3.10.13 (pendiente en
  `.venv-py310`).
- El tipado usa `from __future__ import annotations` y es compatible con
  Python 3.10+. No instalar ni modificar dependencias sin aprobación.
