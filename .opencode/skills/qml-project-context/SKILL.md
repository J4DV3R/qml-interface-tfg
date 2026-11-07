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
5. `src/hybrid_model.py` calcula expectativas Pauli Z y usa parameter-shift
   dentro de una función de autograd de PyTorch.
6. `src/training.py` ejecuta el bucle clásico con Adam o SGD.
7. `src/metrics.py` calcula accuracy, precision, recall y F1 para etiquetas
   `-1/+1`.
8. `src/classical_baseline.py` entrena una regresión logística externa al motor
   QML, con la misma partición y transformaciones.

## Estado actual del código

- **Implementado:** QCNN con codificación angular o por fase, pooling lógico
  `4 -> 2 -> 1`, expectativa Z, parameter-shift, simulación exacta, shots,
  ruido depolarizante, semilla, Adam/SGD, métricas binarias y baseline.
- **Implementado:** datasets Iris binario y Breast Cancer Wisconsin con PCA a
  cuatro componentes ajustado solo con train.
- **Implementado:** experimentos `compare_execution.py`,
  `diagnose_convergence.py` y `run_final_qcnn.py`.
- **Planificado, no implementado:** motor QCNN puro sin PyTorch
  (`src/qcnn_engine.py`), optimizadores sin gradientes (`src/optimizers.py`),
  actualización cuántica de pesos (`src/quantum_update.py`), runner
  `experiments/run_quantum_training.py` y pruebas
  `tests/test_quantum_training.py`.
- **Histórico:** los resultados de la etapa híbrida están congelados en
  `results/initial-hybrid-adam/`.

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

## Unidades de comparación

- **Época:** una pasada completa por train con pérdida, gradiente y una
  actualización. Aplica a Adam y QGD.
- **Paso:** actualización de QGD; equivale a una época en datos y coste de
  simulación.
- **Evaluación:** una ejecución de la pérdida sobre train, sin gradiente ni
  actualización. Aplica a COBYLA y SPSA.
- **Iteración:** una actualización del optimizador; SPSA consume dos
  evaluaciones por iteración.

No compares épocas de Adam con evaluaciones de COBYLA como si fueran
equivalentes.

## Limitaciones conocidas

- El parameter-shift es el principal coste: reejecuta el circuito por cada
  aparición de puerta parametrizada.
- El pooling no reduce la memoria del vector de estado.
- Solo clasificación binaria; no hay multiclase ni minibatches.
- El ruido es depolarizante después de `RY`, `RZ` y `CNOT`; no hay ruido de
  medición, amplitude damping ni phase damping.
- Simulación local en CPU; no es hardware cuántico real.
- Pocas semillas y una sola partición de test en los resultados históricos.
- El plan QGD, COBYLA y SPSA está pendiente de implementación y evaluación.

## Dónde mirar

- Decisiones técnicas: `docs/TFG-technical-log.md`.
- Protocolo: `docs/experimental-protocol.md`.
- Plan no híbrido: `docs/quantum-training-plan.md`.
- Resultados: `results/initial-hybrid-adam/`,
  `results/reproductions/legacy/` y `results/quantum-training/`.
- Memoria: `memory/main.tex`, `memory/preamble.tex` y `memory/chapters/`.
