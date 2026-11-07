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
6. **Ruta QGD M1-M4:** `src/qcnn_engine.py` es el motor NumPy/Qibo sin Torch;
   `src/quantum_training.py` integra el entrenamiento y
   `src/quantum_update.py` ejecuta un circuito reversible por peso.
7. `src/metrics.py` conserva métricas Torch del legado; el entrenamiento QGD
   calcula métricas binarias NumPy sobre train/validation y el runner calcula
   test una sola vez después de seleccionar por validation.
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
- **Implementado y validado unitariamente (M1-M4):** motor puro, optimizadores
  autónomos COBYLA/SPSA/Nelder-Mead, actualización QGD modular, bucle y CLI
  (`src/qcnn_engine.py`, `src/optimizers.py`, `src/quantum_update.py`,
  `src/quantum_training.py`, `experiments/run_quantum_training.py`). La última
  validación registrada pasó 52 pruebas mediante
  `qml_env\Scripts\python.exe -m unittest discover -s tests -v`; `compileall`
  y `git diff --check` también pasaron.
- **No evaluado:** no se ha ejecutado el CLI ni hay JSON nuevo bajo
  `results/quantum-training/`.
- **Frontera actual:** primer gradiente parameter-shift exacto/sin ruido;
  gradientes de pasos posteriores con circuitos y shots. El circuito QGD se
  simula en estado-vector exacto y el host decodifica el código de salida por
  `argmax` (no se usa una puerta de medida ni shots en ese update). El host
  también acumula derivadas, pérdidas, selección por validation y control del
  bucle. No describirlo como entrenamiento totalmente cuántico.
- **Desviación QGD:** actualización modular reversible que conserva el registro
  del gradiente; gradientes fuera de rango se saturan. No equivale al clipping
  propuesto; requiere una decisión sobre esta variante concreta. El objetivo del
  tutor, según la aclaración del responsable, es preparar ejecución en hardware
  cuántico real y mover el ajuste de pesos al circuito, evitando el ajuste
  clásico. La implementación actual aún no satisface ese objetivo de extremo a
  extremo.
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
- El pooling no reduce la memoria del vector de estado.
- Solo clasificación binaria; no hay multiclase ni minibatches.
- El ruido es depolarizante después de `RY`, `RZ` y `CNOT`; no hay ruido de
  medición, amplitude damping ni phase damping.
- Simulación local en CPU; no es hardware cuántico real.
- Pocas semillas y una sola partición de test en los resultados históricos.
- El método QGD nuevo aún no tiene evaluación experimental ni ejecución en
  hardware real. El objetivo de hardware y de ajuste de pesos en circuito viene
  indicado por el tutor; la variante modular concreta y las operaciones host
  admisibles necesitan definirse y contrastarse antes de afirmar cumplimiento.
- `src/quantum_training.py` no importa Torch, pero el runner carga
  `src.datasets`, que importa Torch y devuelve tensores; estos se convierten a
  NumPy en la frontera del entrenamiento.
- `requirements.txt` contiene una lista amplia similar a un freeze del entorno,
  no un manifiesto mínimo; `scikit-learn` no está fijado y `torchvision` no tiene
  import directo en `src/` ni `experiments/`. No limpiar ni actualizar
  dependencias sin aprobación.

## Dónde mirar

- Decisiones técnicas: `docs/TFG-technical-log.md`.
- Protocolo: `docs/experimental-protocol.md`.
- Plan no híbrido: `docs/quantum-training-plan.md`.
- Resultados: `results/initial-hybrid-adam/`,
  `results/reproductions/legacy/` y `results/quantum-training/`.
- Protocolo QGD y decisiones abiertas: `docs/quantum-training-plan.md`,
  `docs/TFG-technical-log.md` sección 23.
- Memoria: `memory/main.tex`, `memory/preamble.tex` y `memory/chapters/`.

## Entorno y dependencias comprobados

- Usa `qml_env\Scripts\python.exe` en este checkout. El `python` global
  (Python 3.14.3) no tiene Qibo.
- Último entorno comprobado: Python 3.14.3, Qibo 0.1.11, NumPy 2.5.3,
  SciPy 1.18.1, Torch 2.14.0, scikit-learn 1.9.1; `pip check` informó que no
  hay dependencias rotas.
- Imports directos necesarios: NumPy, Qibo, SciPy (optimizadores), Torch
  (legado/loader) y scikit-learn (datasets/baseline). Matplotlib/Jupyter aparecen
  en `requirements.txt`, pero no se encontraron imports directos en código ni
  notebooks versionados.
- El tipado usa sintaxis moderna (`X | Y`); documentar Python >=3.10 si se
  ordena la instalación. El mínimo no está declarado actualmente.
