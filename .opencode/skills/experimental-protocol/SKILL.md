---
name: Protocolo experimental
description: "Preparación, ejecución y registro reproducible de experimentos; lectura correcta de resultados JSON, particiones, semillas y directorios de resultados."
---

# Protocolo experimental

## Antes de ejecutar

1. Confirma que el usuario ha pedido explícitamente el experimento.
2. Lee `docs/experimental-protocol.md` y, si aplica,
   `docs/quantum-training-plan.md`.
3. Ejecuta `git status --short --branch` y anota la revisión del código.
4. Congela la configuración: dataset, particiones, semilla de dataset,
   semillas de inicialización, codificación, optimizador, presupuesto, shots y
   ruido.
5. Comprueba que el comando escribe en el directorio correcto y que no
   sobrescribe resultados históricos.
6. Pide aprobación si el comando puede tardar mucho, consumir recursos o
   modificar datos.

## Directorios de resultados

- `results/initial-hybrid-adam/`: resultados históricos congelados. No
  sobrescribir.
- `results/reproductions/legacy/`: repeticiones de experimentos antiguos.
- `results/quantum-training/`: resultados del nuevo entrenamiento cuántico.

Conserva siempre los JSON de origen y los logs. No los edites a mano para
«corregir» cifras. Si hace falta un reanálisis, crea un informe aparte.

## Comandos de referencia

```powershell
# Pruebas del proyecto (solo cuando se soliciten; usar el entorno de este checkout)
qml_env\Scripts\python.exe -m unittest discover -s tests -v

# Diagnóstico de convergencia exacta en Iris
qml_env\Scripts\python.exe -m experiments.diagnose_convergence `
  --output results/reproductions/legacy/qcnn-convergence.json

# Convergencia exacta en Breast Cancer
qml_env\Scripts\python.exe -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 10 20 30 --seeds 19 `
  --output results/reproductions/legacy/breast-cancer-qcnn-convergence.json

# Variabilidad entre semillas
qml_env\Scripts\python.exe -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 30 --seeds 19 23 29 `
  --output results/reproductions/legacy/breast-cancer-qcnn-seeds.json

# Condición reducida con shots y ruido
  qml_env\Scripts\python.exe -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 5 --seeds 1 2 --shots 50 --noise 0.01 `
  --output results/reproductions/legacy/breast-cancer-qcnn-noisy.json

# Ruta M1-M4: no ejecutar sin aprobación explícita; no se ha evaluado aún
qml_env\Scripts\python.exe -m experiments.run_quantum_training `
  --dataset iris --max-steps 1 --shots 100 --noise 0.0 `
  --learning-rate 0.1 --qgd-bits 5 --encoding angle `
  --dataset-seed 42 --initialization-seed 19 --execution-seed 101 `
  --output results/quantum-training/iris-angle-1step-shots100.json
```

No ejecutes el runner QGD, la batería completa ni amplíes shots, épocas o
semillas sin autorización. La ruta QGD conserva la actualización modular y
saturación de gradientes; no equivale al clipping propuesto y el tutor aún no la
ha aceptado.

## Esquemas JSON observados

Los resultados históricos usan dos familias:

1. Barridos: claves `sweep`, `runs`, `summaries`, `baseline`.
2. Batería focalizada: claves `dataset`, `model`, `training`, `results`,
   `baseline`.
3. Runner QGD M1-M4 (solo cuando exista JSON): `dataset`, `seeds`, `model`,
   `initial_weights`, `qgd`, `training`, `best_weights`, `final_weights`, `test`,
   `circuit_executions_estimate` y `classical_quantum_boundary`.

Antes de leer un JSON, comprueba sus claves reales. No supongas que todos los
archivos tienen el mismo esquema.

Campos que deben conservarse o verificarse:

- Dataset: nombre, semilla, fracciones, número de componentes.
- Modelo: qubits, codificación, ejecución (shots, ruido, semillas).
- Entrenamiento: optimizador, learning rate, épocas o presupuesto.
- Resultados individuales: historial de pérdida, pérdida de validación,
  métricas de validación y test, tiempo.
- Ruta QGD: distingue gradiente inicial exacto, gradientes posteriores con shots,
  actualización state-vector exacta sin medición/shots, y contadores estructurales
  de ejecuciones (estimaciones, no instrumentación).
- Baseline: mismas particiones y transformaciones.
- Revisión del código y comando de ejecución.

## Reglas de lectura

- No inventes valores que no estén en el JSON.
- Calcula agregados solo a partir de los resultados individuales.
- No mezcles condiciones con distinta partición, semilla, shots, ruido o
  presupuesto.
- Distingue semilla de dataset, semilla de inicialización y semilla de
  ejecución.
- Reporta media, desviación o intervalo solo si hay datos suficientes.
- No presentes una diferencia pequeña como concluyente con pocas semillas.
- Si falta un metadato, indícalo; no lo rellenes por inferencia.

## Registro de una ejecución

Incluye como mínimo:

- Comando exacto.
- Revisión de Git.
- Configuración efectiva.
- Ruta del JSON y del log.
- Tiempo de reloj.
- Resultados individuales y agregados.
- Limitaciones y observaciones.
- Si la ejecución se interrumpió, estado parcial y causa.

## Después de ejecutar

1. Comprueba que el JSON es válido y contiene la configuración esperada.
2. No modifiques el JSON de origen.
3. Registra la evidencia en `docs/TFG-technical-log.md` o en el documento que
   corresponda, a través del escritor.
4. Si el resultado contradice la documentación, repórtalo; no lo maquilles.
