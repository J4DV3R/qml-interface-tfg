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

# Runner M1-M5 (solo simulador Qibo; no constituye ejecución QPU)
# Para finite-shots con ruido cero, añadir --measured-execution.
# No ejecutar perfiles largos sin congelar coste y obtener autorización.
# ADVERTENCIA: con --qgd-bits 5 --learning-rate 0.1 el delta entero es 0
# para los 32 códigos de gradiente; el update solo cuantiza. No usar como
# evidencia de aprendizaje. La revisión de b/lr está pendiente del tutor.
qml_env\Scripts\python.exe -m experiments.run_quantum_training `
  --dataset iris --max-steps 1 --shots 100 --measured-execution --noise 0.0 `
  --learning-rate 0.1 --qgd-bits 5 --encoding angle `
  --dataset-seed 42 --initialization-seed 19 --execution-seed 101 `
  --output results/quantum-training/iris-angle-1step-shots100.json
```

La configuración `b=5/lr=0.1` es un no-op; los screens S0/G0/U0 y los perfiles
M5 la usan. La suite completa actual es de 92 pruebas (10 de noviembre de 2026;
sección 29 del log). En Qibo 0.1.11 el canal depolarizante se muestrea una vez
por ejecución del circuito, no por shot; los shots no promedian ruido.
`requirements.txt` está aprobado para el commit de cierre del 10 de noviembre de
2026; no lo modifiques de nuevo sin nueva aprobación.

El objetivo acordado con el tutor es preparar ejecución en hardware cuántico
real y llevar el ajuste aritmético de pesos a circuitos cuanto sea posible. No
está pendiente la dirección general; sí deben definirse el backend y la
implementación concreta de actualización. M5 implementa actualización aritmética
reversible medida mediante counts, pero el adaptador disponible es solo el
simulador Qibo. No hay backend de proveedor, transpilación ni validación de
recursos del dispositivo. La regla modular difiere del clipping propuesto, por
lo que no se debe asumir que esa regla concreta está aprobada.

Para la nueva arquitectura, sigue las etapas y condiciones propuestas en la
sección «Protocolo propuesto para orientar la ruta QGD a QPU» de
`docs/experimental-protocol.md`. No ejecutes una matriz extensa: primero prueba
un peso, escala a todos, valida circuitos y mide coste. Antes de cualquier
ejecución larga o costosa, congela configuración y solicita aprobación
específica. Distingue claramente estimaciones de circuitos, trabajos enviados y
shots.

## Esquemas JSON observados

Los resultados históricos usan dos familias:

1. Barridos: claves `sweep`, `runs`, `summaries`, `baseline`.
2. Batería focalizada: claves `dataset`, `model`, `training`, `results`,
   `baseline`.
3. Runner QGD M1-M5: `dataset`, `provenance`, `seeds`, `model`,
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
- Ruta QGD: distingue gradiente inicial exacto o medido, modo medido/no medido
  del update, shots, y contadores estructurales de ejecuciones (estimaciones, no
  instrumentación).
- Baseline: mismas particiones y transformaciones.
- Revisión del código y comando de ejecución.

El CLI registra en `provenance` el `HEAD`, si el árbol estaba sucio, las rutas
modificadas y el `command_argv`. Un `HEAD` con árbol sucio no identifica por sí
solo todos los cambios ejecutados; conserva y consulta también ese estado.

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
