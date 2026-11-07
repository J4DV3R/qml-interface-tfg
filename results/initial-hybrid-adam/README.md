# Resultados históricos: primera etapa híbrida (QCNN + PyTorch + Adam)

**Fecha de archivo:** 6 de noviembre de 2026.
**Código de referencia:** commit `8b0e51d`.
**Método:** Qibo (backend numpy, estado puro) + parameter-shift + PyTorch
autograd + Adam.
**Estado:** congelado. No sobrescribir estos archivos con ejecuciones nuevas.

Este directorio conserva todos los resultados generados durante la primera
etapa del TFG, antes de sustituir el entrenamiento híbrido por el plan de
entrenamiento cuántico descrito en
[`docs/quantum-training-plan.md`](../../docs/quantum-training-plan.md).

Los resultados nuevos deben guardarse en `results/quantum-training/` para no
mezclar métodos, configuraciones ni conclusiones.

## Archivos

| Archivo | Contenido |
|---|---|
| `qcnn-validation.json` | Batería reducida de validación técnica: 2 épocas, 50 shots, ruido 0.0/0.01, dos semillas en las condiciones muestreadas |
| `iris-qcnn-medium.json` | Prueba intermedia sobre Iris: 3 épocas, 50/100 shots, ruido 0.0/0.01, 9 ejecuciones y 5 resúmenes |
| `qcnn-convergence.json` | Diagnóstico de convergencia exacta sobre Iris con 3, 10, 20 y 30 épocas |
| `qcnn-noisy-convergence-5epochs.json` | Comparación exacta frente a 50 shots con ruido 0.01 durante 5 épocas sobre Iris |
| `breast-cancer-qcnn-convergence.json` | Convergencia exacta sobre Breast Cancer con PCA a 4 componentes y 10, 20 y 30 épocas |
| `breast-cancer-qcnn-seeds.json` | Variabilidad entre tres inicializaciones con 30 épocas exactas y codificación angular |
| `breast-cancer-qcnn-angle.json` | Referencia angular con las mismas tres semillas que la prueba de variabilidad |
| `breast-cancer-qcnn-phase.json` | Comparación con codificación por fase y las mismas tres semillas |
| `breast-cancer-qcnn-noisy.json` | Condición reducida con 5 épocas, 50 shots, ruido 0.01 y dos semillas |
| `*.log` | Salidas de ejecución y errores de los experimentos correspondientes |

## Resultados principales

- Iris, convergencia exacta: accuracy de test `0.533`, `0.667`, `0.667` y
  `0.733` para 3, 10, 20 y 30 épocas.
- Breast Cancer, convergencia exacta: accuracy de test `0.651`, `0.744` y
  `0.826` para 10, 20 y 30 épocas.
- Breast Cancer, tres semillas angulares: `0.826`, `0.872` y `0.849`
  (media `0.849`).
- Breast Cancer, tres semillas en fase: `0.826`, `0.872` y `0.895`
  (media `0.864`).
- Baseline logístico sobre Breast Cancer: accuracy `0.907` y F1 `0.931`.
- Condición ruidosa sobre Breast Cancer: accuracy de test `0.744` y `0.616`
  para las semillas 1 y 2, con unos 26 minutos por ejecución.

## Notas de conservación

- Los JSON incluyen la configuración, las métricas y el baseline de regresión
  logística cuando corresponde.
- `breast-cancer-qcnn-noisy.json` conserva el esquema antiguo de metadatos
  (`seed` en lugar de `seeds`); se mantiene tal cual como registro histórico.
- `breast-cancer-qcnn-seeds.log` está vacío y
  `breast-cancer-qcnn-seeds-error.log` solo contiene el aviso informativo de
  inicialización de Qibo.
- No se han recalculado ni corregido las cifras al archivar. Cualquier
  reanálisis debe hacerse en un informe aparte, sin modificar los originales.
