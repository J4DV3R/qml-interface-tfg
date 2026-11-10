# Resultados del entrenamiento no híbrido

Directorio reservado para los resultados del plan descrito en
[`docs/quantum-training-plan.md`](../../docs/quantum-training-plan.md):
motor Qibo puro, QGD discreto, COBYLA y SPSA.

No mezclar estos archivos con los resultados históricos de
`results/initial-hybrid-adam/`, que corresponden a la etapa híbrida con
PyTorch y Adam.

Convención propuesta de nombres:

```text
results/quantum-training/<metodo>/<dataset>-<encoding>-<presupuesto>-<condicion>.json
```

Ejemplo:

```text
results/quantum-training/qgd/breast-cancer-angle-100steps-exact.json
```

Los perfiles exploratorios M5 se guardan directamente en este directorio con
prefijo `m5-profile-`. Son ejecuciones de dos pasos para comprobar el flujo y
estimar coste; no deben tratarse como evaluación de convergencia o rendimiento.
Los cinco usan QGD `b=5/lr=0.1`, configuración en la que el delta entero es cero
para los 32 códigos de gradiente y los pesos solo se cuantizan; además, el ruido
de Qibo se muestrea una vez por ejecución del circuito, no por shot. Su
configuración, tiempos y limitaciones están registrados en la sección 25 de
[`docs/TFG-technical-log.md`](../../docs/TFG-technical-log.md).

Los screens S0, G0 y U0 se guardan con prefijo `screen-`. Son ejecuciones de
dos pasos para comprobar canal, conteos, reproducibilidad y coste; tampoco son
una evaluación de rendimiento. Los 12 JSON válidos declaran `screening=true` y
no evalúan `test`. S0 quedó incompleto (falta la segunda referencia exacta), G0
completó el bloque pareado definido y U0 tiene tres fallos de decodificación y
una sola ejecución válida. Los logs de las ejecuciones fallidas se conservan
aunque no exista JSON asociado. El inventario y los tiempos están en la sección
28; el diagnóstico del no-op, la limitación de ruido y las decisiones
pendientes, en la sección 29 de
[`docs/TFG-technical-log.md`](../../docs/TFG-technical-log.md). No se deben
sobrescribir ni presentar como resultados confirmatorios.
