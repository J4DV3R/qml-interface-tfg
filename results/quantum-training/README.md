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
