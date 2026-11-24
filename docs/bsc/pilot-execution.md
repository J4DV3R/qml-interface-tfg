# Piloto de ejecución en MareNostrum5-Ona

## Objetivo

Validar el entorno de ejecución BSC con un circuito mínimo y, si funciona,
probar un único checkpoint de la QCNN de Iris. Esta primera fase no pretende
demostrar convergencia ni comparar optimizadores.

## Tabla de ejecución

| Fase | Circuito o datos | Parámetros iniciales | Comprobación | Estado |
|---|---|---|---|---|
| 1 | Instalación y runcard | Entorno `quantum_blue` o `quantum_red`, 100 shots | Importar Qililab y obtener counts | Pendiente de acceso |
| 2 | Operación mínima | Bell o RY, 1000 shots | Transpilación, ejecución, suma de counts | Pendiente de acceso |
| 3 | Una muestra QCNN | 4 qubits, 1 muestra Iris, 1000 shots | Layout, native gates y exacto vs. hardware | Pendiente de acceso |
| 4 | Inferencia Iris | 15 muestras de test, 1000 shots | Métricas raw y mitigadas, una partición | Pendiente de acceso |
| 5 | Inferencia multi-checkpoint | 8 checkpoints Iris, 1000 shots | 248 circuitos estimados | Opcional |
| 6 | Entrenamiento corto | SPSA o secuencial, presupuesto fijo | Sólo después de validar fases 1–4 | Opcional |

## Adaptador disponible

El repositorio incluye `src/qililab_adapter.py`. La llamada pública es:

```python
result = qililab.execute(circuit, runcard, nshots=1000)
counts = result.counts()
```

El adaptador:

- importa Qililab sólo al ejecutar;
- recibe el runcard sin guardar credenciales;
- ejecuta cada circuito individualmente;
- acepta `result.counts()` o un diccionario;
- normaliza claves binarias;
- valida que el total de counts sea exactamente el número de shots.

## Variables externas

No debe añadirse ninguna clave o runcard al código o al repositorio.
Configuración recomendada:

```bash
export RUNCARD_BLUE=/ruta/suministrada/al/runcard
export RUNCARD_RED=/ruta/suministrada/al/runcard
export CHIP_ID=<chip-suministrado>
export EDGES=<qubit-physical-qubit>
```

La cuenta, QoS y partición Slurm deben quedar fuera del repositorio o
completarse por el tutor en `slurm/ona_inference.sbatch`.

## Comando de inferencia

```bash
python -B -m experiments.run_hardware_inference \
  results/variational/B-confirm-iris/I1-exact/sequential-angle/run-angle-sequential-init19.json \
  --executor qililab \
  --runcard "$RUNCARD_BLUE" \
  --qililab-adapter-module src.qililab_adapter \
  --chip-id "$CHIP_ID" \
  --edges "$EDGES" \
  --shots 1000 \
  --splits test \
  --max-samples 1 \
  --output results/hardware/pilot-smoke.json
```

## Datos que debe confirmar el tutor

- Runcard de `quantum_blue` o `quantum_red`.
- Versión y módulo Qililab disponible.
- Versión de Qibo disponible.
- Chip, qubits físicos y mapa de acoplamiento.
- Cuenta, QoS y partición Slurm.
- Número máximo estable de shots.
- Formato confirmado de `counts()`.
- Política de lotes y tiempo máximo.
