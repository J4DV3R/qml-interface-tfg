# Plan BSC: ejecución de la QCNN variacional en MareNostrum5-Ona

**Fecha:** 20 de noviembre de 2026. **Estado global:** fase 1 implementada y validada en simulación; **sin acceso al BSC y sin ninguna ejecución en hardware.**

## Contexto verificado

Fuente: guía de acceso de la RES para recursos cuánticos (*Application Guide for Quantum Computing Resources*, v3.3).

- **MareNostrum5 Ona:** dos chips superconductores digitales, de 20 y 35 qubits; la disponibilidad puede variar. Cada chip está conectado a un nodo de 40 CPU para el pre- y postprocesado.
- **Software:** en el BSC los flujos cuánticos se basan en **Qibo y Qililab**. Los circuitos deben ser compatibles con Qibo.
- **Uso de la partición cuántica:** los trabajos que no usan el chip no se aceptan. Las simulaciones hay que pedirlas aparte en MN5-GPP o MN5-ACC.
- **Datos técnicos que pide la solicitud:** qubits, profundidad máxima, shots, conjunto de puertas y duración del trabajo más largo.
- **Tiempo de referencia:** un trabajo con profundidad 1000 y 10 000 shots tarda unos 20 s, y 1000 trabajos independientes unas 5,5 h, sobre todo por la sobrecarga de conexión.

No se han podido leer de forma automática la página de Slurm del BSC ni la API detallada de Qililab. La topología, las puertas nativas, la calidad de los qubits y los nombres de cuenta, QOS y partición siguen **pendientes**.

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Confirmar con el tutor y preparar la solicitud RES (Ona + MN5-GPP) | **Propuesto**. Borrador en `res-application-draft.md` (no enviado); datos técnicos en `cost-estimates.md` |
| 1 | Preparación sin acceso: estimador, transpilador, ejecutor por lotes, mock ruidoso, mitigación de lectura, inferencia en seco, scripts Slurm | **Implementado y validado en simulación** |
| 2 | Puesta en marcha en el BSC: entorno, runcard, adaptador Qililab y pruebas mínimas | Propuesto; necesita acceso |
| 3 | H1: inferencia en el chip de checkpoints de la fase C (pre-registrada) | Propuesto |
| 4 | H2 (opcional): entrenamiento corto en el chip con Iris, presupuesto fijo | Propuesto |
| 5 | Documentación y capítulo de la memoria | Propuesto |

**Fuera de alcance:** QGD en hardware (sus puertas multicontroladas hacen la profundidad inviable), control a nivel de pulso y más de 4 qubits en el chip.

## Fase 1: entregables

| Archivo | Qué hace |
|---|---|
| `src/hardware_cost.py` | Circuitos por paso y por muestra (33 / 2 / 40), coste de inferencia y entrenamiento, y dos modelos de tiempo **supuestos**: cota sin lotes según la referencia de la RES, y escenario con lotes |
| `src/transpilation.py` | Transpilador mínimo, porque Qibo 0.1.11 no trae `qibo.transpiler`: layout óptimo por distancia, SWAPs por camino mínimo, descomposición a `{RX, RY, RZ, CZ}` y fusión de rotaciones. El circuito se compacta a los qubits físicos usados |
| `src/hardware_execution.py` | Protocolo `BatchCircuitExecutor` y `execute_batch`. `HardwareMockExecutor` es un simulador que solo acepta puertas nativas sobre pares acoplados, con ruido depolarizante local por puerta y error de lectura asimétrico. Incluye calibración y mitigación de lectura, y `QililabExecutor` como **esqueleto no funcional** |
| `experiments/run_hardware_inference.py` | Inferencia de checkpoints guardados. Recalcula particiones, transpila, ejecuta por lotes, calibra la lectura al principio y al final, y da métricas exactas, brutas y mitigadas, con procedencia y coste. No selecciona nada con test |
| `experiments/estimate_hardware_cost.py` | Genera `cost-estimates.md/json`, los datos técnicos para la solicitud |
| `experiments/make_variational_jobs.py` | Genera la lista de trabajos para los *job arrays* |
| `slurm/` | `mn5_gpp_array.sbatch`, `ona_inference.sbatch` (marcadores `<...>`) y README del entorno |
| `tests/test_hardware_preparation.py` | 10 pruebas: coste, descomposiciones exactas, marginal conservada en 6 topologías, mock, mitigación y ejecución en seco |

**Validación (20-nov-2026, árbol sucio sobre `25a0e69`):**
- Comando: `qml_env/Scripts/python.exe -B -m unittest tests.test_hardware_preparation tests.test_circuit_execution tests.test_variational_training tests.test_fast_qcnn tests.test_partition_analysis`.
- Resultado: 47 pruebas OK.
- Revisión previa al push (20-nov, árbol basado en `b9499d0`): **suite completa, 138 pruebas OK**. Se añadió una prueba que compara el estimador de coste con los circuitos que cuenta un entrenamiento real (sección 36 del log).
- La suite completa no se ha reejecutado.

## Datos para la solicitud (de `cost-estimates.md`)

- **Qubits:** 4 lógicos y 1 medido.
- **Profundidad:**
  - Lógica: 25 en angular y 26 en fase, con 11 CNOT.
  - Nativa, en topologías genéricas: 26–38 si hay acoplamiento con triángulo, línea, anillo o rejilla (0–2 SWAPs); 65–68 en estrella (6 SWAPs).
- **Shots:** 1000 por circuito.
- **Puertas:** RY, RZ, H y CNOT en lógico; RX/RY/RZ y CZ supuestas como nativas.
- **Escenarios:**

  | Escenario | Circuitos | Horas sin lotes (cota) | Horas con lotes (supuesto) |
  |---|---|---|---|
  | H1 Iris (8 checkpoints) | 248 | 1,4 | 0,02 |
  | H1 Breast Cancer | 1376 | 7,6 | 0,09 |
  | H2 SPSA con minilotes | 3515 | 19,5 | 0,24 |
  | H2 secuencial, 10 barridos | 28 165 | 156 | 1,9 |

  Entrenar con Adam como en simulación no es viable: 1,3 millones de circuitos en Breast Cancer.

## Ejemplo de ejecución en seco (mock, 2 checkpoints de Iris, línea de 5 qubits)

Se usaron 64 circuitos, 1000 shots, $p_{1q}=0{,}001$, $p_{2q}=0{,}01$ y error de lectura 0,02/0,04.
- La accuracy exacta coincide con la guardada en los JSON.
- El error medio de $\langle Z\rangle$ frente al exacto baja de 0,11–0,13 (bruto) a 0,08–0,09 (mitigado).
- Es una **prueba de funcionamiento con ruido supuesto**, no un resultado de hardware.

## Pendiente para la fase 2

- Implementar `QililabExecutor` con la API documentada por el BSC: runcard, conexión, ejecución por lotes, conversión a *counts* y metadatos de calibración.
- Sustituir las topologías genéricas por el mapa de acoplamiento real y las puertas nativas reales.
- Medir los tiempos reales y reemplazar el modelo `batched-assumption`.
- Comprobar que la versión de Qibo del BSC es compatible con la 0.1.11.
