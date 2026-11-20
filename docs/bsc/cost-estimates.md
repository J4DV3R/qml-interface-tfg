# Estimaciones técnicas para la solicitud RES (generado)

Generado por `experiments/estimate_hardware_cost.py`. Topologías **genéricas e hipotéticas**; puertas nativas supuestas `{RX, RY, RZ, CZ}`. Profundidad nativa: máximo sobre 20 circuitos con pesos y datos aleatorios.

## Tamaño del circuito (4 qubits lógicos, 1 qubit medido)

| Cod. | Topología | Prof. lógica | Puertas 2q lógicas | SWAPs | Prof. nativa | CZ | Puertas 1q | Qubits físicos |
|---|---|---|---|---|---|---|---|---|
| angle | full-4 | 25 | 11 | 0 | 26 | 11 | 25 | 4 |
| angle | line-5 | 25 | 11 | 2 | 35 | 17 | 37 | 4 |
| angle | ring-5 | 25 | 11 | 2 | 35 | 17 | 37 | 4 |
| angle | star-5 | 25 | 11 | 6 | 65 | 29 | 59 | 4 |
| angle | grid-20 | 25 | 11 | 2 | 35 | 17 | 37 | 4 |
| angle | grid-35 | 25 | 11 | 2 | 35 | 17 | 37 | 4 |
| phase | full-4 | 26 | 11 | 0 | 29 | 11 | 37 | 4 |
| phase | line-5 | 26 | 11 | 2 | 38 | 17 | 49 | 4 |
| phase | ring-5 | 26 | 11 | 2 | 38 | 17 | 49 | 4 |
| phase | star-5 | 26 | 11 | 6 | 68 | 29 | 71 | 4 |
| phase | grid-20 | 26 | 11 | 2 | 38 | 17 | 49 | 4 |
| phase | grid-35 | 26 | 11 | 2 | 38 | 17 | 49 | 4 |

## Escenarios (1000 shots por circuito)

| Escenario | Circuitos | Shots totales | Horas (cota sin lotes) | Horas (supuesto con lotes) |
|---|---|---|---|---|
| H1-smoke | 17 | 17,000 | 0.1 | 0.01 |
| H1-iris | 248 | 248,000 | 1.4 | 0.02 |
| H1-breast-cancer | 1,376 | 1,376,000 | 7.6 | 0.09 |
| H2-iris-sequential-10-steps | 28,165 | 28,165,000 | 156.5 | 1.88 |
| H2-iris-spsa-100-steps-mb10 | 3,515 | 3,515,000 | 19.5 | 0.24 |
| H2-iris-adam-100-steps (reference, not proposed) | 232,515 | 232,515,000 | 1291.8 | 15.51 |
| simulation-equivalent: BC adam 100 steps (not proposed) | 1,321,985 | 1,321,985,000 | 7344.4 | 88.13 |

## Modelos de tiempo (supuestos)

- **unbatched-upper-bound**: 20 s por trabajo, 1 circuitos por trabajo, 0 µs por shot. RES guide reference: ~20 s per job (depth 1000, 10k shots), dominated by overhead; each circuit treated as its own job. Upper bound.
- **batched-assumption**: 20 s por trabajo, 500 circuitos por trabajo, 200 µs por shot. Assumption, not a BSC figure: 20 s overhead per connection, 500 circuits per connection and 200 microseconds per shot (typical superconducting repetition time). Must be replaced by measured values once access is granted.
