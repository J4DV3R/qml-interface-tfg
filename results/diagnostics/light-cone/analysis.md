# Diagnóstico del cono de luz de la QCNN (generado)

Generado por `experiments/diagnose_light_cone.py`. Propiedad del circuito, independiente de los datos.

## Codificación angle (40 entradas y pesos aleatorios)

- Características sin influencia en la salida: [3] (cambio máximo por característica: [0.8147, 0.841, 1.1062, 0.0]).
- Pesos `[capa, k]` sin influencia: [[1, 1], [1, 2], [1, 3]].
- Apariciones con contribución al gradiente: 7 de 16.
- Grado trigonométrico efectivo por peso `capa,k`: {'0,0': [2], '0,1': [1], '0,2': [2], '0,3': [1], '1,0': [1], '1,1': [0], '1,2': [0], '1,3': [0]}.
- Circuitos por muestra: parameter-shift 33 (actual) frente a 15 (solo apariciones vivas); secuencial 40 (actual) frente a 19 (grado efectivo).
- Diferencia máxima simulador rápido frente a Qibo: 6.7e-16.

| Aparición | Contribución máxima |
|---|---|
| layer0-pair01-w0 | 0.8095 |
| layer0-pair01-w1 | 0.8693 |
| layer0-pair01-w2 | 0.5418 |
| layer0-pair01-w3 | 0.2426 |
| layer0-pair12-w0 | 0.7803 |
| layer0-pair12-w1 | 0.0000 |
| layer0-pair12-w2 | 0.7132 |
| layer0-pair12-w3 | 0.0000 |
| layer0-pair23-w0 | 0.0000 |
| layer0-pair23-w1 | 0.0000 |
| layer0-pair23-w2 | 0.0000 |
| layer0-pair23-w3 | 0.0000 |
| layer1-pair02-w0 | 0.9057 |
| layer1-pair02-w1 | 0.0000 |
| layer1-pair02-w2 | 0.0000 |
| layer1-pair02-w3 | 0.0000 |

## Codificación phase (40 entradas y pesos aleatorios)

- Características sin influencia en la salida: [3] (cambio máximo por característica: [1.5762, 1.5912, 0.6742, 0.0]).
- Pesos `[capa, k]` sin influencia: [[1, 1], [1, 2], [1, 3]].
- Apariciones con contribución al gradiente: 7 de 16.
- Grado trigonométrico efectivo por peso `capa,k`: {'0,0': [2], '0,1': [1], '0,2': [2], '0,3': [1], '1,0': [1], '1,1': [0], '1,2': [0], '1,3': [0]}.
- Circuitos por muestra: parameter-shift 33 (actual) frente a 15 (solo apariciones vivas); secuencial 40 (actual) frente a 19 (grado efectivo).
- Diferencia máxima simulador rápido frente a Qibo: 5.3e-16.

| Aparición | Contribución máxima |
|---|---|
| layer0-pair01-w0 | 0.7288 |
| layer0-pair01-w1 | 0.7676 |
| layer0-pair01-w2 | 0.4733 |
| layer0-pair01-w3 | 0.3620 |
| layer0-pair12-w0 | 0.7043 |
| layer0-pair12-w1 | 0.0000 |
| layer0-pair12-w2 | 0.5337 |
| layer0-pair12-w3 | 0.0000 |
| layer0-pair23-w0 | 0.0000 |
| layer0-pair23-w1 | 0.0000 |
| layer0-pair23-w2 | 0.0000 |
| layer0-pair23-w3 | 0.0000 |
| layer1-pair02-w0 | 0.5579 |
| layer1-pair02-w1 | 0.0000 |
| layer1-pair02-w2 | 0.0000 |
| layer1-pair02-w3 | 0.0000 |
