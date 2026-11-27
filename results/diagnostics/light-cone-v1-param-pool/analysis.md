# Diagnóstico del cono de luz de la QCNN, arquitectura `v1-param-pool` (generado)

Generado por `experiments/diagnose_light_cone.py`. Propiedad del circuito, independiente de los datos.

Pesos entrenables: 12 (forma [2, 6]); apariciones parametrizadas: 25; CNOT: 11; qubit de lectura: 0.

## Codificación angle (40 entradas y pesos aleatorios)

- Características sin influencia en la salida: [] (cambio máximo por característica: [0.7322, 0.5893, 0.7766, 1.2973]).
- Pesos `[capa, k]` sin influencia: [].
- Apariciones con contribución al gradiente: 25 de 25.
- Grado trigonométrico efectivo por peso `capa,k`: {'0,w0': [3], '0,w1': [3], '0,w2': [3], '0,w3': [3], '0,pool_keep': [4], '0,pool_discard': [2], '1,w0': [1], '1,w1': [1], '1,w2': [1], '1,w3': [1], '1,pool_keep': [2], '1,pool_discard': [1]}.
- Circuitos por muestra: parameter-shift 51 (actual) frente a 51 (solo apariciones vivas); secuencial 62 (actual) frente a 62 (grado efectivo).
- Diferencia máxima simulador rápido frente a Qibo: 4.4e-16.

| Aparición | Contribución máxima |
|---|---|
| 000-layer0-ry0-w0 | 0.4430 |
| 001-layer0-rz0-w1 | 0.4546 |
| 002-layer0-ry1-w2 | 0.6763 |
| 003-layer0-rz1-w3 | 0.3907 |
| 004-layer0-ry1-w0 | 0.4042 |
| 005-layer0-rz1-w1 | 0.6731 |
| 006-layer0-ry2-w2 | 0.5649 |
| 007-layer0-rz2-w3 | 0.2179 |
| 008-layer0-ry2-w0 | 0.4453 |
| 009-layer0-rz2-w1 | 0.2617 |
| 010-layer0-ry3-w2 | 0.6155 |
| 011-layer0-rz3-w3 | 0.3168 |
| 012-layer0-ry0-pool_keep | 0.8300 |
| 013-layer0-ry1-pool_discard | 0.5270 |
| 014-layer0-ry0--pool_keep | 0.8402 |
| 015-layer0-ry2-pool_keep | 0.5767 |
| 016-layer0-ry3-pool_discard | 0.4902 |
| 017-layer0-ry2--pool_keep | 0.7778 |
| 018-layer1-ry0-w0 | 0.8402 |
| 019-layer1-rz0-w1 | 0.3993 |
| 020-layer1-ry2-w2 | 0.6244 |
| 021-layer1-rz2-w3 | 0.4988 |
| 022-layer1-ry0-pool_keep | 0.6302 |
| 023-layer1-ry2-pool_discard | 0.5699 |
| 024-layer1-ry0--pool_keep | 0.5727 |

## Codificación phase (40 entradas y pesos aleatorios)

- Características sin influencia en la salida: [] (cambio máximo por característica: [0.9894, 0.6471, 0.5218, 0.7398]).
- Pesos `[capa, k]` sin influencia: [].
- Apariciones con contribución al gradiente: 25 de 25.
- Grado trigonométrico efectivo por peso `capa,k`: {'0,w0': [3], '0,w1': [3], '0,w2': [3], '0,w3': [3], '0,pool_keep': [4], '0,pool_discard': [2], '1,w0': [1], '1,w1': [1], '1,w2': [1], '1,w3': [1], '1,pool_keep': [2], '1,pool_discard': [1]}.
- Circuitos por muestra: parameter-shift 51 (actual) frente a 51 (solo apariciones vivas); secuencial 62 (actual) frente a 62 (grado efectivo).
- Diferencia máxima simulador rápido frente a Qibo: 5.4e-16.

| Aparición | Contribución máxima |
|---|---|
| 000-layer0-ry0-w0 | 0.3675 |
| 001-layer0-rz0-w1 | 0.5474 |
| 002-layer0-ry1-w2 | 0.5004 |
| 003-layer0-rz1-w3 | 0.5331 |
| 004-layer0-ry1-w0 | 0.6743 |
| 005-layer0-rz1-w1 | 0.5241 |
| 006-layer0-ry2-w2 | 0.3380 |
| 007-layer0-rz2-w3 | 0.3882 |
| 008-layer0-ry2-w0 | 0.3631 |
| 009-layer0-rz2-w1 | 0.3848 |
| 010-layer0-ry3-w2 | 0.3890 |
| 011-layer0-rz3-w3 | 0.2444 |
| 012-layer0-ry0-pool_keep | 0.5806 |
| 013-layer0-ry1-pool_discard | 0.5917 |
| 014-layer0-ry0--pool_keep | 0.7090 |
| 015-layer0-ry2-pool_keep | 0.5309 |
| 016-layer0-ry3-pool_discard | 0.3536 |
| 017-layer0-ry2--pool_keep | 0.4340 |
| 018-layer1-ry0-w0 | 0.7090 |
| 019-layer1-rz0-w1 | 0.3495 |
| 020-layer1-ry2-w2 | 0.5398 |
| 021-layer1-rz2-w3 | 0.3300 |
| 022-layer1-ry0-pool_keep | 0.5451 |
| 023-layer1-ry2-pool_discard | 0.6499 |
| 024-layer1-ry0--pool_keep | 0.5156 |
