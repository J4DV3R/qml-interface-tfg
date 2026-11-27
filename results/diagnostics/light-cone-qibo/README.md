# Diagnóstico: cono de luz del ansatz `qibo` (QuantumCNN)

**Fecha:** 26 de noviembre de 2026. **Estado: validado numéricamente.** Es una propiedad del circuito, independiente de los datos y del entrenamiento. No se ha entrenado ni evaluado nada.

Comando: `qml_env/Scripts/python.exe -B -m experiments.diagnose_light_cone --architecture qibo --output-dir results/diagnostics/light-cone-qibo` (40 entradas y pesos aleatorios por codificación, semilla 2026, revisión base `b5abe03` con cambios sin confirmar). Salidas: `analysis.json` y `analysis.md`.

## Circuito

Es el ansatz de `qibo.models.QuantumCNN` con el ansatz de dos qubits por defecto, transcrito del código fuente de la rama `main` de Qibo; la clase no existe en Qibo 0.1.11. Para 4 qubits y 2 capas:

- **Capa 1:**
  - convolución de 15 parámetros `RX RY RZ ⊗ RX RY RZ · RZZ · RYY · RXX · RX RY RZ ⊗ RX RY RZ` sobre los pares `(0,1), (2,3), (1,2), (3,0)`;
  - pooling de 6 parámetros `R(sumidero) R(fuente) CNOT(fuente, sumidero) R(sumidero)⁻¹` de los qubits fuente 0 y 1 a los sumideros 2 y 3.
- **Capa 2:** convolución sobre `(2,3)` y pooling de 2 a 3.
- **Lectura:** ⟨Z⟩ del qubit 3.

Son 21 pesos por capa, compartidos por todos los pares de la capa: 42 pesos, 102 apariciones parametrizadas y 3 CNOT.

Diferencias deliberadas con la clase de Qibo:
- no hay sesgo entrenable;
- se usa nuestra codificación angular o de fase en lugar de un estado inicial;
- se entrena con nuestros optimizadores (MSE sobre −1/+1) en lugar de `qibo.optimizers`.

## Resultado

Con las dos codificaciones (angular y fase):

| Elemento | Resultado |
|---|---|
| Características con influencia | **x0, x1, x2, x3** |
| Pesos con influencia | **40 de 42**: no influyen `pool5` de la capa 1 ni de la capa 2 |
| Apariciones con contribución al gradiente | 99 de 102 (las 3 apariciones de `pool5`) |
| Circuitos por muestra, parameter-shift | 205 (199 sin las apariciones muertas) |
| Circuitos por muestra, secuencial | 246 (238 con el grado efectivo) |
| Simulador rápido frente a Qibo | ≤ 1,1e-15 |

## Por qué `pool5` no influye

`pool5` es la `RZ` del qubit fuente, la última del selector de base de la fuente, justo antes de la CNOT de pooling. `RZ` es diagonal en la base computacional y conmuta con el control de una CNOT. Después, el qubit fuente no vuelve a usarse ni se mide, así que esa rotación es invisible para la salida. Son 2 de los 42 pesos redundantes. Esto no es un error de transcripción: se deduce del propio circuito de Qibo.

## Limitaciones

- Solo describe qué elementos pueden influir en la salida, no si el modelo clasifica mejor. La comparación con `v1` y `v1-param-pool` está pendiente de la tutoría (decisión 6).
- Coste: unas 6 veces más circuitos por muestra y paso que `v1`, aunque cada circuito tiene menos CNOT (3 frente a 11), que es relevante para el hardware.
