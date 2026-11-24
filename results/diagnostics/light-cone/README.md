# Diagnóstico: cono de luz de la QCNN de 4 qubits

**Fecha:** 24 de noviembre de 2026. **Estado: validado numéricamente.** Es una propiedad del circuito actual, independiente de los datos y del entrenamiento, y afecta por igual a las rutas híbrida, QGD y variacional, que construyen el mismo circuito.

Comando: `qml_env/Scripts/python.exe -B -m experiments.diagnose_light_cone --output-dir results/diagnostics/light-cone` (40 entradas y pesos aleatorios por codificación, semilla 2026). Salidas: `analysis.json` y `analysis.md`.

## Resultado

Con las dos codificaciones (angular y fase):

| Elemento | Influye en la salida ⟨Z⟩ del qubit 0 |
|---|---|
| Características x0, x1, x2 | Sí |
| **Característica x3** | **No** (cambio máximo 0) |
| Pesos de la capa 1 (w0–w3) | Sí |
| Peso w0 de la capa 2 | Sí |
| **Pesos w1, w2 y w3 de la capa 2** | **No** |
| Apariciones de pesos con contribución al gradiente | **7 de 16** |

- El simulador rápido y el constructor de circuitos de Qibo coinciden (diferencia ≤ 7e-16).
- El modelo usa en realidad **3 de las 4 características** y **5 de los 8 pesos**.

## Por qué ocurre

Secuencia de puertas del circuito (codificación angular):

```
RY RY RY RY | conv capa 1: par (0,1): RY0 RZ0 CNOT(0,1) RY1 RZ1 CNOT(1,0)
                          par (1,2): RY1 RZ1 CNOT(1,2) RY2 RZ2 CNOT(2,1)
                          par (2,3): RY2 RZ2 CNOT(2,3) RY3 RZ3 CNOT(3,2)
            | pooling 1:  CNOT(1,0)  CNOT(3,2)
            | conv capa 2: par (0,2): RY0 RZ0 CNOT(0,2) RY2 RZ2 CNOT(2,0)
            | pooling 2:  CNOT(2,0)
            | medida: Z en el qubit 0
```

1. **Las CNOT duplicadas se anulan.**
   - La convolución termina con `CNOT(derecha → izquierda)`, y el pooling hace `CNOT(descartado → conservado)`. Cuando el qubit descartado es el «derecho» del último par, las dos puertas son la misma CNOT y CNOT·CNOT = identidad.
   - Pasa con `CNOT(3,2)` en la capa 1 (la `CNOT(1,0)` intermedia actúa sobre otros qubits y conmuta) y con `CNOT(2,0)` en la capa 2.
2. **Capa 2.**
   - Tras la cancelación, el qubit 0 solo recibe `RY(w0)`, `RZ(w1)` y actúa como control de `CNOT(0,2)`. Ser control no cambia la distribución de Z del qubit 0, y `RZ` es diagonal en la base de medida, así que no altera las probabilidades.
   - Por tanto, solo `w0` importa, y la segunda capa equivale a una rotación de un qubit.
3. **Capa 1, par (2,3).**
   - La información del qubit 3 solo volvería al qubit 2 mediante `CNOT(3,2)`, que se cancela, y el qubit 2 solo vuelve a influir en el 0 a través de la cancelada `CNOT(2,0)`.
   - Por eso `x3` no tiene ningún efecto, y el par (2,3) no contribuye.

## Consecuencias

- **Para los resultados:** todos los resultados evaluados (fases A–C, legado híbrido y perfiles M5) siguen siendo válidos *para este circuito*, pero describen un modelo con 3 características y 5 parámetros efectivos.
  - Esto puede explicar en parte que la QCNN quede por debajo de la regresión logística, que usa las 4 características. Es una **hipótesis**: no se ha contrastado.
  - En Breast Cancer, x3 es la cuarta componente PCA. En Iris, la cuarta característica en el orden de scikit-learn es la anchura del pétalo.
- **Para el coste:**
  - Con el circuito actual, el parameter-shift necesitaría 15 circuitos por muestra en lugar de 33 (solo las apariciones vivas), y la reconstrucción secuencial 19 en lugar de 40 (con el grado efectivo de cada peso).
  - Para el hardware, el qubit 3 y sus puertas podrían eliminarse, quedando un circuito efectivo de 3 qubits.
  - Ninguna de las dos optimizaciones está implementada.
- **Corrección posible (propuesta, sin implementar):** cambiar el pooling o el final del bloque de convolución para que no repitan la misma CNOT. Por ejemplo, un pooling con una rotación controlada parametrizada, del estilo de la QCNN de la literatura [REF pendiente de verificar], o una convolución que no termine en `CNOT(derecha → izquierda)`.
  - Cambiar el circuito cambia el modelo, así que habría que repetir la fase C: unas 7,5 h de CPU en simulación exacta.
  - **Es una decisión de la tutoría.**

## Limitaciones

- La comprobación es numérica: 40 entradas y pesos aleatorios por codificación, con tolerancia 1e-9.
- El mecanismo descrito se deduce de la secuencia de puertas y es coherente con todos los valores medidos.
