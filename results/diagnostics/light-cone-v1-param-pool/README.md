# Diagnóstico: cono de luz de la arquitectura `v1-param-pool`

**Fecha:** 26 de noviembre de 2026. **Estado: validado numéricamente.** Es una propiedad del circuito, independiente de los datos y del entrenamiento. No se ha entrenado ni evaluado nada.

Comando: `qml_env/Scripts/python.exe -B -m experiments.diagnose_light_cone --architecture v1-param-pool --output-dir results/diagnostics/light-cone-v1-param-pool` (40 entradas y pesos aleatorios por codificación, semilla 2026, revisión base `b5abe03` con cambios sin confirmar). Salidas: `analysis.json` y `analysis.md`.

## Circuito

La convolución y la disposición son las de `v1`. Cada pooling `CNOT(descartado → superviviente)` se sustituye por

```
RY(a) en el superviviente · RY(b) en el descartado · CNOT(descartado, superviviente) · RY(−a) en el superviviente
```

Es la versión mínima, con una sola CNOT, del pooling parametrizado de `qibo.models.QuantumCNN`. Los parámetros `a` y `b` se comparten en cada capa: 4 + 2 = 6 pesos por capa, 12 en total. Hay 25 apariciones parametrizadas y el mismo número de CNOT que `v1` (11).

## Resultado

Con las dos codificaciones (angular y fase):

| Elemento | `v1` (sección 37) | `v1-param-pool` |
|---|---|---|
| Características con influencia | x0, x1, x2 (x3 no) | **x0, x1, x2, x3** |
| Pesos con influencia | 5 de 8 | **12 de 12** |
| Apariciones con contribución al gradiente | 7 de 16 | **25 de 25** |
| Circuitos por muestra, parameter-shift | 33 | 51 |
| Circuitos por muestra, secuencial | 40 | 62 |
| Simulador rápido frente a Qibo | ≤ 7e-16 | ≤ 5,5e-16 |

El grado trigonométrico efectivo de cada peso coincide con su número de apariciones, de modo que la reconstrucción exacta del optimizador secuencial no desperdicia evaluaciones.

## Por qué funciona

En `v1` la última CNOT de la convolución, `CNOT(3,2)`, va seguida directamente de la CNOT de pooling `CNOT(3,2)`, y como `CNOT·CNOT = I` ambas se anulan. Con el pooling parametrizado, entre las dos CNOT hay una rotación `RY(b)` en el qubit de control y una `RY(a)` en el destino: el producto ya no es la identidad (salvo si `a = b = 0`), y el qubit descartado vuelve a influir en el superviviente.

## Limitaciones

- Solo describe qué elementos pueden influir en la salida, no si el modelo clasifica mejor. La comparación con `v1` y con `qibo` requiere el protocolo de la fase C y está pendiente de la tutoría (decisión 6).
- Coste: aproximadamente 1,5 veces más circuitos por paso que `v1`.
