# Fase A del benchmark variacional: ajuste de hiperparámetros (sin test)

**Fecha de ejecución:** 12 de noviembre de 2026. **Estado:** ejecutado; es una fase de **ajuste sobre validation**, no una evaluación de rendimiento. **Test no se ha tocado** (`--screening`; cada JSON declara `test: {evaluated: false}`).

**Código:** revisión `bc877bdfa2fbbd9a270e33a059366ff1565fa94a` con árbol de trabajo sucio (correcciones de la auditoría; log técnico, sección 30). Hash SHA-256 del diff sin commit: `de51dbcf89d418e91b063360751e10f90594896c21296757113fef4942513c6c` (registrado en cada `manifest.json`).

## Configuración común

Breast Cancer (PCA 4 ajustado en train), partición 70/15/15 con semilla 42 (398/85/86; las huellas SHA-256 de las particiones están en cada manifiesto). QCNN de 4 qubits con codificaciones angular y fase. Simulación exacta (`fast-numpy`, idéntica a Qibo, sin shots ni ruido). 60 pasos. Semillas de inicialización 101, 102 y 103 (distintas de las de confirmación), semilla de ejecución 101. Checkpoint por mínima pérdida de validation, incluido el paso 0.

Cada subdirectorio corresponde a una invocación:

```text
qml_env\Scripts\python.exe -B -m experiments.run_variational_benchmark --dataset breast-cancer --encodings angle phase --init-seeds 101 102 103 --max-steps 60 --screening <opciones> --output-dir results/variational/A-tuning-bc-exact/<nombre>
```

| Subdirectorio | Opciones |
|---|---|
| `gd-lr0.2`, `gd-lr0.5`, `gd-lr1.0` | `--optimizers gd --gd-lr {0.2,0.5,1.0}` |
| `adam-lr0.02`, `adam-lr0.05`, `adam-lr0.1` | `--optimizers adam --adam-lr {0.02,0.05,0.1}` |
| `spsa-a0.1`, `spsa-a0.2`, `spsa-a0.4` | `--optimizers spsa --spsa-a {0.1,0.2,0.4} --spsa-c 0.1` |
| `sequential` | `--optimizers sequential` |

Los logs (UTF-8, con `EXIT_CODE` y `WALL_SECONDS`) están en `logs/`: las 10 invocaciones terminaron con `EXIT_CODE=0`. El tiempo de pared por invocación (6 entrenamientos) fue de 73–77 s para GD/Adam, 11–12 s para SPSA y 116 s para el secuencial.

## Análisis

`analysis.md`, `analysis.json` y `curves.csv` se generaron con:

```text
qml_env\Scripts\python.exe -B -m experiments.analyze_variational_benchmark <los 10 subdirectorios> --output .../analysis.md --json .../analysis.json --curves-csv .../curves.csv --reference adam
```

## Selección según la regla del protocolo

La regla fijada antes de ejecutar fue elegir, por optimizador, el valor con menor media de la mejor pérdida de validation (media de las dos codificaciones y de las 3 semillas). Valores calculados a partir de `analysis.json`:

| Optimizador | Valor | Media mejor val. loss | Comentario |
|---|---|---:|---|
| GD | `lr=0.5` | 0,4585 | `lr=1.0` da 0,4597: diferencia mínima |
| Adam | `lr=0.1` | 0,4257 | `lr=0.05` da 0,4274: diferencia mínima |
| SPSA | `a=0.4` | 0,5211 | **En el borde de la rejilla** y aún mejorando en el paso 60 |
| Secuencial | — | 0,4235 | Sin hiperparámetros; en fase alcanza el óptimo en 1–2 barridos |

Referencias en validation (85 casos): clase mayoritaria con accuracy 0,6235; regresión logística con accuracy 0,9647.

## Ampliación de la rejilla de SPSA (mismo día, sin test)

Como el óptimo de SPSA quedaba en el borde de la rejilla, se añadieron `spsa-a0.8` y `spsa-a1.6` con la misma configuración (`--optimizers spsa --spsa-a {0.8,1.6} --spsa-c 0.1`). Ambas invocaciones terminaron con `EXIT_CODE=0` (≈12 s cada una). Entre la ejecución inicial y esta ampliación solo cambió documentación (log técnico, sección 31, y este README); por eso el hash del diff registrado en sus manifiestos es distinto (`dcebe122…654a`). El código de `src/` y `experiments/` no cambió. El análisis conjunto de los 12 subdirectorios (72 entrenamientos) está en `analysis-v2.md`, `analysis-v2.json` y `curves-v2.csv`; el análisis inicial se conserva sin cambios.

| SPSA `a` | Mejor val. loss angular | Mejor val. loss fase | Media |
|---:|---:|---:|---:|
| 0.1 | 0,6408 | 0,8242 | 0,7325 |
| 0.2 | 0,5347 | 0,7055 | 0,6201 |
| 0.4 | 0,4637 | 0,5786 | 0,5211 |
| 0.8 | 0,4399 | 0,5069 | 0,4734 |
| 1.6 | 0,4459 | 0,4670 | **0,4565** |

Con la regla del protocolo, la selección pasa a **SPSA `a=1.6`**, que sigue en el borde de la rejilla. En angular el mejor valor es `a=0.8` (1.6 ya no mejora y alcanza su óptimo antes, en el paso 48); en fase sigue mejorando con 1.6 y el óptimo está en el último paso. La interpretación más probable es que SPSA está limitado por el número de pasos con su ganancia decreciente, más que por el valor de `a`. Como un paso de SPSA cuesta unas 16 veces menos circuitos que uno de GD/Adam, la comparación justa en la fase B es con un presupuesto común de circuitos (`--max-optimization-circuits`), no con el mismo número de pasos.

**Selección final de la fase A:** GD `lr=0.5`, Adam `lr=0.1`, SPSA `a=1.6` (`c=0.1`) y secuencial sin hiperparámetros.

## Limitaciones

- n = 3 semillas por condición: las diferencias entre GD 0.5 y 1.0, o entre Adam 0.05 y 0.1, están dentro de la variabilidad y no son concluyentes.
- Varias configuraciones tienen su óptimo en el último paso (Adam 0.02/0.05 angular, GD en fase y SPSA): 60 pasos no bastan para la convergencia y la fase B usará 100.
- El óptimo de SPSA está en el extremo de la rejilla; conviene ampliarla (`a ∈ {0.8, 1.6}`) antes de la confirmación.
- El optimizador secuencial alcanza el óptimo de validation muy pronto en fase y luego empeora, lo que sugiere sobreajuste a train o a la estructura de coordenadas; la selección por validation lo controla.
- Las métricas de validation se usaron para elegir hiperparámetros: no son estimaciones insesgadas del rendimiento. Solo la fase B, con semillas nuevas y test una vez, puede estimarlo.
- Simulación exacta: no incluye shots ni ruido.
