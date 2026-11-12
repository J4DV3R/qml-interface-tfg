# Fase B en Iris binario: replicación de la confirmación con test una vez

**Pre-registro (escrito antes de ejecutar, 12 de noviembre de 2026).** Replica el diseño de `../B-confirm-bc/README.md` sobre el Iris binario del proyecto (`load_binary_iris_splits`: setosa = −1, versicolor = +1).

## Decisiones fijadas antes de ejecutar

- **Hiperparámetros transferidos de Breast Cancer, sin reajuste en Iris:** GD `lr=0.5`, Adam `lr=0.1`, SPSA `a=1.6`, `c=0.1`, `A=6`, y secuencial con 2048 puntos de rejilla. Motivo: validation tiene 15 casos (1 caso = 6,7 puntos), así que un ajuste en Iris apenas discriminaría entre valores y añadiría grados de libertad. Esto mide si la configuración elegida en un dataset se transfiere a otro.
- **Datos:** 4 características originales con min-max ajustado en train; partición 70/15/15 con semilla 42 (70/15/15 muestras; train 35/35, validation 8/7). Las huellas SHA-256 se registran en cada manifiesto.
- **Semillas:** inicialización 19, 23, 29, 31, 37, 41, 43 y 47 (n = 8), las mismas que en Breast Cancer; ejecución 101.
- **Codificaciones:** angular y fase. **Checkpoint:** mínima pérdida de validation, con el paso 0 incluido. Test evaluado una sola vez, más un diagnóstico exacto sin ruido.
- **Ejecución de humo previa** (1 semilla, 5 pasos, `--screening`, sin test, fuera del repositorio): solo comprobó que el runner funciona con Iris y registró las baselines de validation (clase mayoritaria 0,5333; regresión logística 1,0000).

| Bloque | Directorio | Condición | Optimizadores | Presupuesto |
|---|---|---|---|---|
| I1 (principal) | `I1-exact/` | exacto | GD, Adam, SPSA, secuencial | 100 pasos |
| I2 | `I2-shots1000/` | 1000 shots | GD, Adam, SPSA, secuencial | 100 pasos |
| I3 | `I3-shots1000-noise/` | 1000 shots, despolarizante local `p=0.001`, error de lectura 0.02 | GD, Adam, SPSA, secuencial | 100 pasos |
| I4 (mismo coste) | `I4-equal-budget-exact/` | exacto | SPSA, secuencial | 231.000 circuitos de optimización (lo que gastan GD/Adam en 100 pasos: 70 × 33 × 100); máx. 1650 pasos; métricas de train cada 10 pasos |

En I4, GD y Adam se toman de I1.

## Análisis pre-especificado

Es el mismo que en Breast Cancer:

- Por condición: media ± sd e IC95 % bootstrap de accuracy, accuracy balanceada, F1 y MSE de test, junto con las baselines evaluadas en test una vez.
- Comparaciones pareadas frente a Adam: diferencia media, IC bootstrap y Wilcoxon, sin corrección por multiplicidad y con carácter exploratorio.
- Angular frente a fase, descriptivo.
- Coste hasta el checkpoint.

## Expectativas y limitaciones declaradas de antemano

- Setosa y versicolor son **linealmente separables**: la regresión logística ya alcanza 1,0 en validation. Se espera un **efecto techo**, con accuracy de test cercana a 1 en varias condiciones. Por eso las comparaciones entre optimizadores tendrán muy poca capacidad de discriminación. El valor principal del bloque es: (a) comprobar que la línea variacional resuelve un problema fácil, y (b) dar un segundo dataset para la conclusión «sin diferencias afirmables entre optimizadores».
- Test tiene **15 casos**: un caso equivale a 6,7 puntos de accuracy. La MSE de test, continua, será más informativa que la accuracy.
- No es comparable directamente con los perfiles M5 de Iris (QGD, 2 pasos, ruido `legacy_shared_trajectory`, otra semilla de inicialización). Cualquier contraste con ellos será cualitativo.
- n = 8; una partición; simulación (`fast-numpy`); el ruido es de sensibilidad, no una calibración de hardware.

---

# Resultados (añadidos tras la ejecución, 12 de noviembre de 2026)

**Estado: evaluado** en simulación. Cifras derivadas de los JSON mediante `*-analysis.{md,json}`, generados con `experiments.analyze_variational_benchmark --reference adam` (I4 combina `I4-equal-budget-exact/*` con `I1-exact/gd-*` e `I1-exact/adam-*`).

## Ejecución y procedencia

- Las 28 invocaciones (224 entrenamientos) terminaron con `EXIT_CODE=0` (`logs/`). Tiempo de pared por invocación:

| Bloque | Tiempo por invocación |
|---|---|
| I1, I2 e I4 | 15–59 s |
| I3 | 66–71 s con SPSA; 1484–1955 s con los demás |

- Revisión `bc877bd` con árbol sucio y hash del diff `1916248db836…`. El hash difiere del de Breast Cancer (`035098290b95…`) solo por cambios en la documentación. Los hashes del código no versionado (`provenance.untracked_code_sha256`) son idénticos a los de `../B-confirm-bc/`, y no se editó código versionado entre ambas ejecuciones.

## Baselines (test, 15 casos)

| Baseline | Accuracy | Accuracy balanceada | F1 |
|---|---:|---:|---:|
| Clase mayoritaria | 0,4667 | 0,5000 | 0,6364 |
| Regresión logística | **1,0000** | 1,0000 | 1,0000 |

## Test: accuracy y MSE (media ± sd, n = 8; los IC están en `*-analysis.md`)

| Bloque | Cod. | Adam | GD | Secuencial | SPSA |
|---|---|---|---|---|---|
| I1 exacto | angular | 0,883 ± 0,047 / MSE 0,407 | **1,000 ± 0,000** / MSE 0,278 | 0,925 ± 0,043 / 0,371 | 0,883 ± 0,099 / 0,367 |
| I1 exacto | fase | 0,900 ± 0,071 / 0,393 | 0,900 ± 0,036 / 0,361 | 0,958 ± 0,035 / 0,361 | 0,900 ± 0,036 / 0,377 |
| I2 1000 shots | angular | 0,858 ± 0,024 / 0,435 | 0,975 ± 0,071 / 0,289 | 0,942 ± 0,024 / 0,355 | 0,900 ± 0,087 / 0,343 |
| I2 1000 shots | fase | 0,917 ± 0,069 / 0,392 | 0,917 ± 0,059 / 0,354 | 0,942 ± 0,043 / 0,395 | 0,883 ± 0,059 / 0,394 |
| I3 shots + ruido | angular | 0,858 ± 0,075 / 0,411 | 0,975 ± 0,071 / 0,308 | 0,942 ± 0,024 / 0,375 | 0,858 ± 0,090 / 0,377 |
| I3 shots + ruido | fase | 0,908 ± 0,079 / 0,397 | 0,942 ± 0,043 / 0,348 | 0,967 ± 0,036 / 0,394 | 0,933 ± 0,050 / 0,357 |
| I4 mismo coste | angular | (I1) | (I1) | (= I1) | 0,892 ± 0,094 / 0,356 |
| I4 mismo coste | fase | (I1) | (I1) | (= I1) | 0,900 ± 0,036 / 0,371 |

## Comparaciones pareadas frente a Adam (accuracy de test; diferencia [IC95 %], Wilcoxon p)

| Bloque | Cod. | GD − Adam | Secuencial − Adam | SPSA − Adam |
|---|---|---|---|---|
| I1 | angular | +0,117 [0,083, 0,133], p=0,016 | +0,042 [0,000, 0,083], p=0,19 | 0,000 [−0,083, 0,083], p=1,00 |
| I1 | fase | 0,000 [−0,050, 0,050], p=1,00 | +0,058 [0,000, 0,108], p=0,125 | 0,000 [−0,067, 0,058], p=1,00 |
| I2 | angular | +0,117 [0,058, 0,158], p=0,016 | +0,083 [0,067, 0,108], p=0,0078 | +0,042 [−0,008, 0,100], p=0,25 |
| I2 | fase | 0,000 [−0,033, 0,033], p=1,00 | +0,025 [−0,033, 0,083], p=0,63 | −0,033 [−0,083, 0,017], p=0,41 |
| I3 | angular | +0,117 [0,058, 0,167], p=0,031 | +0,083 [0,025, 0,133], p=0,0625 | 0,000 [−0,058, 0,067], p=0,94 |
| I3 | fase | +0,033 [−0,008, 0,075], p=0,31 | +0,058 [−0,008, 0,133], p=0,31 | +0,025 [−0,042, 0,092], p=0,38 |
| I4 | angular | (= I1) | (= I1) | +0,008 [−0,075, 0,083], p=0,81 |
| I4 | fase | (= I1) | (= I1) | 0,000 [−0,067, 0,058], p=1,00 |

## Coste hasta el checkpoint (I1, media de circuitos de optimización)

| Cod. | Adam | GD | Secuencial | SPSA |
|---|---:|---:|---:|---:|
| Angular | 100 774 | 180 758 | 46 900 | 6 388 |
| Fase | 58 616 | 102 506 | 4 900 | 8 610 |

## Interpretación (exploratoria)

1. **No hubo efecto techo, contra lo esperado en el pre-registro.** La QCNN queda entre 0,858 y 1,000 de accuracy de test, muy por encima de la clase mayoritaria (0,467) pero **por debajo de la regresión logística (1,000)** en todas las condiciones salvo GD angular exacto. La pérdida de validation se queda en torno a 0,27–0,51 en un problema linealmente separable: el modelo, con 8 parámetros y lectura de ⟨Z⟩, tiene una capacidad de ajuste limitada.
2. **GD angular supera a Adam angular en I1, I2 e I3** (+11,7 puntos, unos 1,75 casos de 15; p = 0,016–0,031; MSE de test 0,28 frente a 0,41), **aunque ambos llegan a la misma pérdida de validation** (0,3378 frente a 0,3379 en I1). Adam acierta 13/15 en 7 de 8 semillas y GD 15/15 en las 8. Interpretación más probable: los dos optimizadores llegan a mínimos distintos que 15 casos de validation no distinguen y que generalizan de forma diferente en esta partición. No era una hipótesis pre-especificada, hay 20 comparaciones de accuracy sin corregir y en fase no se repite. Es una observación a verificar con más particiones, no una conclusión sobre GD frente a Adam.
3. **El secuencial está igual o por encima de Adam** en todas las condiciones (+0,025 a +0,083). Solo I2 angular tiene p < 0,05. Llega al checkpoint con menos circuitos: unas 12 veces menos en fase (4 900 frente a 58 616) y unas 2,1 veces menos en angular.
4. **SPSA es comparable a Adam** y el presupuesto común (I4) no cambia nada: en Iris SPSA no está limitado por el número de pasos.
5. **Codificación:** a diferencia de Breast Cancer, aquí no hay un patrón consistente; la fase es igual o mejor en 8 de las 12 combinaciones de I1–I3.
6. **Shots y ruido** no cambian las conclusiones dentro de la resolución de 15 casos de test.
7. **Contraste cualitativo con los perfiles M5 de Iris** (QGD, 2 pasos, 0,333–0,467 de accuracy de test): la línea variacional sí aprende el problema. No es una comparación controlada.

## Comparación con Breast Cancer (`../B-confirm-bc/`)

- **Coinciden:** la QCNN supera claramente a la clase mayoritaria; queda a la altura de la regresión logística (Breast Cancer, diferencia de un caso) o por debajo (Iris); el secuencial llega al checkpoint con muchos menos circuitos sin perder accuracy; los shots y el ruido de sensibilidad apenas cambian la accuracy.
- **Difieren:** el orden de los optimizadores (en Breast Cancer Adam/GD/secuencial empatan y SPSA queda peor en fase; en Iris Adam angular es el peor) y el de las codificaciones (angular claramente mejor en Breast Cancer; mixto en Iris).
- **Conclusión prudente:** no hay un optimizador que sea consistentemente mejor en los dos datasets; el secuencial es el más eficiente en circuitos en ambos.

## Desviaciones y limitaciones

- **Expectativa pre-registrada incumplida:** se esperaba un efecto techo y no lo hubo. El análisis se mantuvo tal como estaba pre-especificado.
- **Hiperparámetros transferidos de Breast Cancer:** Adam `lr=0.1` puede no ser adecuado para Iris; no se ajustó, por diseño.
- 15 casos de test (1 caso = 6,7 puntos) y 15 de validation; una partición; n = 8; una semilla de ejecución; sin corrección por multiplicidad.
- En I4 el secuencial vuelve a coincidir con I1 (checkpoint previo al límite del presupuesto).
- Simulación en CPU; ruido de sensibilidad no calibrado; sin QPU. Línea variacional, no QGD.
