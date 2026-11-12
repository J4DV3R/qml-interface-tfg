# Fase B del benchmark variacional: confirmación con test una vez

**Pre-registro (escrito antes de ejecutar, 12 de noviembre de 2026).** Hiperparámetros fijados en la fase A (`../A-tuning-bc-exact/README.md`), sin volver a ajustarlos: GD `lr=0.5`, Adam `lr=0.1`, SPSA `a=1.6`, `c=0.1`, `A=6` (el mismo valor efectivo que en la fase A) y secuencial sin hiperparámetros.

## Diseño

- **Datos:** Breast Cancer, PCA 4 ajustado en train, partición 70/15/15 con semilla 42 (398/85/86).
- **Semillas de inicialización (nuevas, no usadas en la fase A):** 19, 23, 29, 31, 37, 41, 43 y 47 (n = 8). Semilla de ejecución 101.
- **Codificaciones:** angular y fase.
- **Checkpoint:** mínima pérdida de validation, con el paso 0 como candidato. Test evaluado **una sola vez** por entrenamiento tras seleccionar el checkpoint, más un diagnóstico exacto sin ruido del mismo checkpoint que nunca se usa para seleccionar.

| Bloque | Directorio | Condición | Optimizadores | Presupuesto |
|---|---|---|---|---|
| B1 (principal) | `B1-exact/` | exacto | GD, Adam, SPSA, secuencial | 100 pasos |
| B2 | `B2-shots1000/` | 1000 shots | GD, Adam, SPSA, secuencial | 100 pasos |
| B3 | `B3-shots1000-noise/` | 1000 shots, despolarizante local `p=0.001`, error de lectura 0.02 | GD, Adam, SPSA, secuencial | 100 pasos |
| B4 (mismo coste) | `B4-equal-budget-exact/` | exacto | SPSA, secuencial | 1.313.400 circuitos de optimización (lo que gastan GD/Adam en 100 pasos: 398 × 33 × 100); máx. 1650 pasos; métricas de train cada 10 pasos |

En B4, GD y Adam se toman de B1, porque con 100 pasos consumen exactamente ese presupuesto.

## Análisis pre-especificado

- Por condición: media ± sd e IC95 % bootstrap sobre las 8 semillas de accuracy, accuracy balanceada, F1 y MSE de test, junto con las baselines de clase mayoritaria y de regresión logística (evaluadas en test una vez).
- Comparaciones pareadas frente a Adam con las mismas semillas: diferencia media, IC bootstrap y Wilcoxon (n ≥ 6). Un IC que cruce 0 no permite afirmar una diferencia. Las cuatro comparaciones por bloque no se corrigen por multiplicidad y se interpretan como exploratorias.
- Comparación angular frente a fase, descriptiva.
- Coste: circuitos de optimización hasta el checkpoint.
- No se volverán a ajustar hiperparámetros a la vista del test.

## Limitaciones conocidas de antemano

n = 8 sigue siendo exploratorio. Test tiene 86 casos, así que un caso equivale a 1,16 puntos de accuracy. El ruido es un modelo de sensibilidad en simulación, no una calibración de hardware. La simulación usa `fast-numpy` (idéntico a Qibo según las pruebas).

---

# Resultados (añadidos tras la ejecución, 12 de noviembre de 2026)

**Estado: evaluado** en simulación, con las limitaciones indicadas. Todo lo que sigue se deriva de los JSON de cada subdirectorio a través de los análisis generados; ninguna cifra se ha escrito a mano sin su fuente.

## Ejecución y procedencia

- Las 28 invocaciones (`logs/<bloque>--<optimizador>-<codificación>.log`) terminaron con `EXIT_CODE=0`: 224 entrenamientos en total. Tiempo de pared por invocación (8 entrenamientos), con 6 procesos en paralelo y `OMP_NUM_THREADS=1`:

| Bloque | Tiempo por invocación |
|---|---|
| B1 y B2 | 27–283 s |
| B3 (ruido, matriz de densidad) | 841–919 s con SPSA; 11 746–14 650 s con los demás |
| B4 | 138–165 s |

- Revisión `bc877bd` con árbol sucio. Hash del diff `035098290b95…` en todos los manifiestos de la fase B.
- **Desviación de procedencia respecto a la fase A.** Entre la ampliación de SPSA y la fase B cambiaron dos archivos de código no versionados:
  - `src/fast_qcnn.py` (`e1db0dd4…` → `69217cf5…`): la aplicación de puertas de un qubit pasó de `einsum` a combinaciones 2×2 explícitas, solo por rendimiento.
  - `experiments/run_variational_benchmark.py` (`4720a4a1…` → `10bcae12…`): nueva opción `--spsa-stability`, necesaria para fijar `A=6` con 100 pasos.

  Con el código de la fase B, `tests.test_fast_qcnn` y `tests.test_variational_training` (21 pruebas, incluida la equivalencia con Qibo con error < 1e-10) pasaron el 12 de noviembre de 2026. Los hashes por archivo están en cada `manifest.json` (`provenance.untracked_code_sha256`).
- Análisis generados con `experiments.analyze_variational_benchmark --reference adam`:

| Bloque | Entrada | Salida |
|---|---|---|
| B1 | `B1-exact/*` | `B1-exact-analysis.{md,json}`, `B1-exact-curves.csv` |
| B2 | `B2-shots1000/*` | `B2-shots1000-analysis.*`, `B2-shots1000-curves.csv` |
| B3 | `B3-shots1000-noise/*` | `B3-shots1000-noise-analysis.*`, `B3-shots1000-noise-curves.csv` |
| B4 | `B4-equal-budget-exact/*` más `B1-exact/gd-*` y `B1-exact/adam-*` | `B4-equal-budget-analysis.*`, `B4-equal-budget-curves.csv` |

## Baselines (test, 86 casos, evaluadas una vez)

| Baseline | Accuracy | Accuracy balanceada | F1 |
|---|---:|---:|---:|
| Clase mayoritaria | 0,6279 | 0,5000 | 0,7714 |
| Regresión logística (PCA 4) | 0,9070 | 0,8750 | 0,9310 |

## Accuracy de test (media ± sd [IC95 % bootstrap], n = 8)

| Bloque | Cod. | Adam | GD | Secuencial | SPSA |
|---|---|---|---|---|---|
| B1 exacto | angular | 0,9186 ± 0,0000 | 0,9157 ± 0,0054 | 0,9070 ± 0,0124 | 0,9099 ± 0,0161 |
| B1 exacto | fase | 0,8852 ± 0,0041 | 0,8823 ± 0,0041 | 0,8866 ± 0,0054 | 0,8459 ± 0,0184 |
| B2 1000 shots | angular | 0,9099 ± 0,0082 | 0,9026 ± 0,0107 | 0,9055 ± 0,0115 | 0,9026 ± 0,0060 |
| B2 1000 shots | fase | 0,8852 ± 0,0075 | 0,8895 ± 0,0088 | 0,8881 ± 0,0060 | 0,8576 ± 0,0309 |
| B3 shots + ruido | angular | 0,9099 ± 0,0103 | 0,9070 ± 0,0088 | 0,9012 ± 0,0139 | 0,8953 ± 0,0124 |
| B3 shots + ruido | fase | 0,8837 ± 0,0062 | 0,8779 ± 0,0224 | 0,8881 ± 0,0060 | 0,8648 ± 0,0284 |
| B4 mismo coste | angular | (B1) | (B1) | 0,9070 ± 0,0124 | 0,9157 ± 0,0082 |
| B4 mismo coste | fase | (B1) | (B1) | 0,8866 ± 0,0054 | 0,8692 ± 0,0269 |

Los IC, la accuracy balanceada, el F1, el MSE, el diagnóstico exacto sin ruido del mismo checkpoint y el coste están en cada `*-analysis.md`.

## Comparaciones pareadas frente a Adam (accuracy de test; diferencia media [IC95 %], Wilcoxon p)

| Bloque | Cod. | GD − Adam | Secuencial − Adam | SPSA − Adam |
|---|---|---|---|---|
| B1 | angular | −0,0029 [−0,0073, 0,0000], p=0,50 | −0,0116 [−0,0203, −0,0029], p=0,125 | −0,0087 [−0,0203, −0,0015], p=0,25 |
| B1 | fase | −0,0029 [−0,0073, 0,0000], p=0,50 | +0,0015 [−0,0029, 0,0058], p=1,00 | **−0,0392 [−0,0494, −0,0276], p=0,0078** |
| B2 | angular | −0,0073 [−0,0145, −0,0015], p=0,25 | −0,0044 [−0,0145, 0,0073], p=0,63 | −0,0073 [−0,0145, −0,0000], p=0,125 |
| B2 | fase | +0,0044 [−0,0029, 0,0116], p=0,50 | +0,0029 [−0,0044, 0,0087], p=0,69 | −0,0276 [−0,0451, −0,0101], p=0,0625 |
| B3 | angular | −0,0029 [−0,0131, 0,0058], p=0,59 | −0,0087 [−0,0145, −0,0029], p=0,0625 | −0,0145 [−0,0276, −0,0015], p=0,0625 |
| B3 | fase | −0,0058 [−0,0233, 0,0073], p=1,00 | +0,0044 [0,0015, 0,0087], p=0,25 | −0,0189 [−0,0407, 0,0000], p=0,20 |
| B4 | angular | (= B1) | (= B1) | −0,0029 [−0,0087, 0,0000], p=1,00 |
| B4 | fase | (= B1) | (= B1) | −0,0160 [−0,0363, −0,0015], p=0,25 |

Con n = 8, el menor p posible de Wilcoxon es 0,0078 (las 8 semillas con el mismo signo). Hay 26 comparaciones de accuracy de test sin corrección por multiplicidad.

## Coste hasta el checkpoint (circuitos de optimización, B1 exacto, media)

| Cod. | Adam | GD | Secuencial | SPSA |
|---|---:|---:|---:|---:|
| Angular | 692 818 | 372 677 | 189 050 | 72 336 |
| Fase | 691 177 | 939 081 | 29 850 | 76 118 |

Coste por paso, derivado de `optimization_circuits` en los JSON:

| Optimizador | Circuitos por paso | Cálculo |
|---|---:|---|
| GD y Adam | 13 134 | 398 × 33 |
| SPSA | 796 | 398 × 2 |
| Secuencial | 15 920 | 398 × 40 |

## Interpretación (exploratoria)

1. **La QCNN aprende y supera con claridad a la clase mayoritaria.** En todas las condiciones la accuracy de test está entre 0,846 y 0,919, frente a 0,628. Las mejores condiciones (angular con Adam/GD) quedan **a la altura de la regresión logística** (0,9186 frente a 0,9070: un caso de 86 de diferencia, sin significado estadístico). No se puede afirmar que la QCNN supere a la baseline clásica.
2. **Entre GD, Adam y secuencial no hay diferencias de accuracy de test que se puedan afirmar.** Las diferencias medias son ≤ 1 caso de test. Algunos IC bootstrap excluyen 0 por muy poco, en ambos sentidos (por ejemplo, el secuencial en fase B3, +0,0044), pero ningún Wilcoxon baja de 0,0625 y no hay un patrón consistente entre bloques. El secuencial angular en B1 queda ~1 caso por debajo de Adam (IC [−0,0203, −0,0029], p=0,125), lo que no se considera una diferencia establecida.
3. **SPSA con 100 pasos rinde peor en fase**: −3,9 puntos en B1, con las 8 semillas por debajo de Adam. **Con presupuesto común de circuitos (B4) la diferencia se reduce** a −1,6 puntos (p=0,25) en fase y a −0,3 puntos en angular. La desventaja de SPSA a 100 pasos refleja sobre todo su menor número de evaluaciones, no un techo de calidad. En fase sigue siendo el optimizador más variable (sd 0,027).
4. **Eficiencia: el optimizador secuencial alcanza el checkpoint con muchos menos circuitos**, sin perder accuracy de test: en fase, 29 850 frente a 691 177 de Adam, unas 23 veces menos; en angular, unas 3,7 veces menos. Esta cifra es *a posteriori*: en la práctica hace falta un criterio de parada (patience) para cobrar ese ahorro. Por paso, el secuencial es más caro que GD/Adam.
5. **Codificación:** la angular supera a la de fase en las 13 combinaciones de bloque y optimizador, aunque su pérdida de validation es mayor. La diferencia va de 1,3 a 6,4 puntos de accuracy de test (1–5 casos). Es una observación descriptiva, sin test formal pre-especificado.
6. **Shots y ruido:** la accuracy de test apenas cambia (Adam angular: 0,9186 exacto, 0,9099 con 1000 shots, 0,9099 con ruido). La mejor pérdida de validation baja con shots (Adam angular: 0,4287 → 0,4215) porque es el mínimo de estimaciones ruidosas y tiene un sesgo optimista. Sube con ruido (0,4420) porque el error de lectura contrae los valores esperados en un factor 0,96 y el depolarizante los acerca a 0. Por eso las pérdidas de validation de B2/B3 no se comparan con B1. A este nivel de ruido simulado el entrenamiento es robusto, pero esto no dice nada sobre hardware real.

## Desviaciones y limitaciones observadas

- **El pre-registro suponía que el secuencial era más barato por paso que GD/Adam; no lo es** (15 920 frente a 13 134 circuitos). En B4 el secuencial se detuvo por presupuesto en el paso 83, antes que en B1. Su checkpoint coincide con el de B1 porque el óptimo se alcanzó antes (pasos 2–30), así que sus métricas en B4 son idénticas a las de B1.
- El control de presupuesto se aplica **después** de cada paso. El secuencial lo superó en un paso parcial: 1 321 360 frente a 1 313 400, un 0,6 % más. SPSA lo agotó exactamente (1650 × 796).
- SPSA en B4 mantuvo `A=6` (pre-registrado) aunque ejecutó 1650 pasos. El valor convencional, 0,1 × pasos, sería 165. Es un posible ajuste no explorado.
- Una sola partición y un solo dataset; 86 casos de test; n = 8 semillas de inicialización con una única semilla de ejecución. Sin corrección por multiplicidad.
- Simulación en CPU; el ruido es un modelo local depolarizante más un error de lectura simétrico, no una calibración de dispositivo. No hay ejecución en QPU.
- Estos resultados pertenecen a la línea variacional (híbrida). No son resultados de QGD ni sustituyen a los históricos de `results/initial-hybrid-adam/`.
