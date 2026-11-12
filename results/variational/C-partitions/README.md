# Fase C: robustez frente a la partición (10 particiones por dataset)

**Pre-registro (escrito antes de ejecutar, 12 de noviembre de 2026).**

## Objetivo

Las fases B (`../B-confirm-bc/`, `../B-confirm-iris/`) usan una sola partición por dataset (semilla 42). Esta fase intenta separar el efecto del optimizador del efecto de la partición, comprobar si las observaciones de las secciones 32 y 33 del log se mantienen y medir el ahorro real de circuitos que se obtiene con un criterio de parada (patience).

## Diseño

- **Datasets:** Breast Cancer (PCA 4 ajustado en train) e Iris binario (setosa/versicolor).
- **Particiones:** semillas de dataset **1 a 10**, nuevas. Partición 70/15/15 estratificada; normalización y PCA ajustados solo en train. Las huellas SHA-256 van en cada manifiesto. La partición 42 (fases A/B) **no** entra en el análisis principal, porque en Breast Cancer sirvió para ajustar hiperparámetros.
- **Semillas de inicialización:** 19, 23, 29, 31, 37, 41, 43 y 47. Semilla de ejecución: 101.
- **Condición:** solo simulación **exacta** (`fast-numpy`), 100 pasos, checkpoint por mínima pérdida de validation (con el paso 0 incluido), test evaluado una vez.
- **Hiperparámetros fijos (los de la fase A, sin reajuste):** GD `lr=0.5`, Adam `lr=0.1`, SPSA `a=1.6`, `c=0.1`, `A=6`, secuencial con 2048 puntos.
- **Codificaciones:** angular y fase. **Optimizadores:** GD, Adam, SPSA y secuencial.
- Total: 2 datasets × 10 particiones × 4 optimizadores × 2 codificaciones = 160 invocaciones de 8 semillas cada una, 1280 entrenamientos.
- Directorios: `<dataset>/ds<semilla>/<optimizador>-<codificación>/`; logs en `logs/`.

## Análisis principal (pre-especificado)

- **Unidad de análisis: la partición** (n = 10 por dataset). Para cada dataset × codificación × optimizador se promedian primero las 8 semillas de inicialización dentro de cada partición. Después se reportan la media, la sd y el IC95 % bootstrap sobre las 10 particiones.
- **Métrica principal:** accuracy de test. Secundarias: accuracy balanceada, F1 y MSE de test.
- **Comparaciones frente a Adam:** diferencia por partición (media pareada sobre las 8 semillas), IC95 % bootstrap sobre particiones y Wilcoxon de rangos con signo sobre las 10 particiones. **Corrección de Holm** dentro de cada dataset sobre las 6 comparaciones (3 optimizadores × 2 codificaciones).
- **Baselines por partición** (clase mayoritaria y regresión logística, evaluadas en test una vez), más la diferencia QCNN − logística por partición.
- **Comprobación de las observaciones previas**, con Wilcoxon sobre particiones y Holm dentro de este bloque de 4 contrastes:
  - O1 (Iris): GD angular > Adam angular en accuracy de test.
  - O2 (Breast Cancer): SPSA en fase < Adam en fase con 100 pasos.
  - O3 (Breast Cancer): angular > fase con Adam.
  - O4 (ambos): el secuencial alcanza el checkpoint con menos circuitos de optimización que Adam (por codificación).
- **Análisis secundario:** el mismo análisis añadiendo la partición 42 (n = 11), solo como sensibilidad.

## Patience (secundario, reconstruido a posteriori)

En modo exacto, la trayectoria de cada optimizador no depende del criterio de parada: la parada solo trunca el bucle, y el generador aleatorio de SPSA consume lo mismo hasta ese punto. El entrenamiento con `patience = P` equivale exactamente a truncar el historial de 100 pasos en el primer paso en que lleva P pasos sin mejorar la validation. Su checkpoint es el mejor hasta ese paso, y sus pesos están guardados en el historial.

- Valores: P ∈ {5, 10, 20}.
- Se reporta: paso de parada; circuitos de optimización consumidos al parar; fracción de entrenamientos en que el checkpoint coincide con el de 100 pasos; diferencia en la pérdida de validation; y accuracy de test del checkpoint con patience. Esta última se evalúa una vez por regla con el simulador exacto a partir de los pesos guardados y nunca se usa para elegir P.
- La equivalencia con una ejecución real con `--patience` se valida con una prueba unitaria antes de analizar.

## Limitaciones declaradas de antemano

- Las particiones comparten muestras: no son conjuntos de test independientes. La variabilidad entre particiones mide la sensibilidad a la partición, no la incertidumbre frente a una población nueva.
- Iris tiene 15 casos de test por partición.
- Solo simulación exacta: sin shots, ruido ni QPU.
- Los hiperparámetros se ajustaron en la partición 42 de Breast Cancer y se transfieren a Iris.

---

## Resultados (ejecutado el 12 de noviembre de 2026)

**Estado: evaluado en simulación exacta.** Todo lo que sigue se deriva de `analysis.md` / `analysis.json` (análisis principal, n = 10 particiones por dataset). La sensibilidad con la partición 42 (n = 11) está en `analysis-with-42.md` / `analysis-with-42.json`.

### Ejecución y procedencia

- 160/160 invocaciones terminaron con `EXIT_CODE=0`: 1280 entrenamientos, 6 procesos en paralelo con `OMP_NUM_THREADS=1`. La última terminó a las 21:53. La suma de los tiempos de pared por invocación es 27 070 s.
- Las 160 tienen `git_revision` `bc877bd`, el mismo hash del diff (`c1ddf4ac9293…`) y el mismo hash del código no versionado que las fases B.
- El hash del diff difiere del de Iris B (`1916248db836…`) solo por ediciones de documentación (`AGENTS.md`, `README.md`, log técnico). Ningún `.py` de `src/`, `experiments/` ni `tests/` se modificó después de las 11:27, antes de Iris B y de esta fase.
- Hash del diff restringido a `src experiments tests`, calculado al terminar: `69180fb6a596…`.
- El script de análisis `experiments/analyze_partitions.py` y su prueba `tests/test_partition_analysis.py` se añadieron al repositorio **después** de la ejecución, para no alterar la procedencia.
  - La prueba comprueba que la patience reconstruida coincide con un entrenamiento real con `patience=P` (GD, Adam, SPSA y secuencial; P ∈ {2, 3, 5}). Pasó junto con `tests.test_variational_training` y `tests.test_fast_qcnn`: 24 pruebas OK.
  - El análisis reproduce las métricas de test almacenadas con error máximo 0 antes de evaluar los checkpoints con patience.

Comandos:

```
qml_env/Scripts/python.exe -B -m experiments.analyze_partitions results/variational/C-partitions --output results/variational/C-partitions/analysis.md --json results/variational/C-partitions/analysis.json
qml_env/Scripts/python.exe -B -m experiments.analyze_partitions results/variational/C-partitions results/variational/B-confirm-bc/B1-exact results/variational/B-confirm-iris/I1-exact --output results/variational/C-partitions/analysis-with-42.md --json results/variational/C-partitions/analysis-with-42.json
```

**Desviación respecto al pre-registro.** El pre-registro habla de un bloque de 4 contrastes, pero O4 se definió por dataset y codificación. El análisis aplica Holm sobre 7 contrastes: O1, O2, O3 y cuatro O4. Las p ajustadas de O1–O3 serían idénticas con 4 contrastes (resumiendo O4 por su p máxima), porque las cuatro O4 tienen la p mínima posible (0,002) y ocupan los primeros puestos del procedimiento.

### Accuracy de test (media ± sd sobre 10 particiones)

| Dataset | Cod. | Adam | GD | Secuencial | SPSA | Logística | Mayoritaria |
|---|---|---|---|---|---|---|---|
| Breast Cancer | angular | 0,911 ± 0,022 | 0,910 ± 0,027 | 0,909 ± 0,023 | 0,904 ± 0,023 | 0,937 ± 0,018 | 0,628 |
| Breast Cancer | fase | 0,874 ± 0,049 | 0,859 ± 0,040 | 0,874 ± 0,046 | 0,853 ± 0,034 | 0,937 ± 0,018 | 0,628 |
| Iris | angular | 0,903 ± 0,075 | 0,905 ± 0,105 | 0,928 ± 0,060 | 0,881 ± 0,092 | 1,000 ± 0,000 | 0,500 ± 0,035 |
| Iris | fase | 0,855 ± 0,134 | 0,817 ± 0,145 | 0,844 ± 0,130 | 0,826 ± 0,137 | 1,000 ± 0,000 | 0,500 ± 0,035 |

### Comparaciones frente a Adam (accuracy de test, diferencia por partición)

- **Ninguna comparación es significativa tras Holm en ninguno de los dos datasets** (6 por dataset).
- En Breast Cancer, SPSA queda por debajo de Adam sin corregir: −0,008 en angular, con IC [−0,013, −0,003] y p = 0,016, y −0,021 en fase, con IC [−0,036, −0,007] y p = 0,020. Tras Holm, p = 0,094 y 0,098. GD en fase: −0,015, con IC [−0,025, −0,005] y Holm p = 0,16.
- En Iris todas las diferencias tienen un IC que cruza 0 (Holm p ≥ 0,42). La dispersión entre particiones (sd 0,06–0,14) es mayor que cualquier diferencia entre optimizadores.

### Observaciones de las fases B (Holm sobre el bloque)

| Id | Contraste | Diferencia [IC95 %] | Particiones +/− | p | Holm p | Resultado |
|---|---|---|---|---|---|---|
| O1 | Iris: GD angular − Adam angular | +0,003 [−0,038, +0,040] | 4/3 | 0,84 | 0,84 | **No se replica** (en la fase B fue +0,117) |
| O2 | Breast Cancer: SPSA fase − Adam fase | −0,021 [−0,036, −0,007] | 2/8 | 0,020 | 0,059 | Misma dirección; no significativa tras Holm |
| O3 | Breast Cancer: Adam angular − Adam fase | +0,037 [+0,013, +0,062] | 9/1 | 0,037 | 0,074 | Misma dirección; no significativa tras Holm |
| O4 | Secuencial − Adam, circuitos hasta el checkpoint | Breast Cancer angular −459 819; fase −627 716; Iris angular −85 488; fase −52 754 | 0/10 en los cuatro | 0,002 | 0,014 | **Se replica** en los cuatro casos |

- En O4, la razón de medias (Adam / secuencial) es 3,7 (Breast Cancer angular), 9,4 (Breast Cancer fase), 3,2 (Iris angular) y 5,3 (Iris fase).
- En la sensibilidad con la partición 42 (n = 11), O2 y O3 tienen Holm p = 0,029 y 0,049, y SPSA frente a Adam en Breast Cancer, Holm p = 0,047 (angular) y 0,049 (fase). Esa partición sirvió para ajustar los hiperparámetros, por lo que este resultado no sustituye al análisis principal.

### QCNN frente a regresión logística

- **La QCNN queda por debajo de la regresión logística en ambos datasets y en casi todas las particiones.**
  - Breast Cancer: −0,026 (Adam angular) a −0,084 (SPSA fase); entre 0 y 1 de 10 particiones alcanzan a la logística; todos los IC excluyen 0.
  - Iris: −0,073 (secuencial angular) a −0,183 (GD fase); la logística acierta los 15 casos de test en las 10 particiones.
- En la fase B, Adam angular en Breast Cancer (0,919) superaba a la logística (0,907). La partición 42 resultó favorable a la QCNN; en las 10 particiones nuevas, Adam angular no alcanza a la logística en ninguna.
- La QCNN supera con claridad a la clase mayoritaria en todos los casos.

### Codificación (descriptivo, salvo O3)

La codificación angular tiene mayor accuracy media que la de fase con los cuatro optimizadores en ambos datasets: en Breast Cancer +0,035 a +0,051 y en Iris +0,048 a +0,088. Solo O3 se contrastó formalmente. En Iris esto contrasta con la fase B, donde la fase era igual o mejor en 8 de 12 combinaciones.

### Patience (secundario, reconstruido a posteriori)

- **Secuencial: la parada temprana no cambia el resultado.**
  - Con P = 5 para entre los pasos 8 y 13 y consume el 8–13 % de los circuitos de 100 pasos.
  - Con P ≥ 10, la diferencia en accuracy de test frente al checkpoint de 100 pasos está entre −0,006 y 0. Con P = 5 es 0 salvo en Iris fase (−0,018).
- **Adam: para antes de 100 pasos en el 96–100 % de los entrenamientos, pero pierde accuracy si P es pequeña.**
  - P = 5: −0,035 en Iris angular y −0,045 en Iris fase.
  - P = 20: entre −0,013 y +0,002, consumiendo el 42–57 % de los circuitos.
- **SPSA** con P = 5 pierde hasta −0,051 (Iris fase); con P = 20 la pérdida es ≤ 0,003.
- **GD** para antes solo en el 10–79 % de los entrenamientos.
- **Coste realista con la misma regla para todos (P = 10), circuitos consumidos al parar, Adam frente a secuencial:**
  - Breast Cancer angular: 350 185 frente a 301 286 (1,2×).
  - Breast Cancer fase: 483 824 frente a 221 288 (2,2×).
  - Iris angular: 78 165 frente a 42 315 (1,8×).
  - Iris fase: 63 438 frente a 38 395 (1,7×).
  - En esas condiciones el secuencial no pierde accuracy y Adam pierde hasta 0,032 (Iris angular).
  - La ventaja con parada realista es menor que la de O4, porque O4 cuenta circuitos hasta el mejor checkpoint (velocidad de convergencia) y no el coste de detectar la parada.
- Los valores de P no se eligieron con test; la accuracy de test de cada regla se evaluó una sola vez.

### Interpretación

1. **El orden de los optimizadores depende de la partición.** Las diferencias de accuracy de las fases B no se mantienen como efectos significativos con 10 particiones. GD > Adam en Iris (O1) era propio de la partición 42.
2. **El hallazgo robusto es de coste, no de accuracy.** El secuencial alcanza el mismo nivel que Adam con 3,2–9,4 veces menos circuitos hasta el checkpoint (en las 40 comparaciones partición × caso), Con P = 10 mantiene la accuracy (diferencia ≥ −0,006) usando el 14–19 % de los circuitos de sus 100 pasos.
3. **La QCNN de 4 qubits no alcanza a una regresión logística** sobre las mismas 4 características. La conclusión de las fases B («a la altura de la logística») era optimista por la partición.
4. La codificación angular es, en media, mejor que la de fase en ambos datasets. Formalmente solo está contrastada para Adam en Breast Cancer (O3, Holm p = 0,074).

**Limitaciones.**
- Las particiones comparten muestras.
- Iris tiene 15 casos de test por partición.
- Los hiperparámetros se ajustaron en la partición 42 de Breast Cancer.
- Solo simulación exacta: sin shots, ruido ni QPU.
- Wilcoxon con n = 10 tiene una p mínima de 0,002.
