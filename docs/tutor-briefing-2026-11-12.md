# Resumen para la tutoría: estado del TFG y decisiones pendientes

**Fecha:** 12 de noviembre de 2026; actualizado el 20 de noviembre con el plan de ejecución en el BSC (sección 8) y **el 24 de noviembre** con el borrador de la solicitud RES y un hallazgo sobre la arquitectura del circuito (sección 9). **Base:** commits `dec0f0b`–`cbbf2a0` sobre `bc877bd` y el diagnóstico del 24 de noviembre (secciones 30–37 de `docs/TFG-technical-log.md`). Guía de fundamentos para preparar la reunión: `docs/reports/guia-fundamentos-tutoria.pdf`. Versión PDF con figuras: `docs/reports/tutor-briefing-2026-11-12.pdf`; informe detallado de resultados: `docs/reports/variational-results.pdf`.

Estados usados: **Propuesto** (solo en plan), **Implementado** (código sin evaluar), **Validado** (prueba concreta superada) y **Evaluado** (experimento con resultados persistidos y analizados).

## 0. Qué ha cambiado desde la última reunión

1. **Auditoría independiente:** los resultados QGD previos no entrenaban (pesos inmóviles con `b=5`, `lr=0.1`), las pruebas de QGD eran vacuas y el ruido estaba mal simulado. Corregido y validado (sección 3).
2. **Nueva línea variacional:** simulador rápido idéntico a Qibo y 4 optimizadores (GD, Adam, SPSA y secuencial). Fases A, B y C evaluadas en simulación; la fase C (10 particiones, 1280 entrenamientos, pre-registrada) es el resultado principal (secciones 4 y 7).
3. **Preparación para el BSC sin acceso:** transpilador, ejecución por lotes, simulador que imita el chip, mitigación de lectura y borrador de la solicitud RES (sección 8).
4. **Hallazgo del 24 de noviembre:** la salida del circuito actual ignora la cuarta característica y 3 de los 8 pesos, por dos CNOT que se anulan entre convolución y pooling (sección 9).
5. **Documentación e informes:** log técnico (secciones 30–37), informe de resultados con figuras (`docs/reports/variational-results.pdf`) y suite completa de 138 pruebas OK.

## 1. Qué se necesita decidir

1. Qué línea de entrenamiento es la **principal** en la memoria: la variacional (circuito + optimizador clásico) o la QGD (actualización de pesos dentro del circuito).
2. Si se acepta la **actualización modular** de QGD como sustituta del clipping, y con qué rejilla y qué `b/lr`.
3. Qué significa «no híbrido» para el TFG: sin PyTorch, sin Adam, o actualización ejecutada como circuito.
4. Qué resultados serán **oficiales** en la memoria.
5. **Ejecución en hardware en el BSC** (sección 8):
   - si entra en el TFG y con qué alcance (solo inferencia o también entrenamiento corto);
   - quién figura como responsable de la solicitud a la RES;
   - si hay un proyecto financiado que permita pedir acceso por la vía rápida (*fast-track*).
6. **Arquitectura del circuito** (sección 9): ¿se corrige la cancelación de CNOT y se repite la fase C (unas 7,5 h de CPU), o se mantiene el circuito actual y se documenta como un modelo de 3 características efectivas? Conviene decidirlo antes de enviar la solicitud RES y de escribir la memoria.

## 2. Estado en una tabla

| Componente | Estado | Evidencia |
|---|---|---|
| QCNN de 4 qubits (codificación angular/fase, convolución, pooling lógico) | Validado; **la salida solo usa 3 características y 5 pesos** (sección 9) | Pruebas unitarias: suite completa, 138/138 OK el 20 de noviembre; `results/diagnostics/light-cone/` |
| Legado híbrido PyTorch + Adam | Evaluado (histórico, n = 3) | `results/initial-hybrid-adam/` |
| QGD M1–M5 (actualización modular reversible) | Implementado y validado unitariamente; **no evaluado** | Log, secciones 28–30; auditoría |
| Ruido local corregido (matriz de densidad, shots independientes) | Validado | Log, sección 30 |
| Línea variacional (GD, Adam, SPSA, secuencial tipo Rotosolve/NFT) | **Evaluado en simulación** | `results/variational/` (log, secciones 31–34) |
| Preparación para hardware (transpilación, ejecución por lotes, mitigación de lectura, simulador que imita el chip) | Implementado y validado en simulación | `docs/bsc/README.md`; log, secciones 35–36; incluida en la suite completa |
| Ejecución en hardware real (BSC, MareNostrum5-Ona) | Propuesto | Sin acceso ni ejecuciones; el adaptador de Qililab es un esqueleto |

## 3. Problemas encontrados y corregidos (auditoría independiente)

- **Toda ejecución QGD previa usaba una configuración que no mueve los pesos.** Con `b=5`, `lr=0.1`, el desplazamiento entero es 0 para los 32 códigos de gradiente: los pesos solo se cuantizan. Los perfiles M5 y los screens S0/G0/U0 no evalúan, por tanto, QGD como método de entrenamiento. Ahora el código detecta esta situación, avisa y la registra en el JSON.
- **Las pruebas de la aritmética QGD eran vacuas:** usaban configuraciones con desplazamiento 0. Se añadieron pruebas exhaustivas con configuraciones que sí mueven pesos (circuito frente a la referencia clásica), y pasan.
- **Ruido mal modelado en Qibo 0.1.11.**
  - El canal depolarizante compartía una única trayectoria de ruido entre todos los shots, así que no promediaba realizaciones.
  - En matriz de densidad, ese mismo canal despolariza todo el registro, no solo el qubit objetivo.
  - En ejecución repetida, todos los shots reutilizan la muestra del primero.
  - Se sustituyó por un canal local exacto. Los JSON antiguos con ruido no son estimaciones promediadas.
- **Métricas M5 sin referencia trivial:** en Iris, la accuracy de test (0,333–0,467) está **por debajo** de la clase mayoritaria (0,533).

## 4. Resultados de la línea variacional (evaluado en simulación)

**Protocolo:**
- Ajuste de hiperparámetros solo con validation (fase A).
- Confirmación con 8 semillas de inicialización nuevas y test evaluado una sola vez (fase B), bajo cuatro condiciones: exacta, 1000 shots, 1000 shots con ruido local (`p=0.001`) y error de lectura del 2 %, y presupuesto común de circuitos.
- Baselines de clase mayoritaria y regresión logística.
- Comparaciones pareadas con IC bootstrap y Wilcoxon.

| Dataset (test) | QCNN, rango de accuracy | Clase mayoritaria | Regresión logística |
|---|---|---|---|
| Breast Cancer, PCA 4 (86 casos) | 0,846–0,919 | 0,628 | 0,907 |
| Iris setosa/versicolor (15 casos) | 0,858–1,000 | 0,467 | 1,000 |

La tabla corresponde a la partición 42 (fases B). La fase C (sección 7), con 10 particiones nuevas, matiza estas lecturas.

- La QCNN **aprende** y supera claramente a la clase mayoritaria. En la partición 42 quedaba a la altura de la regresión logística, pero **con 10 particiones queda por debajo de ella en ambos datasets** (sección 7).
- **No hay un optimizador consistentemente mejor.** En Breast Cancer, GD, Adam y el secuencial empatan. En Iris con codificación angular, GD superó a Adam en la partición 42, pero **no se replica** con 10 particiones (sección 7).
- **El optimizador secuencial** (reconstrucción exacta del coste por coordenada) alcanza el mejor checkpoint con muchos menos circuitos: 3,7–23 veces menos que Adam en Breast Cancer y 2–12 veces menos en Iris, sin perder accuracy. Es el único hallazgo que se mantiene en la fase C. Es relevante para el hardware, donde el coste se mide en ejecuciones.
- **1000 shots y el ruido de sensibilidad** apenas cambian la accuracy de test (solo se midió en la partición 42).

Limitaciones: simulación en CPU, ruido no calibrado, pocos casos de test en Iris, corrección por multiplicidad solo en la fase C, y sin QPU.

## 5. Situación de QGD respecto al objetivo de hardware real

Lo implementado (estado: implementado y validado unitariamente):

- El host calcula el gradiente por parameter-shift a partir de las medidas.
- El gradiente se codifica en un registro de `b` qubits.
- Un circuito aritmético reversible suma el desplazamiento al registro del peso (módulo 2^b).
- El resultado se mide y se decodifica.

Limitaciones del diseño, no solo de su evaluación:

1. **La entrada y la salida del circuito de actualización son clásicas.** El gradiente llega ya calculado en el host y el peso se mide y vuelve al host. El circuito ejecuta una suma modular que el host podría hacer directamente, de modo que no elimina el bucle clásico ni la actualización clásica. Solo la traslada a un circuito con coste y ruido añadidos.
2. **Coste en hardware.** Las puertas son multicontroladas de hasta 2b qubits (`max_qubits_per_arithmetic_gate`) y en un dispositivo real habría que descomponerlas. No hay transpilación, routing ni validación de conectividad.
3. **Regla de actualización.** La rejilla tiene los extremos ±π duplicados y no representa el cero. La regla modular no equivale al clipping del plan original y su aceptación está pendiente.
4. **No hay ninguna ejecución QGD con pesos que se muevan.** No se puede afirmar nada cuantitativo sobre su rendimiento.

Llevar la actualización de pesos al circuito sin ningún paso clásico, por ejemplo estimando el gradiente coherentemente, exigiría algoritmos que en general necesitan hardware tolerante a fallos. Esta afirmación todavía **no tiene referencias verificadas** en `memory/bibliography.bib`, así que debe tratarse como pendiente de bibliografía.

## 6. Opciones

| Opción | Qué implica | Coste | Riesgo |
|---|---|---|---|
| **A. Variacional como línea principal; QGD como prototipo documentado** | Memoria con resultados evaluados (fases B y C). QGD se presenta como exploración de la actualización en circuito, con sus límites y sin cifras de rendimiento | Bajo: los experimentos ya están hechos | Se aparta de la intención inicial de «actualización en circuito» |
| **B. QGD como línea principal** | Fijar `b/lr` que muevan pesos, corregir la rejilla, decidir la política de decodificación y evaluar QGD con el mismo protocolo (pareado frente a la variacional) | Medio-alto: implementación y experimentos | Es probable que no supere a la línea variacional; la limitación 1 de la sección 5 persiste |
| **C. Ambas, con comparación controlada** | A + una evaluación de QGD con una configuración que mueva pesos, con el mismo protocolo | Medio | Alcance mayor; requiere decidir antes la regla QGD |

**Recomendación técnica:** opción **A**, o bien **C** si el tutor considera imprescindible una cifra de QGD. En ambos casos, el paso de mayor valor hacia el «hardware real» sería una ejecución pequeña de la línea variacional en un QPU, por ejemplo la inferencia de un checkpoint entrenado. La sección 8 concreta este paso en el BSC.

## 7. Fase C: robustez frente a la partición

**Estado: evaluado en simulación exacta.**

**Diseño:**
- 10 particiones nuevas por dataset (semillas 1–10) × 4 optimizadores × 2 codificaciones × 8 semillas de inicialización: 1280 entrenamientos.
- Los hiperparámetros son los de la fase A, sin reajuste.
- La unidad de análisis es la partición, con corrección de Holm.
- El diseño se pre-registró antes de ejecutar.
- Fuente: `results/variational/C-partitions/README.md` y `analysis.md`.

| Accuracy de test (media ± sd, 10 particiones) | Mejor QCNN | Peor QCNN | Regresión logística | Clase mayoritaria |
|---|---|---|---|---|
| Breast Cancer | 0,911 ± 0,022 (Adam, angular) | 0,853 ± 0,034 (SPSA, fase) | 0,937 ± 0,018 | 0,628 |
| Iris | 0,928 ± 0,060 (secuencial, angular) | 0,817 ± 0,145 (GD, fase) | 1,000 ± 0,000 | 0,500 |

Qué se mantiene y qué no:

1. **Optimizadores: ninguna diferencia de accuracy frente a Adam es significativa tras Holm.** En Iris, la variación entre particiones (sd 0,06–0,14) supera cualquier diferencia entre optimizadores.
2. **GD > Adam en Iris: no se replica** (+0,003, IC [−0,038, 0,040]). Era propia de la partición 42.
3. **SPSA < Adam y angular > fase en Breast Cancer:** misma dirección que en la fase B, pero no significativas tras Holm (p = 0,059 y 0,074).
4. **Secuencial más barato en circuitos: se replica en las 40 combinaciones** de partición × dataset × codificación (Holm p = 0,014). Necesita entre 3,2 y 9,4 veces menos circuitos hasta el checkpoint.
5. **QCNN frente a regresión logística:** queda por debajo en casi todas las particiones.
   - En Breast Cancer, de −0,026 a −0,084.
   - En Iris, de −0,07 a −0,18.
   - La partición 42 era favorable a la QCNN.
6. **Parada temprana (patience), reconstruida del historial:**
   - Con una regla P = 10 para todos, el secuencial para con el 14–19 % de los circuitos de 100 pasos y no pierde accuracy.
   - Adam con P = 10 pierde hasta 0,032 en Iris.
   - Con esa regla, el secuencial consume entre 1,2 y 2,2 veces menos circuitos que Adam hasta parar.

**Lectura para la decisión de la sección 1:**
- La línea variacional está evaluada de forma reproducible. Su resultado defendible no es «la QCNN supera al clásico», sino:
  1. que aprende, por encima de la clase mayoritaria;
  2. que queda por debajo de una regresión logística que recibe las mismas 4 características, aunque la salida del circuito solo use 3 (sección 9);
  3. que el optimizador secuencial reduce el número de ejecuciones del circuito sin perder accuracy, lo que importa en hardware.
- Esto refuerza la opción A. Para la memoria convendría tomar como oficiales los resultados de la fase C (varias particiones) y presentar los de las fases B como un caso concreto.

## 8. Ejecución en hardware real: plan BSC (añadido el 20 de noviembre)

**Qué ofrece el BSC** (guía de acceso de la RES para recursos cuánticos, v3.3):
- **MareNostrum5-Ona** tiene dos chips superconductores digitales, de 20 y 35 qubits. Cada uno va conectado a un nodo de 40 CPU para el pre- y postprocesado.
- Los trabajos se programan con **Qibo + Qililab**. Qililab es la librería de Qilimanjaro, el fabricante de los chips, que traduce un circuito Qibo a pulsos sobre el chip. Nuestro circuito ya está en Qibo.
- **La partición cuántica solo acepta trabajos que usen el chip.** Las simulaciones hay que pedirlas aparte (MN5-GPP), en la misma solicitud.
- La solicitud pide qubits, profundidad, shots, puertas y duración.
- El acceso se concede por convocatoria de la RES: estándar (5 meses) o vía rápida, que exige un proyecto financiado.

**Qué se puede llevar al chip:** solo la línea variacional. QGD no es viable, porque sus puertas multicontroladas hacen el circuito demasiado profundo.

**Hecho sin acceso (fase 1; implementado y validado en simulación; suite completa de 138 pruebas OK):**
- Transpilador propio (Qibo 0.1.11 no trae uno). Reubica los qubits, añade SWAPs donde el chip no tiene conexión y descompone el circuito en puertas RX/RY/RZ/CZ. El resultado medido coincide exactamente con el circuito original.
- Ejecución de circuitos por lotes, porque en el BSC el coste está dominado por cada conexión.
- Un simulador que se comporta como el chip: solo acepta puertas nativas sobre qubits conectados, añade ruido y error de lectura.
- Calibración y corrección del error de lectura.
- Inferencia de los checkpoints entrenados, ya probada en seco.
- Scripts Slurm para MN5 y para Ona.
- El adaptador de Qililab es todavía un **esqueleto**: falta la API concreta del BSC.

**Datos técnicos para la solicitud** (`docs/bsc/cost-estimates.md`; topologías de chip **hipotéticas**):

| Concepto | Valor |
|---|---|
| Qubits | 4 (se mide 1) |
| Profundidad del circuito lógico | 25–26, con 11 CNOT |
| Profundidad ya transpilada | 26–38 con conexiones en línea, anillo o rejilla (0–2 SWAPs); 65–68 si el chip tiene forma de estrella (6 SWAPs) |
| Shots | 1000 por circuito |
| Inferencia de 8 modelos | 248 circuitos (Iris) y 1376 (Breast Cancer): como máximo 1,4 y 7,6 h |
| Entrenamiento corto en el chip | SPSA con minilotes, 3515 circuitos; secuencial con 10 barridos, 28 165 |
| Entrenar como en simulación (Adam) | 1,3 millones de circuitos en Breast Cancer: **no es viable** |

Los tiempos máximos usan la referencia de la guía (unos 20 s por trabajo, contando cada circuito como un trabajo). Los tiempos con lotes son una suposición hasta poder medirlos.

**Plan por fases:**

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Decisión del tutor y solicitud a la RES (Ona + MN5-GPP) | Borrador listo (`docs/bsc/res-application-draft.md`), no enviado |
| 1 | Preparación sin acceso | **Hecho** (validado en simulación) |
| 2 | Puesta en marcha en el BSC: entorno, adaptador de Qililab y pruebas mínimas | Necesita acceso |
| 3 | H1: inferencia en el chip de los checkpoints de la fase C, pre-registrada. Mide el efecto real del hardware con muy pocos circuitos | Propuesto |
| 4 | H2 (opcional): entrenamiento corto en Iris con presupuesto fijo, secuencial frente a SPSA | Propuesto |
| 5 | Documentación y capítulo de la memoria | Propuesto |

**¿Se sale del alcance?**
- No en el tema: es justo el objetivo de preparar la ejecución en hardware real.
- El riesgo está en los plazos: la convocatoria, la evaluación, la disponibilidad de los chips y la calibración.
- La propuesta es que **el TFG no dependa del chip**:
  - si el acceso llega a tiempo, la memoria incluye resultados en hardware (H1 y quizá H2);
  - si no llega, incluye la plataforma preparada para hardware y validada en simulación.

**Preguntas para la tutoría:**
1. ¿Entra el BSC en el TFG? ¿Con H1 solo, o también H2?
2. ¿Quién figura como responsable de la solicitud? ¿Hay un proyecto financiado para pedir la vía rápida?
3. ¿Hay contacto en el BSC o en Qilimanjaro para conocer la topología del chip, sus puertas nativas y la API de Qililab antes de enviar la solicitud?
4. ¿Qué fecha de entrega tiene la memoria?

## 9. Hallazgo: el circuito actual solo usa 3 características y 5 pesos (24 de noviembre)

**Estado: validado numéricamente** (`experiments/diagnose_light_cone.py`; `results/diagnostics/light-cone/`; log, sección 37).

**Qué pasa:**
- La convolución termina con `CNOT(derecha → izquierda)` y el pooling aplica `CNOT(descartado → conservado)`. Cuando el qubit descartado es el derecho del último par, las dos son **la misma CNOT, y dos CNOT seguidas se anulan**.
- Ocurre con `CNOT(3,2)` en la capa 1 y con `CNOT(2,0)` en la capa 2.

**Consecuencias:**

| Elemento | ¿Influye en la predicción? |
|---|---|
| Características x0, x1, x2 | Sí |
| Característica x3 (4.ª componente PCA en Breast Cancer; anchura del pétalo en Iris) | **No** |
| Pesos de la capa 1 | Sí |
| Peso w0 de la capa 2 | Sí (la capa 2 queda en una rotación de un qubit) |
| Pesos w1, w2 y w3 de la capa 2 | **No** |

- Afecta a todas las rutas (híbrida, QGD y variacional), porque comparten el circuito.
- Los resultados evaluados siguen siendo válidos para *este* circuito, pero describen un modelo con 3 características y 5 parámetros efectivos.
- **Hipótesis, no contrastada:** puede explicar en parte por qué la QCNN queda por debajo de la regresión logística.
- **Coste** (no implementado): bastarían 15 circuitos por muestra en lugar de 33 (parameter-shift) y 19 en lugar de 40 (secuencial). En hardware, el circuito efectivo es de 3 qubits.

**Opciones para decidir:**

| Opción | Qué implica |
|---|---|
| Corregir el circuito (por ejemplo, un pooling con rotación controlada parametrizada) | Modelo nuevo: hay que repetir la fase C (unas 7,5 h de CPU en exacto), regenerar los informes y revisar la solicitud RES |
| Mantenerlo | Documentar la limitación en la memoria, describir el modelo como de 3 características efectivas y, opcionalmente, podar el qubit 3 para el hardware |

## 10. Fuentes en el repositorio

- `docs/TFG-technical-log.md`: secciones 28–37.
- `results/diagnostics/light-cone/README.md`: hallazgo del cono de luz.
- `docs/bsc/res-application-draft.md`: borrador de la solicitud RES.
- `docs/bsc/README.md` y `docs/bsc/cost-estimates.md`: plan BSC y datos técnicos.
- `docs/experimental-protocol.md`: protocolo variacional.
- `results/variational/A-tuning-bc-exact/README.md`, `results/variational/B-confirm-bc/README.md`, `results/variational/B-confirm-iris/README.md` y `results/variational/C-partitions/README.md`.
- `results/quantum-training/`: perfiles M5, con pesos que no se mueven.
- Auditoría independiente: `AUDIT.md`, en el paquete de traspaso, fuera del repositorio.
