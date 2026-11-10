# Registro técnico del desarrollo del TFG

Este documento registra las decisiones de implementación del prototipo, el
motivo de cada cambio, su funcionamiento y las limitaciones conocidas. Sirve
como material de trabajo para redactar posteriormente la memoria del TFG. No
es todavía la memoria académica final: las afirmaciones experimentales deben
revisarse cuando exista un protocolo definitivo.

## 1. Alcance actual

El proyecto estudia una plataforma modular para ejecutar experimentos
reproducibles de Quantum Machine Learning sobre Qibo, usando PyTorch para el
entrenamiento clásico. La QCNN es el primer algoritmo implementado. La
plataforma deberá poder ampliarse posteriormente con otros algoritmos QML.

El prototipo actual trabaja con simulación local en CPU. Puede ejecutar
circuitos de estado puro de forma exacta o estimar las medidas mediante un
número configurable de shots. También permite añadir ruido depolarizante a
las puertas del circuito.

## 2. Estado inicial y problemas detectados

El primer prototipo permitía construir una codificación, una convolución y una
operación de pooling, pero no demostraba un entrenamiento correcto:

- los pesos se convertían a NumPy con `detach()`, rompiendo el grafo de
  autograd de PyTorch;
- la salida se basaba en una amplitud concreta del estado, en lugar de un
  observable definido;
- el pooling se describía como eliminación física de qubits, aunque Qibo
  conserva el registro completo;
- no había pruebas automáticas;
- no existía una separación entre ejecución exacta, muestreada y ruidosa;
- no había un bucle reutilizable que registrara épocas y pérdida.

La primera fase de desarrollo se centró por ello en hacer verificable el
motor QCNN antes de añadir una interfaz gráfica.

## 3. Codificación de datos

### Cambio

`src/data_encoding.py` implementa:

- codificación angular mediante `RY(pi * x)`;
- codificación por fase mediante `H` seguida de `RZ(2 * pi * x)`;
- validación de características finitas y normalizadas en `[0, 1]`.

### Motivo

Los datasets contienen números clásicos, mientras que el circuito recibe
ángulos de puertas. La codificación define cómo se representa la información
en el estado cuántico y, por tanto, forma parte del diseño experimental.

La codificación angular cambia amplitudes. La codificación por fase cambia
fases relativas y necesita una superposición previa para que esas fases puedan
afectar a una medición posterior. La QCNN actual usa la codificación angular;
la de fase queda disponible para una comparación futura.

### Detalle de implementación

Las funciones `angle_encoding` y `phase_encoding` crean circuitos nuevos.
Sus variantes `add_angle_encoding` y `add_phase_encoding` añaden puertas a un
circuito existente, que es la forma utilizada por la QCNN. El argumento
opcional `shifted_feature` permite desplazar una puerta durante el cálculo de
gradientes por parameter-shift.

## 4. Capa convolucional

### Cambio

`src/qcnn_circuit.py` aplica un filtro de cuatro ángulos sobre pares
adyacentes de qubits mediante rotaciones `RY`, `RZ` y puertas `CNOT`.

### Motivo

Los cuatro parámetros se comparten entre las distintas posiciones del filtro.
Esto mantiene la analogía con una convolución clásica: el mismo filtro se
reutiliza al recorrer la entrada y el número de parámetros no crece con el
número de posiciones.

Como un parámetro puede aparecer en varias puertas, el gradiente total debe
sumar la contribución de cada aparición.

## 5. Pooling

### Cambio

`src/pooling.py` aplica una `CNOT` desde cada qubit descartado al qubit
superviviente y valida que los grupos sean compatibles.

La QCNN mantiene una lista de qubits activos y excluye los descartados de las
capas posteriores y de la medición.

### Limitación

Qibo mantiene fijo el número de qubits del registro. El pooling actual reduce
los qubits activos desde el punto de vista de la arquitectura, pero no elimina
qubits del vector de estado ni reduce su coste de memoria. Esta diferencia debe
explicarse en la memoria para no confundir pooling lógico con reducción física
del registro.

## 6. Salida y gradientes

### Salida

`src/hybrid_model.py` calcula la expectativa del observable Pauli Z:

`<Z> = P(0) - P(1)`

El resultado está en `[-1, 1]` y se puede usar como salida binaria asociando
las etiquetas a `-1` y `+1`.

### Gradientes

La simulación de Qibo se ejecuta fuera del grafo automático de PyTorch. Para
conectar ambos sistemas se implementó una función `torch.autograd.Function`
con un `forward` y un `backward` propios.

El `backward` usa la regla de desplazamiento:

`df/dtheta = (f(theta + pi/2) - f(theta - pi/2)) / 2`

Para la codificación angular, el gradiente respecto a la característica
incluye además el factor `pi`, porque el ángulo es `theta = pi * x`.

Los pesos compartidos se derivan por cada aparición de puerta y sus
contribuciones se acumulan. El cálculo solo ejecuta gradientes de entradas o
pesos cuando PyTorch los necesita, evitando simulaciones innecesarias.

### Limitación de rendimiento

El método de parameter-shift vuelve a ejecutar el circuito muchas veces.
Resulta apropiado para validar circuitos pequeños, pero no es todavía una
implementación optimizada para muchos qubits o lotes grandes.

## 7. Generalización del número de qubits

La arquitectura permite distintos tamaños, pero el pooling actual exige que
el número de qubits y el número de salidas sean potencias de dos compatibles:

`n_qubits -> n_qubits/2 -> ... -> n_outputs`

Por ejemplo, `4 -> 2 -> 1` y `8 -> 4 -> 2 -> 1` son configuraciones válidas.
Tamaños como 3 o 6 qubits no se aceptan todavía. Es una restricción de esta
arquitectura jerárquica, no una imposibilidad general de las QCNN.

## 8. Configuración de ejecución cuántica

### Cambio

`src/execution.py` define `ExecutionConfig`:

- `shots=None`: simulación exacta de estado puro;
- `shots` positivo: estimación de expectativas mediante mediciones;
- `depolarizing_probability`: probabilidad de ruido depolarizante;
- `seed`: semilla para reproducir el muestreo.

El ruido requiere shots positivos. En modo muestreado, la expectativa se
estima como `(N0 - N1) / N`. Aumentar shots reduce la variabilidad estadística
aproximadamente como `1/sqrt(N)`, pero aumenta el coste total.

Actualmente el ruido solo se aplica después de `RY`, `RZ` y `CNOT`. Todavía no
se incluyen ruido de medición, amplitude damping o phase damping.

## 9. Entrenamiento clásico

### Cambio

`src/training.py` define `TrainingConfig` con:

- número de épocas;
- tasa de aprendizaje;
- optimizador `adam` o `sgd`.

`train_qcnn` ejecuta el ciclo:

1. poner a cero los gradientes;
2. ejecutar la QCNN;
3. calcular MSE;
4. ejecutar `backward`;
5. actualizar los pesos;
6. guardar la pérdida de la época.

La tasa de aprendizaje controla el tamaño de la actualización de los pesos.
No es un parámetro de ruido: el carácter exploratorio también depende de la
inicialización, el optimizador, los minibatches y, en modo cuántico
muestreado, la variabilidad de los shots.

## 10. Dataset Iris

### Cambio

`src/datasets.py` carga Iris mediante `scikit-learn`, selecciona `setosa` y
`versicolor`, conserva sus cuatro características y asigna etiquetas `-1` y
`+1`. Las cuatro características se normalizan a `[0, 1]`, por lo que encajan
directamente en una QCNN de cuatro qubits.

El experimento contiene 100 muestras binarias y utiliza:

`QCNNModel(n_qubits=4)`

### Dataset mayor para el cierre de la primera versión

Se añadió un adaptador para Breast Cancer Wisconsin, con 569 muestras, dos
clases y 30 características originales. Para mantener controlado el coste de
la QCNN se proyectan las características a cuatro componentes mediante PCA.
El `StandardScaler` y el PCA se ajustan exclusivamente con el subconjunto de
entrenamiento; después los componentes se normalizan a `[0, 1]` usando también
solo estadísticas de entrenamiento. Así, cada componente se asigna a uno de
los cuatro qubits sin fuga de información.

Este dataset permite evaluar una generalización más informativa que Iris sin
introducir a la vez una arquitectura de ocho o más qubits. El adaptador se
encuentra en `src/datasets.py` y el corredor reproducible de la batería final
en `experiments/run_final_qcnn.py`.

### Particionado y normalización

El pipeline actual separa primero las 100 muestras mediante particionado
estratificado configurable: por defecto usa 70% para entrenamiento, 15% para
validación y 15% para test. La semilla controla el particionado. Después
calcula mínimos y máximos solo con entrenamiento y aplica esa transformación a
los tres subconjuntos. Esto evita que la distribución del test influya en el
preprocesamiento.
Si una muestra de validación o test queda fuera del rango observado en
entrenamiento, el valor transformado se recorta a `[0, 1]` para respetar el
contrato del codificador cuántico. Este recorte no utiliza información de las
etiquetas ni recalcula la escala.

## 11. Experimento comparativo

`experiments/compare_execution.py` realiza un barrido sobre:

- simulación exacta;
- `shots` de 100, 500 y 1000;
- ruido depolarizante de 0.0, 0.01 y 0.05;
- semillas 1, 2 y 3.

Cada condición utiliza la misma arquitectura, partición de datos, pesos
iniciales, optimizador, tasa de aprendizaje y número de épocas. El modelo se
entrena solo con entrenamiento; la validación se usa para calcular una
pérdida independiente y la accuracy final se calcula sobre test. Se registran
pérdida inicial y final de entrenamiento, pérdida de validación, accuracy de
test y tiempo. Después se calculan medias y desviaciones estándar por
condición.

El experimento actual es una herramienta de validación y exploración. Para la
memoria será necesario acordar el protocolo estadístico y el número de
repeticiones. La separación train/validation/test ya está implementada.

## 12. Validación acumulada

La batería automática en `tests/test_qcnn.py` verifica:

- codificaciones;
- validación de entradas;
- filtro convolucional;
- pooling;
- gradientes;
- actualización de pesos;
- ejecución con shots;
- ejecución con ruido;
- configuración del entrenamiento.

La última ejecución validada produjo:

`Ran 17 tests - OK`

También se verificaron la compilación sintáctica y `git diff --check`.

## 13. Próximas tareas

1. Acordar el protocolo estadístico y el número de repeticiones.
2. Incorporar más modelos de ruido.
3. Comparar codificación angular y por fase si la tutoría lo considera útil.
4. Diseñar la interfaz después de estabilizar el flujo experimental.
5. Incorporar otros algoritmos QML después de validar la QCNN.

## 14. Registro de cambio: pipeline genérico de particionado

**Fecha:** 5 de octubre de 2026.

**Cambio:** se extrajo `split_dataset`, una utilidad reutilizable para
características y etiquetas de cualquier dataset tabular. `SplitConfig`
permite indicar explícitamente las proporciones de entrenamiento, validación y
test, además de la semilla. `load_binary_iris_splits` utiliza ahora esta misma
utilidad y acepta también las tres proporciones.

**Propósito:** evitar fuga de información del test hacia el entrenamiento y
permitir medir generalización en muestras no utilizadas para actualizar los
pesos, manteniendo el mismo protocolo al incorporar otros datasets.

**Validación:** se añadieron pruebas para porcentajes personalizados y para un
dataset genérico. Se comprueban tamaños, reproducibilidad y rangos; el
experimento expone ahora las proporciones mediante `SweepConfig`.

**Limitación:** Iris sigue siendo un dataset pequeño y el experimento todavía
no incluye intervalos de confianza.

## 15. Registro de cambio: cierre experimental de la QCNN

**Fecha:** 5 de octubre de 2026.

**Cambio:** se añadieron métricas binarias comunes (accuracy, precision,
recall y F1), una regresión logística como baseline externo y persistencia
opcional de resultados en JSON. El baseline se entrena exclusivamente con
`train` y se evalúa sobre `test`.

**Propósito:** separar la lógica científica de evaluación del motor cuántico y
obtener una referencia clásica sencilla. La regresión logística no se incluye
en la abstracción futura del motor QML porque no es una implementación de un
modelo cuántico, sino un control experimental.

**Validación:** se añadió una prueba de las métricas y se ejecutó un barrido
reducido de la QCNN con particionado train/validation/test. Los resultados
pueden guardarse con `--output` en formato JSON.

## 16. Revisión global de calidad y generalización

**Fecha:** 5 de octubre de 2026.

**Correcciones:** se garantizó que el particionador genérico convierta
cualquier problema binario numérico al contrato de etiquetas `-1/+1` de la
QCNN. Se validan valores finitos en configuraciones, entrenamiento y métricas.
Además, la evaluación de validación con shots reutiliza una única inferencia
para calcular pérdida y métricas, evitando comparar resultados obtenidos con
muestreos distintos.

**Decisiones de diseño:** las métricas permanecen independientes del modelo y
el baseline clásico queda en el módulo experimental. El núcleo QCNN conserva
su responsabilidad de ejecutar circuitos y calcular gradientes; no se añade
todavía una jerarquía abstracta de modelos QML hasta conocer el segundo
algoritmo.

**Limitaciones conocidas:** el particionador y las métricas siguen orientados a
clasificación binaria tabular; la clasificación multiclase queda pendiente. La
QCNN procesa lotes completos, sin minibatches. El cálculo parameter-shift
continúa siendo el principal coste de ejecución.

## 17. Cierre del código y protocolo experimental

**Fecha:** 5 de octubre de 2026.

**Revisión final:** se corrigió la gestión de la semilla de shots. Antes se
reiniciaba en cada llamada al modelo, lo que podía repetir la misma secuencia
entre épocas y evaluaciones. Ahora cada repetición experimental inicializa la
semilla una vez antes del entrenamiento; las ejecuciones internas consumen
muestras consecutivas y la repetición sigue siendo reproducible.

La condición exacta se ejecuta una sola vez, porque no tiene variabilidad de
shots ni ruido cuando los pesos y datos son iguales.

**Documentación:** se creó
`docs/experimental-protocol.md` con la matriz base de Iris, la configuración
de la QCNN, las condiciones exacta/muestreadas/ruidosas, las métricas, el
baseline y los experimentos posteriores.

**Criterio de cierre:** la QCNN queda preparada para la batería experimental
base. Las extensiones como multiclase, minibatches, nuevos datasets y un
segundo algoritmo QML se mantienen fuera de esta primera versión para no
mezclar objetivos.

**Batería previa:** se ejecutaron 5 configuraciones reducidas con 2 épocas,
50 shots cuando procedía, una ejecución exacta y dos semillas muestreadas con
ruido `0.0`/`0.01`. El resultado se guardó en
`results/initial-hybrid-adam/qcnn-validation.json`. Sirve como validación técnica del flujo, no
como resultado definitivo de la memoria.

## 18. Resultado de la prueba intermedia y siguiente experimento

**Fecha:** 5 de octubre de 2026.

Se ejecutó una prueba intermedia con 3 épocas, Adam, tasa de aprendizaje
`0.03`, 50 y 100 shots, ruido `0.0` y `0.01`, y dos semillas en las
condiciones muestreadas. El archivo
`results/initial-hybrid-adam/iris-qcnn-medium.json` contiene 9 ejecuciones y 5 resúmenes.

La accuracy de test de la QCNN quedó entre `0.50` y `0.567` en esta prueba,
mientras que la regresión logística alcanzó `1.0` sobre la misma partición.
La interpretación está limitada por las 3 épocas y por un conjunto test de
solo 15 muestras. Por ello, los resultados validan el flujo, pero no prueban
la convergencia ni permiten concluir que la arquitectura no pueda mejorar.

El coste del ruido fue el hallazgo operativo principal: las condiciones con
ruido `0.01` tardaron aproximadamente 162--164 segundos por repetición, frente
a unos 7 segundos para las condiciones sin ruido. Ejecutar directamente el
barrido completo de shots, ruido, semillas y 30 épocas tendría un coste
desproporcionado.

Como siguiente experimento se planifica un diagnóstico de convergencia solo
con simulación exacta: 3, 10, 20 y 30 épocas, Adam, learning rate `0.03`,
partición y pesos iniciales fijos. Su duración objetivo es inferior a
20 minutos. El resultado decidirá si se fija una configuración de entrenamiento
para un estudio posterior de ruido o si primero hay que revisar la QCNN.

Se implementó `experiments/diagnose_convergence.py`, que guarda la curva de
pérdida de cada configuración y muestra un mensaje al terminar cada número de
épocas. El resultado se guardará en
`results/initial-hybrid-adam/qcnn-convergence.json`, evitando que el progreso dependa de que
termine todo el experimento.

## 19. Batería final propuesta

La prueba `results/initial-hybrid-adam/qcnn-noisy-convergence-5epochs.json` comparó una ejecución
exacta con dos repeticiones de 50 shots y ruido depolarizante `0.01`, usando
cinco épocas. La ejecución exacta redujo la pérdida de `1.208` a `1.086` y
obtuvo `0.533` de accuracy en test. Las dos ejecuciones ruidosas obtuvieron
`0.267` y `0.467` en test, con pérdidas finales `1.173` y `1.176`.

El tiempo fue de unos 11 segundos en exacto y unos 270 segundos por repetición
ruidosa. Esto confirma que shots, ruido y parameter-shift forman el principal
cuello de botella. Por ello, el estudio ruidoso final se limita a pocas
semillas y cinco épocas, y no se considera necesario un barrido exhaustivo
para cerrar la primera versión.

Se añadió `experiments/run_final_qcnn.py` para ejecutar la batería enfocada:

- convergencia exacta en Breast Cancer Wisconsin reducido mediante PCA a
  cuatro componentes;
- variabilidad entre tres inicializaciones;
- comparación de codificación angular y por fase;
- una condición reducida de 50 shots y ruido `0.01`.

El adaptador ajusta estandarización, PCA y normalización solo con train. El
dataset mayor tiene 569 muestras y produce particiones de 398, 85 y 86
ejemplos, por lo que ofrece una evaluación más informativa que las 15
muestras de test de Iris sin aumentar inicialmente el número de qubits.

### Resultado ejecutado del experimento ruidoso

El experimento E se ejecutó sobre Breast Cancer con PCA a cuatro componentes,
codificación angular, Adam, cinco épocas, 50 shots, ruido depolarizante
`0.01` y dos semillas. La pérdida descendió de `1.028` a `0.915` para la
semilla 1 y de `1.068` a `0.944` para la semilla 2. Las accuracies de test
fueron `0.744` y `0.616`, con F1 `0.817` y `0.686`.

La regresión logística alcanzó `0.907` de accuracy y `0.931` de F1 en el mismo
test. Por tanto, la QCNN muestra aprendizaje bajo ruido, pero su rendimiento
medio (`0.680` de accuracy) queda por debajo del baseline y presenta
variabilidad apreciable entre semillas.

La repetición final, realizada con la suspensión automática desactivada,
registró aproximadamente 26,0 minutos para la semilla 1 y 26,1 minutos para
la semilla 2. Estos tiempos de reloj son coherentes entre sí y sustituyen a
los tiempos de la ejecución anterior, que quedó afectada por la suspensión
del equipo. No se utiliza la duración antigua en las conclusiones de coste.

## 20. Resultados exactos finales

La batería exacta sobre Breast Cancer se completó correctamente. La
convergencia con codificación angular produjo:

- 10 épocas: pérdida final `0.894`, accuracy de test `0.651`;
- 20 épocas: pérdida final `0.685`, accuracy de test `0.744`;
- 30 épocas: pérdida final `0.590`, accuracy de test `0.826`.

La mejora monotónica de la pérdida y de la accuracy de test justifica usar 30
épocas como configuración de referencia, sin afirmar que sea un óptimo global.

La comparación angular con tres semillas obtuvo accuracies de test `0.826`,
`0.872` y `0.849`, media `0.849`. La codificación por fase obtuvo `0.826`,
`0.872` y `0.895`, media `0.864`. La regresión logística alcanzó `0.907`.
La fase presenta una ventaja media pequeña, pero las tres semillas no
permiten establecer superioridad estadística.

Los tiempos exactos fueron aproximadamente 5,6 minutos por ejecución angular
y 6,0 minutos por ejecución en fase. Con estos resultados quedan cubiertos
convergencia, variabilidad, comparación de codificaciones y robustez frente a
ruido. La primera versión experimental de la QCNN puede considerarse
cerrada; las mejoras de rendimiento y las arquitecturas de más qubits pasan a
ser trabajo futuro.

## 21. Requisito de entrenamiento no híbrido

**Fecha:** 6 de noviembre de 2026.

**Origen:** reunión de seguimiento del TFG. El núcleo de la QCNN y la pipeline
experimental se consideran válidos, pero se pide evitar que el modelo y el
entrenamiento sean híbridos. En particular, el ajuste de pesos no debe depender
del descenso de gradientes clásico ni de Adam.

**Situación de partida:** Qibo ejecuta el circuito y calcula expectativas, pero
PyTorch mantiene los parámetros, construye el grafo de autograd y Adam actualiza
los pesos. El backward usa parameter-shift, pero el bucle exterior sigue siendo
una optimización clásica con retropropagación.

**Decisión:** sustituir el entrenamiento híbrido por un camino basado en Qibo y
NumPy. La pérdida se evaluará ejecutando circuitos cuánticos y la actualización
de cada peso se realizará mediante un circuito cuántico sobre un registro
discreto de pesos y gradientes (QGD discreto). Se usarán COBYLA y SPSA como
baselines sin gradientes. Adam se conserva únicamente como referencia histórica
de comparación.

**Alcance:** el control clásico del bucle (medición, discretización y parada)
es inevitable en un simulador. La memoria deberá distinguir con precisión qué
parte es cuántica y qué parte sigue siendo clásica, sin presentar el resultado
como un entrenamiento totalmente coherente.

**Plan:** el diseño completo, las alternativas, el protocolo experimental y los
criterios de aceptación están en
[`docs/quantum-training-plan.md`](quantum-training-plan.md). El plan incluye
la extracción de un motor QCNN puro, los optimizadores sin gradientes, el
prototipo QGD, las pruebas y la repetición de experimentos con presupuestos
ampliados y más semillas. También define las unidades de comparación para no
mezclar conceptos: época para Adam y QGD, evaluación para COBYLA e iteración
para SPSA, con equivalencias en pasadas sobre el entrenamiento y en coste de
simulación. La comparación se divide en un bloque A de igualdad de pasadas
sobre los datos y un bloque B de presupuesto ampliado para los métodos sin
gradientes.

**Experimentos previstos:** Breast Cancer con PCA a cuatro componentes, Adam
legado como referencia, COBYLA y SPSA con 300--1000 evaluaciones, QGD con
30--100 pasos, cinco semillas en la batería principal y una condición reducida
con shots y ruido. Se reportarán media, desviación estándar e intervalos de
confianza, evitando afirmaciones de significación con pocas semillas.

**Criterio de cierre:** existir un entrenamiento sin PyTorch, autograd ni Adam;
que la actualización cuántica coincida con la actualización clásica sobre la
rejilla discretizada; que la QCNN reduzca la pérdida y obtenga métricas de test
con el nuevo método; y que los resultados queden persistidos y documentados.

**Archivo histórico:** los resultados de la etapa híbrida se conservan
congelados en `results/initial-hybrid-adam/` y no se sobrescribirán. Las
repeticiones de experimentos antiguos deben guardarse en
`results/reproductions/legacy/`, y los resultados del nuevo entrenamiento en
`results/quantum-training/`.

**Estado:** planificado. No se ha modificado todavía el código de entrenamiento.

## 22. Estado de M1-M3 antes de integrar el runner (registro histórico)

**Fecha:** 7 de noviembre de 2026.

**Alcance — Implementado:** se incorporaron los componentes M1-M3 del diseño:

- `src/qcnn_engine.py` contiene el motor QCNN de Qibo/NumPy, con simulación,
  pérdida y gradiente parameter-shift. `src/hybrid_model.py` delega ahora la
  simulación al motor, manteniendo el wrapper legado.
- `src/optimizers.py` ofrece optimizadores autónomos, incluidos COBYLA y SPSA;
  aún no están conectados a un runner de entrenamiento.
- `src/quantum_update.py` implementa la actualización discreta mediante una
  permutación modular reversible.

**Validación — Validado:** existen pruebas para estos componentes en
`tests/test_quantum_engine.py`, `tests/test_optimizers.py` y
`tests/test_quantum_update.py`. Tras los ajustes menores posteriores en
comentarios y pruebas, la suite completa pasó con 41 pruebas mediante
`qml_env\\Scripts\\python.exe -m unittest discover -s tests -v`. También pasaron
`qml_env\\Scripts\\python.exe -m compileall src tests` y `git diff --check`.

**Desviación y limitación de la actualización:** frente al mapeo con clipping
del plan, la implementación conserva el registro del gradiente y actualiza el
código del peso de forma modular. La conservación del gradiente hace reversible
la transformación. En cambio, la propuesta original
`|theta_code>|grad_code> -> |theta_new_code>|0>` no es inyectiva y no puede ser
unitaria como está formulada. La codificación del gradiente recorta los valores
al intervalo finito configurado; para gradientes fuera de ese intervalo se
pierde magnitud y cambia la actualización efectiva. La variante modular se
desvía del plan y la aprobación del tutor está pendiente; no se registra como
cumplimiento total del diseño ni como aceptación de «no híbrido».

**Estado en la fecha de esta entrada:** `src/training.py` continúa usando
PyTorch con Adam o SGD. En este punto de la secuencia de trabajo no existían
`experiments/run_quantum_training.py` ni `tests/test_quantum_training.py`; por
tanto, todavía no había runner ni entrenamiento integrado. No se ejecutó ningún
experimento nuevo ni se generó un JSON de resultados. M1-M3 estaban
**Implementados** y **Validados** por la suite indicada, pero no había evaluación
experimental. La integración M4 posterior se registra en la sección 23.

**Siguientes tareas:** M4, integrar los componentes en el runner y el flujo de
entrenamiento experimental; M0, obtener la decisión del tutor sobre la variante
modular y sobre el uso de la denominación «no híbrido».

## 23. Integración M4 del entrenamiento con QGD discreto

**Fecha:** 7 de noviembre de 2026.

La sección 22 registra el estado de M1-M3 anterior a esta integración. Sus
menciones a la ausencia del runner y de las pruebas de integración quedan
actualizadas por esta entrada.

**Alcance — Implementado:** M1-M4 están integrados en los módulos
`src/qcnn_engine.py`, `src/optimizers.py`, `src/quantum_update.py` y
`src/quantum_training.py`. La orquestación publica `QuantumTrainingConfig`,
`QuantumTrainingResult`, `train_quantum_qcnn` y `derive_execution_seed`. El
runner de línea de comandos se encuentra en
`experiments/run_quantum_training.py`. Las pruebas asociadas son
`tests/test_quantum_engine.py`, `tests/test_optimizers.py`,
`tests/test_quantum_update.py` y `tests/test_quantum_training.py`.

**Validación — Validado unitariamente:** tras añadir métricas de validación y
pruebas de ruido/flujo de test, la suite completa pasó con 52 pruebas:

```powershell
qml_env\Scripts\python.exe -m unittest discover -s tests -v
```

También terminó correctamente la compilación sintáctica:

```powershell
qml_env\Scripts\python.exe -m compileall src experiments tests
```

`git diff --check` pasó; la salida incluyó únicamente avisos de conversión
LF/CRLF en documentación preexistente. Estas comprobaciones validan el código y
su integración unitaria, no el comportamiento experimental del entrenamiento.

**Flujo y frontera clásica-cuántica:** `src/quantum_training.py` no importa
PyTorch. El runner importa `src.datasets` al cargarse, y este loader importa
PyTorch; la conversión de sus tensores ocurre después, en la frontera de entrada
al entrenamiento. Para inicializar la primera actualización, Qibo ejecuta los circuitos
del gradiente parameter-shift en modo exacto y sin ruido; el host calcula las
diferencias. La actualización ejecuta un circuito reversible de Qibo por cada
peso, en estado-vector exacto y sin shots. El host obtiene el estado de salida
mediante `argmax` y decodifica el código del peso. Tras cada actualización, las
pérdidas de entrenamiento y validación se evalúan con shots. Los gradientes de
las actualizaciones posteriores se estiman con circuitos parameter-shift
muestreados: el host lee las expectativas, calcula las diferencias y acumula las
derivadas. El cálculo de pérdidas, la acumulación, el control del bucle, la
selección del checkpoint con la pérdida de validación y la parada también son
clásicos.

El runner carga los datos a través de `src.datasets`, que conserva tensores
PyTorch por compatibilidad con la pipeline heredada, y los convierte a NumPy en
la frontera con el entrenamiento nuevo. La API `train_quantum_qcnn` no acepta
datos de test. El runner selecciona el checkpoint con validación y evalúa el
test una sola vez al finalizar el entrenamiento.

**Métricas:** el historial registra pérdida y accuracy, precision, recall y F1
para train y validation en cada paso, a partir de la misma pasada de predicción
muestreada. El JSON del runner añade las métricas de test en la evaluación final.
El checkpoint continúa seleccionándose solo por pérdida de validation. El CLI
permite configurar ruido depolarizante para gradientes posteriores y evaluaciones
con shots; el gradiente inicial exacto y el circuito de actualización no usan ese
ruido.

**Límites y desviaciones:** los contadores de ejecuciones de circuitos son
estimaciones estructurales derivadas de la configuración y del número de
muestras; no son instrumentación del tiempo de ejecución. La actualización QGD
usa una permutación reversible con wrap-around modular del peso y conserva el
registro del gradiente. La codificación satura los gradientes fuera del rango
finito configurado; esto no equivale a clipping y altera la magnitud efectiva de
esos gradientes. La variante difiere del plan y requiere contraste con el tutor.
El postprocesamiento de medidas y el control clásico se hacen explícitos. No se
afirma que M4 satisfaga el objetivo de extremo a extremo ni que esté listo o
verificado para hardware real.

**Evaluación — No realizada:** no se ejecutó el CLI ni se lanzó ningún
experimento M4; tampoco se generó un JSON nuevo. La validación unitaria no debe
interpretarse como evaluación experimental.

**Siguientes pasos:** concretar una ruta de backend real compatible con la QCNN,
revisar el circuito de actualización modular y definir qué medición,
postprocesamiento y control clásico son inevitables. Ejecutar un piloto en
simulación o hardware solo cuando se solicite y autorice, con configuración y
destino acordados, sin sobrescribir resultados históricos.

## 24. Aclaración del objetivo de entrenamiento y hardware

**Fecha:** 7 de noviembre de 2026.

El responsable aclara que la intención de evitar un entrenamiento híbrido es
preparar la ejecución en un ordenador cuántico real. La indicación inicial del
tutor fue llevar el entrenamiento lo más posible a circuitos cuánticos y
eliminar el ajuste clásico de los pesos. Esta aclaración establece la dirección
del proyecto; no constituye una validación del código M4 ni una aprobación de su
regla modular concreta.

Por tanto, deben distinguirse:

- **Objetivo acordado:** compatibilidad con ejecución en hardware real y pesos
  actualizados mediante circuitos, evitando una actualización clásica.
- **Estado implementado:** M4 ejecuta circuitos en el simulador Qibo; el circuito
  de actualización se decodifica desde el estado-vector exacto mediante
  postprocesamiento clásico y no usa shots. El módulo de entrenamiento sigue
  dependiendo de lógica clásica para extraer y combinar información de
  mediciones, pérdida, selección y control.
- **Pendiente para acercarse al objetivo:** identificar un backend/dispositivo,
  verificar compatibilidad de las operaciones y restricciones de hardware,
  revisar la actualización modular frente a la intención del tutor, y definir
  explícitamente la frontera clásica necesaria para la lectura de mediciones y
  el control iterativo.

No llamar a la implementación actual «hardware-ready» ni afirmar que satisface
el objetivo solo porque no importa PyTorch o porque la actualización se expresa
como un circuito de simulación.

## 25. M5: executor medido y perfiles exploratorios

**Fecha:** 7 de noviembre de 2026.

**Implementado:** M5 añade `src/circuit_execution.py`, con el contrato
`CircuitExecutor`, perfiles declarativos de recursos y un adaptador local de
simulación Qibo que devuelve counts. `src/quantum_update.py` ejecuta la
actualización aritmética reversible mediante circuitos controlled-X; se cubren
la actualización de un peso y la vectorial. Con executor, los counts se
decodifican en el host; la preparación de entradas y registros, la codificación
del gradiente, el cálculo y acumulación de derivadas, pérdidas, selección por
validation y control del bucle siguen siendo clásicos. La interfaz declara
límites de recursos, pero no transpila ni valida conectividad/base de puertas
contra un dispositivo.

El modo `--measured-execution` activa shots finitos incluso con ruido cero, para
que el gradiente inicial y las predicciones se midan. Un ruido solicitado mayor
que cero también activa el executor. Sin executor y con `shots=None`, los
gradientes y las evaluaciones de la QCNN son exactos y sin ruido, pero cada
actualización usa el circuito medido en el simulador Qibo con un shot
determinista. Con executor se requieren shots finitos. El ruido implementado es
un canal depolarizante local de sensibilidad, aplicado en la simulación a
qubits tocados por puertas; `p=0.001` no representa una calibración física.

**Validación — Validado unitariamente:** en el árbol de trabajo sucio basado en
`164eaee2b05ed53f9950afe8a272ca91f5e292df`, el 7 de noviembre de 2026 pasó la
suite completa con 73 pruebas (conteo histórico; la validación vigente es la
sección 29):

```powershell
qml_env\Scripts\python.exe -m unittest discover -s tests -v
qml_env\Scripts\python.exe -m compileall src experiments tests
git diff --check
```

`compileall` terminó correctamente y `git diff --check` no reportó errores de
espaciado; Git mostró únicamente avisos de conversión LF/CRLF. La suite valida
componentes y contratos de simulación, no compatibilidad con una QPU.

**Perfiles — Ejecutados para caracterizar flujo y coste, no para evaluar
rendimiento:** se generaron los siguientes JSON con partición completa, 100
shots, dos pasos, semillas dataset/inicialización/ejecución `42/19/101` y cuatro
qubits. Las duraciones son las registradas como `training.elapsed_seconds`; los
conteos son estimaciones lógicas estructurales.

| Perfil | Tiempo de entrenamiento | Ejecuciones de entrenamiento estimadas | JSON |
|---|---:|---:|---|
| Iris angular, `p=0` | 5,68 s | 4.806 | [`m5-profile-iris-angle-measured-shots100-noise0-2steps.json`](../results/quantum-training/m5-profile-iris-angle-measured-shots100-noise0-2steps.json) |
| Iris angular, `p=0.001` | 46,84 s | 4.806 | [`m5-profile-iris-angle-measured-shots100-noise0001-2steps.json`](../results/quantum-training/m5-profile-iris-angle-measured-shots100-noise0001-2steps.json) |
| Iris fase, `p=0` | 5,94 s | 4.806 | [`m5-profile-iris-phase-measured-shots100-noise0-2steps.json`](../results/quantum-training/m5-profile-iris-phase-measured-shots100-noise0-2steps.json) |
| Iris fase, `p=0.001` | 51,19 s | 4.806 | [`m5-profile-iris-phase-measured-shots100-noise0001-2steps.json`](../results/quantum-training/m5-profile-iris-phase-measured-shots100-noise0001-2steps.json) |
| Breast Cancer angular, `p=0` | 31,49 s | 27.250 | [`m5-profile-breast-cancer-angle-measured-shots100-noise0-2steps.json`](../results/quantum-training/m5-profile-breast-cancer-angle-measured-shots100-noise0-2steps.json) |

Cada JSON conserva `provenance.command_argv`, revisión y estado del árbol. La
revisión Git registrada es `164eaee2b05ed53f9950afe8a272ca91f5e292df`, con
cambios M5 sin commit. No se generó un log independiente. Los JSON incluyen
test una vez tras seleccionar por validation (15 casos Iris y 86 Breast Cancer);
las métricas de test de estos perfiles no se interpretan como comparación de
rendimiento: hay una inicialización y solo dos pasos por condición.

**Estimación para decidir coste — no ejecutada:** una extrapolación lineal desde
los tiempos de dos pasos da las siguientes proyecciones para 30/60/100 pasos:

| Condición perfilada | 30 pasos | 60 pasos | 100 pasos |
|---|---:|---:|---:|
| Iris angular, `p=0` | 1,4 min | 2,8 min | 4,7 min |
| Iris fase, `p=0` | 1,5 min | 3,0 min | 4,9 min |
| Iris angular, `p=0.001` | 11,7 min | 23,4 min | 39,0 min |
| Iris fase, `p=0.001` | 12,8 min | 25,6 min | 42,7 min |
| Breast Cancer angular, `p=0` | 7,9 min | 15,7 min | 26,2 min |

Para los cinco perfiles disponibles, 30/60/100 pasos y dos o tres semillas
sumarían aproximadamente 7,5–11,2 horas de entrenamiento bajo la hipótesis
lineal; la cifra excluye las condiciones de Breast Cancer no perfiladas y
la evaluación final de test y overhead de inicio. No es una cotización fiable ni
constituye autorización para ejecutar la matriz.

**Límites y siguientes pasos:** no hay backend/proveedor, transpilación,
validación de conectividad, ejecución QPU ni modelo calibrado de ruido. Las
estimaciones de circuitos no son instrumentación de jobs o shots. Los perfiles
solo permiten comparar coste operativo en estas condiciones y no demuestran
convergencia ni generalización. Antes de ampliar el protocolo, acordar el
alcance y el tiempo disponible; después perfilar las condiciones de Breast
Cancer restantes y decidir si se mantienen 30/60/100 pasos y 2–3 semillas.

## 26. Validación de componentes previa a la comparación

**Estado — Validado unitariamente, no evaluado experimentalmente:** en el árbol
sucio basado en HEAD `164eaee2b05ed53f9950afe8a272ca91f5e292df` se registraron
estas ejecuciones:

```text
qml_env\Scripts\python.exe -m unittest tests.test_quantum_update -v
Ran 17 tests in 1.722s — OK

qml_env\Scripts\python.exe -m unittest tests.test_quantum_update tests.test_quantum_training tests.test_circuit_execution -v
Ran 38 tests in 12.652s — OK

qml_env\Scripts\python.exe -m unittest tests.test_qcnn tests.test_quantum_engine -v
Ran 31 tests in 5.468s — OK
```

La primera ejecución incluye cuatro pruebas nuevas con executor fake y counts
controlados; la segunda corresponde al estado anterior a esas cuatro
adiciones. La suite completa no se ejecutó y estas pruebas no constituyen un
experimento. `git diff --check` pasó con solo avisos LF/CRLF. Los fake counts
verifican política de ejecución y propagación de counts, no ruido estadístico ni
comportamiento en QPU.

El protocolo comparativo está propuesto al final de
`docs/experimental-protocol.md`; no se ha ejecutado ni aprobado. El plan
`docs/quantum-training-plan.md` conserva una referencia antigua que presenta
Adam como solo histórico; la armonización documental sigue pendiente. No se
registra aprobación del tutor.

## 27. Prerrequisitos para la comparación emparejada QGD–Adam

**Fecha:** 7 de noviembre de 2026.

**Implementación:** se añadió `experiments/compare_qgd_adam.py` y se modificaron
`src/circuit_execution.py`, `src/hybrid_model.py`, `src/qcnn_engine.py` y
`src/quantum_training.py`, junto con pruebas para el runner, la ejecución, el
entrenamiento, la actualización y el motor QCNN. El runner construye un vector
inicial común: inicializa NumPy, cuantiza a `float32` y promueve a `float64`;
verifica que Adam, cuyos parámetros son `float32`, pueda almacenarlo sin pérdida.
QGD recibe valores numéricamente idénticos. Adam conserva el mismo optimizador
entre épocas y ambos métodos seleccionan el checkpoint con menor MSE de
validation. `--screening` evita acceder a test y fija exactamente dos
épocas/pasos (por defecto, dos); la configuración normal usa 30 por defecto.
`--qgd-only` omite Adam, exige `p_model=0` y solo permite estudiar `p_update`,
por lo que no constituye una comparación entre métodos. El runner evita
sobrescribir rutas existentes y resultados históricos.

**Ruido:** el runner inyecta un executor común en los circuitos QCNN de Adam y
QGD. `p_model` aplica canales depolarizantes locales de un qubit después de
`RY`, `RZ` y `CNOT`, en cada qubit tocado y excluyendo mediciones; cubre también
parameter-shift, predicción y evaluaciones de train, validation y test. Un
`update_executor` independiente aplica `p_update` a los circuitos aritméticos
QGD, cuya transformación cubre toda puerta no medición. Las coberturas son
distintas y no deben interpretarse como un mismo canal físico. Con `p=0` se
pueden usar counts con shots finitos sin canal. `shots=None` solo se admite con
`p_model=p_update=0`; en ese caso, los gradientes y evaluaciones QCNN son
exactos, mientras que el update QGD usa un shot determinista. Todo ello es
sensibilidad en simulador local; no representa hardware ni QPU.

**Validación — Validado unitariamente, no evaluado experimentalmente:** las
comprobaciones dirigidas se ejecutaron sobre un árbol de trabajo sucio basado
en HEAD `164eaee2b05ed53f9950afe8a272ca91f5e292df`. Se registraron 88 pruebas
correctas en estos grupos (suma de grupos dirigidos, no una suite completa; la
validación vigente es la sección 29):

```text
qml_env\Scripts\python.exe -m unittest discover -s tests -p test_compare_qgd_adam.py -v
12 pruebas, OK (0.055 s)
qml_env\Scripts\python.exe -m unittest discover -s tests -p test_circuit_execution.py -v
9 pruebas, OK (0.067 s)
qml_env\Scripts\python.exe -m unittest discover -s tests -p test_quantum_training.py -v
19 pruebas, OK (5.367 s)
qml_env\Scripts\python.exe -m unittest discover -s tests -p test_quantum_update.py -v
17 pruebas, OK (2.095 s)
qml_env\Scripts\python.exe -m unittest discover -s tests -p test_qcnn.py -v
18 pruebas, OK (1.357 s)
qml_env\Scripts\python.exe -m unittest discover -s tests -p test_quantum_engine.py -v
13 pruebas, OK (3.027 s)
```

`git diff --check` pasó sin errores; Git emitió solo avisos LF/CRLF. El árbol
estaba sucio y estas pruebas dirigidas/mockeadas comprueban contratos y
comportamientos acotados: no equivalen a ejecutar el runner, un experimento ni
la suite completa. En el momento de esa validación no se ejecutó
`experiments/compare_qgd_adam.py`, no se generaron resultados comparativos y no
se ejecutaron las matrices S0, G0, U0, S1, G1, G2 o U1; la ejecución posterior
de S0, G0 y U0 se registra en la sección 28. Se retira el hallazgo anterior que
indicaba que el test QGD no recibía `p_model`: el test reutiliza el executor con
su transformación.

**Estado metodológico y límites:** una revisión estática independiente considera
aptos con condiciones S0, G0 y U0 como screenings de dos pasos, sin bloqueo
técnico identificado. La semilla emparejada por identificador no garantiza
realizaciones de ruido idénticas. La instrucción de preparar el código no
constituye aprobación para confirmaciones. Tras los screenings debe revisarse
el coste total y presentarse al responsable antes de G1, G2, S1 o U1. En el
momento de redactar esta sección no se habían ejecutado screenings ni
confirmaciones; la ejecución posterior de S0, G0 y U0 se registra en la sección
28. La aceptación del tutor de la regla modular QGD sigue pendiente.

## 28. Screens S0, G0 y U0: inventario, tiempos y bloqueos

**Fecha de ejecución:** 8 de noviembre de 2026. **Registro:** 9 de noviembre
de 2026.

**Estado — Ejecutado parcialmente; no es evaluación de rendimiento:** este
apartado actualiza el estado de la sección 27. Se ejecutaron los screenings S0,
G0 y U0 del runner `experiments/compare_qgd_adam.py` sobre el árbol de trabajo
sucio basado en HEAD `164eaee2b05ed53f9950afe8a272ca91f5e292df`. Se conservaron
12 JSON válidos y 15 logs en `results/quantum-training/`; los tres logs sin JSON
corresponden a fallos de U0. No se ejecutaron S1, G1, G2, U1 ni hardware/QPU.
Todos los JSON declaran `screening=true`, dos épocas/pasos,
`test.evaluated=false` y la revisión citada; ninguno contiene métricas de test.

**Configuración común:** Iris binario, codificación angular, split estratificado
70/15/15, semilla de dataset `42`, semilla de inicialización `19`, semillas de
ejecución `101` y `102`, MSE frente a etiquetas `-1/+1`, selección de checkpoint
por menor MSE de validation, Adam con `lr=0.03` y QGD con `b=5`,
`weight=[-π,π]`, `grad=[-1,1]` y `lr=0.1`. La inicialización se comparte
exactamente entre métodos. `p_model` afecta a los circuitos QCNN y `p_update`
solo a los circuitos aritméticos de QGD; sus coberturas no son un mismo canal
físico. Cada JSON conserva `provenance`, semillas, configuración, pesos
iniciales y el estado de test; los logs conservan `WALL_SECONDS` y, salvo la
excepción indicada, `EXIT_CODE`.

**Inventario:**

| Bloque | JSON | Configuración ejecutada | Estado |
|---|---:|---|---|
| S0 | 7 | 1 referencia exacta (`shots=None`, `p=0`, ejecución 101) y 6 pares finite-shot (`shots=100/500/1000` × ejecuciones 101/102; `p_model=p_update=0`) | Parcial: falta la segunda referencia exacta prevista; no hay test. |
| G0 | 4 | Pares Adam–QGD con 100 shots, `p_model=0.001/0.01`, `p_update=0`, ejecuciones 101/102 | Completo para el bloque pareado definido; no hay test. |
| U0 | 1 + 3 fallos | QGD-only, `p_model=0`, 100 shots, `p_update=0.001/0.01`, ejecuciones 101/102 | Parcial: solo `p_update=0.001`, ejecución 101 produjo JSON; tres ejecuciones fallaron. El caso `p_update=0` no se lanzó como U0. |

**Tiempos observados:** son medidas de pared y de entrenamiento de este
checkout, no un presupuesto ni una extrapolación.

| Bloque | Pared por ejecución | Adam (entrenamiento) | QGD (entrenamiento) |
|---|---:|---:|---:|
| S0 exacto | 14,278 s | 4,025 s | 4,345 s |
| S0 finite-shot | 17,048–17,901 s | 5,512–5,740 s | 5,732–5,928 s |
| G0 | 99,440–102,606 s | 46,686–48,122 s | 46,981–48,558 s |
| U0 válido | 10,789 s | — | 6,171 s |
| U0 fallos | 7,255–10,374 s | — | — |

El total de pared de los 12 JSON válidos fue 531,969 s (≈8,9 min); incluyendo
los tres fallos, 556,924 s (≈9,3 min). G0 dominó el coste. Estas cifras no
permiten extrapolar 30 pasos, otras semillas, otros datasets ni hardware.

**Validez de los artefactos:** los 12 JSON se cargaron con `json.load` sin
errores. Hay 15 logs: 12 asociados a JSON y 3 de fallo. Uno de los logs
correctos (S0, 100 shots, ejecución 101) no contiene la línea `EXIT_CODE`, pero
termina con `Wrote ...` y `WALL_SECONDS=17.901`; el JSON correspondiente existe.
Los tres logs de fallo contienen `EXIT_CODE=1`. No se modificó ningún JSON ni
log.

**Hallazgos operativos y límites:**

1. **La actualización QGD no movió los pesos más allá de la cuantización
   inicial.** Los 12 JSON comparten exactamente el mismo vector final de QGD;
   ese vector coincide con la cuantización del vector inicial a la rejilla de
   32 niveles. La causa derivada del código es que, con `b=5`, `lr=0.1`,
   `weight=[-π,π]` y `grad=[-1,1]`, el delta entero
   `round(lr·g·(2^b−1)/(weight_max−weight_min))` es cero para los 32 códigos
   de gradiente (valor absoluto máximo ≈0,493). Por tanto, el primer paso solo
   cuantiza los pesos y los pasos siguientes no los mueven. No es un fallo de
   ejecución, pero impide interpretar estos screens como una trayectoria de
   aprendizaje QGD. El mismo patrón aparece en los cinco perfiles M5 de dos
   pasos: los cinco JSON declaran `qgd.n_bits=5` y `qgd.learning_rate=0.1`, y
   sus pesos finales coinciden con la cuantización inicial. La sección 29
   registra la comprobación read-only con `b=5/lr=0.2` y la tabla de compromiso
   `b/lr`. Antes de confirmar debe revisarse la resolución de la rejilla y el
   `lr` de QGD, junto con la aceptación metodológica pendiente.
2. **U0 no está completo.** Tres de las cuatro ejecuciones U0 fallaron con
   `RuntimeError: measured QGD gradient register differs from its input code.`,
   lanzado en `decode_update_counts` (`src/quantum_update.py`) a través de
   `quantum_update_step`. Ocurrió en `p_update=0.001` con ejecución 102 y en
   `p_update=0.01` con ejecuciones 101 y 102. El mensaje indica que el registro
   de gradiente medido no coincide con el código de entrada; la causa concreta
   no se ha aislado. Los logs se conservan y no hay JSON para esos fallos.
3. **La única ejecución U0 válida no demuestra inocuidad del ruido.** La
   ejecución `p_update=0.001`, semilla 101, produjo un historial y unos pesos
   finales idénticos a los de S0 con 100 shots y semilla 101; sus
   probabilidades de modo fueron 1,0. Ese valor es estructural bajo el muestreo
   de ruido de Qibo 0.1.11: el canal se aplica una vez por ejecución del
   circuito y los shots comparten la misma trayectoria de Pauli, de modo que no
   demuestra ausencia de ruido (sección 29). Es una observación de una sola
   semilla y dos pasos; no permite generalizar la causa, la tasa de fallo ni el
   efecto de `p_update`.
4. **G0 completó el bloque pareado definido.** Las cuatro ejecuciones
   (`p_model=0.001/0.01` × semillas 101/102, 100 shots, `p_update=0`)
   produjeron JSON con test no evaluado. No hubo fallos de ejecución en G0.
   Los checkpoints y las pérdidas no se interpretan como comparación de
   rendimiento.
5. **S0 quedó incompleto respecto al recuento previsto.** Hay una referencia
   exacta y seis pares finite-shot. La segunda referencia exacta no se generó;
   aunque la ejecución exacta es determinista, no se ha decidido si basta con
   una repetición o si debe documentarse su reutilización.
6. **Alcance limitado.** Los screens usan un dataset (Iris angular), una
   inicialización, dos semillas de ejecución y dos pasos; no evalúan test, no
   estiman convergencia ni generalización y no representan hardware. El ruido
   es una sensibilidad de simulador local, no una calibración de QPU. Los
   tiempos no autorizan un presupuesto.

**Consecuencia y siguiente puerta:** el trabajo se detiene antes de las
confirmaciones. Antes de S1, G1, G2 o U1 deben revisarse con el tutor la regla
modular y los parámetros de rejilla/`lr`; aislarse o acotarse el fallo de U0;
decidirse la segunda referencia exacta de S0 y si se reutiliza S0 para
`p_update=0`; y presentarse al responsable el coste observado y el alcance de
las confirmaciones. Estos resultados no autorizan ninguna confirmación. No se
modificaron `main.py`, los JSON históricos ni los resultados M5; no se hizo
commit. `git diff --check` no detectó errores; Git emitió solo avisos LF/CRLF.

## 29. Validación actual y decisiones pendientes antes de un commit (10 de noviembre de 2026)

**Estado — Validado unitariamente; screens parciales; no evaluación de rendimiento:** esta sección cierra la revisión del árbol de trabajo sucio basado en HEAD `164eaee2b05ed53f9950afe8a272ca91f5e292df`. No se ha hecho commit, no se han lanzado experimentos nuevos y no se han modificado los JSON ni los logs de `results/quantum-training/`.

**Validación ejecutada:** el 10 de noviembre de 2026 se ejecutaron las comprobaciones siguientes sobre el árbol de trabajo actual:

```text
qml_env\Scripts\python.exe -m unittest discover -s tests -v
Ran 92 tests in 9.774s
OK

qml_env\Scripts\python.exe -m compileall -q src experiments tests
OK

git diff --check
OK (solo avisos LF/CRLF)
```

Los 17 JSON de `results/quantum-training/` se cargaron con `json.load` sin errores. Los 12 screens declaran `configuration.screening=true` y `methods.*.test.evaluated=false`; los cinco perfiles M5 no declaran evaluación de test. Hay 15 logs: 12 asociados a JSON válidos y 3 de fallos de U0. Los 12 screens comparten exactamente la misma matriz final de QGD (dos pasos por cuatro pesos) y esa matriz coincide con la cuantización del vector inicial en la rejilla de 32 niveles. Los cinco JSON M5 declaran `qgd.n_bits=5` y `qgd.learning_rate=0.1`, y sus pesos finales también coinciden con la cuantización inicial.

**Aclaración de conteos:** 73 es la suite completa registrada el 7 de noviembre de 2026 (sección 25); 88 es la suma de los seis grupos dirigidos del 8 de noviembre (sección 27), no una suite completa; 92 es la suite completa actual. Los tres conteos son históricos y no se contradicen.

**Diagnóstico del no-op de QGD:** con `b=5`, `lr=0.1`, `weight=[-π,π]` y `grad=[-1,1]`, el delta entero

```text
round(lr · g · (2^b − 1) / (weight_max − weight_min))
```

es cero para los 32 códigos de gradiente. El máximo absoluto es `0.4934 < 0.5`, de modo que `rint` devuelve cero en todos los casos; el primer código se movería solo por encima de `lr = π/31 ≈ 0.10134` (en el valor exacto, `np.rint(0.5)` redondea a cero por redondeo al par). No es un fallo de ejecución: es un problema de resolución de la rejilla y del `lr`. Una comprobación read-only con `QGDConfig(n_bits=5, learning_rate=0.2)` y pesos/gradientes de prueba coincidió con `classical_grid_update` y movió 16 de los 32 códigos de gradiente; con `b=5/lr=0.1` movió 0 de 32. No se generaron resultados nuevos.

La tabla siguiente resume el compromiso entre resolución, número de códigos que se mueven y coste lógico. Las cotas de puertas son `gate_count_per_circuit_upper_bound` de `estimate_update_resources`: instrucciones lógicas de Qibo sin transpilar, no coste de hardware.

| Configuración | Qubits por update | Códigos de gradiente que mueven | Cota de puertas |
|---|---:|---:|---:|
| `b=5`, `lr=0.1` | 10 | 0/32 | 171 |
| `b=5`, `lr=0.15` | 10 | 12/32 | 651 |
| `b=5`, `lr=0.2` | 10 | 16/32 | 811 |
| `b=4`, `lr=0.5` | 8 | 10/16 | 303 |

`b=6` no está permitido por `QGDConfig` (solo admite 2–5). `b=4/lr=0.5` usa menos qubits, pero una rejilla más gruesa. Ninguna fila es una recomendación aprobada: la configuración oficial y la aceptación de la regla modular siguen pendientes del tutor.

**Limitación del muestreo de ruido en Qibo 0.1.11:** el canal depolarizante no se muestrea por shot. `NumpyBackend.apply_channel` (`qml_env/Lib/site-packages/qibo/backends/numpy.py:250`) sortea cada operador de Pauli una vez por llamada a `apply_channel`, y `execute(nshots=100)` llama al canal una vez por ejecución del circuito. Por tanto, los 100 shots comparten la misma trayectoria aleatoria de Pauli: los shots no promedian realizaciones de ruido. La comprobación empírica con un circuito de un qubit (`X`, `DepolarizingChannel((0,), lam=0.5)`, medida, 100 shots, semillas 0–9) produjo 100 copias del mismo resultado en cada semilla; 3 de 10 semillas invirtieron el bit. En consecuencia, una `mode_probability=1.0` no demuestra ausencia de ruido. La única ejecución U0 válida es una trayectoria sin error que afecte al registro de gradiente; los tres fallos son trayectorias con error. Esto aplica tanto a `p_model` como a `p_update` y no constituye un modelo de errores por shot ni una calibración de hardware.

**Decisiones pendientes antes del commit:**

1. `requirements.txt` — **resuelto:** HEAD está en UTF-16LE (122 líneas); el árbol actual está en UTF-8 (51 líneas) y elimina el stack Jupyter y `torchvision`. El responsable aprobó explícitamente el 10 de noviembre de 2026 incluir este cambio en el commit de cierre. Git sigue viendo el archivo antiguo como binario; el cambio es intencionado y queda registrado aquí.
2. `load_binary_iris()` — **resuelto:** se eliminó de `src/datasets.py` en el árbol actual. No se encontraron usos internos, pero era un loader público retrocompatible; es una ruptura de API. Se decide documentar la ruptura como intencionada y no restaurar la función en esta revisión; los consumidores deben usar `load_binary_iris_splits()`.
3. QGD `b/lr` y regla modular: la configuración `b=5/lr=0.1` es un no-op y la variante modular no es el clipping propuesto. La decisión corresponde al tutor.
4. Política de U0: hay tres fallos de decodificación y una sola ejecución válida. Debe decidirse si se mantiene el rechazo estricto, se relaja la comprobación del registro de gradiente, se reintenta o se registra el fallo como resultado ruidoso.
5. S0: falta la segunda referencia exacta prevista. Debe generarse o documentarse la decisión de reutilizar la existente.

**Documentos de planificación y memoria:** `docs/quantum-training-plan.md` conserva su encabezado histórico del 7 de noviembre y M1-M4; no se editó. Para el estado vigente prevalecen este log y el código. La memoria LaTeX conserva marcadores `\pending{}` y una bibliografía sin referencias verificadas; no se editó y no debe presentarse como finalizada.

**Veredicto de commit:** el árbol es apto para un commit que se presente como «implementación M1-M5 + screens parciales», no como evaluación de rendimiento. Condiciones: (i) no presentar S0/G0/U0 como convergencia, generalización o rendimiento; (ii) incluir `requirements.txt` con la aprobación explícita registrada; (iii) mantener documentada la ruptura de `load_binary_iris()` como intencionada; (iv) conservar los JSON, los logs y los fallos U0; (v) mantener `b/lr` y la política de U0 como decisiones pendientes del tutor. No se ha hecho commit.
