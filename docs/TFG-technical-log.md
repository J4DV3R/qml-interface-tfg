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
`results/qcnn-validation.json`. Sirve como validación técnica del flujo, no
como resultado definitivo de la memoria.

## 18. Resultado de la prueba intermedia y siguiente experimento

**Fecha:** 5 de octubre de 2026.

Se ejecutó una prueba intermedia con 3 épocas, Adam, tasa de aprendizaje
`0.03`, 50 y 100 shots, ruido `0.0` y `0.01`, y dos semillas en las
condiciones muestreadas. El archivo
`results/iris-qcnn-medium.json` contiene 9 ejecuciones y 5 resúmenes.

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
`results/qcnn-convergence.json`, evitando que el progreso dependa de que
termine todo el experimento.

## 19. Batería final propuesta

La prueba `results/qcnn-noisy-convergence-5epochs.json` comparó una ejecución
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
