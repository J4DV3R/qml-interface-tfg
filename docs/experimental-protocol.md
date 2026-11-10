# Protocolo experimental base de la QCNN

Este documento define la primera matriz experimental reproducible para cerrar
la validación de la QCNN. No es todavía el protocolo universal de todos los
modelos QML: fija una referencia común y deja explícitos los parámetros
específicos de la QCNN.

## Objetivos

1. Comprobar que el pipeline completo funciona.
2. Medir el efecto de los shots y del ruido depolarizante.
3. Estimar la estabilidad entre semillas.
4. Contextualizar la QCNN frente a una regresión logística.
5. Conservar resultados reproducibles para la memoria del TFG.

## Datos y particionado

- Dataset principal: Iris binario, `setosa` frente a `versicolor`.
- Características: las cuatro características originales.
- Partición: 70% entrenamiento, 15% validación y 15% test.
- Semilla del particionado: `42`.
- Estratificación: activada.
- Normalización: min-max ajustada únicamente con entrenamiento.
- Etiquetas: conversión al contrato `-1/+1`.

La partición se mantiene fija entre condiciones para que las diferencias
observadas procedan de la ejecución y no de distintos conjuntos de muestras.

## Configuración de la QCNN

- 4 qubits.
- Codificación angular.
- Arquitectura de pooling lógico `4 -> 2 -> 1`.
- Una salida basada en expectativa Pauli Z.
- Gradientes mediante parameter-shift.
- Optimizador principal: Adam.
- Épocas: 30.
- Tasa de aprendizaje: `0.03`.
- Pérdida: MSE contra etiquetas `-1/+1`.
- Pesos iniciales idénticos en todas las condiciones.

SGD queda disponible como control metodológico, pero no forma parte del
barrido base. Comparar optimizadores será un experimento separado si los
resultados justifican estudiarlo.

## Condiciones de ejecución

Se ejecutan estas condiciones:

1. simulación exacta;
2. shots `100`, ruido `0.0`;
3. shots `500`, ruido `0.0`;
4. shots `1000`, ruido `0.0`;
5. shots `100`, ruido `0.01`;
6. shots `500`, ruido `0.01`;
7. shots `1000`, ruido `0.01`;
8. shots `100`, ruido `0.05`;
9. shots `500`, ruido `0.05`;
10. shots `1000`, ruido `0.05`.

Cada condición muestreada se repite con las semillas de ejecución `1`, `2` y
`3`. La simulación exacta se ejecuta una sola vez porque es determinista con
los pesos fijados. En cada repetición muestreada la semilla se inicializa una
vez antes del entrenamiento y la secuencia aleatoria continúa durante todas
las épocas y evaluaciones.

## Evaluación

- `train`: se utiliza para actualizar los pesos.
- `validation`: se utiliza para observar pérdida y métricas durante el
  experimento, sin actualizar los pesos.
- `test`: se utiliza para la evaluación final.

Se registran:

- pérdida inicial y final de entrenamiento;
- pérdida de validación;
- accuracy, precision, recall y F1 en validación y test;
- tiempo de entrenamiento y evaluación;
- configuración completa y semilla.

Para cada condición se calculan media y desviación estándar poblacional de las
repeticiones. Los resultados individuales y agregados se guardan en JSON.

## Baseline

La regresión logística se entrena con el mismo `train` y se evalúa con el
mismo `test`, partición, normalización y métricas. Es una referencia clásica
externa al motor QML, no una implementación de modelo cuántico.

## Experimentos posteriores

Después de esta matriz base se podrán estudiar por separado:

- otra pareja de clases de Iris;
- un segundo dataset binario;
- SGD frente a Adam;
- otras codificaciones;
- más modelos de ruido;
- clasificación multiclase;
- minibatches.

No se combinan estos factores en el barrido base para conservar la
interpretabilidad de los resultados.

## Protocolo propuesto para orientar la ruta QGD a QPU

Esta sección documenta una planificación, no una evaluación experimental. El
objetivo acordado con el tutor es orientar la arquitectura a hardware cuántico
real y eliminar la operación aritmética clásica de actualización de pesos,
llevándola a circuitos en la medida de lo posible. El backend no está
especificado; la interfaz y la configuración deberán ser genéricas y ajustarse
a los recursos disponibles. La actualización deberá probarse primero para un
parámetro y, solo después de verificar su funcionamiento, escalarse al conjunto
de parámetros.

El baseline de referencia es la QCNN actual de cuatro qubits con codificación
angular o por fase. M1-M5 están implementados y validados unitariamente en el
simulador Qibo; no se han ejecutado en hardware. M4 utilizaba una actualización
por estado-vector exacto y `argmax`. M5 implementa una interfaz de ejecución
medida (`CircuitExecutor`) y una actualización aritmética reversible basada en
circuitos controlled-X, cuyos resultados se leen mediante counts y se
decodifican modalmente. El modo `--measured-execution` fuerza ejecución medida
del simulador incluso con ruido cero; sin executor y con `shots=None`, el
gradiente y las evaluaciones de la QCNN son exactos y sin ruido, mientras que
cada actualización QGD se ejecuta mediante el circuito medido del simulador con
un shot determinista. El executor inyectado requiere shots finitos para las
operaciones medidas. Se han generado cinco perfiles
exploratorios cortos en JSON; no constituyen una evaluación de convergencia o
rendimiento. Esta evolución no proporciona adaptador de proveedor ni soporte de
hardware: no hay selección de backend, transpilación/routing genérico o
específico del dispositivo ni resultados QPU.

La actualización aritmética se ejecuta en circuitos, pero el schedule, la
preparación de entradas, la codificación y saturación del gradiente, la lectura
y decodificación de counts, y la pérdida, acumulación de gradientes y control
del entrenamiento mantienen procesamiento clásico. Por tanto, no se describe
como entrenamiento totalmente cuántico ni como hardware-ready. El
`ResourceProfile` declara máximos de qubits y shots, base de puertas y
conectividad; estas dos últimas capacidades todavía no se validan ni se
transpilan genéricamente. Las estimaciones de instrucciones y profundidad son
lógicas, sin transpilation.

### Datos y control de selección

Se proponen dos datasets: Iris binario, con 100 muestras tras filtrar dos
clases, y Breast Cancer Wisconsin, con 569 muestras y reducción a cuatro
componentes mediante PCA. Para ambos se conservará el split oficial
estratificado 70/15/15 y el preprocesamiento se ajustará únicamente con
`train`. Un subset piloto, si resulta necesario por coste, se extraerá solo de
`train`; se registrarán sus índices y tamaño efectivo. No se alterará el split
ni se utilizará `test` para seleccionar datos, pesos, hiperparámetros o
presupuestos.

### Secuencia de evaluación propuesta

1. **Validado unitariamente:** pruebas funcionales de los circuitos de
   actualización para un peso y para todos los pesos; la validación es local y
   no implica compatibilidad QPU.
2. **Pendiente:** simulación exacta sin ruido para estudiar convergencia.
3. **Perfilado exploratorio ejecutado:** finite-shots y ruido cero para Iris
   angular/fase y Breast Cancer angular, solo dos pasos y una semilla.
4. **Perfilado parcial ejecutado:** sensibilidad `p=0.001` para Iris
   angular/fase; no se han perfilado las condiciones ruidosas de Breast Cancer.

En la nueva ruta, el ruido disponible es una transformación local de canal
depolarizante de una qubit con probabilidad `p` después de cada puerta lógica,
aplicada a cada qubit tocado tanto en la QCNN como en la actualización. Es una
prueba de sensibilidad del simulador, no un modelo calibrado ni realista de
hardware. Con `--measured-execution` se pueden comparar shots finitos con ruido
cero y con `p>0` en el mismo modo de medición. `--shots exact` no se admite con
ruido o ejecución medida. Cuando se seleccione un dispositivo, se ejecutará
primero un circuito
smoke y, si es satisfactorio, una QCNN y una actualización reducidas antes de
plantear ejecuciones mayores.

### Presupuesto y coste

Como presupuesto exploratorio se proponen 30, 60 y 100 pasos; no son valores
óptimos establecidos. Se usarán inicialmente 2--3 semillas exploratorias. Solo
después de estimar coste y variabilidad se decidirá si procede una confirmación
con al menos cinco semillas. El protocolo confirmatorio debe establecer primero
la convergencia exacta y después separar codificación, shots y ruido, sin cruzar
plenamente todos los factores. Los perfiles M5 ya ejecutados son únicamente
perfiles de flujo/coste y no sustituyen esas etapas.

La contabilidad estructural disponible para el motor de cuatro qubits y una
salida es de 16 apariciones paramétricas por muestra y gradiente. Con
parameter-shift, esto implica 32 ejecuciones desplazadas y una ejecución base
por muestra y gradiente, es decir, 33 ejecuciones por muestra. Para
aproximadamente 70 muestras de entrenamiento de Iris, son 2.310 ejecuciones
por gradiente; para aproximadamente 398 muestras de Breast Cancer, 13.134.
Estas son estimaciones estructurales por gradiente. Los perfiles M5 de dos pasos
registran el total de gradientes, evaluación y updates, pero siguen siendo
estimaciones lógicas, no mediciones instrumentadas de jobs ni de hardware. El
perfil de dos pasos se eligió para incluir tanto el gradiente inicial como uno
posterior. Antes de autorizar presupuestos mayores se debe revisar el coste
observado y el presupuesto extrapolado; no se ejecutarán matrices grandes sin
autorización específica.

### Variables, métricas y registro

La medida primaria serán las trayectorias de pérdida de `train` y `validation`,
interpretadas con criterios definidos antes de ejecutar. Como medidas
secundarias se registrarán accuracy, precision, recall y F1, tiempo, número de
circuit submissions y número de shots, errores y jobs. Circuit submissions y
shots son cantidades distintas y se informarán por separado. El checkpoint se
seleccionará únicamente con `validation`; en una evaluación confirmatoria,
`test` se consultará una sola vez al final.

Cada JSON y, si se persiste, su log deberán registrar backend, semillas del
dataset, de inicialización y de ejecución, codificación, ruido, shots, tamaño
efectivo de `train`, presupuesto de pasos, revisión del código, estimaciones de
circuitos y recursos, así como errores o interrupciones. Los cinco perfiles
M5 tienen procedencia registrada dentro del JSON, pero no un log independiente y
no constituyen evaluación de convergencia o rendimiento.

## Validación previa ejecutada

Antes del barrido base se ejecutó una batería reducida para verificar el flujo:

- shots: `50`;
- ruido: `0.0` y `0.01`;
- semillas de ejecución: `1` y `2`;
- épocas: `2`;
- 5 ejecuciones totales: una exacta y dos para cada condición muestreada.

El resultado completo se conserva en
`results/initial-hybrid-adam/qcnn-validation.json`. Esta batería confirma el contrato técnico,
pero no debe utilizarse como resultado científico final por su reducido número
de épocas, shots y repeticiones.

## Resultado de la prueba intermedia

Se ejecutó una prueba de mayor tamaño con:

- 3 épocas;
- Adam y tasa de aprendizaje `0.03`;
- una condición exacta;
- 50 y 100 shots;
- ruido `0.0` y `0.01`;
- dos semillas para las condiciones muestreadas.

El resultado se conserva en `results/initial-hybrid-adam/iris-qcnn-medium.json` y contiene 9
ejecuciones y 5 condiciones agregadas. La accuracy de test quedó entre
`0.50` y `0.567`, mientras que la regresión logística obtuvo `1.0` en la
misma partición. Estos valores no permiten evaluar todavía la convergencia,
porque solo se utilizaron 3 épocas y el test contiene 15 muestras.

La ejecución con ruido `0.01` tardó aproximadamente 162--164 segundos por
repetición, frente a unos 7 segundos sin ruido. Por tanto, el ruido
depolarizante combinado con shots y parameter-shift es el cuello de botella
principal. El barrido definitivo completo se pospone hasta comprobar primero
la convergencia en simulación exacta.

## Experimento diagnóstico de convergencia

El siguiente experimento se limita a la ejecución exacta y compara el número
de épocas:

- épocas: `3`, `10`, `20` y `30`;
- una partición fija de Iris;
- una inicialización fija de pesos;
- Adam;
- tasa de aprendizaje `0.03`;
- una ejecución por configuración;
- sin shots ni ruido.

El objetivo es observar si la pérdida de entrenamiento y la pérdida/accuracy
de validación mejoran con las épocas. Se estima que debe durar menos de
20 minutos, porque evita las ejecuciones ruidosas. No pretende comparar shots
ni ruido; solo decidir si tiene sentido lanzar posteriormente el barrido
experimental completo.

Si la validación mejora de forma consistente, se podrá fijar el número de
épocas y ejecutar una matriz de ruido más pequeña. Si permanece estancada,
habrá que revisar la tasa de aprendizaje, el optimizador, la codificación o
la arquitectura antes de aumentar el coste experimental.

El diagnóstico se ejecutará con:

```powershell
python -m experiments.diagnose_convergence `
  --output results/reproductions/legacy/qcnn-convergence.json
```

## Batería focalizada para cerrar la QCNN

Para evitar un barrido combinatorio, se utilizará
`experiments/run_final_qcnn.py`. El nuevo adaptador de Breast Cancer
Wisconsin tiene 569 muestras y ajusta la estandarización, el PCA de cuatro
componentes y la normalización min-max únicamente con entrenamiento. Cada
componente resultante se codifica en un qubit; por defecto se conserva la
arquitectura `4 -> 2 -> 1`.

Los resultados originales de esta etapa están congelados en
`results/initial-hybrid-adam/`. Los comandos siguientes escriben en
`results/reproductions/legacy/` para que una repetición no sobrescriba los
archivos históricos.

Los comandos recomendados desde la raíz del repositorio son:

```powershell
# B: convergencia exacta en el dataset mayor
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 10 20 30 --seeds 19 `
  --output results/reproductions/legacy/breast-cancer-qcnn-convergence.json

# C: variabilidad entre tres inicializaciones, 30 épocas exactas
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 30 --seeds 19 23 29 `
  --output results/reproductions/legacy/breast-cancer-qcnn-seeds.json

# D-angular: referencia reproducible para la codificación angular
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 30 --seeds 19 23 29 `
  --output results/reproductions/legacy/breast-cancer-qcnn-angle.json

# D-fase: misma configuración cambiando solo la codificación
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding phase `
  --epochs 30 --seeds 19 23 29 `
  --output results/reproductions/legacy/breast-cancer-qcnn-phase.json

# E: robustez y coste con dos semillas; usar solo después de validar B
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 5 --seeds 1 2 --shots 50 --noise 0.01 `
  --output results/reproductions/legacy/breast-cancer-qcnn-noisy.json
```

El experimento F (QCNN frente al baseline) queda incluido en cada JSON:
`baseline` usa exactamente la misma partición y las mismas características
reducidas. No se debe interpretar el baseline como parte del motor QML.
Los experimentos D-angular y D-fase solo difieren en la codificación. La
semilla de inicialización se conserva por condición para que la comparación
no mezcle cambios de arquitectura con cambios de pesos iniciales.

El experimento E no debe ampliarse automáticamente a más shots o épocas: el
diagnóstico previo mostró que `50` shots con ruido `0.01` tarda unos cuatro
minutos y medio por ejecución de cinco épocas en Iris. El resultado sirve
para documentar robustez y coste, no para optimizar exhaustivamente el
entrenamiento ruidoso.

### Resultado de E en Breast Cancer

El experimento se ejecutó con 5 épocas, 50 shots, ruido `0.01` y semillas `1`
y `2`. La QCNN obtuvo:

| Semilla | Pérdida inicial | Pérdida final | Val. accuracy | Test accuracy | Test F1 | Tiempo |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.028 | 0.915 | 0.706 | 0.744 | 0.817 | 26,0 min |
| 2 | 1.068 | 0.944 | 0.671 | 0.616 | 0.686 | 26,1 min |

El baseline de regresión logística obtuvo `0.907` de accuracy y `0.931` de
F1 en el mismo test de 86 muestras. La QCNN ruidosa queda por debajo del
baseline en media de accuracy (`0.680` frente a `0.907`), aunque la primera
semilla supera al baseline en accuracy por la variabilidad del muestreo y no
debe interpretarse como una ventaja estable.

Tras repetir la ejecución con la suspensión automática desactivada, ambas
semillas tardaron aproximadamente 26 minutos. Estos son tiempos de reloj
válidos para esta máquina y configuración; no deben extrapolarse directamente
a otros equipos o a más shots y épocas.

## Resultados exactos sobre Breast Cancer

La batería exacta se completó con cuatro qubits, PCA a cuatro componentes,
partición 70/15/15, Adam y tasa de aprendizaje `0.03`.

### Convergencia

| Épocas | Pérdida final | Val. accuracy | Test accuracy | Test F1 | Tiempo |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.894 | 0.694 | 0.651 | 0.776 | 117 s |
| 20 | 0.685 | 0.753 | 0.744 | 0.825 | 230 s |
| 30 | 0.590 | 0.812 | 0.826 | 0.878 | 341 s |

La pérdida y las métricas mejoran de forma consistente al aumentar las
épocas. Se fija provisionalmente 30 épocas para las comparaciones posteriores.

### Variabilidad y codificación

Con 30 épocas, las tres semillas angulares obtuvieron accuracies de test
`0.826`, `0.872` y `0.849`, con media aproximada `0.849`. La codificación por
fase obtuvo `0.826`, `0.872` y `0.895`, con media aproximada `0.864`. El
baseline de regresión logística obtuvo `0.907`.

La fase supera ligeramente a la angular en media, pero la diferencia es
pequeña y no debe tratarse como concluyente con solo tres inicializaciones.
Ambas codificaciones quedan por debajo del baseline en promedio y muestran
que la elección de representación afecta al entrenamiento.

Los tiempos exactos fueron aproximadamente 5,6 minutos por ejecución angular
y 6,0 minutos por ejecución en fase. La fase es algo más costosa, pero la
diferencia no cambia el criterio de cierre.

## Perfiles exploratorios M5

El 7 de noviembre de 2026 se ejecutaron perfiles de dos pasos con 100 shots,
partición completa, semilla de dataset `42`, inicialización `19` y ejecución
`101`. Las condiciones sin ruido usaron `--measured-execution`, por lo que el
gradiente inicial también se muestreó; las condiciones con `p=0.001` activaron
el executor medido mediante el ruido. El tiempo es `training.elapsed_seconds`
del JSON. Los conteos son estimaciones estructurales de ejecuciones lógicas,
no trabajos enviados a un proveedor.

| Dataset | Codificación | `p` | Tiempo para 2 pasos | Ejecuciones estimadas de entrenamiento | Evaluación test estimada | JSON |
|---|---|---:|---:|---:|---:|---|
| Iris binario | Angular | 0 | 5,68 s | 4.806 | 15 | [`m5-profile-iris-angle-measured-shots100-noise0-2steps.json`](../results/quantum-training/m5-profile-iris-angle-measured-shots100-noise0-2steps.json) |
| Iris binario | Angular | 0,001 | 46,84 s | 4.806 | 15 | [`m5-profile-iris-angle-measured-shots100-noise0001-2steps.json`](../results/quantum-training/m5-profile-iris-angle-measured-shots100-noise0001-2steps.json) |
| Iris binario | Fase | 0 | 5,94 s | 4.806 | 15 | [`m5-profile-iris-phase-measured-shots100-noise0-2steps.json`](../results/quantum-training/m5-profile-iris-phase-measured-shots100-noise0-2steps.json) |
| Iris binario | Fase | 0,001 | 51,19 s | 4.806 | 15 | [`m5-profile-iris-phase-measured-shots100-noise0001-2steps.json`](../results/quantum-training/m5-profile-iris-phase-measured-shots100-noise0001-2steps.json) |
| Breast Cancer | Angular | 0 | 31,49 s | 27.250 | 86 | [`m5-profile-breast-cancer-angle-measured-shots100-noise0-2steps.json`](../results/quantum-training/m5-profile-breast-cancer-angle-measured-shots100-noise0-2steps.json) |

Los perfiles son de una sola inicialización y solo dos pasos. Aunque el JSON
contiene métricas de test tras seleccionar por validation, estas cifras no se
usan para comparar rendimiento ni permiten inferir convergencia. El ruido
`p=0.001` es una perturbación local de sensibilidad del simulador, no una
calibración de hardware. La procedencia de cada ejecución registra el comando,
`HEAD` `164eaee2b05ed53f9950afe8a272ca91f5e292df` y el estado sucio del árbol;
no existe un log independiente persistido.

Como guía de coste, una extrapolación lineal simple desde el tiempo observado
para dos pasos produce los siguientes tiempos de entrenamiento para 30/60/100
pasos:

| Condición perfilada | 30 pasos | 60 pasos | 100 pasos |
|---|---:|---:|---:|
| Iris angular, `p=0` | 1,4 min | 2,8 min | 4,7 min |
| Iris fase, `p=0` | 1,5 min | 3,0 min | 4,9 min |
| Iris angular, `p=0.001` | 11,7 min | 23,4 min | 39,0 min |
| Iris fase, `p=0.001` | 12,8 min | 25,6 min | 42,7 min |
| Breast Cancer angular, `p=0` | 7,9 min | 15,7 min | 26,2 min |

Son proyecciones orientativas, no tiempos medidos para esos presupuestos; asumen
escalado lineal y una sola semilla. Breast Cancer con ruido y codificación por
fase todavía no se perfilaron. Ejecutar los cinco perfiles disponibles en los
tres presupuestos con dos o tres semillas supondría, bajo esa misma aproximación,
unas 7,5–11,2 horas de entrenamiento, sin incluir condiciones de Breast Cancer
no perfiladas, la evaluación final de test ni overhead de inicio. La estimación
no autoriza por sí sola esa matriz; cualquier ampliación requiere aprobación
del presupuesto y alcance.

## Protocolo propuesto: comparación de QGD con entrenamiento híbrido

**Estado: propuesto; no ejecutado ni aprobado.** La comparación principal será
QGD frente a Adam. SGD podrá estudiarse como comparador opcional separado. El
entrenamiento híbrido con PyTorch sigue implementado en `src/hybrid_model.py`,
`src/training.py` y disponible mediante el runner legacy. Adam no es solo un
resultado histórico: `results/initial-hybrid-adam/` conserva resultados, pero
la ruta sigue siendo funcional. La caracterización antigua de Adam como solo
histórico en `docs/quantum-training-plan.md` requiere armonización posterior;
ese archivo no se modifica aquí.

Se propone comparar la misma QCNN de cuatro qubits, pooling lógico `4 -> 2 ->
1`, misma codificación dentro de cada contraste, split estratificado 70/15/15,
semilla de dataset `42` y preprocesamiento ajustado solo con `train`. Las
inicializaciones deberán ser idénticas y la selección de checkpoint se basará
solo en `validation`; `test` queda reservado para la evaluación final. Los
runners actuales aún no garantizan pesos iniciales compartidos ni un protocolo
de checkpoint comparable: ambos requisitos deben resolverse y comprobarse
antes de ejecutar.

El primer bloque será de simulación sin ruido. Finite-shots y ruido se
estudiarán después, por separado, porque la semántica de ejecución y los
modelos de ruido de las dos rutas aún no están armonizados.

### Secuencia de datasets y definición de QGD

La secuencia propuesta comienza con Iris binario como piloto rápido y continúa
con Breast Cancer Wisconsin, reducido a cuatro componentes mediante PCA, como
confirmación principal. En ambos casos se conservarán las condiciones de
partición y preprocesamiento indicadas arriba. Esta secuencia no fija un
presupuesto confirmatorio ni constituye una decisión de ejecución.

La variante QGD que se propone evaluar representa cada peso y gradiente en un
registro de `b` bits sobre una rejilla finita. Se cuantizan ambos valores a sus
rejillas configuradas; al codificar un gradiente fuera del intervalo
representable, este se satura (clamp) al extremo correspondiente. El
desplazamiento entero se obtiene del gradiente cuantizado y la tasa de
aprendizaje, y el código del peso se actualiza mediante resta modular, mientras
se conserva el registro del gradiente. Esta regla no es clipping de pesos:
puede producir wrap-around del código de peso en la rejilla modular. Los
parámetros concretos de la rejilla y `b` deberán quedar registrados para cada
ejecución.

La comparación se expresará por pasadas completas sobre `train` y, por
separado, por tiempo, conteos lógicos de circuitos y shots. Épocas/pasadas no
equivalen a pasos o evaluaciones de COBYLA/SPSA, por lo que no se igualarán por
número de iteraciones. Se registrarán pérdidas de `train` y `validation`,
accuracy, precision, recall y F1, junto con configuración, semillas,
revisión y conteos. No se usará `test` para seleccionar pesos, hiperparámetros
ni duración. Se informarán resultados individuales, dispersión e intervalos
adecuados; diferencias pequeñas con pocas semillas no serán concluyentes.
La pérdida propuesta es MSE frente a etiquetas `-1/+1`. Los hiperparámetros
específicos de cada optimizador deberán fijarse o seleccionarse usando solo
`train` y `validation` antes de consultar `test`. Las tasas de aprendizaje
distintas no se considerarán equivalentes por tener el mismo valor numérico ni
se compararán como si fueran el mismo hiperparámetro.

**Puerta de parada:** no lanzar la comparación hasta cerrar o explicitar que
la regla modular QGD es la variante evaluada. Esto no implica afirmar su
aceptación por el tutor. La propuesta tampoco autoriza un presupuesto de
ejecución.

Como proxy únicamente, los perfiles M5 de dos pasos, 100 shots, semilla de
dataset `42`, inicialización `19` y ejecución `101` registran 31,49 s para
Breast Cancer angular `p=0`
([JSON](../results/quantum-training/m5-profile-breast-cancer-angle-measured-shots100-noise0-2steps.json))
y 5,68 s para Iris angular `p=0`
([JSON](../results/quantum-training/m5-profile-iris-angle-measured-shots100-noise0-2steps.json)).
La extrapolación lineal a 30 pasos sería aproximadamente 7,9 min y 1,4 min,
respectivamente; es muy incierta, basada en perfiles de una ejecución de dos
pasos y no equivale a una comparación con Adam. No se extrapola el tiempo
combinado ni se autoriza presupuesto con estas cifras.

## Propuesta: Matriz de ruido y shots

**Estado: propuesta; no ejecutada ni aprobada.** Esta sección diseña un bloque
escalonado de ruido y muestreo para la comparación QGD–Adam. Las condiciones
descritas no son resultados ni autorizan la ejecución de matrices grandes.

### Convención de ruido y separación de parámetros

Se propone definir el ruido como un canal depolarizante local de un qubit,
parametrizado por una probabilidad `p` en cada aplicación, no por circuito.
Después de cada puerta lógica elegible de la QCNN (`RY`, `RZ` o `CNOT`) se
insertaría un canal independiente en cada qubit tocado; las mediciones quedan
excluidas. La convención se mantendría igual en los circuitos de predicción,
parameter-shift y evaluación de ambos métodos. Esta definición no incluye ruido
de lectura, amplitude damping, phase damping ni crosstalk, y `p` no representa
una calibración de QPU.

La ruta legacy de `src/execution.py` y la transformación M5 perteneciente a
`experiments/run_quantum_training.py` conservan convenciones distintas y no
deben tratarse como los runners de la nueva comparación emparejada. Esta última
usa `experiments/compare_qgd_adam.py` y un executor común para los circuitos
QCNN de Adam y QGD, con canales depolarizantes locales tras `RY`, `RZ` y `CNOT`
en cada qubit tocado, excluidas las mediciones. `p_model` se aplica a los
circuitos QCNN, incluidos parameter-shift, predicción y evaluaciones de
train/validation/test. Un `update_executor` separado aplica `p_update` solo a
los circuitos aritméticos QGD; su transformación cubre todas las puertas no
medición y, por tanto, tiene una cobertura distinta. No se debe interpretar
`p_model` y `p_update` como el mismo canal físico. Con `p=0` y shots finitos
puede utilizarse ejecución con counts sin canal. `shots=None` solo se admite
cuando `p_model=p_update=0`: las evaluaciones y gradientes QCNN son exactos,
mientras que el update QGD continúa usando un shot determinista. Son
configuraciones de sensibilidad en simulación local, no ejecución ni calibración
de hardware.

**Limitación del muestreo de ruido (Qibo 0.1.11):** el canal depolarizante se
muestrea una vez por ejecución del circuito, no por shot.
`NumpyBackend.apply_channel` sortea cada operador de Pauli una vez por llamada y
`execute(nshots=...)` llama al canal una vez; los shots comparten la misma
trayectoria. Por tanto, los shots no promedian realizaciones de ruido y una
`mode_probability=1.0` no demuestra ausencia de ruido. Esto afecta a `p_model` y
`p_update` y se detalla en la sección 29 del log técnico.

### Runner comparativo implementado y validado

`experiments/compare_qgd_adam.py` implementa una comparación emparejada que
inicializa un único vector de pesos NumPy, lo cuantiza a `float32` y lo promueve
a `float64`; comprueba que el parámetro `float32` de Adam lo almacena sin
pérdida y entrega valores numéricamente idénticos a QGD. Adam mantiene una
instancia del optimizador entre épocas. Ambos métodos seleccionan checkpoint
por menor MSE de validation. El modo `--screening` omite el acceso y evaluación
de test y fija exactamente dos épocas/pasos (dos por defecto); el modo normal
usa 30 por defecto. `--qgd-only` omite Adam, requiere `p_model=0` y se reserva
para estudiar ruido de actualización: no es una comparación Adam–QGD. Las
salidas no sobrescriben rutas existentes ni resultados históricos.

El código y las pruebas dirigidas están validados; la suite completa actual pasó
92 pruebas el 10 de noviembre de 2026 (sección 29 del log). Los screenings S0, G0 y U0
se ejecutaron parcialmente el 8 de noviembre de 2026; el inventario, los tiempos,
los fallos y las limitaciones se registran en la sección 28 de
`docs/TFG-technical-log.md`. Los 12 JSON válidos no constituyen una evaluación
de rendimiento ni de convergencia y no autorizan confirmaciones. La revisión
estática independiente
consideraba aptos los screenings de dos pasos, pero la ejecución reveló dos
bloqueos: (i) con la configuración propuesta `b=5`, `lr=0.1`, `weight=[-π,π]` y
`grad=[-1,1]`, el delta entero de QGD es cero para todos los códigos de
gradiente, de modo que los pesos solo se cuantizan en el primer paso; (ii) tres
de las cuatro ejecuciones U0 terminaron con `RuntimeError: measured QGD gradient
register differs from its input code.` Las semillas emparejadas por
identificador no aseguran las mismas realizaciones de ruido. Antes de plantear
G1, G2, S1 o U1 debe revisarse la configuración QGD y el fallo U0, decidir si se
repite la segunda referencia exacta de S0 y presentar el coste total al
responsable. Las confirmaciones no se ejecutaron y la aprobación del tutor sobre
la regla modular QGD sigue pendiente.

Para un cribado se invoca el runner `experiments/compare_qgd_adam.py` con
`--screening` y la configuración correspondiente a S0 o G0; para U0 se añade
`--qgd-only` y se mantiene `p_model=0`. Se conservan las semillas de ejecución
indicadas en la matriz y una ruta de salida nueva. No se debe retirar
`--screening` ni aumentar el presupuesto para estas comprobaciones: los comandos
de confirmación se posponen hasta revisar el coste total de los screenings y
presentarlo al responsable. Esta pauta sigue vigente para cualquier repetición;
no autoriza las confirmaciones.

`p_model` designaría el ruido de los circuitos QCNN compartidos por QGD y Adam.
El bloque principal usaría ese ruido común y `p_update=0`, de modo que el
contraste se refiera a la clasificación bajo una condición común. `p_update`
designaría el ruido de los circuitos aritméticos exclusivos de QGD. Un bloque
QGD-only fijaría `p_model=0` y estudiaría `p_update` por separado; no se
interpretaría como comparación Adam–QGD, porque Adam no tiene circuito de
actualización. Valores candidatos, propuestos a partir del repositorio, son
`p_model` en `0`, `0.001` (perfil M5) y `0.01` (protocolo legacy). `p=0.05`,
aunque aparece en el protocolo legacy, queda fuera de la matriz inicial: solo
se consideraría posteriormente como estrés si existe justificación y perfil
de coste QGD. Para `p_update` se propone el núcleo `0` y `0.001`, con `0.01`
solo como screening QGD-only.

### Bloque separado de shot sampling

La referencia `shots=None` se propone únicamente para `p=0`. En QGD, bajo esa
configuración, el gradiente y la evaluación QCNN son exactos, pero cada update
usa un shot determinista. Esa referencia no equivale a ejecución finite-shots.
El bloque shot-only fijaría `p_model=p_update=0` y usaría Iris angular, con 2
pasos, inicialización fija `19` y dos semillas de ejecución `101` y `102`,
emparejadas por identificador entre Adam y QGD. Se propone `shots=100/500/1000`
en este bloque separado del barrido de `p`. Para el bloque de ruido de puertas
se fijarían 100 shots, sin cruzar todos los valores de shots con todos los
valores de `p`.

### Matriz escalonada propuesta

En todas las filas se propone split estratificado 70/15/15 con semilla de
dataset `42`, distinta de la semilla de inicialización. En los screenings se
fija la inicialización en `19`; las semillas de ejecución `101` y `102` se
emparejan por identificador entre Adam y QGD, pero no garantizan trayectorias
aleatorias idénticas entre métodos. Las confirmaciones usarían inicializaciones
`19/23/29` y semillas de ejecución `101/102/103`; G2 usaría cinco
inicializaciones `19/23/29/31/37` y cinco semillas de ejecución `101–105`.
La codificación propuesta es angular. La configuración QGD propuesta es `b=5`,
`weight=[-pi, pi]`, `grad=[-1, 1]` y `lr=0.1`; Adam usaría `lr=0.03`.
**Advertencia:** con `b=5/lr=0.1` el delta entero es cero para los 32 códigos de
gradiente y los pesos solo se cuantizan; no debe congelarse como configuración
confirmatoria. La sección 29 del log registra el diagnóstico y una tabla de
compromiso `b/lr`; cualquier cambio requiere decisión del tutor. `b=6` no está
permitido por `QGDConfig` y `b=4/lr=0.5` usa una rejilla más gruesa. Estos
hiperparámetros se congelarían con datos del piloto, sin consultar `test`, una
vez resuelta la configuración. La pérdida sería MSE;
el checkpoint se seleccionaría con `validation`. En screening no se evaluaría
`test`; en confirmación se consultaría una sola vez al final.

| Etapa propuesta | Diseño; las confirmaciones S1/G1/G2/U1 siguen pendientes y no ejecutadas. Los screenings S0/G0/U0 se ejecutaron parcialmente (sección 28 del log técnico). |
|---|---|
| N0 | Referencia sin ruido: reutilizar la comparación base propuesta. `shots=None` solo con `p=0`; el gradiente/evaluación QGD son exactos, pero su update usa un shot determinista y no equivale a finite-shots. |
| S0 | Shot screen: Iris angular, 2 pasos, `p_model=p_update=0`, inicialización `19`, `shots=100/500/1000` y ejecuciones `101/102`. Dos ejecuciones de referencia exacta previstas más `2 métodos × 3 shots × 2 semillas` dan 14 runs; la referencia exacta y los updates QGD de un shot determinista se distinguen de finite-shots. |
| S1 | Confirmación de shots condicionada: Iris, 30 pasos/épocas, shots seleccionados tras S0, inicializaciones `19/23/29` y ejecuciones `101/102/103`, emparejadas por identificador entre métodos. |
| G0 | Screen de ruido de puertas: Iris angular, 2 pasos, 100 shots, `p_model=0.001/0.01`, `p_update=0` para QGD, inicialización `19` y ejecuciones `101/102` emparejadas entre Adam/QGD. Son 8 runs adicionales (`2 métodos × 2 niveles × 2 semillas`); el control `p_model=0` se reutiliza. `p=0.05` queda fuera de la matriz inicial y solo sería una opción futura de estrés si se justifica. |
| G1 | Confirmación de ruido de puertas condicionada: Iris, 30 pasos/épocas, 100 shots, `p_model=0/0.001/0.01`, `p_update=0`, inicializaciones `19/23/29` y ejecuciones `101/102/103`, emparejadas por identificador entre métodos. |
| G2 | Traslado a Breast Cancer angular (PCA a cuatro componentes): 30 pasos, 100 shots, `p_model=0/0.001`, `p_update=0`, inicializaciones `19/23/29/31/37` y ejecuciones `101–105`; solo tras perfil de coste. `p=0.01` queda condicionado al presupuesto y evaluación del piloto; no se lanzará automáticamente. |
| U0 | Screen QGD-only del ruido de actualización: Iris angular, `p_model=0`, 100 shots, `p_update=0/0.001/0.01`, 2 pasos, inicialización `19` y dos semillas de ejecución `101/102`. Son 6 runs QGD-only; el caso `p_update=0` puede reutilizarse solo si coincide exactamente con S0. No es comparación Adam–QGD. |
| U1 | Solo si U0 lo permite: `p_update=0/0.001`, Iris, 30 pasos, inicializaciones `19/23/29`, `p_model=0`, 100 shots y ejecuciones `101/102/103`. |

### Estado de ejecución de los screenings (8 de noviembre de 2026)

S0 produjo 7 JSON: una referencia exacta y seis pares finite-shot
(`shots=100/500/1000` × ejecuciones `101/102`). Falta la segunda referencia
exacta prevista; no se evaluó test.

G0 produjo 4 JSON pareados (`p_model=0.001/0.01` × ejecuciones `101/102`,
100 shots, `p_update=0`), completos para el bloque definido. No hubo fallos de
ejecución y no se evaluó test.

U0 produjo un JSON válido (`p_update=0.001`, ejecución `101`) y tres fallos con
`RuntimeError: measured QGD gradient register differs from its input code.`
(`p_update=0.001`/`102`; `p_update=0.01`/`101` y `102`). No hay JSON para los
fallos. El caso `p_update=0` no se lanzó como U0 y no se ha decidido si se
reutiliza S0. El detalle, los tiempos y los bloqueos están en la sección 28 del
log técnico. La validación actual del runner es de 92 pruebas (10 de noviembre
de 2026); no se evaluó test en ningún screen, no hubo QPU y esta revisión no
lanzó experimentos nuevos. Los 12 screens y los cinco perfiles M5 usan la
configuración no-op `b=5/lr=0.1`. Además, el ruido de Qibo se muestrea una vez
por ejecución del circuito, no por shot (sección 29 del log).

### Avance, costes y registro

Los screenings S0, G0 y U0 se ejecutaron parcialmente y sirvieron únicamente
para comprobar el canal, la reproducibilidad, los conteos y el tiempo; no
permiten juzgar accuracy ni convergencia. El trabajo se detiene aquí para
revisar el coste y los bloqueos. No se pasará a confirmar ninguna condición
cuyo coste total no se haya presentado al usuario. En particular, no se
extrapola el coste de `p=0.01` ni `p=0.05` para QGD: no existen perfiles QGD de
esos niveles. La matriz principal inicial de ruido comprende
`p_model={0, 0.001, 0.01}` a 100 shots. No se ejecutarán matrices grandes sin
aprobación. No se cruzará cartesiana y exhaustivamente shots, `p`, datasets y
codificaciones; la codificación por fase y SGD quedan para ablaciones
posteriores.

El recuento de cribado indicado para S0, G0 y U0 era un número de runs
derivable de las configuraciones, no un tiempo total ni un presupuesto
autorizado. La referencia exacta `p=0` y el update QGD de un shot determinista
no equivalen a ejecución finite-shots; deben reportarse separadamente y no
combinarse como si fueran la misma condición. La ejecución de los screenings
solo permitió revisar canal, reproducibilidad, conteos y reloj; no es una
evaluación de accuracy ni de convergencia. El coste observado (≈8,9 min para
los 12 JSON válidos; G0 ≈100 s por ejecución) se registra en la sección 28 y no
se extrapola. U0 no está completo por los fallos de decodificación; U1 no puede
plantearse hasta acotar el fallo. Además, el delta entero de QGD es nulo con la
configuración propuesta; hasta revisar `b` y `lr`, las confirmaciones QGD no
medirían una trayectoria de aprendizaje más allá de la cuantización. La
limitación de muestreo de ruido de Qibo 0.1.11 (una trayectoria de Pauli por
ejecución, no por shot) refuerza que los bloques con `p_model`/`p_update` son
sensibilidades de flujo y coste, no estimaciones promediadas de ruido. El
diagnóstico y las opciones están en la sección 29 del log.

Como evidencia de coste disponible, y solo con cautela, los perfiles M5 de
`results/quantum-training/m5-profile-iris-angle-measured-shots100-noise0-2steps.json`
(Iris angular, 100 shots, `p=0`, 2 pasos) y
`results/quantum-training/m5-profile-iris-angle-measured-shots100-noise0001-2steps.json`
(Iris angular, 100 shots, `p=0.001`, 2 pasos) registran 5,68 s y 46,84 s,
respectivamente. El perfil
`results/quantum-training/m5-profile-breast-cancer-angle-measured-shots100-noise0-2steps.json`
(Breast Cancer angular, 100 shots, `p=0`, 2 pasos) registra 31,49 s. El
resultado legacy descrito arriba, en «Resultado de E en Breast Cancer»,
documenta aproximadamente 26 minutos para Breast Cancer con `p=0.01`, 50
shots y 5 épocas; no es directamente comparable con M5. Estos tiempos no
autorizan un presupuesto ni permiten extrapolar rendimiento. Los niveles QGD
`p=0.01` y `p=0.05` no están perfilados.

Para cada condición se propone registrar MSE de `train` y `validation` por
paso, accuracy, precision, recall y F1, tiempo, conteos de circuitos estimados
y reales (si se instrumentan), shots totales, semilla del dataset, de
inicialización y de ejecución, `p_model`, `p_update`, shots, código del modelo
de ruido, revisión del código y comando. `test` se reservaría para una única
evaluación final de cada condición confirmatoria.
