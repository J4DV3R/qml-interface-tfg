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

## Validación previa ejecutada

Antes del barrido base se ejecutó una batería reducida para verificar el flujo:

- shots: `50`;
- ruido: `0.0` y `0.01`;
- semillas de ejecución: `1` y `2`;
- épocas: `2`;
- 5 ejecuciones totales: una exacta y dos para cada condición muestreada.

El resultado completo se conserva en
`results/qcnn-validation.json`. Esta batería confirma el contrato técnico,
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

El resultado se conserva en `results/iris-qcnn-medium.json` y contiene 9
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
  --output results/qcnn-convergence.json
```

## Batería focalizada para cerrar la QCNN

Para evitar un barrido combinatorio, se utilizará
`experiments/run_final_qcnn.py`. El nuevo adaptador de Breast Cancer
Wisconsin tiene 569 muestras y ajusta la estandarización, el PCA de cuatro
componentes y la normalización min-max únicamente con entrenamiento. Cada
componente resultante se codifica en un qubit; por defecto se conserva la
arquitectura `4 -> 2 -> 1`.

Los comandos recomendados desde la raíz del repositorio son:

```powershell
# B: convergencia exacta en el dataset mayor
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 10 20 30 --seeds 19 `
  --output results/breast-cancer-qcnn-convergence.json

# C: variabilidad entre tres inicializaciones, 30 épocas exactas
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 30 --seeds 19 23 29 `
  --output results/breast-cancer-qcnn-seeds.json

# D-angular: referencia reproducible para la codificación angular
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 30 --seeds 19 23 29 `
  --output results/breast-cancer-qcnn-angle.json

# D-fase: misma configuración cambiando solo la codificación
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding phase `
  --epochs 30 --seeds 19 23 29 `
  --output results/breast-cancer-qcnn-phase.json

# E: robustez y coste con dos semillas; usar solo después de validar B
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 5 --seeds 1 2 --shots 50 --noise 0.01 `
  --output results/breast-cancer-qcnn-noisy.json
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
