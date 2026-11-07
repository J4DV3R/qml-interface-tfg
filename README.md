# qml-interface-tfg

El registro técnico de decisiones, explicaciones, validaciones y tareas
pendientes está en [docs/TFG-technical-log.md](docs/TFG-technical-log.md).
El protocolo experimental base está en
[docs/experimental-protocol.md](docs/experimental-protocol.md).

Trabajo de Fin de Grado para desarrollar una plataforma modular de experimentos
reproducibles de Quantum Machine Learning sobre Qibo. El primer algoritmo en
desarrollo es una QCNN; la arquitectura deberá permitir incorporar y comparar
otros algoritmos QML en incrementos posteriores.

## Descenso de gradiente cuántico discreto: M1-M4 integrado

El diseño previsto del descenso de gradiente cuántico discreto (QGD) está
descrito en [docs/quantum-training-plan.md](docs/quantum-training-plan.md). El
estado del código es **implementado y validado unitariamente**, pero todavía no
evaluado mediante experimentos:

- **Implementado (M1-M4):** el motor Qibo/NumPy está en
  [`src/qcnn_engine.py`](src/qcnn_engine.py), los optimizadores en
  [`src/optimizers.py`](src/optimizers.py), la actualización QGD discreta en
  [`src/quantum_update.py`](src/quantum_update.py) y la orquestación del
  entrenamiento en [`src/quantum_training.py`](src/quantum_training.py). El
  runner está en
  [`experiments/run_quantum_training.py`](experiments/run_quantum_training.py).
  COBYLA y SPSA están disponibles como optimizadores autónomos; no se afirma que
  estén integrados en el bucle QGD. El wrapper legado
  [`src/hybrid_model.py`](src/hybrid_model.py) sigue disponible y
  [`src/training.py`](src/training.py) conserva el entrenamiento clásico con
  PyTorch y Adam o SGD.
- **Validado:** las pruebas cubren el motor, los optimizadores, la actualización
  y el entrenamiento en
  [`tests/test_quantum_engine.py`](tests/test_quantum_engine.py),
  [`tests/test_optimizers.py`](tests/test_optimizers.py),
  [`tests/test_quantum_update.py`](tests/test_quantum_update.py) y
  [`tests/test_quantum_training.py`](tests/test_quantum_training.py). La suite
  completa pasó con 52 pruebas mediante
  `qml_env\Scripts\python.exe -m unittest discover -s tests -v`. También
  pasaron `qml_env\Scripts\python.exe -m compileall src experiments tests` y
  `git diff --check`.
- **Flujo y frontera clásica-cuántica:** el primer gradiente se obtiene mediante
  parameter-shift exacto y sin ruido con circuitos de Qibo; el host calcula las
  diferencias. Cada peso se actualiza con un circuito reversible de Qibo,
  ejecutado en estado-vector exacto, sin shots en ese circuito; el host obtiene
  el código de salida por `argmax` y lo decodifica. Después de cada actualización,
  pérdidas y métricas de entrenamiento/validación se calculan en una pasada
  muestreada por muestra. Los gradientes de las actualizaciones posteriores usan
  circuitos desplazados con shots; sus medidas se leen y las derivadas se
  combinan clásicamente. El cálculo de pérdidas,
  la acumulación de gradientes y el control del bucle, la selección de checkpoint
  por validación y la parada también son operaciones clásicas.
- **Límites:** `src/quantum_training.py` no importa PyTorch. El runner importa
  `src.datasets` al cargarse; este loader importa PyTorch y devuelve tensores
  heredados, que se convierten a NumPy en la frontera de entrada.
  La API de entrenamiento no acepta un conjunto test: el runner selecciona el
  checkpoint con validación y evalúa test una sola vez al final. Los contadores
  de circuitos son estimaciones estructurales, no instrumentación de ejecución.
  El objetivo indicado por el tutor es preparar la ejecución en hardware real y
  realizar el ajuste de pesos mediante circuitos, evitando la actualización
  clásica. Esta implementación todavía no satisface ese objetivo de extremo a
  extremo ni ha sido ejecutada en hardware.
- **Sin evaluación experimental:** no se ha ejecutado el runner ni otro
  experimento de M4 y no se ha generado un JSON nuevo. La suite unitaria no
  constituye una evaluación del método.

Ejemplo de ejecución del runner para Iris, un paso QGD, 100 shots en las
evaluaciones muestreadas, codificación angular y semillas explícitas:

```powershell
qml_env\Scripts\python.exe -m experiments.run_quantum_training `
  --dataset iris --max-steps 1 --shots 100 `
  --noise 0.0 --learning-rate 0.1 --qgd-bits 5 --encoding angle `
  --dataset-seed 42 --initialization-seed 19 --execution-seed 101 `
  --output results/quantum-training/iris-angle-1step-shots100.json
```

Este comando es un ejemplo documentado, **no se ha ejecutado** y el JSON de
salida no se ha generado. El runner exige una ruta de salida nueva y rechaza
sobrescribir archivos existentes; también protege los resultados históricos de
`results/initial-hybrid-adam/`.

> **Desviación del plan y aprobación pendiente.** La actualización de
> [`src/quantum_update.py`](src/quantum_update.py) usa una permutación modular
> reversible y conserva el registro del gradiente, en lugar del mapeo con
> clipping y reinicio del registro propuesto en el plan. La propuesta original
> no es inyectiva y, por ello, no puede realizarse como una operación unitaria
> sobre esos registros. Además, la codificación satura los gradientes al rango
> finito configurado, lo que altera su magnitud cuando quedan fuera de ese rango.
> La aprobación del tutor sigue pendiente; esta variante no implica cumplimiento
> total del plan ni aceptación del término «no híbrido».

## Prototipo QCNN actual

El paquete `src` contiene:

- `data_encoding.py`: codificación angular (`RY(pi*x)`) y por fase (`H` seguido
  de `RZ(2*pi*x)`), ambas para características finitas normalizadas a `[0, 1]`.
- `qcnn_circuit.py`: filtros cuánticos locales compartidos sobre pares de
  qubits adyacentes.
- `pooling.py`: CNOT desde el qubit que se va a descartar hacia el superviviente.
  Qibo mantiene fijo el tamaño del registro; el pooling reduce los qubits
  activos en las capas y medidas siguientes, pero no reduce la memoria del
  vector de estado ni realiza una reducción física del circuito.
- `hybrid_model.py`: módulo PyTorch que devuelve expectativas Z sobre los
  qubits supervivientes. Calcula gradientes de entrada y pesos mediante la
  regla de desplazamiento de parámetros; los parámetros compartidos acumulan
  el gradiente de cada aparición de puerta.
- `execution.py`: configuración inmutable para simulación exacta (`shots=None`)
  o muestreada (`shots` positivo), con semilla opcional y ruido depolarizante.
- `training.py`: configuración del bucle clásico (`epochs`, optimizador y
  `learning_rate`) y función de entrenamiento que devuelve la pérdida por época.
- `metrics.py`: métricas binarias independientes del modelo.
- `classical_baseline.py`: regresión logística de referencia, aislada de la
  implementación del motor cuántico.
- `datasets.py`: proporciona un particionador genérico para datasets
  tabulares y un adaptador para Iris; divide de forma estratificada,
  reproducible y normaliza usando solo entrenamiento. También incluye Breast
  Cancer Wisconsin con estandarización y PCA ajustados únicamente en train.

El modo predeterminado simula circuitos de estado puro en CPU y calcula las
expectativas exactamente. También se puede configurar un número de `shots` para
estimarlas mediante muestreo, y una probabilidad de ruido depolarizante que se
aplica después de las puertas `RY`, `RZ` y `CNOT`. El ruido requiere `shots`
positivos. Ninguno de estos modos representa todavía una ejecución en hardware
cuántico real. Para entrenar una clasificación binaria se pueden mapear etiquetas
a `-1` y `+1` y usar, por ejemplo, una pérdida MSE sobre la expectativa Z de
salida. El backward vuelve a ejecutar el circuito por cada aparición de puerta
parametrizada; con shots y ruido esto también hace ruidosa la estimación del
gradiente, por lo que el prototipo está pensado para circuitos pequeños.

```python
import torch

from src.hybrid_model import QCNNModel

model = QCNNModel(n_qubits=4)
features = torch.tensor([[0.1, 0.5, 0.8, 0.2]], dtype=torch.float32)
targets = torch.tensor([[1.0]], dtype=torch.float32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(features), targets)
loss.backward()
optimizer.step()
```

El entrenamiento reutilizable se puede configurar así:

```python
from src.training import TrainingConfig, train_qcnn

history = train_qcnn(
    model,
    features,
    targets,
    TrainingConfig(epochs=20, optimizer="adam", learning_rate=0.01),
)
```

Para usar mediciones muestreadas:

```python
from src.execution import ExecutionConfig

model = QCNNModel(
    n_qubits=4,
    execution_config=ExecutionConfig(
        shots=1000,
        depolarizing_probability=0.01,
        seed=42,
    ),
)
```

## Pruebas

Con las dependencias de `requirements.txt` instaladas:

En este checkout el entorno verificado es `qml_env`. El Python global no tiene
Qibo; usa `qml_env\Scripts\python.exe` (o activa ese entorno antes de ejecutar
los comandos). `requirements.txt` es una lista amplia similar a un freeze, no un
manifiesto mínimo; `scikit-learn` no tiene versión fijada. No se han modificado
las dependencias en esta revisión.

```powershell
qml_env\Scripts\python.exe -m unittest discover -s tests -v
```

## Experimento comparativo

El experimento [compare_execution.py](experiments/compare_execution.py)
realiza un barrido reproducible sobre la misma QCNN pequeña, datos, pesos
iniciales y configuración clásica. Compara simulación exacta con combinaciones
de `shots` (`100`, `500`, `1000`) y ruido depolarizante (`0.0`, `0.01`, `0.05`).
Cada condición se repite con tres semillas y se muestran los resultados
individuales junto con medias y desviaciones estándar de las pérdidas de
entrenamiento y validación, métricas de test y tiempo:

```powershell
python -m experiments.compare_execution
```

Para guardar los resultados reproducibles en JSON:

```powershell
python -m experiments.compare_execution --output results/reproductions/legacy/iris_qcnn.json
```

Los resultados originales de esta etapa están congelados en
`results/initial-hybrid-adam/`; los comandos de reproducción escriben en
`results/reproductions/legacy/` para no sobrescribirlos.

La batería reducida de validación utilizada antes del experimento base está
en [results/initial-hybrid-adam/qcnn-validation.json](results/initial-hybrid-adam/qcnn-validation.json). No representa
todavía el resultado final: utiliza solo 2 épocas, 50 shots, una ejecución
exacta y dos semillas para cada condición muestreada.

La prueba intermedia está en
[results/initial-hybrid-adam/iris-qcnn-medium.json](results/initial-hybrid-adam/iris-qcnn-medium.json). Utilizó 3
épocas, 50/100 shots, ruido `0.0`/`0.01` y dos semillas muestreadas. La QCNN
obtuvo una accuracy de test entre `0.50` y `0.567`, mientras que la regresión
logística obtuvo `1.0` en la misma partición. Las 3 épocas no permiten evaluar
la convergencia. Además, el ruido multiplicó el tiempo hasta aproximadamente
162--164 segundos por repetición.

Antes del barrido definitivo se realizará un diagnóstico de convergencia en
simulación exacta con 3, 10, 20 y 30 épocas, pesos y partición fijos, Adam y
learning rate `0.03`. El objetivo es mantenerlo por debajo de 20 minutos y
decidir si la QCNN aprende con más épocas antes de invertir en simulaciones
ruidosas.

Para ejecutarlo y guardar también el progreso por configuración:

```powershell
python -m experiments.diagnose_convergence `
  --output results/reproductions/legacy/qcnn-convergence.json
```

El experimento usa un problema binario real basado en Iris: 100 muestras,
cuatro características y dos especies. Se divide en entrenamiento, validación
y test de forma estratificada; los mínimos y máximos se calculan solo con
entrenamiento antes de alimentar cuatro qubits. Las proporciones se pueden
configurar en `SweepConfig`:

```python
from experiments.compare_execution import SweepConfig, run_sweep

result = run_sweep(SweepConfig(
    train_fraction=0.6,
    validation_fraction=0.2,
    test_fraction=0.2,
    dataset_seed=123,
))
```

Para otros datasets tabulares binarios se puede reutilizar
`split_dataset(features, targets, SplitConfig(...))`, conservando las mismas
garantías de particionado, semilla, estratificación y normalización. Los
targets numéricos binarios se convierten al contrato común `-1/+1`. La
clasificación multiclase todavía no está implementada. Sigue siendo un
experimento inicial:
El experimento incluye una regresión logística como referencia externa al
motor QML. Este baseline no se introduce en la abstracción del motor porque
su función es exclusivamente contextualizar los resultados cuánticos.

## Batería final de cierre

El script [run_final_qcnn.py](experiments/run_final_qcnn.py) permite repetir
la convergencia, la variabilidad entre semillas, la comparación angular/fase y
una prueba ruidosa reducida. Para consultar todos los comandos y la
interpretación metodológica, véase
[experimental-protocol.md](docs/experimental-protocol.md).

Ejemplo para el dataset mayor:

```powershell
python -m experiments.run_final_qcnn `
  --dataset breast_cancer --encoding angle `
  --epochs 10 20 30 --seeds 19 `
  --output results/reproductions/legacy/breast-cancer-qcnn-convergence.json
```

Breast Cancer Wisconsin tiene 569 muestras y 30 variables originales. El
adaptador ajusta PCA a cuatro componentes solo con train, normaliza esos
componentes a `[0, 1]` y los asigna uno a uno a los cuatro qubits. La
codificación por fase se selecciona con `--encoding phase`; la angular es el
valor por defecto.
