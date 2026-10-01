# qml-interface-tfg

Trabajo de Fin de Grado para desarrollar una plataforma modular de experimentos
reproducibles de Quantum Machine Learning sobre Qibo. El primer algoritmo en
desarrollo es una QCNN; la arquitectura deberá permitir incorporar y comparar
otros algoritmos QML en incrementos posteriores.

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

El prototipo simula circuitos de estado puro en CPU. No usa shots ni ruido y no
representa una ejecución en hardware cuántico. Para entrenar una clasificación
binaria se pueden mapear etiquetas a `-1` y `+1` y usar, por ejemplo, una
pérdida MSE sobre la expectativa Z de salida. El backward exacto actual vuelve
a ejecutar el circuito por cada aparición de puerta parametrizada; es adecuado
para validar circuitos pequeños, no para escalar todavía a muchos qubits o
lotes grandes.

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

## Pruebas

Con las dependencias de `requirements.txt` instaladas:

```powershell
python -m unittest discover -s tests -v
```
