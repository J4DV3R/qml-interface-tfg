import torch
import torch.nn as nn
import numpy as np
from qibo.models import Circuit
from qibo import gates

class QCNNModel(nn.Module):
    def __init__(self, n_qubits):
        super(QCNNModel, self).__init__()
        self.n_qubits = n_qubits
        
        # Definimos los pesos de la red como parámetros entrenables de PyTorch.
        # Inicializamos los ángulos aleatoriamente entre 0 y 2*pi.
        self.weights = nn.Parameter(torch.randn(n_qubits, requires_grad=True) * 0.01)
        
    def forward(self, x):
        """
        Paso hacia adelante (Forward pass):
        Recibe los datos clásicos (x), construye el circuito de Qibo con los pesos actuales 
        y simula el resultado cuántico.
        """
        batch_size = x.shape[0]
        outputs = []
        
        # Iteramos sobre cada muestra del lote (batch)
        for i in range(batch_size):
            features = x[i].tolist()
            
            # 1. Creamos el circuito para esta muestra
            circuit = Circuit(self.n_qubits)
            
            # 2. Aplicamos codificación de datos (Angle Encoding)
            for q_idx, val in enumerate(features):
                theta = val * np.pi
                circuit.add(gates.RY(q_idx, theta=theta))
                
            # 3. Aplicamos nuestra capa de convolución usando los pesos de PyTorch
            # (Convertimos el tensor de PyTorch a un array de numpy para Qibo)
            current_weights = self.weights.detach().numpy()
            for q_idx in range(self.n_qubits):
                circuit.add(gates.RY(q_idx, theta=current_weights[q_idx]))
                
            # Entrelazamos qubits vecinos (CNOT)
            for q_idx in range(self.n_qubits - 1):
                circuit.add(gates.CNOT(q_idx, q_idx + 1))
                
            # 4. Ejecutamos la simulación cuántica en Qibo
            # (Ejecución sobre el estado final del circuito)
            state = circuit.execute()
            
            # 5. Medimos el valor esperado del último qubit (ej. en la base Z)
            # Para simplificar la prueba, cogemos la probabilidad del estado base
            # o una función de salida simulada.
            measurement = torch.tensor([float(torch.abs(torch.tensor(state.state()[0]))**2)], requires_grad=True)
            outputs.append(measurement)
            
        return torch.stack(outputs)

# --- Pequeña prueba del modelo híbrido ---
if __name__ == "__main__":
    # Creamos un batch de 2 muestras, cada una con 3 características clásicas (ej. 3 píxeles)
    datos_prueba = torch.tensor([[0.1, 0.5, 0.9], [0.8, 0.2, 0.0]], dtype=torch.float32)
    
    modelo = QCNNModel(n_qubits=3)
    
    # Probamos el forward pass
    resultado = modelo(datos_prueba)
    print("Salida del modelo híbrido (PyTorch + Qibo):")
    print(resultado)