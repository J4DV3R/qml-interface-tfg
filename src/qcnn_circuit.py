from qibo.models import Circuit
from qibo import gates

def add_conv_layer(circuit, n_qubits, weights):
    """
    Aplica una capa de convolución cuántica 1D.
    Utiliza rotaciones parametrizadas (pesos) y entrelaza qubits adyacentes.
    """
    # 1. Aplicamos las rotaciones (estos 'weights' serán actualizados por PyTorch luego)
    for i in range(n_qubits):
        circuit.add(gates.RY(i, theta=weights[i]))
        
    # 2. Entrelazamos los qubits adyacentes para que compartan información
    # Esto actúa como el "filtro" de la convolución que mezcla píxeles vecinos
    for i in range(n_qubits - 1):
        circuit.add(gates.CNOT(i, i + 1))
        
    return circuit

# --- Pequeña prueba para ver si funciona ---
if __name__ == "__main__":
    n_qubits = 3
    circuito = Circuit(n_qubits)
    
    # Nos inventamos 3 pesos aleatorios simulando los parámetros de una red neuronal
    pesos_entrenables = [0.12, 0.45, 0.88] 
    
    # Añadimos la convolución al circuito vacío
    circuito_convolucionado = add_conv_layer(circuito, n_qubits, pesos_entrenables)
    
    print("Capa de Convolución:")
    print(circuito_convolucionado.draw())