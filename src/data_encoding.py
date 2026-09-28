import numpy as np
from qibo.models import Circuit
from qibo import gates

def angle_encoding(features):
    """
    Toma una lista de características clásicas (números entre 0 y 1) 
    y devuelve un circuito cuántico de Qibo con esos datos codificados.
    """
    n_qubits = len(features)
    
    # 1. Inicializamos un circuito vacío con tantos qubits como datos tengamos
    circuit = Circuit(n_qubits)
    
    # 2. Aplicamos una rotación Ry a cada qubit basada en el valor clásico
    for i, x in enumerate(features):
        # Mapeamos el valor clásico [0, 1] a un ángulo [0, pi]
        theta = x * np.pi 
        
        # Añadimos la puerta Ry al qubit 'i' con el ángulo 'theta'
        circuit.add(gates.RY(i, theta=theta))
        
    return circuit

# --- Pequeña prueba para ver si funciona ---
if __name__ == "__main__":
    # Imaginemos que tenemos 3 datos clásicos (ej. 3 píxeles)
    datos_clasicos = [0.0, 0.5, 1.0]
    
    circuito_codificado = angle_encoding(datos_clasicos)
    
    print("Circuito de Codificación:")
    print(circuito_codificado.draw())