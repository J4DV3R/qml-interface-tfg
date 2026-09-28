from qibo.models import Circuit
from qibo import gates

def add_pooling_layer(circuit, qubits_to_keep, qubits_to_discard):
    """
    Aplica una capa de pooling cuántico.
    Conecta los qubits que vamos a descartar con los que vamos a conservar
    y reduce el número de qubits activos a la mitad.
    """
    # Por cada par, usamos una puerta controlada (ej. CNOT o CZ) 
    # para transferir la información del qubit que vamos a "eliminar" 
    # al qubit que vamos a "conservar".
    for keep, discard in zip(qubits_to_keep, qubits_to_discard):
        # Puerta CNOT donde el que se descarta controla al que se conserva
        circuit.add(gates.CNOT(discard, keep))
        
    # Nota conceptual: En la simulación, los qubits descartados simplemente 
    # dejan de formar parte de las capas activas de convolución siguientes.
    return circuit

# --- Pequeña prueba ---
if __name__ == "__main__":
    # Imaginemos que venimos de 4 qubits y queremos quedarnos con 2
    circuito = Circuit(4)
    
    q_conservar = [0, 1]      # Qubits que sobreviven
    q_descartar = [2, 3]      # Qubits cuya información se comprime y se desecha
    
    circuito_pooling = add_pooling_layer(circuito, q_conservar, q_descartar)
    
    print("Capa de Pooling:")
    print(circuito_pooling.draw())