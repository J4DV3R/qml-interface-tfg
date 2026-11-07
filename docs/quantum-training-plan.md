# Plan de entrenamiento no híbrido de la QCNN

**Fecha:** 6 de noviembre de 2026.
**Origen:** reunión de seguimiento del TFG.
**Estado:** planificado; no implementado todavía.

## 1. Decisión de la reunión

El núcleo de la QCNN y la pipeline experimental se consideran válidos. Sin
embargo, el modelo y el entrenamiento actuales son híbridos: Qibo construye y
ejecuta el circuito, pero PyTorch mantiene los pesos, calcula el grafo de
autograd y Adam realiza la actualización clásica.

El requisito acordado es **eliminar el entrenamiento híbrido basado en
retropropagación y Adam**. El ajuste de los pesos debe realizarse mediante
algún circuito cuántico o un mecanismo equivalente, no mediante descenso de
gradientes clásico.

Una vez estabilizado el nuevo entrenamiento, se repetirán los experimentos con
más épocas o presupuestos de evaluación, porque los resultados actuales con 30
épocas y tres semillas no son suficientes para conclusions sólidas.

## 2. Situación de partida

- Modelo: QCNN `4 -> 2 -> 1`, 8 pesos en dos capas convolucionales.
- Simulación: Qibo, backend numpy, CPU, estado puro.
- Entrenamiento actual: `loss.backward()` sobre una función
  `torch.autograd.Function`; el backward ejecuta parameter-shift.
- Optimizador: Adam, tasa de aprendizaje `0.03`, pérdida MSE.
- Resultados de referencia: Breast Cancer con PCA a 4 componentes y 30 épocas.
- Limitaciones conocidas: coste del parameter-shift, dependencia de PyTorch,
  ausencia de minibatches y una sola partición de test.

## 3. Interpretación de «no híbrido» y alcance

En un simulador de propósito general no es posible eliminar por completo el
control clásico: alguien debe preparar el circuito, lanzar ejecuciones, leer
mediciones y decidir cuándo parar. Por eso conviene distinguir tres niveles:

| Nivel | Significado | Viabilidad |
|---|---|---|
| **A. Sin modelo clásico** | Se elimina PyTorch del modelo y del entrenamiento. La pérdida se evalúa con Qibo y los pesos se actualizan con un optimizador sin derivadas (COBYLA, SPSA, Nelder-Mead). | Alta; es el mínimo exigible |
| **B. Actualización cuántica de pesos** | Además de A, cada actualización de peso se realiza con un circuito cuántico sobre un registro discreto de pesos y gradientes. | Alta para 8 pesos con precisión de 4--5 bits |
| **C. Búsqueda cuántica global** | Grover/QAOA sobre todo el espacio de pesos discretizado, con oráculos cuánticos de pérdida. | Baja; se deja como trabajo futuro |

**Propuesta:** implementar **A + B**. El camino principal será el entrenamiento
con actualización cuántica de pesos (QGD discreto), usando COBYLA y SPSA como
baselines sin gradientes. Adam se conserva únicamente como referencia histórica
y de comparación, no como método del modelo final.

La memoria deberá explicar con claridad qué parte sigue siendo clásica
(medición, discretización, control del bucle) y qué parte es cuántica
(evaluación del circuito, parameter-shift y unitario de actualización).

## 4. Objetivos y preguntas de investigación

### Objetivos

1. Extraer un motor QCNN puro de Qibo/NumPy, sin dependencia de PyTorch.
2. Implementar una actualización cuántica de pesos sobre un registro discreto.
3. Implementar optimizadores sin gradientes como alternativas reproducibles.
4. Repetir los experimentos con más épocas/presupuestos y más semillas.
5. Comparar convergencia, generalización, estabilidad y coste.
6. Documentar el diseño y los resultados para la memoria.

### Preguntas

- ¿Converge la QCNN con actualización cuántica de pesos sin Adam?
- ¿Qué precisión de pesos necesita el QGD para no degradar la accuracy?
- ¿Cómo se compara con COBYLA, SPSA y la referencia Adam?
- ¿La ventaja de la codificación por fase se mantiene con más entrenamiento?
- ¿Cómo afectan los shots y el ruido al nuevo entrenamiento?
- ¿Cuál es el coste real en ejecuciones de circuito y tiempo de reloj?

## 5. Alternativas evaluadas

| Método | Usa gradientes | Actualiza pesos con circuito | Ventajas | Limitaciones |
|---|---|---|---|---|
| Adam + parameter-shift (actual) | Sí | No | Referencia ya validada; converge rápido | Híbrido; PyTorch; no cumple el requisito |
| COBYLA | No | No | Sin gradientes; determinista; fácil de implementar | Muchas evaluaciones; puede caer en mínimos locales |
| SPSA | No | No | Dos evaluaciones por iteración; tolera ruido | Hiperparámetros sensibles; estocástico |
| Nelder-Mead | No | No | Sencillo; sin derivadas | Escala mal; menos robusto que COBYLA |
| **QGD discreto** | Sí, vía parameter-shift cuántico | **Sí** | Elimina Adam; actualización mediante unitario | Discretización; control clásico; oráculo de actualización |
| Grover/QAOA | No | Sí | Búsqueda global | Oráculo de pérdida complejo; coste inviable a corto plazo |

**Recomendación:** QGD como método principal, COBYLA y SPSA como baselines, Adam
solo como referencia.

## 6. Diseño propuesto

### 6.1 Motor QCNN puro

Se extraerá la simulación pura de `src/hybrid_model.py` a un módulo nuevo,
`src/qcnn_engine.py`, sin importar PyTorch:

```python
@dataclass(frozen=True)
class QCNNConfig:
    n_qubits: int
    n_outputs: int = 1
    encoding: str = "angle"
    execution: ExecutionConfig = ExecutionConfig()

def predict(weights, features, config) -> np.ndarray
def mse_loss(weights, features, targets, config) -> float
def parameter_shift_gradient(weights, features, targets, config) -> np.ndarray
```

`src/hybrid_model.py` quedará como envoltorio legado para comparar y para no
romper las pruebas actuales. El nuevo camino de entrenamiento no importará
`torch`.

Los datasets seguirán devolviendo tensores de PyTorch por compatibilidad, pero
el nuevo motor aceptará NumPy y convertirá en la frontera. Si la separación
queda limpia, se valorará devolver NumPy desde `src/datasets.py` y ofrecer un
adaptador Torch.

### 6.2 Optimizadores sin gradientes

Nuevo módulo `src/optimizers.py`:

```python
@dataclass(frozen=True)
class OptimizerConfig:
    method: str                 # "cobyla", "spsa", "nelder-mead"
    max_evaluations: int
    seed: int
    bounds: tuple[float, float] = (-np.pi, np.pi)
    rhobeg: float = 0.1
    tol: float = 1e-4

@dataclass
class OptimizationResult:
    weights: np.ndarray
    loss_history: list[float]
    best_loss: float
    evaluations: int
    elapsed_seconds: float
    method: str
```

- **COBYLA:** `scipy.optimize.minimize(method="COBYLA")`; principal baseline
  determinista.
- **SPSA:** implementación propia con semilla; útil con shots y ruido.
- **Nelder-Mead:** opcional, para contraste.

La función objetivo solo ejecuta circuitos Qibo. No hay `backward()` ni
optimizador de PyTorch.

### 6.3 Actualización cuántica de pesos (QGD discreto)

Cada peso se representa en un registro de `b` qubits con `2^b` niveles dentro
de un rango `[w_min, w_max]`. El gradiente se discretiza en otros `b` qubits.
Para cada peso se construye un unitario de actualización sobre `2b` qubits:

```
|theta_code> |grad_code>  ->  |theta_new_code> |0>
theta_new = clip(theta - learning_rate * gradient, w_min, w_max)
```

El unitario es una matriz de permutación de `2^(2b) x 2^(2b)`. Para `b = 5` son
10 qubits y una matriz de `1024 x 1024`, manejable en CPU. Se aplica un unitario
por peso y se mide el registro de salida para recuperar el nuevo valor.

Ventajas:

- No hay Adam ni autograd.
- La actualización se ejecuta en un circuito Qibo verificable.
- El coste del circuito de actualización es pequeño frente a la simulación de
  la QCNN.

Limitaciones que deben explicarse en la memoria:

- El unitario es una tabla de consulta precalculada, no una implementación
  completa de aritmética cuántica con acarreo.
- La discretización introduce error y puede impedir convergencia fina.
- Sigue existiendo control clásico: medir, discretizar, preparar el siguiente
  circuito y decidir la parada.
- La actualización es por peso; no es un único circuito coherente para todos
  los pesos.

Trabajo futuro: sustituir la tabla por sumadores/restadores cuánticos y
explorar una búsqueda Grover/QAOA para el espacio completo.

### 6.4 Bucle de entrenamiento

Pseudocódigo:

```text
inicializar pesos
para step en 1..max_steps:
    predecir train y calcular pérdida MSE
    calcular gradiente con parameter-shift (Qibo)
    para cada peso:
        codificar peso y gradiente
        aplicar circuito de actualización cuántica
        medir y decodificar nuevo peso
    evaluar validación
    guardar historial y mejor checkpoint por validación
    si no mejora durante paciencia: parar
evaluar test una sola vez con el mejor checkpoint
```

No se usará el test para seleccionar pesos. Se registrarán pérdida de
entrenamiento, pérdida de validación, métricas, número de evaluaciones,
número de ejecuciones de circuito y tiempo.

**Un paso de QGD equivale a una época en uso de datos:** ambos recorren una vez
el conjunto de entrenamiento, calculan la pérdida, obtienen el gradiente y
actualizan los pesos una vez. La diferencia es que QGD sustituye la
actualización de Adam por el circuito cuántico.

### 6.5 Unidades de entrenamiento y comparabilidad

No todos los métodos se controlan con la misma unidad. Para evitar comparaciones
engañosas se usan estas definiciones:

| Unidad | Definición | Métodos |
|---|---|---|
| **Época** | Una pasada completa por el conjunto de entrenamiento, con cálculo de pérdida, gradiente y una actualización de pesos | Adam, QGD |
| **Paso** | Una actualización de QGD: pasada completa, parameter-shift y circuitos de actualización cuántica | QGD |
| **Evaluación** | Una ejecución de la función de pérdida sobre todo el entrenamiento, sin calcular gradiente ni actualizar | COBYLA, SPSA |
| **Iteración** | Una actualización del optimizador. SPSA consume dos evaluaciones por iteración (perturbación positiva y negativa); COBYLA puede consumir varias | COBYLA, SPSA |

Por tanto:

- **Sí se siguen usando épocas** en Adam y en QGD. En QGD se llaman «pasos»,
  pero cada paso equivale a una época en datos y en coste de simulación.
- **COBYLA no usa épocas**: propone vectores de pesos y evalúa la pérdida. Su
  presupuesto se expresa en evaluaciones.
- **SPSA no usa épocas**: cada iteración hace dos evaluaciones y una
  actualización. Su presupuesto se expresará en iteraciones y también en
  evaluaciones equivalentes.

Para comparar de forma justa se reportarán dos ejes:

1. **Pasadas sobre el entrenamiento** (misma exposición a los datos).
2. **Tiempo de reloj y ejecuciones de circuito** (mismo coste computacional).

Una evaluación de COBYLA es mucho más barata que una época de Adam o un paso de
QGD, porque no calcula parameter-shift. Por eso no se comparará «30 épocas de
Adam» con «30 evaluaciones de COBYLA» como si fueran equivalentes.

### 6.6 Protocolo experimental con más épocas

Se mantienen las particiones y semillas actuales para comparar. Para que la
comparación sea justa se plantean **dos bloques**:

- **Bloque A — mismas pasadas sobre el entrenamiento:** todos los métodos ven
  los mismos datos (30, 60 y 100 pasadas). Mide eficiencia de muestra, pero no
  iguala el coste, porque COBYLA y SPSA no calculan gradientes.
- **Bloque B — presupuesto ampliado:** se da a los métodos sin gradientes un
  presupuesto mayor (300--1000 evaluaciones) para comprobar su mejor
  rendimiento alcanzable. La comparación se hace por tiempo de reloj y
  ejecuciones de circuito, no por «épocas».

| Método | Unidad | Bloque A | Bloque B | Semillas |
|---|---|---|---|---|
| Adam legado | época | 30, 60, 100 | — | 19, 23, 29 |
| QGD | paso | 30, 60, 100 | — | 19, 23, 29, 31, 37 |
| COBYLA | evaluación | 30, 60, 100 | 300, 600, 1000 | 19, 23, 29, 31, 37 |
| SPSA | iteración | 15, 30, 50 (30, 60, 100 eval) | 150, 300, 500 (300, 600, 1000 eval) | 19, 23, 29, 31, 37 |
| Mejor método | — | — | 1 presupuesto ampliado | 1, 2 (50 shots, ruido 0.01) |

El bloque A responde a «¿qué método aprovecha mejor cada pasada por los
datos?». El bloque B responde a «¿hasta dónde llega cada método si le damos
más presupuesto, y a qué coste?». Así no se confunde una época de Adam o QGD
con una evaluación de COBYLA.

Datasets:

- Iris binario como validación rápida.
- Breast Cancer Wisconsin con PCA a 4 componentes como experimento principal.

Métricas:

- Pérdida de entrenamiento y validación por paso/época.
- Accuracy, precision, recall y F1 en validación y test.
- Mejor época/paso según validación.
- Tiempo de reloj y número de ejecuciones de circuito.
- Media, desviación estándar y intervalo de confianza bootstrap al 95 %.
- Diferencia pareada por semilla frente a la referencia.

Con cinco semillas no se deben hacer afirmaciones de significación fuertes; se
reportarán intervalos y tamaños de efecto, y se evitará presentar diferencias
pequeñas como concluyentes.

### 6.7 Estructura de código prevista

```text
src/qcnn_engine.py          motor puro Qibo/NumPy
src/optimizers.py           COBYLA, SPSA, Nelder-Mead
src/quantum_update.py       QGD discreto y unitarios de actualización
experiments/run_quantum_training.py   runner y JSON
tests/test_quantum_training.py        pruebas del nuevo camino
docs/quantum-training-plan.md         este plan
```

`src/training.py` y `src/hybrid_model.py` se conservan como legado. Se marcarán
como referencia histórica y no se usarán en los nuevos experimentos.

### 6.8 Pruebas previstas

- La pérdida pura coincide con la del modelo PyTorch con los mismos pesos.
- El gradiente parameter-shift del motor puro coincide con diferencias finitas
  para todas las apariciones de pesos y todas las características, en ambas
  codificaciones y en configuraciones `4->1`, `4->2` y `8->1`.
- COBYLA reduce la pérdida en un problema toy y es reproducible con semilla.
- SPSA reduce la pérdida y respeta el presupuesto de evaluaciones.
- El unitario de actualización es unitario y permuta correctamente.
- La actualización cuántica decodificada coincide exactamente con la
  actualización clásica sobre la misma rejilla.
- El nuevo camino de entrenamiento no importa `torch`.
- El JSON de resultados conserva configuración, semillas, presupuesto,
  historial, métricas y baseline.

### 6.9 Documentación y memoria

- Este plan en `docs/quantum-training-plan.md`.
- Entrada en `docs/TFG-technical-log.md`.
- Secciones pendientes en los capítulos de implementación y experimentación de
  la memoria.
- Actualización del README con el estado del nuevo entrenamiento.
- Esquema de resultados documentado en el protocolo experimental.

## 7. Estimación de coste

Mediciones indirectas a partir de los experimentos actuales:

- Una evaluación forward de Breast Cancer (398 muestras) ronda `0,27 s`.
- Un paso de QGD equivale aproximadamente a una época de Adam en coste de
  simulación: `5,6 min` por 30 pasos.
- COBYLA con 1000 evaluaciones ronda `4,5 min`.
- SPSA con 1000 evaluaciones ronda `4,5 min`.
- Adam con 100 épocas ronda `19 min`.
- El bloque A de COBYLA/SPSA es muy barato (decenas de segundos por ejecución);
  el bloque B es el que domina el coste.

La batería completa con 5 semillas y varios presupuestos puede ocupar entre
4 y 6 horas. Se ejecutará en segundo plano, guardando progreso por configuración
para no depender de que termine todo el proceso.

## 8. Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| El QGD se estanca por discretización | Ajustar `b`, rango y learning rate; comparar con COBYLA; si no converge, usar COBYLA como método principal y QGD como prueba de concepto |
| El coste del parameter-shift es alto | Presupuestos acotados, ejecución en segundo plano, guardado incremental |
| El ruido hace inviable el QGD | Reservar QGD ruidoso a una condición reducida; usar SPSA para ruido |
| Pocas semillas | Añadir semillas y reportar intervalos, sin sobreinterpretar |
| Deriva de alcance | Congelar el protocolo antes de ejecutar; separar experimentos de implementación |
| Pérdida de reproducibilidad | Semillas explícitas, JSON versionado, comandos documentados |

## 9. Criterios de aceptación

1. Existe un camino de entrenamiento que no usa PyTorch, autograd ni Adam.
2. La actualización de pesos se realiza mediante un circuito Qibo verificable.
3. La actualización cuántica coincide con la actualización clásica sobre la
   rejilla discretizada.
4. La QCNN reduce la pérdida y obtiene métricas de test con el nuevo método.
5. Se ejecutan presupuestos ampliados y varias semillas, con resultados
   persistidos y analizados.
6. Las pruebas actuales siguen pasando y las nuevas cubren el motor puro, los
   optimizadores y el QGD.
7. La memoria recoge la decisión, el diseño, las limitaciones y los resultados.

## 10. Planificación propuesta

| Semana | Trabajo |
|---|---|
| 1 | Extraer motor puro, función de pérdida, pruebas de equivalencia |
| 2 | COBYLA/SPSA, QGD discreto, pruebas del unitario |
| 3 | Experimentos Iris y Breast Cancer con presupuestos ampliados |
| 4 | Semillas adicionales, ruido, análisis estadístico y documentación |
| 5 | Volcado en la memoria y revisión con el tutor |

## 11. Decisiones abiertas para el tutor

1. ¿Se acepta QGD con medición y control clásico como «entrenamiento no
   híbrido», o se exige un único circuito coherente?
2. ¿COBYLA/SPSA son aceptables como baselines sin gradientes?
3. ¿Cuántas semillas y qué presupuesto se consideran suficientes?
4. ¿Se conserva Adam como referencia comparativa en la memoria?
5. ¿Se prioriza QGD sobre el modelo completo o se limita a un modelo reducido
   como prueba de concepto?

## 12. Trazabilidad

- Reunión: 6 de noviembre de 2026.
- Repositorio: `C:\Users\jadve\Documents\UPO\qml-interface-tfg`.
- Punto de partida: commit `8b0e51d`.
- Registro técnico: `docs/TFG-technical-log.md`, sección 21.
- Estado: planificado, pendiente de implementación y ejecución.
