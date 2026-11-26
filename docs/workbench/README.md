# QML Workbench: interfaz de la plataforma

**Estado (25 de noviembre de 2026): implementado y validado** (`tests/test_workbench.py`, 17 pruebas OK en el entorno local, Python 3.14.3). Python 3.10.13 aún no se ha validado. Es una aplicación de escritorio con PySide6, basada en flujos de nodos al estilo de Orange y organizada según el patrón MVC.

## Uso

Para crear el entorno de CPython 3.10.13 e instalar las dependencias de la
interfaz, sigue las instrucciones de
[Entorno objetivo](../../README.md#entorno-objetivo).

```bash
.venv-py310/Scripts/python.exe -m src.workbench
```

- Se abre con la plantilla **«Clasificación QML»**: Dataset → QML → Procesado de resultados → Resultados, ya conectados. Es un flujo normal y se puede editar.
- **Configurar un nodo:** se selecciona y se usa el panel «Parámetros». Los campos se generan a partir del esquema del nodo y del algoritmo elegido. Los cambios se aplican con «Aplicar», y los errores (por ejemplo, porcentajes que no suman 100) se muestran antes de aplicar.
- **Ejecutar:** menú *Ejecutar* o F5. La ejecución va en segundo plano, con progreso en cada nodo, y se puede cancelar con Esc. Solo se recalcula lo que ha cambiado.
- **Ver resultados:** aparecen en los dos paneles inferiores al terminar, o al seleccionar un nodo que tenga informe. Con doble clic en «Resultados» se abre el informe completo, con métricas, curvas, matrices de confusión y configuración.
- **Editar el flujo:**
  - para añadir nodos, se arrastran desde la paleta (o doble clic);
  - para conectar, se arrastra de una salida a una entrada; solo se aceptan tipos compatibles, y los ciclos se rechazan;
  - para desconectar, se arrastra desde una entrada ya conectada;
  - la tecla Supr elimina lo seleccionado.
- **Proyectos:** se guardan y abren como `*.qmlflow` (JSON).
- **Sin interfaz:** `.venv-py310/Scripts/python.exe -m src.workbench.run --template clasificacion-qml` o `... run proyecto.qmlflow`.

## Diseño visual

La composición imita la del IBM Quantum Composer, solo en diseño y disposición: la paleta de colores es la del sistema Carbon de IBM. La tipografía es **Bahnschrift** (estilo DIN, incluida en Windows 10/11) en la interfaz y las gráficas, y **Cascadia Code** en el código y el registro; si no están instaladas, se usan Segoe UI y Consolas.

- **Menú Ver:** muestra u oculta cada panel (Ctrl+1 a Ctrl+4) y la barra del lienzo, elige qué muestra cada panel, restablece la disposición y cambia el tema.
- **Clic derecho** en los paneles inferiores (o el botón ✕ de su cabecera): limpia las métricas y gráficas. Los resultados siguen en la caché; al seleccionar un nodo con informe o al volver a ejecutar, reaparecen. En el panel derecho, además, permite limpiar o copiar el registro.
- **Parámetros:** los cambios válidos se aplican al momento, sin botón. Antes, si no se pulsaba «Aplicar», la ejecución usaba los valores anteriores; por eso no se exportaban los archivos. «Valores por defecto» restablece el nodo.
- **Exportación:**
  - el JSON contiene el informe completo, incluidas las predicciones por muestra;
  - el CSV incluye las métricas y las baselines, y por defecto usa el formato de Excel en español (`;` como separador, coma decimal, UTF-8 con BOM);
  - la ruta exportada se muestra completa en el registro.

| Zona | Contenido |
|---|---|
| Barra superior | Nombre del proyecto, menús Archivo/Editar/Ver/Ayuda, «Guardar» y el botón principal azul «Ejecutar flujo», que pasa a «Cancelar» durante la ejecución |
| Izquierda, «Nodos» | Fichas de colores por categoría (DS, QML, EV, RES), con búsqueda |
| Centro | Barra de herramientas del lienzo (ajustar, zoom, ejecutar hasta el nodo, eliminar, ajuste a cuadrícula) y lienzo con nodos en forma de ficha de puerta |
| Derecha | Selector «Parámetros» / «Proyecto (.qmlflow)»: formulario del nodo seleccionado o el archivo del proyecto en vivo, como el panel de OpenQASM |
| Abajo, izquierda | «Accuracy por partición» (QML frente a las baselines) o «Tabla de métricas» |
| Abajo, derecha | «Curvas de entrenamiento», «Matriz de confusión» o «Registro» |
| Barra de estado | Mensajes, progreso y selector de tema claro/oscuro |

## Arquitectura MVC

| Capa | Paquete | Responsabilidad | Importa Qt |
|---|---|---|---|
| **Modelo** | `src/workbench/model/` | Grafo (`workflow.py`), puertos con tipo y tipos de datos (`data.py`), esquemas de parámetros (`parameters.py`), registro de nodos (`node.py`) y de algoritmos (`algorithms/`), motor con caché (`engine.py`), preprocesado sin fugas (`preprocessing.py`) y plantillas (`templates.py`) | No |
| **Controlador** | `src/workbench/controller/` | Único punto por el que la vista cambia o ejecuta el modelo. Emite eventos (`workflow_reset`, `workflow_changed`, `node_state`, `node_progress`, `log`, `run_started`, `run_finished`, `dirty`) y ejecuta en un hilo aparte | No |
| **Vista** | `src/workbench/view/` | Lienzo (`canvas.py`), formularios generados del esquema (`settings.py`), ventana de resultados (`results_view.py`) y ventana principal (`main_window.py`). `GuiDispatcher` lleva los eventos del hilo de ejecución al hilo de la interfaz | Sí |

La vista nunca modifica el modelo directamente: llama al controlador y se redibuja con sus eventos. Por eso el modelo y el controlador se prueban sin interfaz.

## Tipos de datos (puertos)

| Tipo | Contenido |
|---|---|
| `Dataset` | Características, etiquetas, nombres y origen, sin partir ni preprocesar |
| `TrainingResult` | Particiones preprocesadas, predicciones en el checkpoint, historial, checkpoint, estado del modelo y coste |
| `EvaluationReport` | Métricas por partición, baselines, curvas, matrices de confusión y configuración |

## Reglas metodológicas que impone la plataforma

- La partición (estratificada, con porcentajes y semilla) y el preprocesado (estandarización + PCA si hacen falta, y min-max) se ajustan **solo con train**, dentro del nodo QML.
- El checkpoint lo elige el algoritmo **solo con validation**. El nodo QML predice test **una sola vez**, después.
- Las baselines (clase mayoritaria y regresión logística) usan las mismas características y se ajustan en train.
- **Reproducibilidad:** un flujo configurado como la fase C reproduce exactamente sus particiones, su checkpoint y su accuracy de test (prueba `PhaseCParityTests`). Solo difieren los pesos `w1`–`w3` de la capa 2, que no influyen en la salida (log, sección 37): su valor es arbitrario y el desempate depende del orden de las sumas en coma flotante.

## Cómo ampliarla

**Un algoritmo nuevo** se registra en `src/workbench/model/algorithms/` y aparece en el desplegable del nodo QML con su formulario:

```python
@register_algorithm
class MiAlgoritmo(QMLAlgorithm):
    algorithm_id = "mi-algoritmo"
    label = "Mi algoritmo"
    n_features = 4                      # características que necesita (o None)

    def parameter_schema(self, values):  # hiperparámetros
        return [IntParam("max_steps", "Pasos", 50, minimum=1)]

    def fit(self, splits, params, context):  # usa solo train y validation
        ...
        return FitOutcome(predict=..., history=[...], checkpoint={...}, model_state={...})
```

**Un nodo nuevo** se registra en `src/workbench/model/nodes/` y aparece en la paleta:

```python
@register_node
class MiNodo(NodeType):
    type_id, title, category = "mi-nodo", "Mi nodo", "Evaluación"
    inputs = (PortSpec("result", TRAINING_RESULT, "Resultado"),)
    outputs = (PortSpec("report", EVALUATION_REPORT, "Informe"),)

    def parameter_schema(self, values=None):
        return [...]

    def run(self, inputs, params, context):
        return {"report": ...}
```

**Una plantilla nueva** es un archivo `src/workbench/templates/<id>.qmlflow`; se puede crear guardando un proyecto desde la aplicación.

## Limitaciones actuales

- **Algoritmo disponible:** solo la QCNN variacional de 4 qubits. Arrastra el hallazgo del cono de luz: su salida ignora la cuarta característica y 3 de los 8 pesos, y la ventana de resultados lo avisa.
- **Ejecución:** solo simulación (exacta, con shots o con ruido de sensibilidad). La ejecución en el BSC no está conectada al nodo QML.
- **Datos:** clasificación binaria; datasets Iris binario, Breast Cancer o CSV con columnas numéricas.
- **Edición:** no hay deshacer/rehacer, ni nodos de preprocesado o comparación todavía. Son ampliaciones previstas, posibles sin tocar el núcleo.
- **Pruebas de la vista:** se ejecutan con la plataforma Qt `offscreen`. No hay pruebas de usabilidad con usuarios.
