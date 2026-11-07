# Instrucciones del proyecto qml-interface-tfg

## 1. Identidad y objetivo

Este repositorio desarrolla el TFG «Plataforma para la experimentación con
Quantum Machine Learning» sobre Qibo. La QCNN es el primer algoritmo
implementado; la arquitectura debe permitir incorporar otros algoritmos QML en
incrementos posteriores. El proyecto incluye código Python, pruebas,
experimentos, resultados JSON, documentación Markdown y una memoria LaTeX en
`memory/`.

## 2. Fuentes de verdad y precedencia

Para afirmar algo sobre el estado del proyecto, usa este orden:

1. El código y las pruebas del repositorio.
2. Los resultados JSON persistidos y sus logs.
3. `docs/TFG-technical-log.md` y `docs/experimental-protocol.md`.
4. `README.md`.
5. `docs/quantum-training-plan.md` y las notas de reunión: son planes, no
   pruebas de implementación.
6. La memoria LaTeX: es un documento en construcción; contiene marcadores
   `\pending{}` y `\placeholderfigure{}`.

Si dos fuentes se contradicen, no elijas una por conveniencia: informa de la
contradicción, cita ambas y pide una decisión. No inventes una decisión de
reunión, una validación ni un resultado.

## 3. Estado del entrenamiento: legado híbrido y nueva ruta QGD

- **Legado híbrido (implementado y evaluado históricamente):** Qibo construye y
  ejecuta el circuito; PyTorch mantiene los pesos, el grafo de autograd y Adam o
  SGD actualiza los parámetros. El backward usa parameter-shift. Está en
  `src/hybrid_model.py` y `src/training.py`.
- **Nueva ruta de entrenamiento con QGD (M1-M4 implementada y validada
  unitariamente, no evaluada experimentalmente):** `src/qcnn_engine.py` aporta el
  motor Qibo/NumPy; `src/optimizers.py`, COBYLA/SPSA/Nelder-Mead autónomos;
  `src/quantum_update.py`, una actualización modular reversible; y
  `src/quantum_training.py` integra el bucle. El CLI es
  `experiments/run_quantum_training.py`. La suite completa pasó 52 pruebas en la
  última validación registrada. No se ha ejecutado el CLI ni generado un JSON de
  resultados nuevos.
- Frontera actual: el primer gradiente se calcula mediante parameter-shift
  exacto y sin ruido; en pasos posteriores los circuitos parameter-shift usan
  shots. Qibo ejecuta los circuitos de actualización en estado-vector exacto,
  sin shots; el host lee el estado de salida del simulador y decodifica los
  códigos. El postprocesamiento de medidas, pérdida, acumulación de gradientes,
  selección por validación y control del bucle también es clásico. El módulo
  `src/quantum_training.py` no importa PyTorch, aunque el runner recibe los datos
  de `src/datasets.py` y convierte sus tensores a NumPy.
- **Desviación metodológica pendiente:** QGD actualiza en una rejilla modular,
  conserva el registro del gradiente y satura gradientes fuera de rango; no usa
  clipping de pesos. El plan original de actualización con clipping no es una
  permutación unitaria como está formulado. La aceptación del tutor de esta
  variante concreta sigue pendiente. El objetivo indicado por el tutor es
  preparar la ejecución en hardware cuántico real y llevar al circuito el ajuste
  de pesos, evitando una actualización clásica. La implementación actual aún no
  cumple ese objetivo de extremo a extremo; no presentarla como hardware-ready,
  totalmente cuántica ni evaluada.
- Los resultados históricos están congelados en `results/initial-hybrid-adam/`.
  No los sobrescribas ni los presentes como resultados del método nuevo.

Etiqueta cada afirmación con uno de estos estados:

| Estado | Significado |
|---|---|
| **Propuesto** | Aparece en un plan, una nota o una propuesta; no hay código. |
| **Implementado** | El código existe, pero no necesariamente se ha ejecutado o validado. |
| **Validado** | Una prueba o comprobación concreta pasó sobre una revisión concreta. |
| **Evaluado** | Se ejecutó un experimento, se persistieron los resultados y se analizaron con sus limitaciones. |

No escribas «se ha implementado» si solo existe en el plan. No escribas «se ha
validado» sin comando, prueba y revisión. No escribas «se ha evaluado» sin un
JSON de resultados y su configuración.

## 4. Mapa del repositorio

- `src/data_encoding.py`: codificación angular `RY(pi*x)` y por fase
  `H + RZ(2*pi*x)`; valida características finitas en `[0, 1]`.
- `src/qcnn_circuit.py`: filtros convolucionales compartidos sobre pares de
  qubits adyacentes.
- `src/pooling.py`: CNOT del qubit descartado al superviviente; el registro de
  Qibo sigue teniendo el mismo tamaño (pooling lógico, no reducción física).
- `src/hybrid_model.py`: envoltorio PyTorch; expectativa Pauli Z y
  parameter-shift; legado híbrido.
- `src/qcnn_engine.py`: motor QCNN de Qibo/NumPy, pérdidas y gradientes
  parameter-shift sin importar PyTorch.
- `src/quantum_update.py`: actualización QGD modular reversible; no equivale
  al clipping propuesto y su aceptación está pendiente.
- `src/quantum_training.py`: bucle integrado QGD, gradiente inicial exacto,
  gradientes posteriores con shots y selección por validación; el postproceso y
  control siguen siendo clásicos.
- `src/optimizers.py`: COBYLA, SPSA y Nelder-Mead como optimizadores autónomos;
  no integrados en el bucle QGD.
- `src/execution.py`: `ExecutionConfig` para simulación exacta, shots, ruido
  depolarizante y semilla.
- `src/training.py`: bucle clásico con Adam o SGD; legado híbrido.
- `src/metrics.py`: accuracy, precision, recall y F1 para etiquetas `-1/+1`.
- `src/classical_baseline.py`: regresión logística externa al motor QML.
- `src/datasets.py`: particionado estratificado, normalización solo con train,
  Iris binario y Breast Cancer con PCA ajustado solo con train.
- `experiments/compare_execution.py`: barrido comparativo de ejecución.
- `experiments/diagnose_convergence.py`: diagnóstico de convergencia exacta.
- `experiments/run_final_qcnn.py`: batería focalizada de cierre.
- `experiments/run_quantum_training.py`: CLI para la nueva ruta QGD; no
  ejecutado todavía.
- `tests/test_qcnn.py`: pruebas unitarias del prototipo.
- `tests/test_quantum_engine.py`, `tests/test_optimizers.py`,
  `tests/test_quantum_update.py` y `tests/test_quantum_training.py`: pruebas de
  componentes e integración QGD.
- `results/initial-hybrid-adam/`: resultados históricos congelados.
- `results/reproductions/legacy/`: repeticiones de experimentos antiguos.
- `results/quantum-training/`: resultados del nuevo entrenamiento (vacío por
  ahora, con README).
- `docs/TFG-technical-log.md`: registro técnico de decisiones.
- `docs/experimental-protocol.md`: protocolo experimental base.
- `docs/quantum-training-plan.md`: plan del entrenamiento no híbrido.
- `memory/`: memoria LaTeX modular; `main.tex`, `preamble.tex`, capítulos,
  anexos y `bibliography.bib` (todavía sin referencias verificadas).

## 5. Orquestación de agentes

El agente principal por defecto es `coordinator`. Los especialistas son
subagentes:

| Agente | Uso | Edición |
|---|---|---|
| `coordinator` | Planifica, delega, serializa ediciones e integra entregas. | Ninguna; solo delega. |
| `qml-developer` | Código Python/Qibo en `src/`, `experiments/`, `tests/` y `main.py`. | Solo esas rutas. |
| `methodology` | Diseño experimental, particiones, fuga de datos, métricas, semillas, reproducibilidad, estadística y límites. | Solo lectura; excepción: `docs/reviews/methodology/` cuando se le asigne. |
| `thesis-writer` | Documentación Markdown y capítulos LaTeX en español académico. | `docs/`, `memory/` y `README.md`, con excepciones. |
| `reviewer` | Auditoría independiente de código, metodología, evidencia y coherencia. | Ninguna; solo lectura. |
| `literature-researcher` | Búsqueda y verificación de fuentes primarias/académicas. | Ninguna; solo lectura. |

Reglas:

1. El coordinador es el único responsable de integrar el resultado y comunicar
   el estado final.
2. No ejecutes dos agentes que puedan editar el mismo archivo a la vez. El
   coordinador debe serializar las ediciones y asignar un único dueño por
   archivo.
3. No delegues tareas triviales que el coordinador pueda resolver leyendo o
   inspeccionando el estado.
4. Cada delegación debe incluir: objetivo, alcance, archivos de entrada,
   entregable esperado, restricciones y evidencia requerida.
5. El revisor, el investigador bibliográfico y el metodólogo no editan código,
   resultados, documentación general ni memoria. El metodólogo solo puede crear
   artefactos en `docs/reviews/methodology/` cuando el coordinador se lo pida
   expresamente.
6. El escritor no edita código, resultados, JSON de origen, el plan
   `docs/quantum-training-plan.md` ni los artefactos del metodólogo.
7. El desarrollador no edita documentación, memoria, resultados ni
   dependencias. Si un cambio requiere documentación, devuelve notas para que
   el coordinador las pase al escritor.

Modelos configurados (OpenCode v2.2.5, catálogo verificado):

- `coordinator`: `openai/gpt-6-luna` (ChatGPT Luna 6).
- `qml-developer`: `opencode/deepseek-v4.1-flash-free` (DeepSeek Flash V4.1).
- `methodology`, `thesis-writer`, `reviewer`, `literature-researcher`:
  `openai/gpt-6-luna`.
- Fallback: si un modelo no está disponible en una instalación, OpenCode usa el
  modelo seleccionado por la sesión. No inventes IDs ni cambies credenciales.
- Si el modelo gratuito de un agente informa límite de uso, no lo relances en
  bucle ni cambies su configuración global. Para tareas pequeñas, continúa en la
  sesión actual si es viable; para trabajo grande, informa el bloqueo y pide
  autorización para cambiar la estrategia.

## 6. Reglas de edición y concurrencia

Antes de editar:

1. Ejecuta `git status --short --branch` y revisa los cambios no confirmados.
2. Lee los archivos objetivo y los documentos de contexto aplicables.
3. Comprueba si otro agente está editando el mismo archivo. Si hay duda,
   detente y pide al coordinador que serialice.
4. No borres, muevas ni sobrescribas cambios existentes para «limpiar» el
   repositorio.
5. No hagas commits, pushes, instalaciones, despliegues ni cambios de
   configuración global.
6. No ejecutes experimentos completos ni la batería completa de pruebas salvo
   que el usuario lo pida explícitamente.

## 7. Reglas experimentales y trazabilidad

- Trata `docs/experimental-protocol.md` como el protocolo de referencia. El
  plan QGD registra componentes M1-M4 implementados, pero la evaluación
  experimental y la aceptación metodológica/tutoría siguen pendientes.
- Mantén las particiones train/validation/test y la semilla del dataset. La
  normalización, el PCA y cualquier ajuste se calculan solo con train.
- No uses test para seleccionar pesos, hiperparámetros ni la mejor época.
- Distingue la semilla del dataset, la semilla de inicialización de pesos y la
  semilla de ejecución (shots/ruido). No las mezcles.
- Cada resultado debe conservar: configuración completa, dataset, particiones,
  semillas, codificación, optimizador, presupuesto, comando, revisión del
  código, resultados individuales, agregados, tiempo y limitaciones.
- Conserva los JSON de origen. Genera tablas y figuras a partir de ellos; no
  escribas métricas a mano si puedes derivarlas del JSON.
- No presentes una media o una diferencia como concluyente con pocas semillas.
  Reporta dispersión, intervalos y limitaciones.
- Si un experimento falla o se interrumpe, conserva los logs y los resultados
  parciales; no los borres.

## 8. Reglas de documentación y memoria

- La memoria está en español académico claro. Respeta la estructura de
  `memory/main.tex` y los comandos de `memory/preamble.tex`.
- No elimines `\pending{}`, `\placeholderfigure{}` ni `\placeholdertable{}` sin
  sustituirlos por contenido verificado.
- Cada afirmación experimental debe citar el archivo de resultados y la
  configuración que la respalda. Cada afirmación bibliográfica debe tener una
  entrada localizable en `memory/bibliography.bib` con metadatos verificados.
- No escribas una referencia en `.bib` sin verificar autores, título, año,
  venue, DOI o URL. Si no puedes verificarla, déjala como pendiente.
- No inventes citas, resultados, métricas, decisiones de reuniones ni
  validaciones.
- Mantén la coherencia entre `README.md`, `docs/`, la memoria y el código.
- Compila la memoria solo cuando se solicite y con aprobación:
  `pdflatex main.tex`, `bibtex main`, `pdflatex main.tex`, `pdflatex main.tex`
  desde `memory/`. Si LaTeX no está disponible, indícalo.

## 9. Seguridad, Git y secretos

- No leas ni muestres `.env`, claves, certificados, `service.json` ni secretos.
- No modifiques la configuración global del usuario ni credenciales.
- No ejecutes `git push`, `git commit`, `git reset`, `git clean`, `git checkout`,
  `git switch`, `git restore`, `git rebase`, `git merge`, `git stash` ni
  operaciones equivalentes.
- No instales paquetes, no actualices OpenCode y no reinicies servicios.
- Los comandos de shell que puedan modificar datos, consumir recursos o tardar
  mucho deben pedir aprobación. Las operaciones destructivas están denegadas en
  `opencode.jsonc`.
- No envíes mensajes a terceros ni publiques archivos.

## 10. Comandos y skills disponibles

Comandos:

- `/plan-task <objetivo>`: planifica sin editar.
- `/implement <cambio>`: implementa código acotado con `qml-developer`.
- `/validate <cambio>`: valida con pruebas ligeras y revisión metodológica.
- `/record-results <json>`: verifica y registra resultados sin modificar los
  originales.
- `/write-results <sección>`: redacta documentación o memoria con evidencia.
- `/review-thesis <alcance>`: auditoría independiente de solo lectura.

Skills del proyecto:

- `qml-project-context`: arquitectura, módulos, decisiones y limitaciones.
- `experimental-protocol`: preparación, ejecución y registro reproducible.
- `academic-writing-latex`: redacción Markdown/LaTeX, citas, tablas y figuras.
- `evidence-and-results`: trazabilidad de afirmaciones y evidencias.

Carga una skill cuando la tarea lo requiera; no repitas todo el contexto del
TFG en cada mensaje.

## 11. Dependencias y entorno

- El manifiesto `requirements.txt` contiene una lista amplia similar a un freeze
  (incluye Jupyter y paquetes no importados directamente); no es una lista
  mínima curada.
- Imports directos de la aplicación: NumPy, Qibo, SciPy, Torch y scikit-learn.
  `scikit-learn` aparece sin versión fijada; `torchvision` no tiene import
  directo localizado en `src/` ni `experiments/`.
- Entorno verificado de este checkout: `qml_env\Scripts\python.exe`, Python
  3.14.3, Qibo 0.1.11, NumPy 2.5.3, SciPy 1.18.1, Torch 2.14.0,
  scikit-learn 1.9.1; `pip check` informó que no hay dependencias rotas.
- El Python global no tiene Qibo. Ejecuta pruebas/comandos con `qml_env\Scripts\python.exe`
  o activa explícitamente el entorno del proyecto.
- No instales, elimines ni fijes dependencias sin aprobación; una futura limpieza
  debería separar dependencias runtime de notebooks/desarrollo y declarar Python
  >=3.10 (por la sintaxis de tipos actual).

## 12. Validación rápida

- Pruebas del proyecto: `qml_env\Scripts\python.exe -m unittest discover -s tests -v`.
- Comprobación de sintaxis: `qml_env\Scripts\python.exe -m compileall src experiments tests`
  o `qml_env\Scripts\python.exe -m py_compile <archivo>`.
- Estado de Git: `git status --short --branch` y `git diff --check`.
- Configuración de OpenCode: `opencode debug config` y
  `opencode debug agents`.
- No ejecutes la batería completa de experimentos para validar la configuración
  de OpenCode. Las pruebas y experimentos se ejecutan solo cuando la tarea lo
  requiera y con aprobación.

## 13. Limitaciones conocidas

- El pooling es lógico: Qibo mantiene el registro completo y no reduce la
  memoria del vector de estado.
- El parameter-shift vuelve a ejecutar el circuito por cada aparición de puerta
  parametrizada; es el principal cuello de botella.
- La clasificación implementada es binaria; no hay multiclase ni minibatches.
- El ruido actual es depolarizante después de `RY`, `RZ` y `CNOT`; no incluye
  ruido de medición, amplitude damping ni phase damping.
- La simulación es local en CPU y no representa hardware cuántico real.
- Los resultados históricos usan pocas semillas y un único test; no permiten
  afirmaciones fuertes de significación.
- La ruta QGD M1-M4 está implementada y validada unitariamente, pero no evaluada
  experimentalmente; la actualización modular difiere del clipping propuesto y
  la aceptación del tutor sigue pendiente.
- La memoria contiene marcadores pendientes y la bibliografía aún no tiene
  referencias verificadas.
