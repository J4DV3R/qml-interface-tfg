---
description: "Planifica una tarea del TFG sin editar archivos ni ejecutar experimentos."
agent: coordinator
subagent: false
---

Planifica la tarea: $ARGUMENTS

## Contrato

- **Lee:** `AGENTS.md`, `docs/TFG-technical-log.md`,
  `docs/experimental-protocol.md`, `docs/quantum-training-plan.md`, el código y
  los resultados directamente relacionados.
- **Puede modificar:** nada. Es una tarea de solo lectura y planificación.
- **No puede:** editar archivos, ejecutar experimentos, instalar dependencias,
  hacer commits o cambiar configuración global.

## Pasos

1. Ejecuta `git status --short --branch` y respeta los cambios existentes.
2. Reformula el objetivo y el criterio de aceptación.
3. Clasifica el estado actual: propuesto, implementado, validado o evaluado.
4. Identifica archivos, evidencias y dependencias relevantes.
5. Descompón el trabajo en subtareas concretas.
6. Para cada subtarea indica: especialista, alcance, archivos de entrada,
   entregable, restricciones y evidencia requerida.
7. Detecta riesgos, contradicciones y decisiones abiertas.

## Evidencia de salida

Devuelve:

- Objetivo y criterio de aceptación.
- Estado actual con fuentes consultadas.
- Plan de subtareas y especialistas.
- Archivos que se prevé tocar y dueño de cada uno.
- Riesgos, dependencias y decisiones pendientes.
- Siguiente comando recomendado (`/implement`, `/validate`, etc.).
