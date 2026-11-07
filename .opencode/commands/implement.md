---
description: "Implementa un cambio de código acotado con el agente qml-developer."
agent: qml-developer
subagent: true
---

Implementa el cambio: $ARGUMENTS

## Contrato

- **Lee:** `AGENTS.md`, `qml-project-context`, `experimental-protocol`,
  `evidence-and-results`, los archivos objetivo y las pruebas relacionadas.
- **Puede modificar:** `src/**`, `experiments/**`, `tests/**` y `main.py`.
- **No puede:** editar `docs/`, `memory/`, `results/`, `requirements.txt` ni
  `.opencode/`; instalar dependencias; ejecutar experimentos completos; hacer
  commits o pushes.

## Pasos

1. Ejecuta `git status --short --branch` y lee los archivos objetivo.
2. Confirma que el cambio está dentro del alcance; si no, detente y explica.
3. Implementa el cambio mínimo y coherente con el estilo existente.
4. Añade o ajusta pruebas solo si el alcance lo permite.
5. Ejecuta las pruebas relacionadas con aprobación; no la batería completa.
6. Revisa `git diff --check` y el diff final.
7. Si hace falta documentación, prepara una nota para el coordinador; no edites
   docs ni memoria.

## Evidencia de salida

Devuelve:

- Archivos modificados y resumen del cambio.
- Comandos ejecutados y salida relevante.
- Pruebas realizadas y resultado; si no se ejecutaron, dilo.
- Estado: implementado, validado o pendiente.
- Riesgos, limitaciones y notas para documentación.
- Diff o referencias `archivo:línea`.
