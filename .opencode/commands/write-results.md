---
description: "Redacta resultados verificados en Markdown o LaTeX con el agente thesis-writer."
agent: thesis-writer
subagent: true
---

Redacta: $ARGUMENTS

## Contrato

- **Lee:** `AGENTS.md`, `academic-writing-latex`, `evidence-and-results`, el
  JSON de resultados, la configuración, el protocolo, el registro técnico y los
  capítulos afectados.
- **Puede modificar:** `docs/**`, `memory/**` y `README.md`, salvo
  `docs/quantum-training-plan.md`, `docs/opencode-orchestration-setup.txt`,
  `docs/reviews/methodology/**`, PDFs y artefactos de compilación.
- **No puede:** editar código, pruebas, experimentos, JSON de origen,
  configuración de OpenCode ni dependencias; inventar datos o citas.

## Pasos

1. Ejecuta `git status --short --branch` y lee los archivos objetivo.
2. Localiza la evidencia exacta de cada afirmación.
3. Redacta en español académico claro y con la terminología del proyecto.
4. Distingue propuesto, implementado, validado y evaluado.
5. Añade tablas o figuras solo desde datos trazables.
6. Añade citas solo con entradas verificables; si no, déjalas pendientes.
7. Respeta `\pending{}` y los marcadores de figura/tabla hasta tener contenido.
8. Si se pide compilar, ejecuta LaTeX con aprobación y reporta el resultado.

## Evidencia de salida

Devuelve:

- Archivos modificados y secciones.
- Afirmaciones añadidas y su evidencia.
- Citas y estado de verificación.
- Tablas y figuras con su fuente.
- Pendientes y limitaciones.
- Resultado de compilación o motivo por el que no se compiló.
