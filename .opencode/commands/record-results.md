---
description: "Verifica y registra un resultado experimental sin modificar los JSON de origen."
agent: coordinator
subagent: false
---

Registra y verifica el resultado: $ARGUMENTS

## Contrato

- **Lee:** el JSON de origen, su log, la configuración, el comando de ejecución,
  la revisión del código, `docs/experimental-protocol.md` y
  `docs/TFG-technical-log.md`.
- **Puede modificar:** nada directamente. Puede delegar en `thesis-writer` la
  actualización de documentación o memoria.
- **No puede:** modificar el JSON de origen, inventar métricas, ejecutar
  experimentos, hacer commits o cambiar configuración global.

## Pasos

1. Ejecuta `git status --short --branch` y localiza el JSON y su log.
2. Comprueba que el JSON es válido y que contiene la configuración esperada.
3. Identifica el esquema: `sweep/runs/summaries/baseline` o
   `dataset/model/training/results/baseline`.
4. Verifica dataset, particiones, semillas, codificación, optimizador,
   presupuesto, shots, ruido y tiempo.
5. Calcula agregados solo desde los resultados individuales.
6. Comprueba que el baseline usa la misma partición y transformaciones.
7. Redacta un resumen verificado y, si procede, pide al `thesis-writer` que lo
   incorpore a `docs/` o `memory/`.
8. No sobrescribas ni «corrijas» el JSON original.

## Evidencia de salida

Devuelve:

- Ruta del JSON y del log.
- Revisión del código y comando.
- Configuración efectiva.
- Resultados individuales y agregados.
- Baseline y comparación, si existe.
- Limitaciones y estado (evaluado o pendiente).
- Archivos documentales actualizados, si los hubo.
