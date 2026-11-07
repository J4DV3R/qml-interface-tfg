---
description: "Auditoría independiente y de solo lectura de la tesis, el código y la evidencia."
agent: reviewer
subagent: true
---

Revisa: $ARGUMENTS

## Contrato

- **Lee:** `AGENTS.md`, las skills del proyecto, `README.md`, `docs/`, la
  memoria, el código, las pruebas y los resultados JSON.
- **Puede modificar:** nada. Es una auditoría de solo lectura.
- **No puede:** editar archivos, ejecutar pruebas o experimentos, instalar
  dependencias, hacer commits o cambiar configuración global.

## Pasos

1. Ejecuta `git status --short --branch` y revisa el diff si existe.
2. Comprueba la coherencia entre código, resultados, documentación y memoria.
3. Verifica que cada afirmación experimental tiene archivo y configuración.
4. Verifica que cada cita existe y es localizable.
5. Comprueba que se distingue propuesto, implementado, validado y evaluado.
6. Busca fuga de datos, uso indebido de test, semillas mezcladas y métricas
   incomparables.
7. Revisa que el plan `docs/quantum-training-plan.md` no se presente como
   implementación.
8. Ordena los hallazgos por severidad.

## Evidencia de salida

Devuelve:

- Hallazgos críticos, mayores y menores con `archivo:línea`.
- Evidencia y recomendación para cada hallazgo.
- Contradicciones y afirmaciones sin soporte.
- Limitaciones de la propia revisión.
- Veredicto: apto, apto con cambios o no apto.
