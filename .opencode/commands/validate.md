---
description: "Valida un cambio con pruebas ligeras y revisión metodológica, sin ejecutar experimentos completos."
agent: coordinator
subagent: false
---

Valida el cambio: $ARGUMENTS

## Contrato

- **Lee:** `AGENTS.md`, el diff, las pruebas relacionadas, el protocolo y los
  resultados aplicables.
- **Puede modificar:** nada directamente. Puede pedir al `qml-developer` que
  ejecute pruebas ligeras y al `methodology` que revise el diseño.
- **No puede:** ejecutar experimentos completos, ampliar presupuestos, instalar
  dependencias, hacer commits o cambiar configuración global.

## Pasos

1. Ejecuta `git status --short --branch` y revisa el diff.
2. Identifica las pruebas mínimas que cubren el cambio.
3. Pide al `qml-developer` que ejecute solo esas pruebas, con aprobación.
4. Pide al `methodology` una revisión de solo lectura si el cambio afecta a
   particiones, métricas, semillas, preprocesamiento o interpretación.
5. Comprueba que no hay fuga de datos ni uso de test para seleccionar el modelo.
6. Integra los resultados y detecta contradicciones.
7. Si falta evidencia, no declares la validación como superada.

## Evidencia de salida

Devuelve:

- Alcance validado y archivos revisados.
- Comandos ejecutados, salida y número de pruebas.
- Hallazgos del metodólogo por severidad.
- Estado: validado, validado con reservas o no validado.
- Limitaciones y siguiente paso.
