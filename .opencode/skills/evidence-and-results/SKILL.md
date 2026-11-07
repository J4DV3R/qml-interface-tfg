---
name: Evidencia y resultados
description: "Trazabilidad de afirmaciones a código, fuentes, comandos y archivos de resultados; reglas contra resultados, métricas y citas inventadas."
---

# Evidencia y resultados

## Principio

Toda afirmación debe poder rastrearse hasta una evidencia localizable. Si no
hay evidencia, la afirmación es una hipótesis o una propuesta, no un resultado.

## Tipos de evidencia

| Tipo | Ejemplo |
|---|---|
| Código | `src/datasets.py:105` ajusta el mínimo solo con train |
| Prueba | `qml_env\Scripts\python.exe -m unittest discover -s tests -v` → resumen final con todos los tests en `OK` |
| Resultado | `results/initial-hybrid-adam/breast-cancer-qcnn-convergence.json` |
| Configuración | `dataset.seed=42`, `epochs=30`, `optimizer=adam`, `shots=null` |
| Revisión | commit o `git rev-parse HEAD` |
| Fuente | DOI, URL oficial o entrada `.bib` verificada |
| Documento | `docs/experimental-protocol.md`, sección concreta |

## Formato de una afirmación experimental

Incluye, cuando exista:

- Qué se midió.
- Dataset y partición.
- Configuración: codificación, optimizador, presupuesto, shots, ruido.
- Semilla o semillas.
- Archivo JSON de origen.
- Comando y revisión del código.
- Métrica y valor.
- Limitaciones.

Ejemplo correcto: «Con Breast Cancer, PCA a cuatro componentes, 30 épocas,
Adam y semilla de inicialización 19, la accuracy de test fue `0.826`
(`results/initial-hybrid-adam/breast-cancer-qcnn-convergence.json`)».

Ejemplo incorrecto: «La QCNN alcanza un 85 % de accuracy».

## Reglas contra resultados inventados

- No escribas una métrica que no esté en un JSON o en una salida de comando.
- No calcules una media sin los resultados individuales.
- No mezcles condiciones distintas para obtener un número más favorable.
- No omitas que un resultado es histórico, preliminar o de una sola semilla.
- No presentes una diferencia como significativa sin análisis adecuado.
- No atribuyas un resultado al método nuevo si se obtuvo con el legado híbrido.
- No copies cifras de la memoria si no coinciden con el JSON; investiga la
  discrepancia.

## Reglas contra citas inventadas

- Cada cita debe tener una entrada localizable.
- Verifica autores, título, año, venue y DOI/URL.
- No inventes páginas, volúmenes ni identificadores.
- No cites un preprint como si fuera revisado por pares.
- Si no puedes verificar, marca la referencia como pendiente.

## Contradicciones

Si el código, los resultados y la documentación no coinciden:

1. Cita las dos fuentes y la discrepancia exacta.
2. Indica qué está verificado y qué no.
3. No elijas una versión por conveniencia.
4. Pide una decisión al usuario o al coordinador.
5. No «arregles» el documento ni el JSON para ocultar la contradicción.

## Lista de comprobación

- ¿La afirmación tiene archivo y línea o JSON?
- ¿La configuración está completa?
- ¿La partición es la misma?
- ¿Las semillas están identificadas?
- ¿Se distingue propuesto, implementado, validado y evaluado?
- ¿La limitación está escrita?
- ¿La cita existe y está verificada?
- ¿La tabla o figura se generó a partir de datos trazables?
