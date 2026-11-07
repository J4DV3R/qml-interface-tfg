---
name: Redacción académica y LaTeX
description: "Normas para redactar Markdown y LaTeX en español académico, citar fuentes, construir tablas y figuras, usar terminología coherente y compilar la memoria."
---

# Redacción académica y LaTeX

## Voz y estilo

- Español académico claro, formal y directo.
- Frases completas; evita abreviaturas informales y emoticonos.
- Define cada sigla en su primera aparición: QML, QCNN, QGD, PCA, SPSA, etc.
- Distingue hecho, interpretación y limitación.
- No uses «demuestra», «prueba» o «confirma» si la evidencia es descriptiva o
  con pocas semillas.
- Mantén la terminología existente del repositorio; no introduzcas sinónimos
  para el mismo concepto.

## Estados de una afirmación

Usa de forma explícita:

- **Propuesto**: plan, diseño o decisión sin código.
- **Implementado**: el código existe, pero no necesariamente se ejecutó.
- **Validado**: una prueba concreta pasó sobre una revisión concreta.
- **Evaluado**: hay resultados persistidos y analizados con limitaciones.

Ejemplo correcto: «El plan de QGD está propuesto; el código actual todavía usa
PyTorch y Adam». Ejemplo incorrecto: «La QCNN ya entrena con QGD».

## Citas y bibliografía

- Cita con `\cite{clave}` en LaTeX y con enlaces o referencias en Markdown.
- Las entradas viven en `memory/bibliography.bib`; el estilo es IEEEtran.
- No escribas una entrada sin verificar autores, título, año, venue, volumen,
  páginas, DOI o URL.
- No inventes citas ni reutilices una referencia para afirmar algo que no dice.
- No uses Wikipedia como fuente primaria.
- Si una referencia no se puede verificar, déjala como pendiente y explica qué
  falta.

## Tablas y figuras

- Usa `booktabs`, `tabularx` o `longtable` según el caso.
- Toda tabla y figura debe tener título, fuente y unidades.
- Genera tablas y figuras a partir de JSON o datos trazables; no copies cifras
  a mano sin comprobarlas.
- No presentes una figura provisional como resultado final.
- En LaTeX, respeta `\placeholderfigure{}` y `\placeholdertable{}` hasta tener
  el contenido real.

## Memoria LaTeX

- `memory/main.tex` es el punto de entrada; no reordenes capítulos sin motivo.
- `memory/preamble.tex` define paquetes, formato y comandos auxiliares.
- No elimines `\pending{}` sin sustituirlo por contenido verificado.
- Respeta el formato numérico del preámbulo (`es-nodecimaldot`); no cambies el
  formato global sin revisar el efecto en todo el documento.
- Para compilar, desde `memory/`:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

- Comprueba `main.log` si hay errores. Si LaTeX no está instalado, indícalo; no
  inventes que la compilación fue correcta.

## Documentación Markdown

- Usa encabezados jerárquicos, tablas y bloques de código con lenguaje.
- Enlaza rutas relativas existentes; si una ruta es futura, márcala como
  pendiente.
- No dupliques el protocolo experimental completo; enlaza al documento fuente.
- Mantén la coherencia con `README.md`, `docs/` y la memoria.

## Revisión antes de entregar

1. ¿Cada afirmación experimental tiene archivo y configuración?
2. ¿Cada cita existe y está verificada?
3. ¿Se distingue planificado de implementado y evaluado?
4. ¿Las tablas y figuras tienen fuente?
5. ¿La terminología es consistente?
6. ¿La compilación se comprobó o se marcó como pendiente?
