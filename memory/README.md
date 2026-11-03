# Plantilla de la memoria

Esta carpeta contiene la plantilla modular de la memoria del TFG. El
documento presenta el sistema de información y la plataforma de
experimentación QML como eje del trabajo; la QCNN aparece como el primer
modelo integrado y validado.

## Estructura

- `main.tex`: punto de entrada del documento.
- `preamble.tex`: paquetes, formato y comandos auxiliares.
- `frontmatter/`: portada, resumen y glosario.
- `chapters/`: capítulos principales.
- `appendices/`: manuales, configuraciones y diagramas.
- `bibliography.bib`: referencias bibliográficas.

La bibliografía aparece como un capítulo propio después de las conclusiones y
antes de los anexos. Las referencias deben añadirse en `bibliography.bib` y
citarse desde los capítulos mediante `\cite{clave}`.

## Edición

Se recomienda abrir `memory/main.tex` en Overleaf o en VS Code con LaTeX
Workshop. Para redactar, sustituye progresivamente los comandos
`\pending{...}` por texto académico. Las figuras provisionales se sustituyen
por archivos reales mediante `\includegraphics`.

La plantilla no contiene todavía el texto definitivo de la memoria. Los
marcadores indican qué debe tratarse en cada apartado y permiten revisar el
alcance antes de redactar.

## Compilación

Desde esta carpeta, una instalación local de LaTeX puede compilar el documento
con:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

En Overleaf, sube la carpeta `memory/` o el contenido de la carpeta como un
proyecto y establece `main.tex` como documento principal.

Antes de entregar, deben confirmarse con el tutor la portada institucional,
la extensión, el estilo bibliográfico, los requisitos de anexos y cualquier
formato específico de la convocatoria.
