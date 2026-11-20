# Scripts Slurm para el BSC (fase 1 del plan BSC)

**Estado: implementado, no ejecutado en el BSC.** Los valores entre `<...>` son marcadores que hay que rellenar con la documentación del BSC cuando se conceda el acceso. Las páginas de usuario del BSC no se han podido leer de forma automática.

| Script | Máquina | Uso |
|---|---|---|
| `mn5_gpp_array.sbatch` | MareNostrum 5 GPP (CPU) | Simulaciones de la línea variacional como *job array*; sustituye a la ejecución local con `xargs -P` de la fase C |
| `ona_inference.sbatch` | MareNostrum5-Ona (chip cuántico) | Inferencia de checkpoints entrenados (fase 3). **No funcional** hasta implementar `QililabExecutor` |

## Por qué dos máquinas

La guía de acceso de la RES para recursos cuánticos indica que en la partición cuántica solo se aceptan trabajos que usen el chip. Cada chip lleva asociado un nodo de 40 CPU para el pre- y postprocesado. Las simulaciones hay que pedirlas aparte en MN5-GPP o MN5-ACC, dentro de la misma solicitud.

## Generar la lista de trabajos

```bash
qml_env/Scripts/python.exe -B -m experiments.make_variational_jobs --datasets breast-cancer iris --dataset-seeds 1 2 3 4 5 6 7 8 9 10 --output-root results/variational/C-partitions --jobs-file slurm/jobs/phaseC.txt
```

Después se envía con `sbatch --array=1-160%40 --export=ALL,JOBS_FILE=slurm/jobs/phaseC.txt slurm/mn5_gpp_array.sbatch`.

## Entorno (pendiente de confirmar)

- **Sin Internet en los nodos de cálculo**, lo habitual en MN5: preparar las dependencias antes. La opción más segura es descargar las ruedas en un equipo con red (`pip download -r requirements.txt -d wheelhouse`), copiarlas a MN5 e instalarlas con `pip install --no-index --find-links wheelhouse -r requirements.txt` en un entorno virtual. La versión de Python debe estar disponible como módulo en MN5.
- La **procedencia** (`collect_provenance`) llama a `git` dentro del repositorio: hay que clonar el repositorio en MN5, no copiar solo los archivos, para que cada JSON registre la revisión y el hash del diff.
- **Ona:** el BSC proporciona Qibo y Qililab. Hay que comprobar si la versión de Qibo del BSC es compatible con la 0.1.11 usada aquí. La construcción de circuitos es Qibo estándar; el transpilador propio (`src/transpilation.py`) no depende de `qibo.transpiler`.

## Logs

Cada tarea escribe `EXIT_CODE`, `WALL_SECONDS` y `HOST` en `slurm/logs/runs/`, con el mismo criterio que los logs de las fases A–C.
