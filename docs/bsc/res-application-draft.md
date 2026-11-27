# Borrador de la solicitud RES (recursos cuánticos + MN5-GPP)

**Estado: borrador para revisar con el tutor. No enviado.** Fecha: 20 de noviembre de 2026.

- **Base:** guía de acceso de la RES para recursos cuánticos, v3.3, que fija los apartados y límites de caracteres; datos técnicos de `docs/bsc/cost-estimates.md`; tiempos medidos en los logs de `results/variational/`.
- **Idioma:** el texto de los campos está en inglés, como la guía y el formulario. Confirmar si se admite español.
- **Datos personales:** los campos de equipo y CV usan marcadores `<...>`.
- **Referencias bibliográficas:** marcadas como `[REF]`. No hay ninguna verificada en `memory/bibliography.bib` y no deben enviarse sin verificar.

## Decisiones previas (para la tutoría)

| Decisión | Propuesta | Motivo |
|---|---|---|
| Responsable (*team leader*) | El estudiante, con el tutor como miembro del equipo | La guía lo permite expresamente en un TFG |
| Tipo de actividad | *Standard Activity* con la opción para usuarios sin experiencia en HPC (menos de un año) | El *fast-track* exige la referencia de un proyecto financiado; la opción de usuario novel elimina la evaluación de experiencia previa en HPC |
| Máquinas | MareNostrum5-Ona (chip) y MareNostrum5 GPP (simulación) | La partición cuántica solo acepta trabajos que usen el chip; las simulaciones deben pedirse aparte |
| Asignación | «Only selected machine» para Ona; «Any machine» aceptable para la simulación | El experimento necesita el chip concreto; la simulación puede ir a otra máquina CPU |
| Área científica | La más próxima (física o informática), según la lista del formulario | Si ninguna encaja, la guía indica consultar a `applications@bsc.es` |
| Proyecto financiado | Indicar si existe; si no, dejar 2b vacío | Es opcional en la actividad estándar |

---

## Paso 1. Información general

**a) Activity title** (propuesta):

> Hardware validation of a variational quantum convolutional neural network: noise impact and circuit-efficient training on MareNostrum5-Ona

**b) Area:** `<elegir en el formulario>`

**c) Type of application:** Standard Activity (novel user, less than one year of HPC expertise).

## Paso 2. Descripción del proyecto

### 2a) Brief description of the project (máx. 5000 caracteres)

> This activity is part of a Bachelor's thesis (TFG) at `<UNIVERSIDAD>` titled "A platform for experimentation with Quantum Machine Learning", supervised by `<TUTOR>`. The platform is built on Qibo and implements, as its first algorithm, a quantum convolutional neural network (QCNN) [REF] for binary classification. The model has 4 qubits, angle (RY) or phase (H + RZ) data encoding, two layers of translation-shared two-qubit convolutional filters (8 trainable parameters) and logical pooling. Its output is the Pauli-Z expectation value of one qubit.
>
> We have completed a simulation study of how this model should be trained when executed as a standard variational algorithm (quantum circuit evaluation plus classical parameter update). It compares gradient descent and Adam with parameter-shift gradients [REF], SPSA [REF], and a sequential, coordinate-wise optimizer that exactly reconstructs the cost as a trigonometric function of each parameter (Rotosolve/NFT-type) [REF]. The protocol was pre-registered: hyperparameters tuned on validation data only, test data evaluated once, majority-class and logistic-regression baselines, and 10 independent data partitions × 8 initialization seeds per dataset (1,280 training runs, Breast Cancer with 4 PCA components and binary Iris). Per-partition statistics use Holm correction. All code (180 unit tests), results and provenance (code revision, data fingerprints, seeds) are kept in a version-controlled repository.
>
> Main simulation findings:
> 1. The QCNN learns both tasks well above the majority class, but stays below a logistic regression on the same four features (by 0.026–0.084 in accuracy on Breast Cancer).
> 2. No optimizer is significantly better in accuracy after correction.
> 3. The sequential optimizer reaches its best checkpoint with 3.2–9.4 times fewer circuit executions than Adam in all 40 partition/dataset/encoding combinations, and keeps its accuracy when combined with early stopping.
>
> On a quantum processor the cost is measured in circuit executions, so finding 3 is directly relevant to hardware. What simulation cannot tell is how the real device changes these conclusions: gate and readout errors, calibration drift and the overhead of submitting circuits. This activity uses MareNostrum5-Ona to measure that. We will:
> - evaluate on the chip the checkpoints already trained and selected in simulation (no test-based selection);
> - quantify the accuracy loss with respect to exact simulation and how much readout-error mitigation recovers;
> - if the measured execution rate allows it, run a short on-chip training with the two circuit-frugal optimizers.
>
> The circuits are small (4 qubits, logical depth 25–26, 11 two-qubit gates), which makes them a well-controlled probe of present hardware for a QML workload. The results will form the hardware chapter of the thesis and a reproducible benchmark within the platform.

### 2b) Grants and funded projects

`<Indicar el proyecto financiado del grupo, si existe. Si no, dejar vacío; no es obligatorio en la actividad estándar.>`

### 2d) Specific activity proposed (máx. 10 000 caracteres)

> Need for RES resources. The activity needs (i) access to a superconducting quantum processor programmable with Qibo, available at MareNostrum5-Ona, and (ii) CPU time for noisy simulations on MareNostrum5 GPP. Training with an exact density-matrix noise model is expensive. In our measurements on a workstation, one noisy configuration set for Breast Cancer (4 optimizers × 2 encodings × 8 seeds, 1000 shots) took 22 CPU-hours for a single data partition.
>
> Already available (no RES time needed). Trained checkpoints for 10 partitions per dataset; a transpiler mapping the QCNN to a coupling map and a {RX, RY, RZ, CZ} gate set with SWAP routing (validated: the measured distribution is preserved to 1e-10 on line, ring, star and grid topologies); batched execution; readout-error calibration and mitigation; a hardware-inference driver that records exact, raw and mitigated metrics with full provenance. A device-like simulator (native gates on coupled qubits only, local depolarizing noise, asymmetric readout error) has been used for dry runs. A Qililab adapter (`src/qililab_adapter.py`) and minimal 1--2 qubit smoke test (`experiments/run_qililab_smoke.py`) are implemented against the public `qililab.execute` API, pending on-site validation with the BSC environment and runcard.
>
> WP1 — Set-up and smoke test (Ona, about 1 h). Connect the Qibo/Qililab backend. Map the transpiler to the chip's real coupling map and native gates. Run readout calibration and one test-set inference (17 circuits, 1000 shots), and check the results against exact simulation within the expected noise.
>
> WP2 — H1: inference of trained checkpoints (Ona, main experiment). 8 checkpoints per dataset (one initialization per optimizer and encoding, fixed before execution). Each checkpoint is evaluated on the validation and test samples of its partition: 15 + 15 (Iris) and 85 + 86 (Breast Cancer) circuits, with 1000 shots per circuit. Readout calibration runs at the beginning and the end of each block. That is 1,641 circuits per repetition. The block is repeated on two different days to measure calibration drift, giving about 3,300 circuits and 3.3 M shots. Reported: accuracy, balanced accuracy and MSE on the chip (raw and readout-mitigated) against exact simulation and against the device-like simulator with the chip's reported error rates. No hyperparameter, checkpoint or threshold is selected using test data. The analysis will be pre-registered before execution.
>
> WP3 — H2: short on-chip training (Ona, conditional). Binary Iris, SPSA with mini-batches of 10 samples for 100 steps, with validation at every step: 3,515 circuits, 1000 shots. If batched submission makes it affordable, also 10 sweeps of the sequential optimizer (28,165 circuits) for a fixed-budget comparison. We compare against identical simulated runs (exact and noisy). WP3 runs only after WP2, and only if the measured execution rate fits within the allocation.
>
> WP4 — Noise-model simulations (MN5 GPP). Using the error rates reported for the chip, we repeat the 10-partition study with shots and noise to predict hardware behaviour and to interpret WP2/WP3. Measured cost on a workstation: about 79,900 CPU-seconds per partition for Breast Cancer and 10,900 for Iris (single-threaded jobs), so about 250 CPU-hours for 10 partitions. We request 2,000 core-hours to cover the different CPU speed, possible whole-node accounting and one re-run.
>
> Time on Ona. The application guide gives about 20 s per job (depth 1000, 10,000 shots), mostly connection overhead. Treating every circuit as an independent job as a conservative upper bound, WP1+WP2 need about 19 h (3,300 circuits) and WP3-SPSA about 20 h; we therefore request 40 h of system time. With batched submission the quantum execution time itself is estimated at under one hour for WP1–WP2, and the remainder would allow the sequential run in WP3. We will report the measured per-circuit and per-job times.
>
> Deliverables. Result files (JSON with configuration, revision, seeds, calibration and per-sample outputs), an analysis of the noise impact and of circuit-efficient training on real hardware, and the hardware chapter of the thesis.

### 2e) Computational algorithms and codes outline (máx. 3000 caracteres)

> Python 3 workflow based on Qibo (circuit construction) and Qililab (execution on the BSC chips). The variational loop runs on the classical node and only circuit executions are sent to the chip.
>
> - Number of qubits required: 4 (1 measured). Any 4 connected qubits of the chip are sufficient.
> - Maximum circuit depth: logical depth 25–26 (11 CNOT). After routing to {RX, RY, RZ, CZ}, native depth ≤ 38 on line, ring or grid couplings (≤ 2 SWAPs) and ≤ 68 on a star coupling (6 SWAPs). To be confirmed with the chip's native gate set.
> - Number of shots: 1000 per circuit (up to 10,000 for readout calibration).
> - Gate set used: RY, RZ, H, CNOT (logical); RX/RY rotations, virtual RZ and CZ after transpilation. Measurement in the computational basis.
> - Estimated duration of the longest job: a batch of 1,376 circuits × 1000 shots (Breast Cancer inference). About 5 min under an assumed 200 µs per shot plus 20 s overhead; if each circuit must be submitted separately, about 20 s per circuit (upper bound).
>
> Own codes (validated with 180 unit tests): QCNN builder; transpiler (layout minimizing interaction distance, shortest-path SWAP routing, native decomposition, rotation merging); batched executor interface and Qililab adapter; readout calibration/mitigation; training (GD, Adam, SPSA, sequential reconstruction) with validation-only checkpointing; hardware-inference and smoke-test drivers with provenance. Classical simulation uses NumPy (vectorized state-vector and density-matrix simulators validated against Qibo to 1e-10). HPC simulations run as Slurm job arrays of independent single-core jobs.

## Paso 3. Software

### 3d) Other requested software

> Python 3 with NumPy, SciPy, scikit-learn, NetworkX and Matplotlib (pinned in requirements.txt), and Qibo. The project currently pins Qibo 0.1.11; we will adapt to the Qibo/Qililab versions provided at BSC. No proprietary software.

### 3e) Proprietary software

Ninguno.

## Paso 4. Equipo

| Campo | Contenido |
|---|---|
| Team leader | `<NOMBRE DEL ESTUDIANTE>`, `<UNIVERSIDAD>`, `<EMAIL>` |
| Otros miembros | `<NOMBRE DEL TUTOR>, <UNIVERSIDAD>, <EMAIL>` |
| CV del responsable (máx. 2500 caracteres) | `<Grado y curso; formación en computación cuántica y ML; experiencia con Python/Qibo; descripción breve del TFG y de la plataforma; sin experiencia previa en HPC (por eso la opción de usuario novel)>` |
| Publicaciones | Ninguna (opcional) |

## Paso 5. Recursos

| Campo del formulario | MareNostrum5-Ona | MareNostrum5 GPP |
|---|---|---|
| Total requested time | **40** (en horas, no en miles, como exige la guía) | **2** (miles de horas-núcleo) |
| Number of processors needed for each job | 1000 (valor que exige la guía) | 1 (trabajos independientes de un núcleo; ajustar a 112 si MN5 contabiliza nodos completos) |
| Estimated number of jobs to submit | 40 (igual al tiempo total, como exige la guía) | ≈ 400 |
| Average job duration (hours) | 1 | ≤ 4 (máximo medido: 4,1 h en el caso más caro) |
| Total memory used by the job (GB) | 1 | 1 |
| Disk / scratch / tape (GB) | 256 / 256 / 256 (mínimo de la guía) | 256 / 256 / 256 (uso real previsto < 5 GB) |
| Asignación | Only selected machine | Any machine |

## Pasos 6–8. Resumen público

> We will run a small quantum convolutional neural network (4 qubits) on the MareNostrum5-Ona quantum computer. The model was trained and validated in simulation; the aim is to measure how real hardware noise changes its accuracy, how much simple error mitigation recovers, and whether circuit-efficient training strategies remain advantageous on a real device. The work is part of a Bachelor's thesis that builds an open platform for quantum machine learning experiments.

---

## Origen de las cifras

| Cifra | Fuente |
|---|---|
| Profundidad, puertas, SWAPs; circuitos de H1/H2 | `docs/bsc/cost-estimates.md` (`experiments/estimate_hardware_cost.py`) |
| 1641 circuitos por repetición de WP1+WP2 | 17 + 248 + 1376 (escenarios H1-smoke, H1-iris y H1-breast-cancer) |
| 22 h por partición (Breast Cancer, con ruido) | Suma de `WALL_SECONDS` de `results/variational/B-confirm-bc/logs/B3*` (79 909 s); Iris: `I3*` (10 856 s) |
| 3,2–9,4 veces menos circuitos; −0,026 a −0,084 frente a la logística | `results/variational/C-partitions/analysis.md` |
| 180 pruebas | Log técnico, sección 41 |
| 20 s por trabajo y 5,5 h por 1000 trabajos | Guía de acceso de la RES v3.3 |

## Pendiente antes de enviar

- [ ] Decisiones de la tabla inicial, con el tutor.
- [ ] **Decidir si se evalúan las arquitecturas alternativas** (secciones 37 y 40 del log: `v1` ignora x3 y 3 de los 8 pesos; `v1-param-pool` y `qibo` ya están implementadas y validadas unitariamente en `src/qcnn_architectures.py`). Si se adopta otra arquitectura, ampliar el transpilador, regenerar `cost-estimates.md` y revisar 2a, 2d y 2e (profundidad, puertas, resultados citados). Si se mantiene `v1`, describir el modelo como de 3 características efectivas y valorar podar el qubit 3.
- [ ] Verificar e incluir las referencias `[REF]`: QCNN, parameter-shift, SPSA y Rotosolve/NFT.
- [ ] Rellenar los datos personales, el CV y la universidad.
- [ ] Confirmar con el BSC la topología y las puertas nativas del chip, y si MN5 GPP contabiliza nodos completos.
- [ ] Comprobar la fecha límite de la convocatoria de la RES.
- [ ] Revisar los límites de caracteres en el formulario real.
