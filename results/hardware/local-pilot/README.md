# Piloto local simulado (`results/hardware/local-pilot/`)

**Estado: prueba en seco en simulador (`HardwareMockExecutor`), no ejecución en hardware cuántico real.**

## Archivo registrado

- [`mock-sequential-iris-init19.json`](mock-sequential-iris-init19.json)
  - **Tipo:** `hardware-inference` (`status: "dry-run (simulator mock)"`).
  - **Checkpoint de entrada:** `results/variational/B-confirm-iris/I1-exact/sequential-angle/run-angle-sequential-init19.json` (seleccionado por validation durante el entrenamiento; este piloto no selecciona hiperparámetros ni pesos).
  - **Configuración:** `--executor mock --topology line --device-qubits 5 --shots 100 --calibration-shots 100 --noise-1q 0.001 --noise-2q 0.01 --readout-p01 0.02 --readout-p10 0.04 --splits test --max-samples 2 --seed 2026`.
  - **Revisión registrada:** `62618f12411b8e095005761af4feeac9ede07115` (árbol sucio con `M requirements.txt`).

## Limitaciones

- Simulación local con topología genérica (`line`, 5 qubits) y parámetros de ruido/lectura supuestos (no calibración de MareNostrum5-Ona).
- Muestra mínima de humo (`--max-samples 2`, 100 shots): sirve únicamente para verificar el flujo extremo a extremo y el esquema JSON de `experiments/run_hardware_inference.py`, no para evaluar rendimiento.
