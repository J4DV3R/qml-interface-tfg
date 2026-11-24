#!/bin/bash
# Hardware inference of trained checkpoints on a MareNostrum5-Ona chip (BSC plan, phase 3).
#
# NOT FUNCTIONAL YET unless the tutors provide a compatible external adapter.
# RUNCARD and CHIP_ID must come from the BSC quantum documentation.
# EDGES must be the real chip coupling map, not a generic topology.
# The RES guide states that jobs on the quantum partition must use the chip;
# simulations belong on MN5-GPP/ACC (slurm/mn5_gpp_array.sbatch).
#
# Usage:
#   sbatch --export=ALL,RUNS="<run-*.json ...>",OUTPUT=...,ADAPTER_MODULE=...,RUNCARD=...,CHIP_ID=...,EDGES=...,SHOTS=1000 slurm/ona_inference.sbatch
#
#SBATCH --job-name=qcnn-ona-inference
# The tutors must add and fill these directives for the target MN5 environment:
# #SBATCH --account=REAL_ACCOUNT
# #SBATCH --qos=REAL_QUANTUM_QOS
# #SBATCH --partition=REAL_QUANTUM_PARTITION
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

: "${RUNS:?set RUNS to the run JSON files to evaluate}"
: "${OUTPUT:?set OUTPUT to the result JSON path}"
: "${ADAPTER_MODULE:?set ADAPTER_MODULE to the external Python adapter module}"
: "${RUNCARD:?set RUNCARD to the supplied chip runcard path or identifier}"
: "${CHIP_ID:?set CHIP_ID to the supplied chip identifier}"
: "${EDGES:?set EDGES to the real chip coupling map, e.g. 0-1,1-2,2-3}"
: "${SHOTS:?set SHOTS to a positive shot count}"
: "${REPO_DIR:=$SLURM_SUBMIT_DIR}"

cd "$REPO_DIR"
# Environment/module names and setup commands must be supplied by the tutors.
python -B -c "import sys; print(sys.executable)"

START=$(date +%s)
set +e
# shellcheck disable=SC2086
python -B -m experiments.run_hardware_inference $RUNS \
  --executor qililab --runcard "$RUNCARD" \
  --qililab-adapter-module "$ADAPTER_MODULE" --chip-id "$CHIP_ID" \
  --edges "$EDGES" --shots "$SHOTS" --output "$OUTPUT"
CODE=$?
set -e
echo "EXIT_CODE=$CODE WALL_SECONDS=$(( $(date +%s) - START )) HOST=$(hostname)"
exit $CODE
