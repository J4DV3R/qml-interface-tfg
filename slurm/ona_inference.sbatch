#!/bin/bash
# Hardware inference of trained checkpoints on a MareNostrum5-Ona chip (BSC plan, phase 3).
#
# NOT FUNCTIONAL YET: src.hardware_execution.QililabExecutor is a skeleton and
# every value marked <...> must come from the BSC quantum documentation
# (account, partition/QOS, modules, runcard, chip and its coupling map).
# The RES guide states that jobs on the quantum partition must use the chip;
# simulations belong on MN5-GPP/ACC (slurm/mn5_gpp_array.sbatch).
#
# Usage (after filling the placeholders):
#   sbatch --export=ALL,RUNS="<run-*.json ...>",OUTPUT=results/hardware/H1/<name>.json slurm/ona_inference.sbatch
#
#SBATCH --job-name=qcnn-ona-inference
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QUANTUM_QOS>
#SBATCH --partition=<QUANTUM_PARTITION>
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

: "${RUNS:?set RUNS to the run JSON files to evaluate}"
: "${OUTPUT:?set OUTPUT to the result JSON path}"
: "${RUNCARD:=<RUNCARD_PATH>}"
: "${EDGES:=<CHIP_COUPLING_MAP, e.g. 0-1,1-2,2-3>}"
: "${SHOTS:=1000}"
: "${REPO_DIR:=$SLURM_SUBMIT_DIR}"

cd "$REPO_DIR"
# module load <qibo/qililab modules provided by the BSC>
# source <environment with qibo + qililab>

START=$(date +%s)
set +e
# shellcheck disable=SC2086
python -B -m experiments.run_hardware_inference $RUNS \
  --executor qililab --runcard "$RUNCARD" --edges "$EDGES" \
  --shots "$SHOTS" --output "$OUTPUT"
CODE=$?
set -e
echo "EXIT_CODE=$CODE WALL_SECONDS=$(( $(date +%s) - START )) HOST=$(hostname)"
exit $CODE
