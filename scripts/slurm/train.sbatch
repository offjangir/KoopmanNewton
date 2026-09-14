#!/usr/bin/env bash
#SBATCH --job-name=physwm
#SBATCH --account=becx-dtai-gh      # set to your allocation
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#
# Single-GPU training of one config on a GH200.
#   sbatch scripts/slurm/train.sbatch configs/kepler_ar.yaml
set -euo pipefail

CONFIG=${1:?usage: sbatch train.sbatch <config.yaml> [overrides...]}
shift || true

eval "$(conda shell.bash hook)"   # conda must be on PATH
conda activate physwm-gpu

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$SLURM_SUBMIT_DIR"

srun python scripts/run_experiment.py --config "$CONFIG" ${@:+--set "$@"}
