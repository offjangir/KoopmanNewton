#!/usr/bin/env bash
#SBATCH --job-name=physwm-sweep
#SBATCH --account=becx-dtai-gh      # set to your allocation
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-149%8
#SBATCH --output=slurm-sweep-%A_%a.out
#
# Array job over generated configs (one GH200 GPU each; 8 concurrent).
# First: python scripts/gen_configs.py --out configs/generated
#   sbatch scripts/slurm/sweep.sbatch configs/generated
set -euo pipefail

DIR=${1:-configs/generated}
eval "$(conda shell.bash hook)"   # conda must be on PATH
conda activate physwm-gpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$SLURM_SUBMIT_DIR"

mapfile -t CONFIGS < <(ls "$DIR"/*.yaml | sort)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID -> $CFG"
srun python scripts/run_experiment.py --config "$CFG"
