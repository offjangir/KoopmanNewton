#!/bin/bash
# Tier 1 evaluation of one Tier 0 model on med (CPU). Usage: sbatch --export=ALL,BLOCK=<b> tier1_eval.sbatch
#SBATCH -p med
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH --exclude=gh205,csr122,csr066,csr067,csr072,csr124,csr058
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=
T=/weka/home/jhu/ycai71/physwm/runs/paper_repro_t1
/weka/projects/hbharad2/ycai71/envs/paper_repro/bin/python /weka/home/jhu/ycai71/physwm/scripts/paper_repro/tier1_eval.py \
  $BLOCK $T/data /weka/home/jhu/ycai71/physwm/runs/paper_repro_t0/checkpoints $T/model_$BLOCK > $T/logs/eval_b$BLOCK.out 2>&1
echo "exit $?"
