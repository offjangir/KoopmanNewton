#!/bin/bash
#SBATCH -J PAPERT0_smoke
#SBATCH -p h200,h100,a100,l40s
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH --exclude=gh205,csr122,csr066,csr067,csr072,csr124
#SBATCH -o /weka/home/jhu/ycai71/physwm/runs/paper_repro_t0/logs/smoke_%j.out
PY=/weka/projects/hbharad2/ycai71/envs/paper_repro/bin/python
R=/weka/home/jhu/ycai71/physwm/runs/paper_repro_t0
S=/weka/home/jhu/ycai71/physwm/scripts/paper_repro
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
nvidia-smi --query-gpu=name --format=csv,noheader
cd $R/smoke
for b in 1 100 20; do
  $PY $S/run_blocksize.py --blocks $b --n_steps 301 --ckpt_dir $R/smoke/ckpt --timing_file $R/smoke/timing.jsonl 2>&1 | awk '{print strftime("%T"), $0; fflush()}' > $R/smoke/run_b$b.out
done
cat $R/smoke/timing.jsonl
