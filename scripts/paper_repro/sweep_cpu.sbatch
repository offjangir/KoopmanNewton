#!/bin/bash
# One block size of the paper's Fig. 6 sweep on a med CPU node. Usage: sbatch --export=ALL,BLOCK=<b> sweep_cpu.sbatch
#SBATCH -p med
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 02:30:00
#SBATCH --exclude=gh205,csr122,csr066,csr067,csr072,csr124
PY=/weka/projects/hbharad2/ycai71/envs/paper_repro/bin/python
R=/weka/home/jhu/ycai71/physwm/runs/paper_repro_t0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=
lscpu | grep "Model name"
cd /weka/home/jhu/ycai71/newton-kepler
$PY /weka/home/jhu/ycai71/physwm/scripts/paper_repro/run_blocksize.py --blocks $BLOCK --n_steps 20001 \
  --ckpt_dir $R/checkpoints --timing_file $R/timing.jsonl 2>&1 | awk '{print strftime("%T"), $0; fflush()}' > $R/logs/sweep/run_b$BLOCK.out
echo "exit ${PIPESTATUS[0]}"
