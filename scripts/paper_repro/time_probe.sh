#!/bin/bash
# Usage: time_probe.sh <workdir> <threads> <block> [<block> ...]
# Runs each block size as its own process in parallel, 101 steps (two evaluations), with timestamps.
PY=/weka/projects/hbharad2/ycai71/envs/paper_repro/bin/python
S=/weka/home/jhu/ycai71/physwm/scripts/paper_repro
W=$1; T=$2; shift 2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
cd $W; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; lscpu | grep "Model name"
for b in "$@"; do
  ( $PY $S/run_blocksize.py --blocks $b --n_steps 101 --ckpt_dir $W/ckpt --timing_file $W/timing.jsonl 2>&1 | awk '{print strftime("%T"), $0; fflush()}' > $W/run_b$b.out ) &
done
wait; cat $W/timing.jsonl
