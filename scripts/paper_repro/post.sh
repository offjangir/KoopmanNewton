#!/bin/bash
# Post-processing on med (CPU): copy results, run the paper's fig6 notebook unchanged, make tables.
set -e
PY=/weka/projects/hbharad2/ycai71/envs/paper_repro/bin
R=/weka/home/jhu/ycai71/physwm/runs/paper_repro_t0
P=/weka/home/jhu/ycai71/newton-kepler
mkdir -p $R/results/kepler_cv_blocksize
cp -n $P/results/kepler_cv_blocksize/*.npz $R/results/kepler_cv_blocksize/
N=$R/notebook; mkdir -p $N/plots/app2
ln -sfn $R/results $N/results; ln -sfn $P/data_cv $N/data_cv
cp $P/fig6_block_size_kepler_newton.ipynb $N/fig6_block_size_kepler_newton.ipynb
cd $N
$PY/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 \
  --output fig6_executed.ipynb fig6_block_size_kepler_newton.ipynb
$PY/python /weka/home/jhu/ycai71/physwm/scripts/paper_repro/make_table.py $R/results/kepler_cv_blocksize $R/tables \
  $R/paper_notebook_outputs/cell8_stream.txt
