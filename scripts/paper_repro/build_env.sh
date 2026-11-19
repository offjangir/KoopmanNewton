#!/usr/bin/env bash
# Build the Python environment for the paper reproduction (Tier 0).
# Base: the move3d vla Python (torch 2.7.1). Adds the paper's missing requirements.
# Run on a med node: srun -p med ... bash build_env.sh
set -euo pipefail
BASE=/weka/projects/hbharad2/ycai71/move3d/envs/vla/bin/python3.11
ENV=/weka/projects/hbharad2/ycai71/envs/paper_repro
export PIP_CACHE_DIR=/weka/projects/hbharad2/ycai71/pip_cache
[ -x $ENV/bin/python ] || $BASE -m venv --system-site-packages $ENV
$ENV/bin/pip install -q scikit-learn matplotlib nbconvert nbclient ipykernel
$ENV/bin/python -m ipykernel install --prefix $ENV --name python3 --display-name "Python 3"
$ENV/bin/python - <<'PY'
import torch, numpy, scipy, sklearn, matplotlib, yaml, tqdm, nbconvert, sys
print("python", sys.version.split()[0])
for m in (torch, numpy, scipy, sklearn, matplotlib, yaml, tqdm, nbconvert):
    print(m.__name__, m.__version__)
print("torch cuda", torch.version.cuda)
PY
