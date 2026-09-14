#!/usr/bin/env bash
# Environment setup for NCSA Delta-AI (NVIDIA GH200, aarch64 / ARM Grace).
#
# The GPU/CUDA aarch64 ("sbsa") PyTorch wheels live on the PyTorch index, NOT on
# default PyPI (which historically ships CPU-only for aarch64). cu128 wheels
# include sm_90 (Hopper/GH200). Run this on a login node; run training via SLURM.
#
#   bash scripts/env_setup.sh
set -euo pipefail

eval "$(conda shell.bash hook)"   # conda must be on PATH

ENV=${1:-physwm-gpu}
conda create -y -n "$ENV" python=3.11
conda activate "$ENV"

test "$(uname -m)" = "aarch64" || { echo "expected aarch64"; exit 1; }

# GPU-enabled aarch64 + CUDA 12.8 wheels (sm_90 for GH200)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy matplotlib pyyaml pytest

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
assert torch.__version__.endswith(("cu128", "cu126")), "got a non-CUDA wheel!"
PY

# install this package (editable)
pip install -e .
echo "Done. Activate with: conda activate $ENV"
