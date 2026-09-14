#!/usr/bin/env bash
# Sigma contrast for one law on a single GPU: {ar, flow, diffusion} x sigma {0, 0.1}.
#
#   bash scripts/run_contrast.sh pendulum                 # -> runs/contrast_pendulum/
#   bash scripts/run_contrast.sh kepler runs/h5b          # the headline Kepler contrast
set -e
DOMAIN=${1:?usage: run_contrast.sh <domain> [out_root]}
OUT_ROOT=${2:-runs/contrast_${DOMAIN}}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
# position-only (2) for 2nd-order laws where the Verlet force test applies;
# full state (none) for 1st-order laws measured by invariant conservation.
case "$DOMAIN" in
  kepler|gr_kepler|pendulum) OBS=2 ;;
  *) OBS=none ;;
esac
COMMON="data.domain=$DOMAIN data.online=true data.online_workers=4 data.obs_dims=$OBS data.context=2 \
 data.n_train=300 data.n_val=200 data.n_test=600 model.d_model=256 model.n_layers=6 model.attn_window=2 \
 train.steps=10000 train.batch_size=512 train.device=cuda train.eval_every=0 train.ckpt_every=0"
for M in ar flow diffusion; do
  for S in 0.0 0.1; do
    OUT=$OUT_ROOT/${M}_s${S}
    [ -f "$OUT/final.pt" ] && { echo "skip $OUT"; continue; }
    echo "=== $DOMAIN $M sigma=$S ==="
    python scripts/run_experiment.py --config configs/kepler_ar.yaml --set $COMMON \
      method.kind=$M method.ctx_noise_sigma=$S train.out_dir=$OUT name=${DOMAIN}_${M}_s${S} 2>&1 | tail -1
  done
done
echo "=== DONE training $DOMAIN contrast ==="
