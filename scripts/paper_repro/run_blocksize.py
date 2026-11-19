"""Thin wrapper around the paper's kepler_cv_blocksize.py (arXiv 2602.06923, repo commit d1e37bd).

It calls the paper's own main() path (seeds, then sweep_parameters) with the paper's settings.
The only additions:
  * --blocks picks the block-size list (the paper ran [1,2,5,10,20,50,100] and [60,70,80,90]
    as two separate invocations of main()).
  * --n_steps overrides n_steps for a smoke test only.
  * After each block size, the final model state_dict and its config go to --ckpt_dir.
    The wrapper keeps a reference to the model that the paper's setup_model() builds.
  * Wall time per block size goes to --timing_file.
Run with cwd = a directory that holds data_cv/ (the paper repo for real runs).
No training code or setting is changed.
"""
import argparse
import json
import os
import sys
import time

PAPER = "/weka/home/jhu/ycai71/newton-kepler"

ap = argparse.ArgumentParser()
ap.add_argument("--blocks", required=True, help="block sizes separated by '+', e.g. 1+2+5")
ap.add_argument("--n_steps", type=int, default=20001)
ap.add_argument("--ckpt_dir", required=True)
ap.add_argument("--timing_file", required=True)
args = ap.parse_args()
blocks = [int(b) for b in args.blocks.split("+")]

sys.path.insert(0, PAPER)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import kepler_cv_blocksize as K  # noqa: E402  (module import sets seed 1, as in the paper)

_models = []
_orig_setup = K.setup_model


def _setup_and_keep(*a, **kw):
    m = _orig_setup(*a, **kw)
    _models.append(m)
    return m


K.setup_model = _setup_and_keep

_orig_train = K.train_one_model


def _train_and_save(**kw):
    t0 = time.time()
    res = _orig_train(**kw)
    dt = time.time() - t0
    model = _models[-1]
    c = K.GPTConfigCV
    cfg = dict(block_size=c.block_size, n_layer=c.n_layer, n_head=c.n_head, n_embd=c.n_embd,
               dropout=c.dropout, bias=c.bias, input_dim=c.input_dim, attention_alpha=c.attention_alpha,
               noise_scale=kw["noise_scale"], lr=kw["lr"], num_trajectories=kw["num_trajectories"],
               n_steps=kw["n_steps"], prob_freq=kw["prob_freq"], loss_mask=kw["loss_mask"], seed=kw["seed"],
               paper_commit="d1e37bd")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, f"model_block_size_{kw['block_size']}.pt")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": cfg}, path)
    with open(args.timing_file, "a") as f:
        f.write(json.dumps({"block_size": kw["block_size"], "n_steps": kw["n_steps"], "seconds": dt,
                            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}) + "\n")
    print(f"[wrapper] block_size={kw['block_size']} took {dt:.1f} s, checkpoint {path}", flush=True)
    return res


K.train_one_model = _train_and_save

# Same as the paper's main(), with the block list from --blocks.
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
K.sweep_parameters(blocks, [10000], [0.1], ['all'], n_steps=args.n_steps, prob_freq=100, seed=seed)
