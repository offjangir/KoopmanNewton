"""Example rollouts with marks, for context lengths 2 and 100 (paper models, arXiv 2602.06923).

Usage: python tier1_example_rollouts.py <tier1_dir> <ckpt_dir> <which: training|far> <out_png>
training: 6 of the paper's training orbits (first 3 with e > 0.6 and first 3 with e < 0.3 in data_cv chunk 0).
far: the same 6 held-out far orbits as example_rollouts_far.png (first 3 of C_ecc, first 3 of C_far).
The rollout follows the paper's generate_trajectory_and_compute_error: 50 known points, then 50 predicted
points, each fed back (sliding window of the context length for context < 100, model.generate for 100).
The rollout error in each title is the paper function's mean distance error over the 50 predicted points.
"""
import importlib.util
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

PAPER = "/weka/home/jhu/ycai71/newton-kepler"
sys.path.insert(0, PAPER)
import kepler_cv_blocksize as K  # noqa: E402

T, ckpt_dir, which, out_png = sys.argv[1:5]
torch.set_grad_enabled(False)

if which == "training":
    trajs = torch.load(f"{PAPER}/data_cv/trajectories_chunk_000000.pt").numpy()
    spec = importlib.util.spec_from_file_location("p0", f"{PAPER}/data_cv/orbital_params_chunk_000000.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    e = np.array([p["e"] for p in mod.orbital_params])
    a = np.array([p["a"] for p in mod.orbital_params])
    pick = list(np.where(e > 0.6)[0][:3]) + list(np.where(e < 0.3)[0][:3])
    ex, ee, aa = trajs[pick], e[pick], a[pick]
else:
    d1, d2 = np.load(f"{T}/data/C_ecc.npz"), np.load(f"{T}/data/C_far.npz")
    ex = np.concatenate([d1["trajectories"][:3], d2["trajectories"][:3]])
    ee = np.concatenate([d1["e"][:3], d2["e"][:3]])
    aa = np.concatenate([d1["a"][:3], d2["a"][:3]])


def rollout(model, b, x):
    seq = torch.from_numpy(x[:, :50]).float()
    if b < K.num_points_per_trajectory:
        for _ in range(50):
            p, _ = model.forward(seq[:, -b:], None)
            seq = torch.cat([seq, p[:, -1:, :]], dim=1)
    else:
        seq = model.generate(seq, max_new_tokens=50)
    return seq.numpy()


def arrow(ax, pts, i, color):
    ax.annotate("", xy=pts[i + 1], xytext=pts[i],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=12))


fig, axs = plt.subplots(2, 6, figsize=(19, 7.6))
for row, b in enumerate((2, 100)):
    ck = torch.load(f"{ckpt_dir}/model_block_size_{b}.pt", weights_only=False)
    model = K.setup_model(block_size=b, n_layer=ck["config"]["n_layer"], n_embd=ck["config"]["n_embd"], device="cpu")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    pred = rollout(model, b, ex)
    for k in range(6):
        st = K.generate_trajectory_and_compute_error(model, torch.from_numpy(ex[k:k + 1, :50]).float(), ex[k:k + 1], 50,
                                                     block_size=b)
        err = float(np.mean(st["position_errors"]))
        mine = float(np.mean(np.linalg.norm(pred[k, 50:100] - ex[k, 50:100], axis=-1)))
        assert abs(err - mine) < 1e-4, (err, mine)
        ax = axs[row, k]
        tr, pr = ex[k], pred[k]
        ax.plot(tr[49:, 0], tr[49:, 1], color="0.65", lw=1.2)
        ax.plot(tr[:50, 0], tr[:50, 1], color="k", lw=1.6)
        ax.plot(pr[49:, 0], pr[49:, 1], color="C3", lw=1.4)
        arrow(ax, tr, 20, "k")
        arrow(ax, tr, 40, "k")
        arrow(ax, pr, 62, "C3")
        arrow(ax, pr, 85, "C3")
        arrow(ax, tr, 75, "0.65")
        ax.plot(*tr[49], "o", color="k", ms=7)
        ax.plot(*pr[50], "o", mfc="white", mec="C3", mew=1.8, ms=8)
        ax.plot(0, 0, "+", color="k", ms=11, mew=2)
        ax.set_aspect("equal", "datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"eccentricity {ee[k]:.2f}, semi-major axis {aa[k]:.2f}\nrollout error {err:.3f}", fontsize=9)
        if k == 0:
            ax.set_ylabel(f"context length {b}", fontsize=12)
handles = [Line2D([], [], color="k", lw=1.6, label="50 known points"),
           Line2D([], [], color="C3", lw=1.4, label="model rollout (50 predicted points)"),
           Line2D([], [], color="0.65", lw=1.2, label="true orbit (next 50 points)"),
           Line2D([], [], ls="", marker="o", color="k", ms=7, label="last known point"),
           Line2D([], [], ls="", marker="o", mfc="white", mec="C3", mew=1.8, ms=8, label="first predicted point"),
           Line2D([], [], ls="", marker="+", color="k", ms=11, mew=2, label="Sun")]
title = ("training orbits of the models" if which == "training"
         else "held-out far orbits (eccentricity 0.9-0.95, or semi-major axis 3-4)")
fig.suptitle(f"Example rollouts on {title}. Arrows show the direction of motion.", fontsize=13)
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=10, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(out_png, dpi=130)
print("saved", out_png)
