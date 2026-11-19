"""Collect the Tier 1 results of the 11 models into CSV tables, HTML table rows and figures.

Usage: python tier1_aggregate.py <tier1_dir>
Kepler score = mean of the best-layer R2 of a, b, LRL_x, LRL_y. Newton score = same for F_magnitude, Fx, Fy.
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

T = sys.argv[1]
out = f"{T}/tables"
os.makedirs(out, exist_ok=True)
# columns of the main page tables, in order
MAIN = ["reference_in_sample_training",
        "step1_fit_training_score_A",
        "step2_in_sample_BmA", "step3_split_by_orbit_BmA",
        "step1_fit_training_score_C_all", "step2_in_sample_C_all", "step3_split_by_orbit_C_all"]
LABEL = {"reference_in_sample_training": "in-sample on training orbits (paper's way)",
         "step1_fit_training_score_A": "training-orbit probes on held-out orbits from the paper's range",
         "step2_in_sample_BmA": "in-sample on wider orbits",
         "step3_split_by_orbit_BmA": "fit and score on different wider orbits",
         "step1_fit_training_score_C_all": "training-orbit probes on held-out far orbits",
         "step2_in_sample_C_all": "in-sample on held-out far orbits",
         "step3_split_by_orbit_C_all": "fit and score on different held-out far orbits"}
SETS = ["train", "A", "BmA", "C_all"]
SETLABEL = {"train": "training orbits", "A": "held-out orbits from the paper's range",
            "BmA": "wider orbits outside the paper's range", "C_all": "held-out far orbits",
            "C_ecc": "far orbits, eccentricity 0.9-0.95", "C_far": "far orbits, semi-major axis 3-4"}
KEP, NEW = ["a", "b", "LRL_x", "LRL_y"], ["F_magnitude", "Fx", "Fy"]

R = {}
for f in glob.glob(f"{T}/model_*/results.json"):
    r = json.load(open(f))
    R[r["block_size"]] = r
blocks = sorted(R)
steps_all = list(R[blocks[0]]["probe_steps"].keys())


def best(d, t):
    return max(d[l][t] for l in d)


scores = {}
with open(f"{out}/probe_r2_per_layer.csv", "w", newline="") as f1, open(f"{out}/scores.csv", "w", newline="") as f2:
    w1, w2 = csv.writer(f1), csv.writer(f2)
    w1.writerow(["context_length", "step", "layer", "target", "r2_last_position"])
    w2.writerow(["context_length", "step", "kepler_score", "newton_score"] + [f"best_{t}" for t in KEP + NEW + ["e"]])
    for b in blocks:
        for s in steps_all:
            d = R[b]["probe_steps"][s]
            for l in d:
                for t, v in d[l].items():
                    w1.writerow([b, s, l, t, v])
            k = np.mean([best(d, t) for t in KEP])
            n = np.mean([best(d, t) for t in NEW])
            scores[(b, s)] = (k, n)
            w2.writerow([b, s, f"{k:.4f}", f"{n:.4f}"] + [f"{best(d, t):.4f}" for t in KEP + NEW + ["e"]])
with open(f"{out}/errors.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["context_length", "orbit_set", "next_step_error", "rollout_error_mean_50_steps"])
    for b in blocks:
        for s in R[b]["errors"]:
            e = R[b]["errors"][s]
            w.writerow([b, s, f"{e['next_step_error']:.5f}", f"{e['rollout_error_mean']:.4f}"])

meta = torch.load("/weka/home/jhu/ycai71/newton-kepler/data_cv/metadata.pt", weights_only=False)
(x0, x1), (y0, y1) = meta["x_range"], meta["y_range"]
frac = {}
for s in ("A", "BmA", "C_ecc", "C_far"):
    X = np.load(f"{T}/data/{s}.npz")["trajectories"]
    o = (X[..., 0] < x0) | (X[..., 0] > x1) | (X[..., 1] < y0) | (X[..., 1] > y1)
    frac[s] = float(o.mean())
json.dump({"x_range": [x0, x1], "y_range": [y0, y1], "fraction_outside": frac}, open(f"{out}/outside_range.json", "w"), indent=1)
print("fraction of positions outside the training x and y range", frac)

for i, name in enumerate(["kepler", "newton"]):
    with open(f"{out}/rows_{name}.html", "w") as f:
        for b in blocks:
            f.write(f"<tr><td>{b}</td>" + "".join(f"<td>{scores[(b, s)][i]:.3f}</td>" for s in MAIN) + "</tr>\n")
with open(f"{out}/rows_rollout.html", "w") as f:
    for b in blocks:
        f.write(f"<tr><td>{b}</td>" + "".join(f"<td>{R[b]['errors'][s]['rollout_error_mean']:.3f}</td>" for s in SETS)
                + "</tr>\n")


def logx(ax):
    lab = [1, 2, 5, 10, 20, 50, 100]
    ax.set_xscale("log")
    ax.set_xticks(lab)
    ax.set_xticklabels([str(v) for v in lab])
    ax.set_xticks([v for v in blocks if v not in lab], minor=True)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("context length")
    ax.grid(alpha=0.3)


STY = {"reference_in_sample_training": ("k", "-"), "step1_fit_training_score_A": ("C0", "-"),
       "step2_in_sample_BmA": ("C1", "-"), "step3_split_by_orbit_BmA": ("C1", "--"),
       "step1_fit_training_score_C_all": ("C3", ":"), "step2_in_sample_C_all": ("C3", "-"),
       "step3_split_by_orbit_C_all": ("C3", "--")}
for i, name in enumerate(["Kepler score", "Newton score"]):
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in MAIN:
        c, ls = STY[s]
        ax.plot(blocks, [scores[(b, s)][i] for b in blocks], marker="o", color=c, ls=ls, label=LABEL[s])
    logx(ax)
    ax.set_ylabel(f"{name} (mean of best-layer R²)")
    ax.set_title(f"{name} against context length")
    lo = min(scores[(b, s)][i] for b in blocks for s in MAIN)
    if lo < -1:
        ax.set_ylim(-1, 1.02)
        ax.text(0.01, 0.01, "Values below -1 are outside the plot.", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out}/{name.split()[0].lower()}_score_vs_context.png", dpi=150)

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for s, c in zip(SETS, ["k", "C0", "C1", "C3"]):
    axs[0].plot(blocks, [R[b]["errors"][s]["rollout_error_mean"] for b in blocks], "o-", color=c, label=SETLABEL[s])
    axs[1].plot(blocks, [R[b]["errors"][s]["next_step_error"] for b in blocks], "o-", color=c, label=SETLABEL[s])
for ax, yl in zip(axs, ["mean distance error over 50 predicted steps", "next-step distance error"]):
    logx(ax)
    ax.set_ylabel(yl)
    ax.set_yscale("log")
axs[0].set_title("Rollout error from 50 known points")
axs[1].set_title("Next-step error")
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{out}/errors_vs_context.png", dpi=150)

fig, axs = plt.subplots(2, 6, figsize=(18, 6.5))
for row, b in enumerate((2, 100)):
    d = np.load(f"{T}/model_{b}/example_rollouts_far.npz")
    for k in range(6):
        ax = axs[row, k]
        tr, pr = d["true"][k], d["pred"][k]
        ax.plot(tr[:, 0], tr[:, 1], color="0.6", lw=1, label="true orbit")
        ax.plot(pr[50:, 0], pr[50:, 1], color="C3", lw=1.2, label="model rollout")
        ax.plot(tr[:50, 0], tr[:50, 1], color="k", lw=1.5, label="50 known points")
        ax.plot(0, 0, "k+")
        ax.set_aspect("equal", "datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("eccentricity 0.9-0.95" if k < 3 else "semi-major axis 3-4", fontsize=9)
        if k == 0:
            ax.set_ylabel(f"context length {b}")
axs[0, 0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(f"{out}/example_rollouts_far.png", dpi=130)

for b in blocks:
    print(b, " ".join(f"{scores[(b, s)][0]:.3f}/{scores[(b, s)][1]:.3f}" for s in MAIN), "|",
          " ".join(f"{R[b]['errors'][s]['rollout_error_mean']:.3f}" for s in SETS))
