"""Make the Tier 0 result tables and one summary figure from the paper's result .npz files.

Usage: python make_table.py <results_dir> <out_dir> <paper_stream_txt>
Writes:
  probe_r2_per_layer.csv  block size, layer, target, R2 on last position (r2_last), R2 on all positions (r2)
  probe_r2_best.csv       block size, best r2_last over layers per target (the notebook's rule), final test error
  summary_vs_block.png    best r2_last against block size, ours (all 11) and paper force values (notebook text)
The probe values are those of the last evaluation (step 20001), as in the paper notebook.
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res_dir, out_dir, paper_txt = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
FORCE = ["F_magnitude", "Fx", "Fy"]
GEOM = ["e", "a", "b", "LRL_x", "LRL_y", "LRL_magnitude"]

rows, best = [], []
files = glob.glob(os.path.join(res_dir, "results_block_size_*_num_trajectories_10000_noise_scale_0.1_loss_mask_all.npz"))
blocks = sorted(int(re.search(r"block_size_(\d+)_", f).group(1)) for f in files)
for b in blocks:
    d = np.load(os.path.join(res_dir, f"results_block_size_{b}_num_trajectories_10000_noise_scale_0.1_loss_mask_all.npz"),
                allow_pickle=True)
    ev = d["eval_results"][-1]
    steps = list(d["eval_steps"])
    pr, gr = ev["probe_results"], ev["geometry_probe_results"]
    rec = {"block_size": b, "n_train_steps": len(d["train_losses"]), "last_eval_step": steps[-1],
           "n_evals": len(steps), "final_test_mean_error": float(np.mean(ev["error_stats_test"]["position_errors"]))}
    for group, names in ((pr, FORCE), (gr, GEOM)):
        for layer, probes in group.items():
            for t in names:
                rows.append([b, layer, t, probes[t]["r2_last"], probes[t]["r2"]])
        for t in names:
            rec[f"best_r2_last_{t}"] = max(group[l][t]["r2_last"] for l in group)
    best.append(rec)

with open(os.path.join(out_dir, "probe_r2_per_layer.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["block_size", "layer", "target", "r2_last", "r2_all"])
    w.writerows(rows)
with open(os.path.join(out_dir, "probe_r2_best.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(best[0].keys()))
    w.writeheader()
    w.writerows(best)

paper = {}
for line in open(paper_txt):
    m = re.match(r"block_size=\s*(\d+): min_error=([\d.]+), R2_F_mag=([\d.]+), R2_Fx=([\d.]+), R2_Fy=([\d.]+)", line)
    if m:
        paper[int(m.group(1))] = [float(x) for x in m.groups()[1:]]
pb = sorted(paper)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
B = np.array([r["block_size"] for r in best])
for t, c in zip(FORCE, ["C0", "C1", "C2"]):
    ax[0].plot(B, [r[f"best_r2_last_{t}"] for r in best], "o-", color=c, label=f"{t} (ours)")
for i, (t, c) in enumerate(zip(FORCE, ["C0", "C1", "C2"])):
    ax[0].plot(pb, [paper[b][i + 1] for b in pb], "x--", color=c, ms=9, label=f"{t} (paper notebook)")
ax[0].set(xscale="log", xlabel="context length (block size)", ylabel="best R² over layers (last position)",
          title="Force probes")
ax[0].legend(fontsize=8)
for t, c in zip(["a", "b", "LRL_x", "LRL_y", "e"], ["C0", "C1", "C2", "C3", "C4"]):
    ax[1].plot(B, [r[f"best_r2_last_{t}"] for r in best], "o-", color=c, label=t)
ax[1].set(xscale="log", xlabel="context length (block size)", ylabel="best R² over layers (last position)",
          title="Orbit-element probes (ours)")
ax[1].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "summary_vs_block.png"), dpi=150)
for r in best:
    print(r)
print("paper", paper)
