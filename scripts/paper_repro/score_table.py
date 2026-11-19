"""Kepler and Newton scores (definitions of the paper's fig6 phase-transition cell), ours and paper.

Kepler score = mean of best-layer last-position R2 of a, b, LRL_x, LRL_y.
Newton score = mean of best-layer last-position R2 of F_magnitude, Fx, Fy.
Ours: from probe_r2_best.csv. Paper Newton: from the stored notebook text.
Paper Kepler: read by pixel from the stored phase-transition image (cell 13), because the
notebook stores these values only in that image.

Pixel reading: the axes frame gives the y range (ylim 0.895 to 1.01, set in the cell).
Markers are the blue (C0, Kepler) and orange (C1, Newton) blobs left after an erosion
that removes the thinner lines. The error estimate comes from two checks:
the dashed line at y = 1.0, and the orange markers against the paper Newton text values.

Usage: python score_table.py <best_csv> <paper_stream_txt> <paper_png> <out_dir>
"""
import csv
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy import ndimage

best_csv, paper_txt, png, out = sys.argv[1:5]

ours = {}
for r in csv.DictReader(open(best_csv)):
    b = int(r["block_size"])
    k = np.mean([float(r[f"best_r2_last_{t}"]) for t in ("a", "b", "LRL_x", "LRL_y")])
    n = np.mean([float(r[f"best_r2_last_{t}"]) for t in ("F_magnitude", "Fx", "Fy")])
    ours[b] = (k, n)

paper_newton = {}
for line in open(paper_txt):
    m = re.match(r"block_size=\s*(\d+): .*R2_F_mag=([\d.]+), R2_Fx=([\d.]+), R2_Fy=([\d.]+)", line)
    if m:
        paper_newton[int(m.group(1))] = np.mean([float(x) for x in m.groups()[1:]])

# ---- pixel reading of the paper image ----
_raw = mpimg.imread(png)
img = _raw[..., :3] * _raw[..., 3:4] + (1 - _raw[..., 3:4]) if _raw.shape[-1] == 4 else _raw[..., :3]
H, W, _ = img.shape
dark = img.sum(-1) < 2.1
gray = np.abs(img - img.mean(-1, keepdims=True)).max(-1) < 0.08
dark &= gray
# axes frame: long dark rows and columns
def maxrun(v):
    best = cur = 0
    for x in v:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


row_runs = np.array([maxrun(dark[r]) for r in range(H)])
col_runs = np.array([maxrun(dark[:, c]) for c in range(W)])
frame_rows = np.where(row_runs > 0.5 * W)[0]
frame_cols = np.where(col_runs > 0.5 * H)[0]
print("row runs top", sorted(zip(row_runs, range(H)))[-8:], "col runs top", sorted(zip(col_runs, range(W)))[-8:])
left = frame_cols[frame_cols < W / 2].mean()
right = frame_cols[frame_cols > W / 2].mean()
top_c = frame_rows[frame_rows < H / 2].mean()
bot_c = frame_rows[frame_rows > H / 2].mean()
ylo, yhi = 0.895, 1.01


def y_of(row):
    return yhi - (row - top_c) / (bot_c - top_c) * (yhi - ylo)


# dashed line at 1.0: rows inside the frame with many dark pixels (dashes), excluding frame rows
inner = dark[:, int(left) + 3:int(right) - 2]
cand = [r for r in range(int(top_c) + 3, int(bot_c) - 3) if inner[r].sum() > 0.4 * inner.shape[1]]
dash_row = float(np.mean(cand)) if cand else float("nan")
dash_err = y_of(dash_row) - 1.0

C0 = np.array([0x1f, 0x77, 0xb4]) / 255.0
C1 = np.array([0xff, 0x7f, 0x0e]) / 255.0


def markers(color):
    m = np.linalg.norm(img - color, axis=-1) < 0.25
    m[:, :int(left) + 1] = False
    m[:, int(right):] = False
    m[:int(top_c) + 1] = False
    m[int(bot_c):] = False
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3)), iterations=1)
    lab, n = ndimage.label(er)
    cs = []
    for i in range(1, n + 1):
        sz = (lab == i).sum()
        if sz >= 6:
            # centre from the un-eroded blob that holds this component
            cy, cx = ndimage.center_of_mass(lab == i)
            cs.append((cx, cy, sz))
    return sorted(cs)


blue, orange = markers(C0), markers(C1)
# x-axis: log scale, xlim 1.5 to 150
def x_of(col):
    return 10 ** (np.log10(1.5) + (col - left) / (right - left) * (np.log10(150) - np.log10(1.5)))


xs = [2, 5, 10, 20, 50, 100]


def assign(cs):
    d = {}
    for cx, cy, sz in cs:
        xv = x_of(cx)
        b = min(xs, key=lambda v: abs(np.log10(v) - np.log10(xv)))
        if abs(np.log10(b) - np.log10(xv)) > 0.05:
            continue  # not on a data x position (for example the legend symbols)
        d.setdefault(b, []).append((cx, cy, sz, xv))
    # A marker that a crossing line splits into parts gives several close components.
    # Merge parts that lie within 8 px of each other (size-weighted centre).
    out = {}
    for b, v in d.items():
        ys = [c[1] for c in v]
        if max(ys) - min(ys) <= 8 and max(c[0] for c in v) - min(c[0] for c in v) <= 8:
            s = sum(c[2] for c in v)
            out[b] = (sum(c[0] * c[2] for c in v) / s, sum(c[1] * c[2] for c in v) / s, s, v[0][3])
    return out, {b: [[float(x) for x in c] for c in v] for b, v in d.items()}


blue_a, blue_all = assign(blue)
orange_a, orange_all = assign(orange)
paper_kepler = {b: y_of(v[1]) for b, v in blue_a.items()}
orange_read = {b: y_of(v[1]) for b, v in orange_a.items()}
orange_err = {b: orange_read[b] - paper_newton[b] for b in orange_read if b in paper_newton}

diag = dict(image_size=[H, W], frame=dict(top=top_c, bottom=bot_c, left=float(left), right=float(right)),
            dash_row=dash_row, dash_value=y_of(dash_row), dash_err=dash_err,
            blue_markers={str(b): [float(x) for x in v] for b, v in blue_a.items()},
            orange_markers={str(b): [float(x) for x in v] for b, v in orange_a.items()},
            blue_components={str(b): v for b, v in blue_all.items()},
            orange_components={str(b): v for b, v in orange_all.items()},
            orange_read=orange_read, orange_err_vs_text=orange_err,
            max_abs_err=float(max([abs(dash_err)] + [abs(v) for v in orange_err.values()])),
            pixel_size_in_r2=(yhi - ylo) / (bot_c - top_c))
json.dump(diag, open(f"{out}/pixel_reading.json", "w"), indent=1, default=float)
print(json.dumps(diag, indent=1, default=float))

blocks = sorted(ours)
with open(f"{out}/scores.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["context_length", "kepler_ours", "kepler_paper_read_from_image", "newton_ours", "newton_paper_text"])
    for b in blocks:
        w.writerow([b, f"{ours[b][0]:.4f}", f"{paper_kepler[b]:.3f}" if b in paper_kepler else "-",
                    f"{ours[b][1]:.4f}", f"{paper_newton[b]:.4f}" if b in paper_newton else "-"])
print(open(f"{out}/scores.csv").read())

# ---- combined plot ----
fig, ax = plt.subplots(figsize=(7, 5.5))
B = np.array(blocks)
ax.plot(B, [ours[b][0] for b in blocks], "o-", color="C0", label="Kepler score, ours")
ax.plot(B, [ours[b][1] for b in blocks], "o-", color="C1", label="Newton score, ours")
pk = sorted(paper_kepler)
ax.plot(pk, [paper_kepler[b] for b in pk], "s--", color="C0", mfc="none", ms=8,
        label="Kepler score, paper (read from image)")
pn = sorted(paper_newton)
ax.plot(pn, [paper_newton[b] for b in pn], "s--", color="C1", mfc="none", ms=8, label="Newton score, paper (notebook text)")
ax.axhline(1.0, ls=":", color="k", lw=1)
ax.set_xscale("log")
lab = [1, 2, 5, 10, 20, 50, 100]
ax.set_xticks(lab)
ax.set_xticks([b for b in blocks if b not in lab], minor=True)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticklabels([str(b) for b in lab])
ax.set_xlabel("context length")
ax.set_ylabel("score (mean of best-layer R²)")
allv = [v for b in blocks for v in ours[b]] + list(paper_kepler.values()) + list(paper_newton.values())
ax.set_ylim(min(allv) - 0.03, 1.01)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
ax.set_title("Kepler and Newton scores against context length")
fig.tight_layout()
fig.savefig(f"{out}/scores_ours_and_paper.png", dpi=150)
