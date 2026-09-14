#!/usr/bin/env python
"""Aggregate the control-task sweep into tables + the paper figure.

The claim under test is that probe R^2 rising for orbital elements while falling for
force shows an emergent Keplerian world model. The controls say what else produces
that pattern: `vel` (needs the same inputs, not conserved) and `sham` (arbitrary
nonlinear functions of the same inputs, no physics at all).
"""
import os, json, glob, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = ["force", "vel", "kepler", "sham"]
LABEL = {"force": "force (Newton)", "vel": "velocity", "kepler": "orbital elts (Kepler)",
         "sham": "sham nonlinear"}
# reference categorical palette, slots 1-4, used in the documented order
COLOR = {"force": "#2a78d6", "vel": "#eb6834", "kepler": "#1baf7a", "sham": "#eda100"}
CTXS = [2, 5, 10, 20, 50, 90]
SIGMAS = [0.0, 0.1]
PARADIGMS = ["ar", "flow", "diffusion"]


def load():
    rows = []
    for f in glob.glob(f"{ROOT}/results_control/*.json"):
        rows.append(json.load(open(f)))
    return rows


def cell(rows, paradigm, sigma, ctx, block, target):
    v = [r[block][target] for r in rows if r["paradigm"] == paradigm
         and abs(r["sigma"] - sigma) < 1e-9 and r["context"] == ctx]
    v = [x for x in v if np.isfinite(x)]
    if not v:
        return np.nan, np.nan, 0
    a = np.array(v)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0), len(a)


def tables(rows):
    out = []
    for p, s in itertools.product(PARADIGMS, SIGMAS):
        out.append(f"\n### {p}  sigma={s}\n")
        out.append("| ctx | n | " + " | ".join(
            f"{t} trained | {t} untr | {t} raw" for t in TARGETS) + " |")
        out.append("|" + "---|" * (2 + 3 * len(TARGETS)))
        for c in CTXS:
            m, _, n = cell(rows, p, s, c, "trained", "force")
            if n == 0:
                continue
            parts = []
            for t in TARGETS:
                tr, se, _ = cell(rows, p, s, c, "trained", t)
                un, _, _ = cell(rows, p, s, c, "untrained", t)
                rw, _, _ = cell(rows, p, s, c, "raw_context", t)
                parts += [f"{tr:.3f}±{se:.3f}", f"{un:.3f}", f"{rw:.3f}"]
            out.append(f"| {c} | {n} | " + " | ".join(parts) + " |")
    return "\n".join(out)


def rff_floor():
    p = f"{ROOT}/runs/rff_floor.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["results"]


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    floor = rff_floor()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True, sharey=True)
    for j, p in enumerate(PARADIGMS):
        for i, s in enumerate(SIGMAS):
            ax = axes[i, j]
            ends = []
            for t in TARGETS:
                xs, ys, es = [], [], []
                for c in CTXS:
                    m, se, n = cell(rows, p, s, c, "trained", t)
                    if n:
                        xs.append(c); ys.append(m); es.append(se)
                if not xs:
                    continue
                ax.errorbar(xs, ys, yerr=es, color=COLOR[t], lw=2, marker="o", ms=5,
                            capsize=2, zorder=3, label=LABEL[t])
                # untrained control, same hue, recessive
                uy = [cell(rows, p, s, c, "untrained", t)[0] for c in xs]
                ax.plot(xs, uy, color=COLOR[t], lw=1.2, ls=":", alpha=0.75, zorder=2)
                # no-network RFF floor on the raw context: architecture-free and
                # nonlinearity-matched. Flat for `kepler` at every context.
                if floor:
                    fy = [floor[str(c)]["rff"][t] for c in xs if str(c) in floor]
                    if len(fy) == len(xs):
                        ax.plot(xs, fy, color=COLOR[t], lw=1.1, ls="--", alpha=0.55,
                                zorder=1)
                ends.append((ys[-1], xs[-1], t))
            # push direct labels apart so the ones that land on top of each other at
            # the right edge (velocity / orbital elts both saturate near 1.0) stay legible
            ends.sort()
            MIN_GAP = 0.055
            for k in range(1, len(ends)):
                if ends[k][0] - ends[k - 1][0] < MIN_GAP:
                    ends[k] = (ends[k - 1][0] + MIN_GAP, ends[k][1], ends[k][2])
            for y_lab, x_end, t in ends:
                ax.annotate(LABEL[t].split(" (")[0], (x_end, y_lab), color=COLOR[t],
                            fontsize=7.5, weight="bold", xytext=(5, 0),
                            textcoords="offset points", va="center",
                            annotation_clip=False)
            ax.set_xscale("log")
            ax.set_xticks(CTXS); ax.set_xticklabels(CTXS)
            ax.grid(alpha=0.18, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(p, fontsize=11, weight="bold")
            if j == 0:
                ax.set_ylabel(f"$\\sigma$={s}\nprobe $R^2$", fontsize=9)
            if i == 1:
                ax.set_xlabel("context length", fontsize=9)
    axes[0, 0].set_ylim(-0.05, 1.12)
    axes[0, 0].set_xlim(1.8, 108)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Probe $R^2$ vs context. Solid = trained, dotted = untrained net, "
                 "dashed = no-network RFF floor on raw input.\nThe orbital-element rise "
                 "never reaches a floor that needs no learning and is already flat at "
                 "ctx=2.", fontsize=11.5, weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    p_out = f"{ROOT}/runs/FIG_control_tasks.png"
    fig.savefig(p_out, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", p_out)


if __name__ == "__main__":
    rows = load()
    print(f"{len(rows)} runs loaded")
    md = tables(rows)
    open(f"{ROOT}/runs/control_tables.md", "w").write(md)
    print(md)
    figure(rows)
