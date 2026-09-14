#!/usr/bin/env python
"""H3 threshold: force-term R^2 vs context-noise sigma for the clean laws
(Kepler, pendulum), all 3 paradigms. Pulls checkpoints from the various run dirs
(sigma 0 and 0.1 from the contrasts; 0.01, 0.05 from the sigma_* fill)."""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physwm.train.checkpoint import load_run
from physwm.eval.behavioral import behavioral_score

GROUPS = {
    "kepler": {"0.0": "runs/h5b/{m}_s0.0", "0.01": "runs/sigma_kepler/{m}_s0.01",
               "0.05": "runs/sigma_kepler/{m}_s0.05", "0.1": "runs/h5b/{m}_s0.1"},
    "pendulum": {"0.0": "runs/contrast_pendulum/{m}_s0.0", "0.01": "runs/sigma_pendulum/{m}_s0.01",
                 "0.05": "runs/sigma_pendulum/{m}_s0.05", "0.1": "runs/contrast_pendulum/{m}_s0.1"},
}
SIGS = ["0.0", "0.01", "0.05", "0.1"]


def main():
    out = {}
    for dom, grp in GROUPS.items():
        print(f"\n=== {dom}: force-term R^2 vs sigma ===")
        print(f"{'method':10s}" + "".join(f"{'s='+s:>10s}" for s in SIGS))
        for meth in ("ar", "flow", "diffusion"):
            row = {}
            for s in SIGS:
                run = grp[s].format(m=meth)
                if not os.path.exists(os.path.join(run, "final.pt")):
                    row[s] = float("nan"); continue
                cfg, ds, m = load_run(run)
                row[s] = behavioral_score(m, ds, dom, cfg.data.context)["value"]
            out[f"{dom}/{meth}"] = row
            print(f"{meth:10s}" + "".join(f"{row.get(s, float('nan')):>10.3f}" for s in SIGS))
    json.dump(out, open("runs/sigma_threshold.json", "w"), indent=2)
    print("\n-> runs/sigma_threshold.json")


if __name__ == "__main__":
    main()
