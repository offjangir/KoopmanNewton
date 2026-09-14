"""Linear probing: does the model encode the instantaneous *law* (force/accel)
or the trajectory *invariants*?

This is the direct generalization of the paper's Newtonian-force vs
Keplerian-orbital-elements probe.

**Leakage discipline** (important -- windows from one trajectory are highly
correlated, so a naive window-level split inflates R^2):

1. The probe is *fit* on ``fit_split`` and *scored* on a **different split**
   (``eval_split``), whose trajectories are disjoint by construction.
2. The best layer is chosen on a held-out *trajectory-level* slice of
   ``fit_split`` only -- the eval split is never used for model selection.
3. All internal splits group by trajectory id, never by window.

``eval_split`` pointing at a held-out OOD regime gives the **counterfactual /
off-distribution force probe**: can the readout still decode the force at states
from a regime the model never trained on?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def ridge_fit(Xtr, Ytr, alpha=1e-2):
    """Closed-form ridge with intercept. Returns (W, x_mean, y_mean)."""
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - xm, Ytr - ym
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, xm, ym


def r2_score(Y, pred) -> float:
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum() + 1e-12
    return float(1 - ss_res / ss_tot)


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=1e-2) -> float:
    """Multi-output ridge R^2. X:(n,d) Y:(n,k)."""
    W, xm, ym = ridge_fit(Xtr, Ytr, alpha)
    return r2_score(Yte, (Xte - xm) @ W + ym)


@dataclass
class ProbeResult:
    r2: dict            # target -> R^2 on eval_split (layer chosen on fit_split only)
    best_layer: dict    # target -> selected layer
    fit_split: str = ""
    eval_split: str = ""
    n_fit_traj: int = 0
    n_eval_traj: int = 0

    def summary(self) -> str:
        return "  ".join(f"{k}={v:.3f}" for k, v in self.r2.items())


@torch.no_grad()
def _collect(method, dataset, split, context, device, n_traj, max_samples, rng):
    """Return (X reps (M,Lyr,d), law (M,L), inv (M,I), traj_id (M,))."""
    states = dataset.splits[split].states[:n_traj]
    law = dataset.splits[split].law[:n_traj]
    inv = dataset.splits[split].invariants[:n_traj]
    norm = dataset.normalize(states).astype(np.float32)
    N, T, _ = norm.shape
    idx = [(n, t) for n in range(N) for t in range(context - 1, T)]
    if len(idx) > max_samples:
        sel = rng.choice(len(idx), max_samples, replace=False)
        idx = [idx[i] for i in sel]
    ctx = np.stack([norm[n, t - context + 1: t + 1] for n, t in idx]).astype(np.float32)
    laws = np.stack([law[n, t] for n, t in idx]).astype(np.float64)
    invs = np.stack([inv[n, t] for n, t in idx]).astype(np.float64)
    tid = np.array([n for n, _ in idx])

    method.eval()
    reps = []
    for i in range(0, len(ctx), 512):
        b = torch.from_numpy(ctx[i:i + 512]).to(device)
        reps.append(method.hidden_states(b).float().cpu().numpy())
    X = np.concatenate(reps, 0).astype(np.float64)
    return X, laws, invs, tid


def _targets(dataset, law, inv) -> dict:
    out = {"force": law}
    for j, name in enumerate(dataset.invariant_names):
        out[f"inv_{name}"] = inv[:, j:j + 1]
    return out


def excess_over_control(model_r2: dict, control_r2: dict) -> dict:
    """Fraction of the *residual* variance explained beyond a control floor.

    ``(R2_model - R2_control) / (1 - R2_control)``. Raw R^2 saturates near 1 and
    is uninterpretable on its own: a linear readout of the raw context already
    reaches R^2~0.93-0.99 for force. This normalizes against that floor --
    0 means "no better than the control", 1 means "explains all remaining
    variance". Negative means worse than the control.

    IMPORTANT -- which control to use:

    * ``control_raw_context`` is **architecture-independent** and is the ONLY
      valid baseline for comparing methods against each other.
    * ``control_untrained`` is architecture-*dependent* and is NOT comparable
      across backbones. A random AR transformer is a strong random-feature map
      (force R^2 ~ 0.92), whereas an untrained DiT is near-identity because
      adaLN-Zero zero-initializes every block gate (force R^2 ~ 0.13). Taking the
      max of the two controls therefore penalizes AR far more than DiT and
      silently scrambles the cross-method ranking. Use it only as a
      within-architecture diagnostic.
    """
    out = {}
    for k, mv in model_r2.items():
        cv = control_r2.get(k)
        if cv is None or cv >= 1.0:
            continue
        out[k] = float((mv - cv) / (1.0 - cv))
    return out


def transfer_scores(Xf, Yf, Xe, Ye, alpha=1e-2) -> dict:
    """Counterfactual/OOD transfer of a readout fit on ``Xf`` and applied to ``Xe``.

    A plain R^2 here can explode to large negative values (we observed -8e3):
    the ridge map extrapolates linearly far outside its training support, so the
    *scale* of the prediction blows up even when its *direction* is fine. We
    therefore report three complementary numbers:

    * ``r2``            -- raw transfer R^2 (harsh; unbounded below)
    * ``r2_calibrated`` -- best affine rescaling of the prediction per output
                           dim, fit on the eval set. Isolates "is the encoded
                           direction still right?" from scale/offset drift.
    * ``corr``          -- mean Pearson r per output dim (scale-free)
    """
    W, xm, ym = ridge_fit(Xf, Yf, alpha)
    pred = (Xe - xm) @ W + ym
    r2 = r2_score(Ye, pred)

    # per-dim affine recalibration: y ~ a*pred + b
    cal = np.empty_like(pred)
    corrs = []
    for j in range(Ye.shape[1]):
        p, y = pred[:, j], Ye[:, j]
        A = np.stack([p, np.ones_like(p)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        cal[:, j] = A @ coef
        sp, sy = p.std(), y.std()
        corrs.append(float(np.corrcoef(p, y)[0, 1]) if sp > 1e-12 and sy > 1e-12 else 0.0)
    return {"r2": r2, "r2_calibrated": r2_score(Ye, cal), "corr": float(np.mean(corrs))}


def probe_raw_context(dataset, fit_split="train", eval_split="test", context=8,
                      n_traj=128, max_samples=4000, alpha=1e-2, seed=0) -> ProbeResult:
    """CONTROL 1: linear readout from the *raw normalized context* (no network).

    The context already contains the states, and force is a smooth function of
    state, so a linear map from raw inputs recovers some force signal for free.
    This is the information floor: a trained model's probe R^2 is only meaningful
    to the extent it exceeds this.
    """
    rng = np.random.default_rng(seed)

    def gather(split):
        states = dataset.splits[split].states[:n_traj]
        law = dataset.splits[split].law[:n_traj]
        inv = dataset.splits[split].invariants[:n_traj]
        norm = dataset.normalize(states).astype(np.float64)
        N, T, _ = norm.shape
        idx = [(n, t) for n in range(N) for t in range(context - 1, T)]
        if len(idx) > max_samples:
            sel = rng.choice(len(idx), max_samples, replace=False)
            idx = [idx[i] for i in sel]
        X = np.stack([norm[n, t - context + 1: t + 1].ravel() for n, t in idx])
        return X, np.stack([law[n, t] for n, t in idx]), np.stack([inv[n, t] for n, t in idx])

    Xf, law_f, inv_f = gather(fit_split)
    Xe, law_e, inv_e = gather(eval_split)
    tgt_f, tgt_e = _targets(dataset, law_f, inv_f), _targets(dataset, law_e, inv_e)
    r2 = {k: ridge_r2(Xf, tgt_f[k], Xe, tgt_e[k], alpha) for k in tgt_f}
    return ProbeResult(r2, {k: -1 for k in r2}, fit_split, eval_split)


def probe_representations(method, dataset, fit_split="train", eval_split="test",
                          context=8, device="cpu", n_traj=128, max_samples=4000,
                          alpha=1e-2, seed=0) -> ProbeResult:
    """Fit a linear readout on ``fit_split``, score it on ``eval_split``.

    ``eval_split`` must differ from ``fit_split`` (disjoint trajectories). Layer
    selection uses a trajectory-level holdout inside ``fit_split`` only.
    """
    if eval_split == fit_split:
        raise ValueError(
            "fit_split and eval_split must differ (probing must be scored on "
            "held-out trajectories to avoid same-trajectory leakage)"
        )
    rng = np.random.default_rng(seed)
    Xf, law_f, inv_f, tid_f = _collect(method, dataset, fit_split, context, device,
                                       n_traj, max_samples, rng)
    Xe, law_e, inv_e, tid_e = _collect(method, dataset, eval_split, context, device,
                                       n_traj, max_samples, rng)
    tgt_f = _targets(dataset, law_f, inv_f)
    tgt_e = _targets(dataset, law_e, inv_e)

    # trajectory-level holdout inside the fit split, used ONLY to pick the layer
    uniq = np.unique(tid_f)
    rng.shuffle(uniq)
    n_sel = max(1, int(0.2 * len(uniq)))
    sel_traj = set(uniq[:n_sel].tolist())
    is_sel = np.array([t in sel_traj for t in tid_f])

    r2, best_layer = {}, {}
    for name, Yf in tgt_f.items():
        best, bl = -np.inf, 0
        for L in range(Xf.shape[1]):
            score = ridge_r2(Xf[~is_sel, L], Yf[~is_sel],
                             Xf[is_sel, L], Yf[is_sel], alpha)
            if score > best:
                best, bl = score, L
        # refit on ALL fit-split data at the chosen layer, score on eval split
        r2[name] = ridge_r2(Xf[:, bl], Yf, Xe[:, bl], tgt_e[name], alpha)
        best_layer[name] = bl

    return ProbeResult(r2, best_layer, fit_split, eval_split,
                       int(len(np.unique(tid_f))), int(len(np.unique(tid_e))))


def probe_transfer(method, dataset, fit_split="train", eval_split="test_ecc_extrap",
                   context=8, device="cpu", n_traj=128, max_samples=4000,
                   alpha=1e-2, seed=0) -> dict:
    """Counterfactual probe with robust transfer metrics (see :func:`transfer_scores`).

    Returns ``{target: {r2, r2_calibrated, corr}}``. Layer is selected on a
    trajectory-level holdout of the fit split, never on the OOD split.
    """
    rng = np.random.default_rng(seed)
    Xf, law_f, inv_f, tid_f = _collect(method, dataset, fit_split, context, device,
                                       n_traj, max_samples, rng)
    Xe, law_e, inv_e, _ = _collect(method, dataset, eval_split, context, device,
                                   n_traj, max_samples, rng)
    tgt_f, tgt_e = _targets(dataset, law_f, inv_f), _targets(dataset, law_e, inv_e)

    uniq = np.unique(tid_f)
    rng.shuffle(uniq)
    sel = set(uniq[:max(1, int(0.2 * len(uniq)))].tolist())
    is_sel = np.array([t in sel for t in tid_f])

    out = {}
    for name, Yf in tgt_f.items():
        bl = max(range(Xf.shape[1]),
                 key=lambda L: ridge_r2(Xf[~is_sel, L], Yf[~is_sel],
                                        Xf[is_sel, L], Yf[is_sel], alpha))
        out[name] = transfer_scores(Xf[:, bl], Yf, Xe[:, bl], tgt_e[name], alpha)
    return out
