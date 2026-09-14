# physwm — Physics World-Models Beyond Kepler→Newton

Do the **learning paradigm** — not just the architecture — and the **kind of
physical law** decide whether a sequence model becomes a *curve-fitter* or a
*physicist*?

This codebase extends *"From Kepler to Newton: Inductive Biases Guide Learned
World Models in Transformers"* (arXiv 2602.06923) along two new axes:

- **More physical laws** — four domains as next-state prediction tasks:
  - **Gravitation**: Kepler two-body (exact), GR-precessing two-body, N-body/3-body
  - **Newtonian**: driven-damped pendulum, double pendulum (chaotic)
  - **Quantum**: two-level Rabi (Bloch), Ehrenfest moments, 1D wavepacket (split-step Fourier)
  - **Relativity**: relativistic charged particle (Boris pusher), Schwarzschild geodesic
- **More learning paradigms** — the autoregressive baseline **plus** conditional
  generative next-state models:
  - `ar` — autoregressive causal transformer (the paper's baseline)
  - `diffusion` — conditional DDPM with a **DiT** backbone
  - `flow` — conditional flow matching / rectified flow
  - `drifting` — Generative Modeling via Drifting (arXiv 2602.04770), one-step
  - `si` — stochastic interpolants (drift + score, ODE/SDE sampler)

All paradigms model the same conditional map `p(s_{t+1} | context window)`, so the
comparison is apples-to-apples and the three inductive biases from the paper apply
uniformly:

| Bias | Where it lives | Flag |
|------|----------------|------|
| spatial smoothness | regression vs binned head | `model.head` |
| stability | noisy-context training | `method.ctx_noise_sigma` |
| temporal locality | short attention / context | `model.attn_window`, `data.context` |

## Probing protocol and controls (read this before trusting any R²)

A bare force-probe R² is **not identifiable**. Two confounds, both measured here:

1. **Finite-difference shortcut.** If velocities are observed, two consecutive
   states linearly determine the acceleration via `(v_t − v_{t−1})/dt` — which
   *is* the force. Raw-context linear readout then hits R² ≈ **0.996** at
   context 2, with no model at all. Use `data.obs_dims: 2` (position-only, as in
   the original paper); the raw floor drops to ≈0.65 but still **grows with
   context** (0.65 → 0.93 at ctx 8).
2. **Random features.** An *untrained* transformer's activations are a random
   nonlinear expansion of the context, and a linear readout off them reaches
   R² ≈ **0.92–0.99** for force. This is often a *stronger* control than the raw
   context.

Every run therefore reports both controls plus

```
excess_over_control = (R²_model − R²_control) / (1 − R²_control)
```

against the stronger control: 0 = training bought nothing, 1 = explains all
remaining variance. **Report excess, not raw R².**

Leakage discipline: the probe is fit on one split and scored on a *different*
split (disjoint trajectories); the layer is selected on a trajectory-level
holdout inside the fit split only. Never split by window — windows from one
trajectory are near-duplicates.

Physics-consistency: `invariant_drift` reconstructs velocity by finite
difference under position-only observation (`pos_vel_layout` domains). It has a
nonzero floor from that discretization (~0.04 in energy at dt=0.2) — compare
model drift against the floor, not against zero.

## Why this addresses the paper's weaknesses
- **Tiny test set** → large held-out test sets + explicit **OOD regimes**
  (parameter extrapolation, stronger nonlinearity, chaotic bands) + a
  **counterfactual off-distribution force probe**.
- **One law** → four physics families with per-domain *law* targets (force/accel)
  vs *invariant* targets (energy, L, LRL vector, norm, γ).
- **Only autoregressive** → four generative paradigms evaluated with the same probe.

## Layout
```
physwm/physics/   simulators (Simulator ABC -> Trajectory: states, law, invariants)
physwm/data/      dataset generation, OOD splits, windowed loaders, newton-kepler data
physwm/models/    ARTransformer (causal, windowed) + DiTBackbone (adaLN-Zero)
physwm/methods/   ar | diffusion | flow | drifting | si  (one Method interface)
physwm/train/     config dataclasses, generic Trainer, checkpoint loading
physwm/eval/      rollout, invariant drift, probes + controls, behavioral force tests
physwm/viz/       plots (loss, rollout, probe bars, trajectory)
scripts/          experiment runners, launchers, analysis (see below), slurm/
configs/          base experiment config (overridden per job with --set)
results_*/        per-job JSON from the probe sweeps
runs/             training runs; only summary figures/tables and analysis.json are tracked
tests/            pytest (physics invariants, data, methods, probe)
```

## Quickstart (CPU)
```bash
pip install -e ".[dev]"          # plus torch (see scripts/env_setup.sh for GH200)
pytest -q                        # physics invariants, data, methods, probe
python scripts/smoke_test.py     # tiny end-to-end over domains x methods -> runs/smoke/
```

## Real runs (Delta-AI GH200)
```bash
bash scripts/env_setup.sh                        # aarch64 + CUDA 12.8 (sm_90) torch
python scripts/gen_configs.py --out configs/generated
sbatch scripts/slurm/train.sbatch configs/kepler_ar.yaml
sbatch scripts/slurm/sweep.sbatch configs/generated    # array over the grid
```
Each run writes `metrics.json` (rollout RMSE/MDE in- and out-of-distribution,
probe R² for force vs each invariant, counterfactual probe, invariant drift) plus
plots to its `runs/<name>/` directory. Set `--account` in the sbatch files to your
allocation.

## Scripts

| Script | What it does |
|--------|--------------|
| `run_experiment.py` | one training + evaluation run from a YAML config (`--set key=value` overrides) |
| `run_contrast.sh` | sigma contrast for one law on one GPU (`run_contrast.sh kepler runs/h5b`) |
| `run_program.py` | the packed multi-GPU experiment program, by named set (`--sets h4 dt --dry`) |
| `sweep_local.py` | sequential domain x method x context sweep with a summary table |
| `control_probe.py` / `run_control_sweep.py` | control-task probe (force / velocity / orbital elements / sham) on the newton-kepler data -> `results_control/` |
| `universal_transition.py` / `run_law_transition.py` | Newton->Kepler context transition per law -> `results_law/` |
| `generality.py` | increment-R² criterion vs sigma / sigma* across laws |
| `rff_floor.py` | random-Fourier-feature information floor on the raw context |
| `probe_vs_behavior.py`, `probe_artifact.py`, `extract_force_law.py` | probe vs behavioral force tests on trained checkpoints |
| `analyze_*.py` | aggregate sweeps into tables/figures under `runs/` |
| `benchmark.py`, `gen_configs.py`, `smoke_test.py` | throughput, config grid, CPU smoke test |

Multi-GPU launchers share `scripts/gpu_pool.py`, which packs several jobs per GPU
and skips jobs whose outputs already exist, so they are safe to re-run.

### Newton-kepler data
`control_probe.py`, `rff_floor.py` and `universal_transition.py` (domain
`kepler_theirs`) run on the trajectories from the original paper's repo. Generate
them with `generate_kepler_cv.py` from
[KindXiaoming/newton-kepler](https://github.com/KindXiaoming/newton-kepler) and point
`NEWTON_KEPLER_DATA` at the resulting `data_cv/` directory (default:
`../newton-kepler/data_cv` next to this repo).

## Status
Physics generators are verified to conserve invariants to machine precision
(Kepler/Rabi/Lorentz ~1e-16; GR-Kepler precesses as intended). The full
train→rollout→probe pipeline runs for every domain × method. Results of the sweeps
run so far are in `results_*/` and the summary files under `runs/`; checkpoints are
not tracked.
