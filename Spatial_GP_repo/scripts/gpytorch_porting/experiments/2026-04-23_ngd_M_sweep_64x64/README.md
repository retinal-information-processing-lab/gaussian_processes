# NGD+Adam M-sweep — 64×64 inducing-point performance characterisation

**Date**: 2026-04-23 (start) → 2026-04-28 (end)
**Status**: COMPLETE. Definitive M-curve for NGD+Adam at 64×64.
**Branch**: `pietro/investigate-default-gpy`

## Data location and commit policy

Two results files exist for this sweep:

- **`results.summary.jsonl`** (824 KB, **committed**): all 1107 records with
  summary stats only (test_r, final_A, wall_time, n_iterations_run, etc.).
  Sufficient for the headline tables in this README and any aggregate
  cross-cell analysis.
- **`results.jsonl`** (80 MB, **gitignored**, lives on `dgx3` only): same
  1107 records WITH full per-iter `train_loss_curve`, `A_curve`,
  `beta_curve`, `iter_time_curve`. Required only for trajectory-level
  analysis (e.g. `investigations/ngd_low_ntrain/diagnose_overfit.py`'s A
  inflation plots). Regenerable from `run_sweep.py` (~17 h on uncontested
  GPU).

If a future session needs the full data: SSH to dgx3, the file is at
`/home/pietro/GP/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/experiments/2026-04-23_ngd_M_sweep_64x64/results.jsonl`.
**Do NOT delete it** without checking with the user first
(CRITICAL RULE 9 in CLAUDE.md).

## TL;DR (added 2026-04-28 after sweep finished)

NGD+Adam matches or exceeds `vargp_direct intl_fixAmp` at every M ≥ 200
on 41 cells × 3 seeds. Plateau at M ≈ 200–300, same as vargp.

| M | NGD mean test_r | vargp mean test_r | Δ NGD−vargp |
|---|---|---|---|
| 50 | 0.815 | 0.817 | −0.002 |
| 100 | 0.828 | 0.830 | −0.002 |
| 200 | 0.843 | 0.838 | **+0.005** |
| 250 | 0.845 | (no vargp ref) | — |
| 300 | 0.845 | 0.838 | **+0.007** |
| 500 | 0.848 | 0.834 | **+0.014** |
| 750 | 0.853 | 0.838 | **+0.015** |
| 1000 | 0.846 | 0.840 | **+0.007** |
| 1500 | 0.846 | 0.840 | **+0.007** |

(NGD seeds {0,1,2}; vargp ref `experiments/2026-04-13_M_sweep_64x64/`,
seeds {0,1,2}, intl_fixAmp + ELBO ES p=15. Both swept on the same
41-cell × 3-seed grid; same dataset, same `fix_Amp=True`,
same `n_train=3160`, same RF init.)

Wall time per run: NGD ~95–160s, vargp ~45–55s — NGD is 2–3×
**slower** here because of GPU contention with another user during the
sweep. On uncontested GPU (Phase 3C), NGD was 4× *faster* than vargp.

**Key conclusion**: NGD has no M-degradation problem — mean_r is flat
from M=200 to M=1500 with no drop, and std actually decreases at large M.
The vargp M-degradation finding (9/41 cells degrade at large M, see
`experiments/2026-04-13_M_sweep_64x64/`) does NOT replicate in NGD.

(For data-starved regimes M=n=50/150/300, see
`investigations/ngd_low_ntrain/` — there NGD genuinely trails.)

---

## Purpose

This is the NGD+Adam analog of `experiments/2026-04-13_M_sweep_64x64/`,
which characterised how test_r depends on M for `vargp_direct`. Running
the exact same grid with NGD+Adam lets us answer:

1. Does NGD+Adam show M-degradation (the known vargp issue at large M on
   9/41 cells)?
2. At what M does NGD+Adam plateau, and how does that compare to vargp's
   plateau at M≈200–300?
3. What is the recommended operating M for the GPyTorch-native training
   mode?

These two sweeps together define the **state of the art** for the project:
- `vargp_direct intl_fixAmp + ELBO ES p=15` — `experiments/2026-04-13_M_sweep_64x64/`
- `ngd+adam (this file)` — `experiments/2026-04-23_ngd_M_sweep_64x64/`

The default `run.mode` in `default_params.json` remains `'default_gpy'`
until the results from this sweep confirm NGD's M-stability.

---

## Grid

| parameter | values |
|-----------|--------|
| cells | 0..40 (all 41) |
| M | 50, 100, 200, 250, 300, 500, 750, 1000, 1500 |
| seeds | 0, 1, 2 |
| **total runs** | **1107** |

Notes:
- M=250 is included (unlike vargp M-sweep which skipped it) because
  `experiments/2026-04-22_ngd_final_verdict_64x64/` (Phase 3C verdict)
  used M=250 with seeds {1,2,3} — different seeds. The M=250 point here
  uses seeds {0,1,2} and serves both as an internal consistency check
  and an independent replication.
- Seeds {0,1,2} match the vargp M-sweep to enable paired cell-level
  comparison.

---

## Training configuration

All 1107 runs use `build_config_from_defaults(mode='ngd', ...)`.
This is NGD's **state-of-the-art config** as validated by the Phase 3C
verdict sweep (SCRAPBOOK §26–33).

### Two explicit non-default overrides (documented here intentionally)

```python
data_path = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
# WHY: build_config_from_defaults defaults to 108x108. This sweep targets
# 64x64, matching the vargp M-sweep and Phase 3C.

ngd_n_iterations = 1500
# WHY: default_params.json["ngd"]["n_iterations"] = 1000. Phase 3C used 1500
# (cap; ELBO ES fires earlier on most cells). We match Phase 3C here so
# the M=250 column reproduces Phase 3C closely.
```

### Everything else: project defaults from `default_params.json`

The values below were frozen at sweep creation from
`build_config_from_defaults(mode='ngd', M=250, cell=0, seed=0, ...)`.
They are copied here verbatim so future readers do not need to dig into
JSON files to know what was used:

```
# Likelihood / init
A_init              = 0.01
lambda0_init        = 1.0

# Kernel
kernel_type         = arc_cosine
beta_init           = 0.1
rho_init            = 0.1
sigma_0_init        = 1.0
rf_init             = ground_truth
bound_rf_center     = True

# NGD optimizer
ngd_lr              = 0.1
ngd_adam_lr         = 0.01

# Early stopping (ELBO-based; NGD-scale patience)
ngd_es_patience         = 200
ngd_es_min_delta_rel    = 0.01
ngd_es_min_iterations   = 50
ngd_es_restore_best     = True

# Data / inducing points
n_train             = 3160
n_val_split         = 0
ip_selection        = random
fix_Amp             = True      # Amp frozen at 1.0

# Numerical
dtype               = float32
device              = cuda
jitter              = 1e-4
cholesky_max_tries  = 3
```

The complete resolved config for one representative run (cell=0, seed=0,
M=250) is frozen in `effective_config.json` (written at sweep launch).

---

## Correspondence with vargp M-sweep

The vargp sweep (`2026-04-13_M_sweep_64x64/`) used:

```
# These are NOT used here:
A_init = 1e-4          # vargp specific; NGD uses 0.01
lambda0_init = -1.0    # vargp specific; NGD uses 1.0
n_estep = 50           # vargp E-step; no equivalent in NGD
n_mstep = 20           # vargp M-step; no equivalent in NGD
```

The grids differ only in that this sweep adds M=250. All other grid
parameters are identical (cells, seeds, n_train, fix_Amp, rf_init,
ip_selection, dataset).

This is a **best-config vs best-config** comparison. Using vargp's
A_init overrides for NGD would not represent the "state-of-the-art NGD
config" — it would represent an untested config.

---

## Estimated runtime

| M | est. s/run | × 123 runs |
|---|-----------|------------|
| 50 | ~1 | ~2 min |
| 100 | ~4 | ~8 min |
| 200 | ~10 | ~21 min |
| 250 | ~14 | ~29 min (Phase 3C measured) |
| 300 | ~19 | ~39 min |
| 500 | ~40 | ~82 min |
| 750 | ~74 | ~152 min |
| 1000 | ~114 | ~234 min |
| 1500 | ~250 | ~513 min |

**Total: ~17 hours.** Runs in background; crash-safe (resumes from
existing `results.jsonl` on restart).

---

## Key questions (pre-registered)

1. **Does mean test_r plateau at M≈200–300 for NGD, as it does for vargp?**
   Positive answer would confirm M=250 is a sound default.
2. **Does NGD show M-degradation (train_r↑ / test_r↓) at large M?**
   If yes: how many cells, at what M, compared to vargp's 9/41?
3. **Is the M=250 point consistent with Phase 3C?**
   Phase 3C mean test_r = 0.844 (seeds {1,2,3}). A matching result here
   (seeds {0,1,2}) would confirm seed stability.

---

## Output

- `results.jsonl` — per-run records (appended, crash-safe)
- `effective_config.json` — frozen resolved config at launch time
- `metadata.json` — git commit, grid, start time
- `run.log` — full stdout/stderr

Post-sweep, `analyze.py` will produce:
- Per-M summary table (mean/std/median test_r across 123 runs)
- M-degradation cell list (cells where test_r drops from peak by > 0.05)
- Paired Δ(NGD − vargp) table at matching M values
