# M-step Optimization — Findings

**Status**: CLOSED (April 2026)
**Branch**: `pietro/investigate-nmstep`
**Final changes**:
- `default_params.json model.lbfgs_tolerance_change: 1e-3` (new key, commit `6bf20eb`)
- `default_params.json training.n_mstep: 10 → 20` + all 8 YAML configs (commit `e5f688e`)

---

## TL;DR for future sessions

**Before you try to "optimize" the M-step again, read this.** Three non-obvious facts about the M-step:

1. **The M-step is NOT the training bottleneck.** It's 14% of total training time on average (6–47% depending on outer iteration). The E+F step dominates at ~1.1 s/iter. Optimizing M-step wall time has diminishing returns; a 38% M-step speedup buys only ~5% total training time.

2. **The expensive cost is kernel evaluation, not linear algebra.** Inputs are 4096-dim images. The arc-cosine kernel builds a pixel-level C matrix (~2500×2500 after masking) and `K̃ = XᵀCX` every closure call. The eigendecomposition of M×M K̃ is sub-millisecond — not the bottleneck (textbook GP analysis assumes scalar-input kernels, not pixel-structured ones). See the full conversation log if present, or `kernels.py:_compute_C_matrix`.

3. **LBFGS tolerances were effectively disabled in float32.** `tolerance_change=1e-9` and `tolerance_grad=1e-7` are below or at the float32 noise floor at typical ELBO magnitudes (~500). What actually terminated LBFGS in early training was the step-size check (`|t*d| < tolerance_change`) hitting float32 precision, not the loss-diff check. In late training, the loss-diff check fires.

---

## What was investigated

Starting question: "can we reduce n_mstep (the LBFGS max_iter for the M-step kernel optimization) to save time?"

Answer evolved through the investigation:
- **Phase 0 (characterization)**: With `n_mstep=20`, LBFGS never hit the ceiling in 514 M-step calls (median 9 iters, max 17). Termination was always via tolerance. This ruled out lowering `n_mstep` as the lever.
- **Follow-up (tolerance calibration)**: The ~1100 closure calls per run showed the TRUE stopping mechanism. Calibrated `tolerance_change` to `1e-3` by offline replay of loss traces, then confirmed empirically.

## Key empirical facts (from the sweeps in `phase0_characterization/`)

- **10 cells × 3 seeds, 64×64 PNAS, `intl_fixAmp` canonical config:**

| Config | Mean test_r | M-step time | Closure calls saved |
|--------|-------------|-------------|---------------------|
| Baseline (`tolerance_change=1e-9`) | 0.7727 | 8.0s | — |
| `tolerance_change=1e-3` (adopted) | 0.7716 | 5.0s | 38% |

- `test_r` delta: mean −0.001 (noise); 6/30 runs have |delta| > 0.01, **concentrated on unstable cells** (0, 15, 20, 39) where seed-level variance is already ~0.05. Well-behaved cells (3, 7, 8, 10, 22, 30) have |delta| ≤ 0.013.
- `tolerance_grad` never triggers at any value up to 1e-1 because gradient magnitudes stay in 1–350 in float32. It is inert. Don't waste time retuning it for float32.
- M-step's own internal ELBO gain is often wiped by eigenspace reprojection at the start of the next outer iteration (the M-step optimizes in a stale eigenspace). 68% of reprojections have negative ELBO delta. This is normal EM behavior, not a bug — the E-step restores it next iteration.

## Untried paths (for future sessions returning to this investigation)

### High priority — worth revisiting

**1. Kronecker factorization of C_smooth**
The RBF spatial smoothing kernel is evaluated on a regular 2D pixel grid. This means `C_smooth` factorizes as a Kronecker product (rows ⊗ cols), reducing the dominant O(N × n_pixels²) cost to O(N × n_rows + N × n_cols). This is the correct structural lever for real M-step speedups — not n_mstep tuning. Implementation would go in `kernels.py:_compute_C_matrix`. Not started.

**2. Redundant kernel recomputation between M-step and E-step**
The M-step's final LBFGS closure evaluates `K_tilde`, `K`, `Kvec` with the converged kernel params. At the start of the next EM iteration, `recompute_eigenspace()` recomputes the same matrices from scratch with those same final params. This is one full redundant kernel evaluation per EM iteration. Fix: cache the M-step's final matrices and pass them to `recompute_eigenspace()`. See `eigenspace_mstep.py` + `eigenspace_training.py`.

### Lower priority — not worth pursuing without a specific motivation

- **LBFGS history_size**: currently 100 for 5–6 params. Reducing to 10 is sufficient (LBFGS needs history ≥ n_params for full quasi-Newton). Never measured whether this changes convergence or per-iteration cost. Low impact expected.
- **Adaptive tolerance schedule**: the adopted `tolerance_change=1e-3` is fixed across all EM iterations. A looser early / tighter late schedule was the deferred "Phase 2" but never characterized. Given that tolerance=1e-3 already matches baseline within 0.001 test_r, the benefit is likely small.
- **Analytical / VJP gradient mode for M-step**: investigation only characterized the `autograd` path. The VJP path (`gradient_mode='vjp'`) computes the same math with explicit formulas. May have different float32 numerical behavior, but no evidence it matters.
- **n_mstep ceiling for unstable cells** (0, 15, 20, 39): LBFGS max was 17 with ceiling 20. Since LBFGS terminates via tolerance before hitting the ceiling, raising n_mstep to 50 would not change behavior. The |delta test_r| > 0.01 observed for these cells is within their normal seed-level variance (~0.05) and is not caused by the tolerance change. Not a real issue.

---

## What NOT to do

1. **Don't lower `n_mstep` below 20.** The tolerance mechanism handles early termination. Lowering the ceiling would clip the 10% of M-steps that legitimately need 14–17 iters (mostly iter 1).
2. **Don't raise `tolerance_change` above 1e-3** without redoing the offline replay. At 1e-2, 85% of the M-step's internal gain is forfeited; at 1e-1, it's essentially skipped entirely.
3. **Don't change `tolerance_grad`.** It's inert in float32. If switching to float64, revisit.
4. **Don't rely on n_mstep reduction as a speedup.** The M-step is not the bottleneck; kernel evaluation is. If you want real training speedups, look at the kernel: Kronecker structure in `C_smooth` (RBF on a regular 2D grid factorizes), or caching invariant components between LBFGS steps.

## Reproducing the results

```bash
# Phase 0 characterization (15 runs, ~15 min)
python investigations/n_mstep_reduction/phase0_characterization/run_phase0.py
python investigations/n_mstep_reduction/phase0_characterization/analyze_phase0.py

# Tolerance calibration sweep (30 runs, ~30 min)
python investigations/n_mstep_reduction/phase0_characterization/run_tolerance_sweep.py
python investigations/n_mstep_reduction/phase0_characterization/analyze_tolerance.py

# Final confirmation: abs_tol=1e-3 vs baseline (30 runs, ~30 min)
python investigations/n_mstep_reduction/phase0_characterization/run_confirmation_abs_1em3.py
```

The confirmation scripts temporarily set `config['lbfgs_tolerance_change_abs']` — this config key no longer exists on the main code path (scaffolding removed after calibration). The scripts will still compile but will silently use the default 1e-3 unless adapted. They're preserved as documentation of the exact calibration procedure.

## Files in this investigation

- `README.md` — scope + phase structure (written at start of investigation)
- `FINDINGS.md` — this file
- `phase0_characterization/run_phase0.py` — 5 cells × 3 seeds characterization
- `phase0_characterization/analyze_phase0.py` — histograms, ELBO traces, cell comparison
- `phase0_characterization/results.jsonl` — Phase 0 output (514 M-step traces)
- `phase0_characterization/run_tolerance_sweep.py` — 10 cells × 3 seeds with full telemetry
- `phase0_characterization/analyze_tolerance.py` — offline what-if tolerance replay
- `phase0_characterization/tolerance_sweep_results.jsonl` — baseline (`tol=1e-9`)
- `phase0_characterization/run_confirmation.py` — `rel_tol=1e-7` (exploration path, noisier)
- `phase0_characterization/confirmation_rel1e7_results.jsonl`
- `phase0_characterization/run_confirmation_abs_1em3.py` — `abs_tol=1e-3` (adopted)
- `phase0_characterization/confirmation_abs1em3_results.jsonl`
- `phase0_characterization/plots/`, `phase0_characterization/tolerance_plots/` — generated plots
