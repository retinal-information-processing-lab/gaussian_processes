# Investigation: Performance Degradation with Large M (Inducing Points)

## First: verify worktree

Before doing anything else, run:

```bash
git worktree list
git branch --show-current
pwd
```

You MUST be in:
- Directory: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_M_degradation`
- Branch: `pietro/investigate-M-degradation`
- Commit: `0ec34fe`

If any of these don't match, STOP and tell the user. Do NOT proceed on the wrong branch or worktree.

## Safety constraints (running with --dangerously-skip-permissions)

This session runs without permission prompts. In exchange, follow these rules strictly:

- **NEVER `cd` out of the working directory** (`gpytorch_porting_M_degradation/`). All paths should be relative or absolute within this tree.
- **NEVER run `git checkout`, `git switch`, or `git worktree` commands.** You are on `pietro/investigate-M-degradation` and must stay there.
- **NEVER modify files outside the working directory.** No edits to `~/.claude/`, no system-level changes.
- **NEVER run `rm -rf` on directories above `investigations/M_degradation/`.** Only clean up files you created.
- **NEVER run `pip install`, `conda install`, or modify the environment.** Everything needed is already installed.
- If you need something outside these bounds, STOP and ask the user.

---

## The problem

In our variational GP for neural encoding (Poisson likelihood, arc-cosine kernel, eigenspace-projected variational inference), **increasing the number of inducing points M beyond ~100-250 degrades test performance** instead of improving it. This is the opposite of what sparse GP theory predicts — more inducing points should give a better approximation to the full GP.

## Training mode

**Focus exclusively on `vargp_direct` mode with `interleave_fstep=True` and `fix_Amp=True`.** This is our best training configuration. Do not test vargp_old or default_gpy — they have separate known issues and are not the production path. Do not test with free Amp or without interleaved F-step — we already know intl+fixAmp is the best algorithm; the question is why large M hurts it.

## Evidence we already have (DO NOT re-run these — trust these numbers)

**Note on reproducibility:** Commits 97bc88f and 0f4dfff (April 2026) changed the random sequence for STA and inducing point selection. Exact bit-for-bit reproduction of pre-April results is not possible. The results below are still trustworthy — the seed change only affects which specific images are selected, not the statistical properties of the selection.

### 1. Original finding (Feb 2026, cell 8 only, seed 123, vanilla vargp_direct, NO interleave, free Amp)

Fixed n_iterations=50:

| M   | n_train=500 | n_train=2000 |
|-----|-------------|--------------|
| 50  | 0.841       | 0.855 (+1.7%)  |
| 100 | 0.869       | 0.767 (-11.7%) |
| 200 | 0.814       | 0.735 (-9.7%)  |

Key pattern: M=50 benefits from more data, M>=100 is *hurt* by it.
Full details: `investigations/performance_loss_ntrain_M/FINDINGS.md`

### 2. Massive 41-cell sweep (Mar 2026, seeds 1-3, n_train=2910, NO interleave, free Amp, old ES)

Experiment file: `experiments/2026-03-24_massive_allcells_64/results.jsonl`

M=300 mean test_r: 0.800, M=2910 mean: 0.817. M=2910 wins on 27/41 cells, but only by +0.017 — with 10x more inducing points we'd expect a much larger gain.

### 3. ES sweep (Apr 2026, M=250, n_train=3160, 3 seeds, intl+fixAmp, ELBO ES p=15)

Experiment file: `experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl`
Config name in JSONL: `64_elbo_intl_fixAmp`, M=250.
Mean test_r = **0.838** — this is our best known result across all configurations.

### 4. Ceiling fits (Apr 2026, n_train=3160, ELBO ES p=15, 3 seeds where noted)

**M=1500 intl+fixAmp (3 seeds):**
- Sweep file: `checkpoints/64x64_ceiling_M1500_intl_fixAmp/sweep_results.jsonl` (114 rows, one per cell×seed)
- Summary: `checkpoints/64x64_ceiling_M1500_intl_fixAmp/ceiling_results.json` (per-cell mean across seeds)
- Mean test_r = 0.829 (38/41 cells; cells 0, 12, 30 failed on all 3 seeds)

**M=1500 vanilla (no interleave, free Amp, seed=0 only):**
- Summary: `checkpoints/64x64_ceiling_M1500/ceiling_results.json`
- Mean test_r = 0.829 (all 41 cells)

**The key result: M=250 intl+fixAmp (0.838) beats M=1500 intl+fixAmp (0.829).** The degradation persists even with the best training algorithm. Something about large M itself is harmful.

---

## What you should investigate

This is a systematic investigation. Create a folder `investigations/M_degradation/` for all scripts and findings. Use `build_config_from_defaults()` from `run_single_mode.py` for all configs (never hardcode parameters). Record all results in JSONL files. Document findings in `investigations/M_degradation/FINDINGS.md` as you go.

Read these files before starting:
- `.claude/CLAUDE.md` — project overview, training modes, critical rules
- `investigations/performance_loss_ntrain_M/FINDINGS.md` — the original (limited) investigation
- `.claude/rules/working_guidelines.md` — how to work on this project

### Phase 1: Characterize the degradation curve (new experiments)

**1a. Dense M sweep on 5 representative cells.** Pick cells spanning the performance range: 2 strong (e.g., cells 1, 3 — test_r > 0.9 at M=250), 2 medium (e.g., cells 8, 9 — test_r ~0.85), 1 weak (e.g., cell 5 — test_r ~0.65). Use seed=0, n_train=3160. Sweep M = [50, 100, 200, 300, 500, 750, 1000, 1500]. All runs use `interleave_fstep=True, fix_Amp=True`.

Log per run: test_r, train_r, train_log_lik, final hyperparameters (A, lambda0, beta, rho, sigma_0, eps_0x, eps_0y), eigenspace dimension n_b, eigenvalue range (min and max of eigvals_b), stopped_early, n_iterations_run, training time.

Plot test_r vs M for each cell. This reveals the shape of the degradation curve — is it monotonic? Does it peak at some M then decline? Does the peak M vary by cell?

**1b. Seed stability check.** For M=[50, 250, 1000] only, run 3 seeds (0, 1, 2) on the same 5 cells. Is the degradation seed-stable or high-variance?

### Phase 2: Diagnose root cause (targeted experiments based on Phase 1)

Based on Phase 1 results, investigate the most promising hypotheses. Prioritize based on what Phase 1 reveals, but here are the candidates:

**H1: Inducing point selection quality.** Random selection from 3160 images means at M=1500, ~47% of the pool is selected. Many images may be redundant (similar natural images), wasting inducing capacity. Test: compare `ip_selection='random'` vs `ip_selection='pivoted'` at M=500 and M=1000 on the same cells. Pivoted Cholesky selects maximally informative inducing points. The existing code has `select_inducing_points_pivoted()` in `utils.py`. If pivoted selection eliminates the degradation, this is likely the root cause.

**H2: Eigenspace rank saturation.** The eigendecomposition of K_tilde (M×M) keeps only eigenvalues > eigval_tol (default 1e-4). With large M, the ratio n_b/M may drop — the effective model capacity plateaus while the optimization problem grows. Check n_b/M ratio from Phase 1 data. Also try eigval_tol=1e-6 at M=1000 on 2-3 cells.

**H3: Optimization insufficiency.** 50 EM iterations with early stopping may fire too early at large M. The variational parameters (m_b, V_b) are n_b-dimensional — more to optimize, but the same iteration budget. Test: run M=1000 with n_iterations=200 on 2-3 cells. Check at which iteration ES fires for each M from Phase 1 data.

**H4: Hyperparameter drift.** Larger M may push kernel hyperparameters to a different (worse) local optimum. Compare final hyperparameters at M=50 vs M=1000 from Phase 1. Especially watch beta (RF size) — if it drifts wider at large M, the RF becomes too broad. Plot hyperparameter trajectories (from `result['curves']`) at different M values.

**H5: Train-test divergence.** Compare train_r vs test_r from Phase 1 at each M. If train_r stays high while test_r drops, that's overfitting. If both drop, it's optimization failure.

### Phase 3: Confirm fix

Once you identify the root cause, run a confirmatory experiment: apply the fix and re-run the M sweep from Phase 1a on the same cells. Show that the degradation is eliminated or substantially reduced.

---

## Script conventions

- Build configs with:
  ```python
  config = build_config_from_defaults(
      mode='vargp_direct',
      M=..., n_train=3160, seed=..., cell=...,
      data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
      interleave_fstep=True, fix_Amp=True,
  )
  result = run_single_config(config)
  ```
- `run_single_config(config)` returns a dict with: `test_r`, `adjusted_r2`, `train_r`, `train_time`, `final_loss`, `stopped_early`, `n_iterations_run`, `curves` (per-iteration metrics), `_model`, `_likelihood`, `_indices_train`
- `result['curves']` contains per-iteration lists: `train_loss`, `train_log_lik`, `train_kl`, `train_r`, `A`, `lambda0`, `beta`, `rho`, `sigma_0`, `eps_0x`, `eps_0y`, `Amp`, `iter_time`
- Eigenspace dimension: `result['_model'].state.n_b`
- Eigenvalues: `result['_model'].state.eigvals_b` (tensor, shape (n_b,))
- For pivoted IP selection: pass `ip_selection='pivoted'` to `build_config_from_defaults()`
- Write results to JSONL (one JSON object per line)
- Create one script per phase, not one monolithic script
- Print progress with `flush=True` (scripts run in background)
- Define cell/seed/M lists as labeled constants at script top with comments explaining choices

## What NOT to do

- Do not re-run fits we already have results for (evidence items 1-4 above). Trust those numbers.
- Do not modify core library files (eigenspace_*.py, kernels.py, etc.) during investigation. If a fix is found, propose it but implement in a follow-up.
- Do not test default_gpy or vargp_old modes — focus on vargp_direct only.
- Do not test without interleave_fstep or with free Amp — focus on the intl+fixAmp configuration only.
- Do not test on 108x108 — use 64x64 only.
- Do not change default_params.json. Pass overrides explicitly.

## Existing code to reuse

- `run_single_mode.py`: `build_config_from_defaults()`, `run_single_config()`, `load_pnas_data()`
- `utils.py`: `select_inducing_points_pivoted()` for pivoted Cholesky IP selection
- `investigations/performance_loss_ntrain_M/` — the old investigation (reference only, limited scope)
- `plotting/plot_training.py` — per-cell training curve style conventions

## Deliverables

1. `investigations/M_degradation/FINDINGS.md` — evolving findings document, updated after each phase
2. `investigations/M_degradation/phase1_m_sweep.py` — Phase 1 sweep script
3. `investigations/M_degradation/phase2_*.py` — Phase 2 hypothesis-specific scripts
4. `investigations/M_degradation/results/` — JSONL outputs and plots
5. A clear conclusion: root cause identified, fix proposed (or "needs further investigation" with specific next steps)
