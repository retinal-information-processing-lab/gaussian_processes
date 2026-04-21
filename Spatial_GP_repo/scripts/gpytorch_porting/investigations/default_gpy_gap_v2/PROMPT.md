# New-session prompt: default_gpy vs. vargp_direct gap — clean restart

**Read this entire document before doing anything.** It is the charter for this
investigation. It replaces the earlier `investigations/default_gpy_gap/`
investigation, which reached wrong conclusions from an underpowered sample.
That folder is preserved as history; do not overwrite it.

---

## What the previous investigation tried, and what went wrong

The prior investigation (`investigations/default_gpy_gap/SCRAPBOOK.md`) claimed a
systematic gap of 0.02–0.18 test_r between `default_gpy` and `vargp_direct` on
cell 8, diagnosed it as "A parameter explosion in joint LBFGS," and proposed
`alternating_fstep=True` (separate LBFGS passes for variational+kernel and for
A/λ₀) as the fix. It was declared "ROOT CAUSE FOUND & FIXED."

It was not. Specifically:

- **The Phase-1 evidence was 2 cells × 3 seeds at M=50, n_train=500.** Too few
  cells. The large gap reported on cell 8 was not a systematic effect.
- **An 8-cell follow-up (cells 0, 1, 3, 5, 8, 10, 15, 22 × 3 seeds) at the same
  config disproved it.** Paired mean Δ test_r (vargp_direct − default_gpy joint)
  = **−0.002**. No systematic gap at that config.
- **`alternating_fstep` catastrophically fails on most cells.** In the same
  8-cell sweep, it failed on 5/8 cells (0, 5, 10, 15, 22) by trapping A at or
  below init (0.01), producing test_r drops of 0.2–0.9 vs. the other modes.
  Cell 10 went to test_r = −0.08. The fix traded one optimizer pathology for
  another.
- **Historical massive-cell data (`experiments/2026-03-24_massive_allcells_64/`,
  492 rows) is invalidated by code churn** — many structural changes since
  `a0787f1` (sigma_0 parameterization flips, LBFGS Hessian corruption fix,
  autograd memory leak fix, IP selection default flip, ES machinery rewrite,
  beta bound tightening, etc.). Cannot be cited as current-code evidence.

**Current position**: the question "is there a gap between `default_gpy` and
`vargp_direct` under current code" has no valid answer yet. That is what this
new investigation exists to answer.

---

## Hard constraints (from `.claude/rules/bewary.md` — still apply)

- **Do not hardcode parameter values.** Any tolerance, threshold, or config
  choice you're tempted to inline must first be raised to the user.
- **Do not change `default_params.json` defaults** as part of this work.
  `alternating_fstep` stays `false`.
- **Any formula that evaluates quantities not pre-specified in this charter
  must be raised to the user before implementing.**
- **Do not redefine the decision criterion** after seeing data. The criterion
  is locked below.
- **Do not expand scope** because the first result looks inconclusive or
  surprising. If the pre-registered analysis returns "inconclusive," report it
  as inconclusive. Do not run more cells or more seeds to chase a desired
  answer. Come back to the user first.
- **Do not re-investigate `alternating_fstep`** beyond the one-seed sanity check
  described below. It is already rejected.

---

## Required reading before writing any code

1. `investigations/default_gpy_gap/SCRAPBOOK.md` — full history of the previous
   investigation, including the final 2026-04-20 section documenting what went
   wrong and why. Read every section. Do not skim.
2. `Spatial_GP_repo/scripts/gpytorch_porting/.claude/CLAUDE.md` — the Critical
   Rules section, the Known Issues section (especially "Stuck-near-init in
   active loop Phase 1" and "Performance Degradation with Large ntrain+M").
3. `Spatial_GP_repo/scripts/gpytorch_porting/.claude/rules/bewary.md`,
   `critical_short_rules.md`, `debugging.md`.
4. `investigations/default_gpy_gap/run_baseline.py` — reference sweep script
   used in the previous investigation. Good starting point structurally; do not
   reuse its hardcoded (M=50, n_train=500) config.
5. `investigations/default_gpy_gap/analyze_8cell_sweep.py` — reference analysis
   script. Good template for the new analysis; borrow freely.

---

## Lessons from the previous investigation — internalize these

These were **owned as mistakes** by the previous session. Do not repeat.

1. **Do not start with a 2-cell sample and declare a finding.** A 2-cell gap is
   a per-seed pathology observation, not a mode comparison. Budget a decisive
   sample up front.
2. **Do not iterate 2 → 8 → 15 → 41 cells reactively.** That pattern lets
   bias-preserving noise into each step's interpretation and produces a
   narrative that keeps shifting. Pick the sample size that answers the
   question and run it once.
3. **Do not label operating points as "degenerate" or "canonical" without data.**
   No one has mapped how `test_r` depends on (M, n_train) for `default_gpy`
   under current code. Report what you measured at the exact config you ran;
   do not extrapolate.
4. **Do not conflate "fix works on cell 8" with "fix generalizes."** Cell 8 had
   a specific seed-42 pathology (A exploding to 0.46); a "fix" that eliminates
   that specific failure mode may introduce a different failure mode elsewhere.
   Always evaluate on a broad cell sample before declaring a fix.
5. **Do not cite historical experiment results without verifying the code
   hasn't changed in ways that would invalidate them.** Check
   `experiments/<name>/metadata.yaml` for `git_commit`, then `git log
   <commit>..HEAD` on the training code paths. If any listed commit affects
   the numerics, the data is not citable.
6. **Always carry over a paired-comparison frame.** Mean comparisons across
   unpaired samples are misleading. Always compute Δ test_r per (cell, seed)
   pair, then aggregate. Report paired mean, std, and per-cell sign.
7. **`A` is a non-leaf property on `PoissonLikelihood`.** `A = exp(raw_A)`. If
   you need to read the gradient, read `likelihood.raw_A.grad`. If you need the
   value, `likelihood.A.item()` is fine.
8. **GPU wall-clock is session-noisy** (thermal, cuBLAS autotune, contention).
   Do not compare absolute s/iter across sessions. Compare modes within a
   session; use relative metrics across sessions.

---

## The question

**Under current code, is there a systematic gap in test_r between
`default_gpy` (joint LBFGS) and `vargp_direct` at operating point
(M=300, n_train=1500, 64×64, arc_cosine kernel, ground-truth RF init, ELBO ES on)?**

---

## Pre-registered decision criterion — locked

Compute the per-(cell, seed) paired difference:

```
Δ(cell, seed) = test_r_{vargp_direct}(cell, seed) − test_r_{default_gpy}(cell, seed)
```

Aggregate:

```
mean_Δ = mean over (cell × seed) of Δ
```

Primary outcome:

- **If `|mean_Δ| > 0.02`** → report "GAP EXISTS at this config, of magnitude
  `mean_Δ` in favor of [vargp_direct / default_gpy]." The sign matters.
- **If `|mean_Δ| ≤ 0.02`** → report "NO GAP DETECTED at this config, within the
  pre-registered threshold of 0.02."

Also report, but do not redefine the decision on:

- `mean_Δ` with paired std, range [min, max], IQR.
- Per-cell mean Δ (3-seed average per cell): how many cells sit on each side of
  the threshold. Shows whether the aggregate is driven by a few cells or
  spread evenly.
- For each of the 15 cells, its 3-seed mean and std for each mode.
- Timing: `train_time` per run, per mode, and mean `s/iter`. Report mode means
  and standard deviations. This is a secondary outcome but we want a concrete
  timing answer from this sweep.

---

## Experimental design — locked

| parameter | value |
|---|---|
| dataset | `datasets/PNAS_64x64_center_crop_no_renorm.npz` (64×64) |
| M (inducing points) | **300** |
| n_train | **1500** |
| cells | **15, stratified by firing rate** — see procedure below |
| seeds | **{42, 123, 789}** for `vargp_direct` and `default_gpy`; **{42}** only for `default_gpy_alt` |
| modes | `vargp_direct`, `default_gpy` (joint), `default_gpy_alt` (sanity only) |
| n_iterations | 50 (matches previous investigation) |
| early stopping | on, ELBO-based, default patience/min_delta (from `default_params.json`) |
| ip_selection | `'random'` (prevents vargp_old confound per CLAUDE.md) |
| kernel | `arc_cosine` (default) |
| rf_init | `'ground_truth'` (default) |
| dtype | float32 (default) |
| device | cuda |
| all other params | `build_config_from_defaults()` defaults — do not override |

**Run count**:
- `vargp_direct`: 15 cells × 3 seeds = 45 runs
- `default_gpy`: 15 cells × 3 seeds = 45 runs
- `default_gpy_alt`: 15 cells × 1 seed (seed 42) = 15 runs, sanity check only
- **Total: 105 runs.**

Budget estimate: at M=300, n_train=1500 with ES on, each run ≈ 20–60 s. Total
budget ≈ 40–100 min on a clean GPU. Run in background and monitor.

---

## Cell selection procedure — execute before the sweep

The 15 cells must be stratified by firing rate: 5 "low", 5 "mid", 5 "high".
Firing rate per cell = mean response (spike count) over the training pool.

Exact procedure (deterministic, reproducible):

1. Load the training pool from `PNAS_64x64_center_crop_no_renorm.npz`. Combine
   `responses_train` + `responses_val` to get the full 3160-response pool
   (matches how the main training code assembles the pool — see
   `run_single_mode.py:687` area).
2. For each cell_id in 0..40, compute `firing_rate[cell_id] = R_pool[:, cell_id].mean()`.
3. Sort cells by firing rate ascending. Split into 3 equal buckets
   (low = indices 0..13, mid = 14..27, high = 28..40).
4. From each bucket, pick 5 cells at evenly-spaced positions within the bucket
   (e.g., indices 1, 3, 6, 9, 12 within each bucket — spread, not endpoints).
5. Write the chosen 15 cells to `cells_used.json` in the investigation folder
   with their firing rates, before starting the sweep. This is the preregistered
   cell list. Do not change it post-hoc.

**Before running the sweep**, print the selected cells + firing rates and ask
the user to confirm the cell list. The user may want to override a selection.
Once the list is confirmed, write it to `cells_used.json` and proceed.

---

## Code that exists vs. code to write

### Exists and should NOT be re-written
- `run_single_mode.py` with `build_config_from_defaults()` — **use this for all
  config construction**. Do not bypass.
- `gpy_training.py` including the slim closure and `alternating_fstep`
  plumbing (landed 2026-04-20). It's production code now; don't touch it
  unless you find a correctness bug.
- `tests/test_gpy_alternating_fstep.py` — covers the slim-closure path.
- `default_params.json` — do not modify.
- `investigations/default_gpy_gap/SCRAPBOOK.md` and its data files — do not
  modify (historical record).

### To write in `investigations/default_gpy_gap_v2/`

1. `select_cells.py` — implements the stratified-selection procedure above.
   Prints the cell list, asks for user confirmation, writes
   `cells_used.json`.
2. `run_sweep.py` — fork of `run_baseline.py`. Adapted to the locked config
   above. Reads `cells_used.json`. Writes `results.jsonl` with one line per
   run. Does not overwrite an existing `results.jsonl` without the user's
   confirmation (check for file before opening in append mode; if non-empty
   and you're starting a fresh sweep, save the old file as
   `results_<timestamp>.jsonl` and start a new one).
3. `analyze.py` — computes and reports everything in the "Pre-registered
   decision criterion" section above. Must produce:
   - A per-cell table (15 rows × modes × 3 seeds mean±std).
   - The paired-Δ summary (mean, std, range, per-cell signs).
   - The binary decision (gap / no gap) against the 0.02 threshold.
   - Timing table (per-mode mean & std of `train_time` and `train_time /
     n_iterations_run`).
   - A per-mode `final_A` distribution summary (catches explosion/freeze).
4. `SCRAPBOOK.md` — investigation narrative, results, decision, lessons.
   Structure described below.

Borrow freely from `investigations/default_gpy_gap/analyze_8cell_sweep.py` and
`run_baseline.py`. You do not need to re-invent infrastructure.

---

## Execution order (required)

Execute these steps strictly in order. **Pause after each for user review
before proceeding to the next.**

1. **Read the required docs** (listed above). Confirm to the user that you've
   read them and list any questions about the charter before touching code.
2. **Implement `select_cells.py`**, run it, show the user the 15 selected
   cells + firing rates. **Wait for user confirmation** before writing
   `cells_used.json`.
3. **Implement `run_sweep.py`**. Show the user the exact config it will use
   and the total run plan. **Wait for user confirmation** before launching.
4. **Launch the sweep** (in background for long runs). Monitor results.jsonl
   line count.
5. **Implement `analyze.py`**. Run it on the completed sweep. Show the user
   the full output. **Wait for user review** before interpretation.
6. **Draft `SCRAPBOOK.md`** with results and the pre-registered decision.
   Show the user. **Wait for user approval** before finalizing.
7. **If the analysis decision is "NO GAP DETECTED"**: the investigation
   concludes here with a written record that this config shows no gap. Do not
   propose follow-ups without the user asking.
8. **If the analysis decision is "GAP EXISTS"**: do not propose fixes. Report
   the finding, the per-cell structure, and A-distribution observations. Ask
   the user how they want to proceed.

---

## What the SCRAPBOOK.md must contain

Required sections, in order:

1. **Charter** — one-paragraph summary of what the investigation is and what
   it explicitly is not.
2. **Relation to the v1 investigation** — short. Points to
   `investigations/default_gpy_gap/SCRAPBOOK.md` for history. Names the v1
   mistakes (from the "Lessons learned" section above) that this v2
   investigation is designed to avoid. Do not duplicate the v1 content.
3. **Pre-registration** — copies of the locked decision criterion and
   experimental design **dated before the sweep ran**. The new session should
   write this section *before* step 4 above and commit it conceptually (even
   if not yet a git commit) as a reference for later.
4. **Cell selection** — the 15 cells with firing rates, and the selection
   procedure. References `cells_used.json`.
5. **Sweep execution** — what was run, when, git commit hash, any notable
   warnings, total wall time.
6. **Results** — per-cell table, paired-Δ summary, decision (gap / no gap),
   A-distribution table, timing table.
7. **`default_gpy_alt` sanity check** — the 15 single-seed runs: did it fail
   on the same cells as at M=50, n_train=500? One-paragraph observation.
8. **Conclusions** — purely what the data supports. No speculation about
   mechanisms unless the user asks. No proposed fixes.
9. **Lessons carried forward** — which lessons from v1 (items 1–8 above) this
   investigation adhered to, and any new ones discovered.

---

## Do NOT do any of the following

- Do not modify `default_params.json` or any file outside
  `investigations/default_gpy_gap_v2/` unless the user explicitly asks.
- Do not expand the cell count, seed count, or operating points mid-sweep.
- Do not attempt a fix if a gap is detected. Report and stop.
- Do not re-investigate `alternating_fstep` beyond the 15-run sanity check.
- Do not rename / restructure the v1 investigation folder.
- Do not run the full 41-cell sweep. 15 is the pre-registered sample; if it's
  inconclusive, that's a finding to discuss, not a trigger to scale up.
- Do not delete or overwrite `results.jsonl` if it already contains runs —
  rename and start fresh.
- Do not start the investigation while the working tree has uncommitted
  changes to training code. If any exist, raise to the user first.

---

## Ground-truth sanity checks the new session must perform before the sweep

1. **GPU availability**: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
   should show no other compute processes. If contended, wait / ask the user.
2. **Code state**: `git status` must show no uncommitted changes in
   `gpy_training.py`, `eigenspace_*.py`, `kernels.py`, `likelihoods.py`,
   `run_single_mode.py`, `default_params.json`. If any do, stop and raise.
3. **Smoke test**: before launching the 105-run sweep, run a single
   (vargp_direct, cell=1, seed=42, M=300, n_train=1500) run as a smoke test
   to confirm the config plumbs through `build_config_from_defaults`. Takes
   under a minute.
4. **Unit tests still pass**: `python -m pytest
   tests/test_gpy_alternating_fstep.py -v` should all pass. Known-failing
   tests (`test_13_default_gpy_elbo_es`, `test_analytical_gradients.py`) are
   pre-existing and unrelated — do not fix them in this investigation.

---

## One-line summary of the task

Run 105 training runs under a pre-registered charter, test whether
`|mean_Δ test_r| > 0.02` at (M=300, n_train=1500), write up the decision, do
not fix anything, do not speculate, do not expand scope.
