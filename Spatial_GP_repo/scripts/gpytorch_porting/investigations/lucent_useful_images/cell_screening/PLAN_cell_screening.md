# PLAN: screen PNAS cells for clean monotonic learning curves (image-pipeline testbed)

**Branch**: `pietro/lucent-useful-images` (gaussian_processes submodule). Commit HERE.
**Scope**: ADDITIVE files under `investigations/lucent_useful_images/cell_screening/`. NO engine edits.
**Goal**: Find **a few testbed cells** whose prediction performance increases **easily and
monotonically with dataset size** (n_train), to demonstrate the lucent image-generation
pipeline without the confound of noisy GP fits.

Read the companion `HANDOFF_cell_screening.md` first for the WHY + the full context.

## FIRST STEP — DISCUSS with the user before running (the user explicitly asked for this)
Do NOT launch the full screen until you have agreed with the user on the cell-quality
methodology. Propose options (recommendation in parentheses) and let the user choose:

- **(a) Performance metric.** test_r = Pearson r on the 30-image test set (what the pipeline
  uses), vs explained variance, vs reliability-corrected r. *(Recommend test_r — matches the
  pipeline and `metrics.compute_pearson_correlation`.)*
- **(b) "Monotonic / easy" score.** Candidates to combine: Spearman ρ(test_r, n_train)
  (1.0 = perfectly increasing); net gain `test_r[300] − test_r[50]`; **max single-step drop**
  (penalizes dips — this is what bit cells 13/8 before); fraction of non-decreasing steps;
  final test_r (must reach a good level). *(Recommend: rank by min over the curve of "no drop
  > 0.03" AND net gain > 0.3 AND final test_r > 0.85; report Spearman + max-drop + net-gain +
  final for every cell so the user can re-threshold.)*
- **(c) The bar for "easy".** e.g. starts non-negative, climbs smoothly to > 0.85, largest dip
  < 0.03. Confirm the numbers with the user.
- **(d) Cell pool.** All 41 PNAS cells **minus the 6 STA-edge-artifact cells (0, 5, 6, 15, 22,
  39)** (near-zero test_r on 108x108, CLAUDE.md). Optionally also drop cell 10 (known
  default_gpy-hard, debugging.md 3.6). Confirm.
- **(e) How many testbed cells.** "a few" — confirm 3–5.
- **(f) n_train grid + max.** The user set **n_train = M = 50, 100, 150, 200, 250, 300**
  (step 50, M = n_train). Confirm the max is 300 (finer/longer is explicitly too slow).
- **(g) Seeds.** The user chose **one seed (42)**. Confirm. (Caveat in the handoff: one seed
  means a single unlucky fit can mislabel a good cell — but the goal is the robustly-EASY
  cells, so one seed is the right screen; just report borderline cells.)

## Fixed constraints (already decided — do not re-litigate)
- **PNAS dataset, `default_gpy`, M = n_train.** The image pipeline uses the PNAS cells (integer
  index 0–40) with default_gpy (the DA utility needs it). This is NOT the closed-loop analysis
  dataset — do not conflate (see handoff).
- **n_train = M = {50,100,150,200,250,300}; seed = 42.** (Subject to (f)/(g) confirmation.)
- **Stay on the pinned 75b207a engine; no engine edits.** All work additive under
  `cell_screening/`. The image pipeline runs on this engine to match the analysis.

## Steps (after the methodology is agreed)
1. **Reuse `../screen_ladder.py`** (already M=n_train, seed=42, per-cell test_r ladder + a
   monotonic flag). Copy/extend into `cell_screening/screen_testbed.py`: the agreed n_train
   grid, the agreed cell pool, the agreed quality score. Training is via
   `gp_models.train_default_gpy(cell, n_train, seed=42)` (M defaults to n_train).
2. **Estimate runtime first** (one cell × full grid), then run all cells. ~34 cells × 6
   n_train ≈ 200 fits; M up to 300 makes the high-n fits the slow ones — likely 30–60 min.
   **Run in the background; save the per-cell ladders to a cache** (csv/pkl) so a crash or a
   re-plot needs no recompute. Free GPU memory between fits (`del`, `gc.collect()`,
   `torch.cuda.empty_cache()`); `nvidia-smi` first (shared GPU).
3. **Rank** cells by the agreed score; pick the few testbed cells. Note RF diversity as a
   secondary criterion (different RF locations/types make a better testbed than 4 near-identical
   cells).
4. **Figure**: test_r-vs-n_train curves for the top cells (and all cells faded behind), fixed
   axes, Liberation Sans. Output to `cell_screening/` (PNG; the folder `.gitignore` ignores
   `*.png` — keep figures regenerable, commit code + the results table + the findings doc).
5. **Document**: `cell_screening/FINDINGS.md` — the full per-cell ladder table, the ranking,
   the chosen testbed cells, and the agreed methodology. Consider updating the memory
   `default-gpy-low-ntrain-unstable` with the M=n_train coarse-grid result (it currently
   records the M=50 finding).

## Deliverable + report (for the originating session to verify, and the user to use)
- The **few testbed cells** with their ladders (the headline).
- `cell_screening/screen_testbed.py` (reproducible, seed-fixed), the results cache, the figure,
  `FINDINGS.md`.
- End your final message with: the chosen cells + their test_r ladders; the methodology the
  user agreed to; how many cells were "clean" vs borderline; a one-line confirmation no engine
  file was touched (`git diff --cached --name-only | grep -v 'investigations/lucent_useful_images/'`
  empty); runtime; anything surprising (e.g. if NO cell is cleanly monotonic — that is itself a
  finding to discuss).

## Tidiness / commits (HARD — shared worktree)
- ~10 sessions share this worktree; other uncommitted WIP exists (`investigations/utility/*`).
  NEVER `git add -A`/`git add .`/`git commit -am`. Stage explicit paths; confirm
  `git diff --cached --name-only` before each commit. Clean messages (repo style: no coauthor
  footer). Env python: `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python`.
