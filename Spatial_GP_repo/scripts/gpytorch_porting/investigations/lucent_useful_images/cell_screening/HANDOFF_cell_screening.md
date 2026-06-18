# Handoff: screen PNAS cells for clean monotonic learning curves

**Branch**: `pietro/lucent-useful-images` (gaussian_processes submodule)
**Date**: 2026-06-18
**Status**: Ready for implementation (discuss methodology with the user first, then run)
**Plan file**: `cell_screening/PLAN_cell_screening.md` (same folder — MANDATORY read)

---

## Motivation

The parent investigation (`lucent_useful_images/`) builds an image-generation pipeline:
synthesize the **most-useful image** to show a retinal cell (the one that maximizes the GP's
information utility), using lucent's Fourier+sigmoid parameterization so the image stays
physically bounded. The intended demonstration is: *as the GP model improves with more
training data (n_train), the synthesized optimal image sharpens from a diffuse probe into a
well-defined receptive-field stimulus.*

That demo needs **testbed cells whose model actually improves smoothly with n_train.** It
doesn't currently have any verified ones. When we ran the pipeline across cells we found the
`default_gpy` fits are NOISY: test_r bounces (e.g. cell 13 across n_train=50..300 step 25 went
0.37, 0.89, 0.94, 0.95, 0.93, **0.77**, 0.95, 0.96, 0.96, **0.78**, 0.96 — two big dips at
n=175 and 275; cell 8 was non-monotonic 0.36/0.70/**0.27**/0.81). With test_r itself
bouncing, we cannot cleanly show "more data -> sharper image." So: **screen the PNAS cells and
find a few with an easy, monotonic test_r increase** to use as the pipeline's testbed. This is
the single biggest blocker flagged at the end of the lucent work.

## Decisions and Rationale

- **PNAS dataset + default_gpy + M = n_train — NOT the closed-loop analysis cells.** The image
  pipeline runs on the static PNAS dataset (cells 0–40) with the `default_gpy` GP (the
  distribution-aware utility requires its covariance). A *different* set of "good cells"
  exists in the `analysis/` superrepo (closed-loop April/August cells like `260415_ch75`,
  `250820_ch245`, with clean warm-start active+random trajectories) — but those are a
  different dataset AND a different engine (legacy varGP), non-comparable cell IDs. Do NOT
  conflate. This screen is PNAS-only. (If lucent "graduates," moving to the closed-loop data
  is a separate future session.)
- **Coarse grid n_train = M = {50,100,150,200,250,300} (step 50), one seed (42).** The user
  set this for tractability: a finer step (25) over 41 cells with M up to 300 is too slow, and
  multiple seeds multiply the cost. The deeper rationale for one seed: we want cells where the
  monotonic increase is *easy* — robust enough to look clean even from a single fit. A cell
  that only looks clean after seed-averaging is not an easy testbed cell.
- **Reuse `screen_ladder.py`, don't rewrite.** It already trains each cell across an n_train
  ladder at M=n_train, seed=42, prints the per-cell test_r ladder, and flags a "CLEAN"
  monotonic ladder. The task is to extend its grid (step 50), its candidate list (all 41 minus
  exclusions), and its scoring (per the methodology the user picks).
- **Methodology is deliberately left OPEN for the user.** How to score "cell quality /
  monotonicity" (Spearman? max-drop? net gain? final level? thresholds?) materially changes
  which cells are picked, and the user explicitly wants to decide this. So the new session's
  FIRST job is to discuss it (plan step "FIRST STEP"), not to pick a metric unilaterally.
- **Stay on the pinned 75b207a engine.** The image pipeline must run on the exact GP code the
  superrepo pins (so the eventual move to closed-loop data is apples-to-apples). So screen on
  this branch/engine; additive files only; no engine edits.

## Critical Subtleties

- **PNAS index vs closed-loop ID.** If you find yourself reading `analysis/cross_sessions/` for
  "good cells", stop — that is the wrong dataset. PNAS cells are integers 0–40 from
  `PNAS_paper_sorted_data.npz`; the screen trains them with `gp_models.train_default_gpy`.
- **One seed mislabels borderline cells.** A single unlucky variational fit can put a dip in an
  otherwise-good cell's ladder (that is exactly what made cell 13 look non-monotonic at n=175/
  275). So: a cell that screens CLEAN with one seed is a safe testbed pick; a cell that screens
  dirty is NOT necessarily bad — it may just have had one bad fit. Report the borderline ones;
  do not over-trust a single dip as disqualifying for an otherwise strong cell.
- **M = n_train means M up to 300** — the high-n fits are the slow, memory-heavy ones. In a
  ~200-fit loop you MUST free GPU memory each iteration (`del model/lik; gc.collect();
  torch.cuda.empty_cache()`) or you will OOM partway (the GPU is shared — `nvidia-smi` first).
  `screen_ladder.py` already trains-and-discards per cell; keep that pattern.
- **The stale-engine confound (know it, don't act on it).** This investigation sits on a GP
  engine 106 commits behind `pietro/workingbranch`; the missing commits include an A-prior /
  adaptive-A_init and the M-degradation work that target *exactly* this fit instability. So the
  noise you are screening against may be partly an old-engine artifact. We deliberately stay on
  75b207a to match the pipeline/analysis — but if the screen finds NO cleanly-monotonic cell,
  raise the engine question to the user rather than concluding "PNAS has no good cells."
- **Don't stage other sessions' WIP.** `git status` shows dirty `investigations/utility/*`
  files (laplace investigation) that are NOT yours — explicit staging only.

## Uncommitted Changes

None of yours — the doc work is committed (`91d8e14`). The only dirty items in the worktree
belong to OTHER sessions and must not be touched/staged:
- ` M investigations/utility/docs/da_utility_theory.md`
- `?? investigations/utility/{HANDOFF_laplace_pointwise_error.md, docs/LAPLACE_VALIDITY_SUMMARY.md, laplace_*, sigma_cap.*, ...}`

(Branch is 6 commits ahead of `origin` — the lucent + LUT + docs commits are local-only; pushing is the user's call.)

## Files to Read First

1. `cell_screening/PLAN_cell_screening.md` — the methodology-choices-to-discuss + the steps + deliverables.
2. `../screen_ladder.py` and `../screen_cells.py` — the EXISTING screening tools to extend (M=n_train, seed=42, per-cell ladder + monotonic flag). This is most of the code already.
3. `../gp_models.py` — `train_default_gpy(cell, n_train, M=None->n_train, seed=42)` returns `(model, likelihood, idx_train, test_r)`; `load_pool()` gives the dataset + global pixel range. Note the dataset path points at the sibling `ClosedLoopProject` repo (pre-existing, out of scope).
4. Memory `default-gpy-low-ntrain-unstable` (in the project memory dir) — the prior finding: at fixed M=50, only ~2/13 cells (13, 3) had a clean positive monotonic ladder; most are negative/non-monotonic at low n_train. Note it was M=50; this screen is M=n_train and a coarser grid.
5. `../README.md` — the parent investigation (the M=n_train convention; the pipeline; the noisy-ladder problem in the cell roster table).
6. `.claude/rules/debugging.md` 3.6 (cell 10 unlearnable with default_gpy) and 3.7 (perf degradation with large ntrain+M); `.claude/CLAUDE.md` (the 6 STA-edge-artifact cells 0,5,6,15,22,39 to exclude).
7. `../standard_utility/FINDINGS.md` (optional context on the broader investigation — not needed for the screen).

Env: `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python` (the Bash default base python has no torch). GPU shared (RTX 4090) — `nvidia-smi` first.

## Caveats and Open Questions

- **A cleanly-monotonic PNAS cell may not exist** at this grid/engine. That is a real possible
  outcome — if so, do NOT silently pick the "least bad" one; report it and raise the
  engine-rebase question (above) to the user.
- **One seed** is a deliberate noise/speed tradeoff (above). The chosen testbed cells should be
  robustly clean; flag any that are borderline.
- **Monotonicity is necessary but not sufficient** for a good testbed cell — also want decent
  final test_r (the model must actually become good) and ideally RF diversity across the picked
  cells. Fold these into the discussion.
- **The methodology is intentionally unfixed** — the metric/threshold choice is the user's to
  make (that is the point of the "discuss first" step).

---

## Continuation Prompt

```
Screen the PNAS cells to find a few testbed cells with an EASY, monotonic increase in test_r
as the training-set size grows. This is for the lucent image-generation pipeline (it needs
cells whose model improves smoothly with data). You are on the gaussian_processes submodule
branch pietro/lucent-useful-images — confirm with `git branch --show-current` and `git status`.

READ FIRST (investigations/lucent_useful_images/):
  1. cell_screening/PLAN_cell_screening.md   (the methodology to DISCUSS + steps + deliverables)
  2. cell_screening/HANDOFF_cell_screening.md (the WHY, the context, the gotchas)
  3. screen_ladder.py + screen_cells.py        (the EXISTING tools to extend — most of the code)
  4. gp_models.py                              (train_default_gpy: M=n_train, seed=42)
  Plus the memory `default-gpy-low-ntrain-unstable` and README.md.

FIRST — DO NOT RUN ANYTHING YET. Discuss with the user (they explicitly asked) the cell-quality
methodology: the performance metric (recommend test_r), the monotonicity/quality score and
thresholds (Spearman / max single-step drop / net gain / final level), the cell pool (all 41
minus the 6 STA-edge cells 0,5,6,15,22,39; maybe minus cell 10), and how many testbed cells
(3-5). Confirm the user's already-set constraints: n_train = M = {50,100,150,200,250,300}
(step 50), one seed (42).

THEN: extend screen_ladder.py into cell_screening/screen_testbed.py with the agreed grid/pool/
score; estimate runtime on one cell; run all cells in the BACKGROUND saving a per-cell ladder
cache; rank; pick the testbed cells (note RF diversity); make a test_r-vs-n_train figure;
write cell_screening/FINDINGS.md.

HARD CONSTRAINTS:
  - ADDITIVE files under investigations/lucent_useful_images/cell_screening/ ONLY. NEVER edit a
    GP engine file (acquisition/utils/utility/kernels/eigenspace/gpy_*/run_single_mode). Verify
    `git diff --cached --name-only | grep -v 'investigations/lucent_useful_images/'` is EMPTY
    before every commit. Engine must stay byte-identical to the pinned 75b207a.
  - Shared worktree: other sessions' WIP is dirty (investigations/utility/*) — NEVER git add -A;
    stage explicit paths; clean commit messages (repo style, no coauthor footer).
  - Free GPU memory between fits (del/gc/empty_cache); nvidia-smi first; M=n_train so M up to 300.
  - Env python: /home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python.

DELIVERABLE: the chosen testbed cells + their ladders, a reproducible screen_testbed.py + cache
+ figure + FINDINGS.md, and a final-message report (chosen cells, the agreed methodology, clean
vs borderline counts, runtime, no-engine-edit confirmation) so the originating session can verify.
If NO cell is cleanly monotonic, report it and raise the stale-engine question — do not silently
pick the least-bad cell.
```
