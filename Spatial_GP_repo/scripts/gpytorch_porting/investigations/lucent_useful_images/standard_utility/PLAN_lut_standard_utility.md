# PLAN: LUT-backed standard utility for the lucent investigation

> **STATUS: DONE (2026-06-18)** — implemented, validated, and independently verified. See
> `LUT_IMPLEMENTATION_REPORT.md`. This plan is kept as the historical spec.

**Branch**: `pietro/lucent-useful-images` (gaussian_processes submodule). Commit HERE.
**Scope**: ADDITIVE files under `investigations/lucent_useful_images/` only. NO engine edits.
**Goal**: Replace the numerically-unstable live `r_max=100` standard utility with the
precomputed LUT, in a **torch-differentiable** form, so the lucent standard-utility
optimization is numerically valid (no 296k-nat blow-ups) and the standard-vs-DA comparison
is fair. **Standard utility only** (DA deferred). No warm-start. No engine edits.

Read the companion `HANDOFF_lut_standard_utility.md` first for the WHY.

## HARD scope rules (violating any of these defeats the purpose)
1. Every new/edited file is under `investigations/lucent_useful_images/`. The only existing
   file you may edit is our own `optimize_image.py` (it lives in that folder). **Do NOT edit
   any GP engine file** (`acquisition.py`, `utils.py`, `utility.py`, `kernels*.py`,
   `eigenspace*.py`, `gpy_*.py`, `run_single_mode.py`, the repo-level `utility.py`/`utils.py`,
   etc.). The submodule must stay byte-identical to the superrepo's pinned commit 75b207a.
   **Verify before EVERY commit:** `git diff --cached --name-only | grep -v 'investigations/lucent_useful_images/'`
   must print NOTHING.
2. **Standalone**: vendor everything you need INTO the repo. No runtime OR test code path may
   `import` from / read a path under the superrepo `analysis/` folder. (A provenance note that
   cites the source path is fine — that's documentation, not a dependency.)

## Steps

### 1 — Vendor the LUT (make it standalone)
- Create `investigations/lucent_useful_images/lut/`.
- Copy from `analysis/figures/utility_landscape/`:
  `lut.npz`, `lut_meta.json`, `lut_select.py`, `lut_interp.py`, `LUT_README.md`
  into `lut/`. (`lut_select.py`/`lut_interp.py` are the numpy reference — vendored so the
  validation test is standalone too.)
- Write `lut/PROVENANCE.md`: source path, the superrepo commit hash that built the LUT, a
  `sha256` of `lut.npz`, and the regenerate command (`build_lut.py` in the source folder).
- **GITIGNORE GOTCHA (confirmed):** the folder `.gitignore` has `*.npz` (and `*.png` etc.),
  so `lut/lut.npz` is ignored by default. It MUST be committed (it is THE standalone
  artifact). Add a negation line to `investigations/lucent_useful_images/.gitignore`:
  `!lut/lut.npz`. Confirm with `git check-ignore lut/lut.npz` returning nothing, then
  `git add lut/lut.npz` and confirm `git status` lists it staged. (`git add -f` works too but
  the `!` negation is cleaner and self-documenting.)

### 2 — Torch-differentiable LUT interpolator
- `lut/lut_utility.py`: load `lut/lut.npz` once (`U`, `mu_axis`, `s2_axis`); expose
  `lut_U(mu_g, sigma2_g) -> U` (torch in, torch out, differentiable w.r.t. both inputs).
- Bilinear interpolation: `mu_axis` is UNIFORM; `s2_axis` is LOG-SPACED (non-uniform) — use
  `torch.searchsorted(s2_axis, sigma2_g)` for the bracket + fractional weight; bilinear-combine
  the 4 corner `U` values. Gradient flows through the fractional weights (correct bilinear grad).
- Out-of-domain policy — **mirror `lut_select.py:LUTUtility` EXACTLY**:
  - clamp `mu_g` to `[mu_axis.min(), mu_axis.max()]` (= [-6, 6]);
  - `sigma2_g > s2_axis.max()` (= 6): continuity-corrected high-rate fallback
    `U = lut_U(mu_g, 6) + [h(mu_g, sigma2_g) - h(mu_g, 6)]`, where
    `h(mu, s2) = 0.5*log1p(exp(mu)*s2)` (torch `log1p`). Monotone, continuous at the seam,
    differentiable.
  - Never returns NaN/Inf (the whole point: a finite, monotone utility everywhere).
- float32 is fine (experiment is float32; LUT built float64; interp error negligible).

### 3 — Validate vs the vendored numpy reference (correctness gate)
- `lut/test_lut_torch.py`: import the VENDORED `lut/lut_select.py:LUTUtility` (numpy/scipy).
  Generate random `(mu_g, sigma2_g)`: in-domain (mu in [-6,6], s2 in [0,6]) AND out-of-domain
  (s2 in [6, 50] to exercise the fallback). Assert torch `lut_U` matches numpy `LUTUtility`:
  - in-domain max abs error <= 1e-5 (both linear; expect ~1e-6),
  - out-of-domain max abs error <= 1e-5.
- Gradient check: `torch.autograd.gradcheck` (float64) on a handful of points, or
  finite-difference vs autograd to ~1e-4.
- PRINT the errors + PASS/FAIL. Record numbers for the report.

### 4 — Standard-utility-via-LUT wrapper
- `lut/lut_utility.py` (or sibling): `lut_standard_utility(model, likelihood, x_candidates)`
  returning a dict shaped like `acquisition.standard_utility`'s output (`{'utility': (N,),
  'mu_g': (N,)}`):
  - moments via the engine helper `acquisition.get_gp_marginal_moments(model, x_candidates)`
    (IMPORT + CALL only — do NOT edit acquisition.py) -> `lambda_m, lambda_v`;
  - `mu_g = A*lambda_m + lambda0`, `sigma2_g = A^2*lambda_v` (A, lambda0 from `likelihood`);
  - `utility = lut_U(mu_g, sigma2_g)`.
- Differentiable w.r.t. x_candidates (moment helper is; `lut_U` is).

### 5 — Wire into our optimize_image
- Edit `optimize_image.py` (OURS — allowed): add `utility_mode='standard_lut'` branch that
  calls `lut_standard_utility` instead of `acquisition.standard_utility`. Keep 'da' / 'standard'.

### 6 — Re-run + 3-way comparison
- Reuse the standard-panels pipeline with the LUT backend (add `--backend lut` to
  `standard_utility/standard_panels.py`, or a sibling `lut_panels.py`). The pool-ranking
  (`pool_utilities_standard`) must use the SAME LUT utility so start = best-available under
  the utility being optimized. Output images to a gitignored folder (`*.png` is ignored).
- Build a Laplace-standard vs LUT-standard vs DA comparison for at least **cell 13** (worst
  Laplace blow-ups: U=295,800 nats, firing 26,605 at n=275; U=28,640 / fr 3,115 at n=50) and
  **cell 3** (the face-vs-foliage content difference).
- Expected: LUT standard-utility values are FINITE and monotone (no blow-ups); firing finite.
  The optimized IMAGES may still be high-contrast — the LUT fixes the NUMERICS, not the
  utility's intrinsic preference (see `FINDINGS.md`). That is correct, not a bug.

## Acceptance criteria (the report must demonstrate each)
1. `git diff --cached --name-only | grep -v 'investigations/lucent_useful_images/'` EMPTY at
   every commit (zero engine edits).
2. `lut/lut.npz` is committed (`git ls-files lut/lut.npz` lists it).
3. torch `lut_U` vs vendored numpy `LUTUtility`: <=1e-5 in-domain AND out-of-domain; gradcheck OK.
4. LUT standard-utility re-run: NO blow-ups (finite utility, e.g. order <=10 nats; finite
   firing) on the cells where Laplace-standard hit 295,800 nats / firing 26,605.
5. No runtime/test path references the superrepo `analysis/` (all reads from vendored `lut/`).

## Report (REQUIRED — for the originating session to verify correctness)
Write `standard_utility/LUT_IMPLEMENTATION_REPORT.md` and end your final message summarizing:
- Files created/edited (paths).
- Validation numbers (in/out-of-domain max error vs vendored `LUTUtility`; gradcheck result).
- Before/after blow-up: per-cell Laplace-standard (295,800 nats / fr 26,605 etc.) -> LUT-standard
  finite numbers (give actuals).
- One-line confirmation NO engine file touched + the `git diff --cached` check output.
- Confirmation `lut/lut.npz` committed.
- Deviations from this plan + why; anything surprising; open questions.

## Tidiness / commits (HARD — shared worktree)
- ~10 sessions share this worktree; OTHER uncommitted WIP exists (`SESSION_LOG.md`,
  `investigations/utility/*`). **NEVER** `git add -A` / `git add .` / `git commit -am`. Stage
  EXPLICIT paths; confirm `git diff --cached --name-only` before each commit.
- Logical commits, clear messages, e.g.:
  1. `lut: vendor lut.npz + numpy reference + provenance (standalone; un-ignore lut.npz)`
  2. `lut: torch-differentiable interpolator + validation vs vendored LUTUtility`
  3. `lut: standard-utility-via-LUT wrapper + optimize_image utility_mode=standard_lut`
  4. `lut: re-run standard panels (LUT backend) + 3-way comparison + report`
- `nvidia-smi` before GPU runs (shared GPU; run light if busy).
