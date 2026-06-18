# Handoff: LUT-backed standard utility for the lucent investigation

**Branch**: `pietro/lucent-useful-images` (gaussian_processes submodule)
**Date**: 2026-06-17
**Status**: DONE (2026-06-18) — implemented + independently verified (see `LUT_IMPLEMENTATION_REPORT.md`). Kept as the historical record of decisions/rationale.
**Plan file**: `standard_utility/PLAN_lut_standard_utility.md` (in this same folder — MANDATORY read)

---

## Motivation

The lucent investigation synthesizes the "most-useful image" for a trained GP cell by
gradient-ascending an acquisition utility under lucent's Fourier+sigmoid parameterization
(bounded, smooth images by construction). As a control for the distribution-aware (DA)
utility, we re-ran the same panels with the **standard utility** `U = H_marg - E[H_noise]`.
It **blew up** — utility to 295,800 nats, predicted firing to 26,605 spikes — at the
badly-fit n_train points.

That blow-up is NOT the utility's real behavior: it is the documented `r_max=100` numerical
trap. The live standard utility (`acquisition.standard_utility` -> `utils.nd_utility_new`)
sums the Poisson count to a fixed `r_max=100` and the `log r!` cancellation fails past the
"~4 sigma gate", producing tens of thousands of nats (see
`analysis/figures/utility_landscape/LUT_README.md`). A precomputed, exact, numerically-stable
**lookup table** of this exact utility already exists for the offline refits. The task is to
make a **torch-differentiable** version of it usable inside the lucent optimization, so the
standard-vs-DA comparison is fair (numerics removed as a confound) — and as a stepping stone
to eventually fixing the same blow-up in the live experiment.

## Decisions and Rationale

- **Use the existing LUT, don't re-derive entropy.** The table is the ground-truth utility
  (exact quadrature, no Laplace, no truncation). `lut_select.py:LUTUtility` is explicitly
  documented as "the UNIVERSAL selection-utility mechanism... reused across August (varGP),
  April (gpytorch-port), and future online experiments." We are completing its intended use,
  not inventing a parallel mechanism. Rejected: re-implementing a stable entropy in torch
  from scratch (pointless — the LUT is the validated artifact).

- **Torch-differentiable twin is the only real new code.** The lucent optimization needs
  gradients of the utility w.r.t. the image. The shipped `LUTUtility` is numpy/scipy
  (`RegularGridInterpolator`) — fine for argmax selection / offline refits, NOT
  differentiable. So we port the interpolation + the documented out-of-domain policy to
  torch (~40-60 lines). Rejected: `scipy` in the optimization loop (no gradients); rejected:
  `torch.nn.functional.grid_sample` (assumes a uniform grid; the sigma2 axis is log-spaced —
  hand-rolled non-uniform bilinear is simpler and exact).

- **Stay on `pietro/lucent-useful-images`; do it ADDITIVELY; do NOT touch the engine.**
  The whole reason this investigation is on this branch (based off 75b207a) is that 75b207a
  is the commit the SUPERREPO PINS for the submodule, and the `analysis/` pipeline runs from
  this same checkout and imports the engine files. We confirmed the engine here is
  byte-identical to the pin (every change is inside `investigations/lucent_useful_images/`).
  Editing `acquisition.py`/`utils.py` would (a) break that byte-identity, and (b) because the
  analysis pipeline imports the live files (not the pinned commit's copy — same folder on
  disk), it would silently change what the analysis runs. So: the LUT-for-lucent work is
  done as NEW files in our folder, calling the engine read-only (`get_gp_marginal_moments`).
  Rejected: a `backend='lut'` switch inside `acquisition.standard_utility` — that edits the
  engine; it is the PROMOTION step and belongs on a separate future branch off
  `pietro/workingbranch`, explicitly OUT OF SCOPE here.

- **Vendor the artifact -> the repo must be standalone.** The LUT lives in the superrepo
  (`analysis/figures/utility_landscape/`). The submodule must not depend on the parent's
  layout (it breaks a standalone checkout and blocks promotion). So we copy `lut.npz` (84 KB,
  a stable artifact) + the numpy reference into our `lut/` folder with a provenance note.
  Rejected: a path reference up into `analysis/` (fragile, non-standalone). Also vendor
  `lut_select.py` so even the validation TEST is standalone (the test imports the vendored
  numpy reference, not the superrepo).

- **Standard utility only; DA later.** The table stores `U` (standard) and `H_marg`
  separately. DA needs `H_marg` at marginal vs conditional moments — doable with the same
  module, BUT the documented `sigma2>6` fallback is the asymptote of `U`, not of `H_marg`, so
  DA-via-LUT needs a separate `H_marg` fallback. The user scoped this to standard utility.

## Critical Subtleties

- **The `*.npz` gitignore will silently drop `lut.npz`.** The folder `.gitignore` ignores
  `*.npz` (and `*.png`/`*.svg`/`*.pdf` — images are not committed). If you `git add lut/` the
  table will NOT be staged and the repo is no longer standalone. You MUST un-ignore it
  (`!lut/lut.npz`) or `git add -f`. Confirm with `git status` that it is actually staged.
  Symptom of getting this wrong: a "standalone" repo that imports a missing file on a fresh
  clone.

- **Same folder on disk = the analysis pipeline sees your edits.** The submodule pin protects
  OTHER clones, not THIS working copy. If you edit an engine file here, any analysis run from
  this `ClosedLoop-standalone-analysis_april26/` checkout imports your edited file. This is
  exactly why the `git diff --cached | grep -v investigations/lucent_useful_images` check is
  non-negotiable before each commit. Symptom: someone's analysis numbers shift and nobody
  knows why.

- **The LUT fixes the numerics, not the utility's taste.** Do not expect the LUT to make the
  standard utility produce "nice" images. It removes the 296k-nat explosion (finite, monotone
  utility) but the standard utility still prefers high-firing / high-contrast images — that's
  its real behavior and the reason DA is the better objective (`FINDINGS.md`). If the LUT
  images are still contrasty, that is correct.

- **Log-spaced sigma2 axis.** `grid_sample` won't work directly; use `searchsorted` on
  `s2_axis` for the bracket. Validate against the vendored numpy `LUTUtility` — that catches
  any axis/interpolation mistake immediately.

- **Shared worktree has other sessions' WIP.** `git status` shows `SESSION_LOG.md` and
  `investigations/utility/*` dirty — NOT yours. Explicit staging only.

## Uncommitted Changes (at handoff time, on this branch)

The lucent investigation (incl. `standard_utility/`) is COMMITTED and clean. The only dirty
items in the worktree are NOT part of this task and belong to other sessions / are pre-existing:
- ` M Spatial_GP_repo/.../SESSION_LOG.md` (a log edit; will get a LUT entry — see below)
- ` M Spatial_GP_repo/.../investigations/utility/docs/da_utility_theory.md` (another session)
- `?? investigations/utility/docs/{LAPLACE_VALIDITY_SUMMARY.md, laplace_*, sigma_cap.*}` etc.
  (another session's laplace investigation — DO NOT TOUCH/STAGE)

Do not stage any of the above. Start clean: your first `git add` is `lut/` files only.

## Files to Read First

1. `standard_utility/PLAN_lut_standard_utility.md` — the step-by-step plan + acceptance criteria + required report.
2. `standard_utility/FINDINGS.md` — what the standard-utility run found, the LUT connection, the blow-up numbers, "make the comparison clean" next steps. This is effectively the spec.
3. `analysis/figures/utility_landscape/LUT_README.md` — what the LUT is, the three numerical traps, the ~4-sigma gate. (Read in place; then vendor.)
4. `analysis/figures/utility_landscape/lut_select.py` — `LUTUtility`: the out-of-domain policy (clamp mu, high-rate fallback `h=0.5*log1p(e^mu*s2)`) you must mirror in torch. Also `lut_interp.py`, `lut_meta.json`.
5. `investigations/lucent_useful_images/optimize_image.py` — the optimizer; see existing `utility_mode='da'|'standard'` branches (add `'standard_lut'`). Note `acquisition.get_gp_marginal_moments` usage (call read-only).
6. `investigations/lucent_useful_images/standard_utility/standard_panels.py` + `compare_da_vs_std.py` — the pipeline to reuse for the LUT re-run.
7. `investigations/lucent_useful_images/README.md` — the parent investigation (M=n_train convention; sample_lambda=False methods note; folder map).

Env: use `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python` (the Bash tool's
default base python has no torch). GPU shared (RTX 4090) — `nvidia-smi` first. Data path is
in `gp_models.py` (points at the ClosedLoopProject sibling repo — pre-existing, not your concern).

## Caveats and Open Questions

- **Out-of-domain fallback is a RANKING proxy** past sigma2=6 ("absolute value not trustworthy"
  per `lut_select.py`). For our GRADIENT use it gives a well-defined, monotone gradient (no
  blow-up) — good — but the absolute utility past sigma2=6 is not a true entropy. The
  optimization can still drift to high sigma2 (the fallback is monotone-increasing), so the
  standard-utility images may still be extreme; that's expected (see the subtlety above).
- **DA-via-LUT is explicitly deferred** (needs an H_marg-specific fallback). Don't attempt it.
- **Promotion to `acquisition.py` is deferred** to a separate branch off `pietro/workingbranch`
  (the engine is 106 commits diverged from our base — that's a later, careful task). Don't do
  it here.
- **No standalone-fix for the dataset path** is in scope. `gp_models.py` still hardcodes the
  PNAS dataset into the ClosedLoopProject sibling repo — a separate pre-existing
  non-standalone reference, flagged but out of scope for this task.

---

## Continuation Prompt

```
Implement the LUT-backed standard utility for the lucent investigation. You are on the
gaussian_processes submodule branch pietro/lucent-useful-images — confirm with
`git branch --show-current` and `git status` before starting.

READ FIRST (in this folder, under investigations/lucent_useful_images/standard_utility/):
  1. PLAN_lut_standard_utility.md   (the step-by-step plan, acceptance criteria, required report)
  2. HANDOFF_lut_standard_utility.md (the WHY, decisions, and the gotchas)
  3. FINDINGS.md                    (what the standard-utility run found + the LUT connection)
Then analysis/figures/utility_landscape/LUT_README.md and lut_select.py (the reference to mirror).

HARD CONSTRAINTS:
  - Commit on pietro/lucent-useful-images. ADDITIVE files under
    investigations/lucent_useful_images/ ONLY. NEVER edit a GP engine file
    (acquisition.py/utils.py/utility.py/kernels/eigenspace/gpy_*/run_single_mode). Verify
    `git diff --cached --name-only | grep -v 'investigations/lucent_useful_images/'` is EMPTY
    before every commit.
  - Standalone: vendor lut.npz + the numpy reference into a new lut/ folder; no runtime/test
    path may reference the superrepo analysis/ folder. The folder .gitignore has *.npz — you
    MUST un-ignore lut.npz (!lut/lut.npz) or git add -f, and confirm it is staged.
  - Standard utility ONLY (DA deferred). No warm-start. No promotion into acquisition.py.
  - Shared worktree: other sessions' WIP is dirty (SESSION_LOG.md, investigations/utility/*) —
    NEVER git add -A; stage explicit paths; tidy, logical commits with clear messages.
  - Env python: /home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python. nvidia-smi
    before GPU runs.

DELIVERABLE: implement steps 1-6 of the plan, then write
standard_utility/LUT_IMPLEMENTATION_REPORT.md AND end your final message with a report (per
the plan's "Report" section) so the originating session can verify: validation errors vs the
numpy LUTUtility, the before/after on the 295,800-nat / firing-26,605 blow-ups, a one-line
confirmation that no engine file was touched (with the git check output), and confirmation
that lut/lut.npz is committed.
```
