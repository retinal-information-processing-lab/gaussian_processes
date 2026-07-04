# Investigation: lucent "most-useful image" generation (continuation)

**Branch**: `pietro/lucent-useful-images` (gaussian_processes submodule)
**Date**: 2026-06-18
**Status**: CONTINUING (new session) — exploratory, with the testbed cells now chosen
**Location**: `investigations/lucent_useful_images/`

This is the top-level "start here" handoff for the whole lucent investigation. Sub-tasks have
their own handoffs (`standard_utility/HANDOFF_lut_standard_utility.md`,
`cell_screening/HANDOFF_cell_screening.md`) — those are DONE; this one is the live thread.

---

## Problem Statement

Synthesize the **most-useful image** to show a retinal ganglion cell — the image that maximizes
the GP's information-gain utility (the active-learning acquisition) — and get **sensible,
physically-bounded** images, not the contrast/border blow-ups every previous attempt produced.
The narrative we want to demonstrate: *as the GP model improves with more training data
(n_train), the synthesized optimal image sharpens from a diffuse probe into a well-defined
receptive-field (RF) stimulus.* Dataset: PNAS (Goldin et al.), cells indexed 0–40, `default_gpy`
GP (the distribution-aware utility requires it). This is the **exploratory, method-development
phase**; the eventual target (if the method graduates) is the real closed-loop data.

## What Was Tried (the journey, with verdicts)

### lucent Fourier+sigmoid parameterization — ADOPTED
- **What**: parameterize the image with lucent's `fft_image` (Fourier `1/f^decay_power` spectral
  prior) + sigmoid (`to_valid_rgb`), affine-mapped from (0,1) to the dataset's physical range
  `[GP_MIN, GP_MAX] = [-2.4013, 2.4780]`.
- **Result**: images bounded BY CONSTRUCTION (sigmoid can't exceed the rail; the 1/f prior
  biases toward smooth, natural-spectrum images). Across every run, saturation ≤ 7% (mostly
  < 2%), zero overblow.
- **Verdict**: this is the fix for the long-standing overblow problem. lucent is imported via
  `sys.path` from a gitignored clone (`lucent_repo/`) — NOT pip-installed (its `kornia<=0.4.1`
  pin would damage the shared env).

### DA (distribution-aware) utility on `default_gpy` — the target objective
- **What**: `acquisition.distribution_aware_utility` (needs `default_gpy`'s covariance).
  `U = H_marg − E_x[H_cond]`; Monte-Carlo over 48 conditioning images from the pool.
- **Result**: natural-ish bounded images. Started from the best-available natural image, the
  result IS that natural image with only the RF region retuned (non-RF pixels get no gradient —
  the kernel's C-matrix only "sees" RF pixels). Firing stays physical (~10–50).
- **Verdict**: the right objective — but the naturalness is NOT from the utility (Finding 2).

### Start point: gray vs natural-image init — natural-start ADOPTED
- gray-start (near-gray Fourier init) isolates the RF structure on a neutral background;
  natural-start (init the FFT params from a chosen natural image's spectrum, via
  `lucent_param.fft_params_from_image01`) keeps the whole image natural and only retunes the RF.
- **Verdict**: natural-start for the "looks natural" deliverable, gray-start/difference maps for
  the "what is the RF" science. They converge to the same RF optimum at high n_train (confident
  model) and differ at low n_train.

### decay_power (spectral prior strength) — 1.0 ADOPTED
- `decay_power=1.0` (natural 1/f) with `sd=0.01` gives a near-gray start + smooth gradients.
  `decay_power ≥ 2` EXPLODES the low-frequency amplitude (scale ~ 108^p), saturating the sigmoid
  into a binary blob BEFORE optimization (vanishing gradients). Higher p needs a proportionally
  smaller `sd`. **Verdict**: keep 1.0; the sd–decay_power coupling is a knob to revisit if more
  spectral control is wanted.

### Standard utility (control) — confirmed DA is essential, then numerically fixed
- `acquisition.standard_utility` (`H_marg − E[H_noise]`, no conditioning) under the same lucent
  param **BLEW UP** (utility to 295,800 nats, firing to 26,605) at badly-fit n_train points — the
  documented `r_max=100` trap — and picks different/higher-contrast images (cell 3 @ n≥200: a
  high-contrast FACE vs DA's foliage).
- **LUT fix (DONE)**: vendored the precomputed utility LUT into `lut/`, made a torch-differentiable
  twin (`lut/lut_utility.py`), wired as `optimize_image` `utility_mode='standard_lut'`. Validated
  to machine precision vs the numpy reference. Blow-up gone (cell 13 max 295,800 → **3.724 nats**);
  standalone. See `standard_utility/LUT_IMPLEMENTATION_REPORT.md`.
- **Verdict**: the standard utility is a noisy/extreme baseline, not a candidate method — but it
  proved the DA conditioning is load-bearing (lucent bounds pixels; DA keeps firing/image sensible).

### Cell screening — DONE, testbed cells chosen
- Screened 34 PNAS cells (all 41 minus the 6 STA-edge cells 0,5,6,15,22,39 minus the
  default_gpy-hard cell 10) across n_train = M = {50,100,150,200,250,300}, seed 42.
- **Honest finding**: NO PNAS cell shows a smooth ~0.3→~0.9 gradual climb. Testbed: **cell 3**
  (0.18→0.83, gradual workhorse), **cell 13** (0.37→0.96, high endpoint via one jump at n=100),
  **cell 36** (0.31→0.73, most gradual but weak endpoint). **Cell 33** (−0.02→0.97) on standby.
- See `cell_screening/FINDINGS.md`. **These are the cells the new session should use.**

## Key Findings

1. **CONFIRMED** — lucent's sigmoid+affine bounds pixels by construction + the 1/f prior smooths;
   no overblow in any run (sat ≤ 7%). (`README.md` "Why previous attempts failed".)
2. **CONFIRMED** — the utility INTRINSICALLY favors UNnatural images (high firing AND high
   epistemic uncertainty = far from training data). **Naturalness comes from the
   parameterization + the natural-image start, NOT the objective.** Proven by the standard-utility
   control (standard+lucent still picks extreme images). The DA conditioning's ρ² term adds
   angular alignment; with the natural start it keeps results sensible. (`standard_utility/FINDINGS.md`;
   `../utility/docs/da_utility_theory.md`.)
3. **CONFIRMED** — DA utility needs `default_gpy` (full covariance). `standard_utility` works with
   both modes. `vargp_direct` DA is the deferred augmented-matrix port.
4. **CONFIRMED** — M = n_train (full inducing set) is the convention. At M=n_train, cell 13
   saturates faster (jump by n=100) than at the old fixed M=50 (gradual 0.37→0.73→0.94).
5. **CONFIRMED** — the standard utility's blow-up is the documented `r_max=100` numerical trap;
   the LUT is the stable fix, now torch-differentiable (`lut/`).
6. **CONFIRMED** — `default_gpy` fits are noisy at low n_train; PNAS has no ideal gradual-high
   cell. Testbed = 3, 13, 36 (compromises in `cell_screening/FINDINGS.md`).
7. **HYPOTHESIS** — the cleanest "model improves → image sharpens" demo is ultimately on the
   closed-loop data (clean warm-start active/random trajectories), not PNAS+default_gpy. (Deferred.)
8. **CONFIRMED** — ALL DA runs so far used `sample_lambda=False` (biased per `why_sample_lambda.tex`);
   the unbiased `sample_lambda=True` has NOT been run.

## Why This Was Stopped

Context handoff to a fresh session — NOT a dead end. The cell-screening result (testbed cells
3, 13, 36) is the natural restart point: re-run the lucent image optimization on these curated
cells and look at the "image sharpens with model quality" story on clean cells, then explore the
open knobs. The investigation is healthy and mid-flow.

## Things Noticed But Not Acted Upon

1. **`sample_lambda=True`** (unbiased DA) — flagged by the user (the `why_sample_lambda.tex`
   Jensen-bias argument), never run. Adds MC noise to quantify.
2. **DA-noise quantification** — run several MC realizations (different 48-image conditioning
   subsets + λ draws) and compare optimized images / ΔU spread. Flagged, not done.
3. **No objective "is this the cell's preferred stimulus" metric** — could correlate the
   optimized RF region with the cell's STA. The repo has STA tools and ground-truth RF centers
   (`datasets/rf_centers_ground_truth.npz`, see `.claude/CLAUDE.md`) — this would turn "it looks
   like an RF" into a number.
4. **No firing / Laplace-validity guard** in the optimizer (we monitor `z_safe` / σ² but never
   reject a step). DA stayed valid; `standard_lut` is bounded by the LUT.
5. **Faint "satellite spots"** in gray-start images = the unoptimized near-gray FFT-init texture
   in non-RF pixels, not RF structure (the natural/difference figures avoid it).
6. **cell 36's weak endpoint** (test_r 0.73) — its n=300 optimal image may still be diffuse;
   `cell 33` is the ready swap.
7. **Dataset path** in `gp_models.py` points at the sibling `ClosedLoopProject` repo (pre-existing
   non-standalone reference, out of scope).

## Uncommitted Changes

`SESSION_LOG.md` (the gpytorch_porting one) — my pending log entries (cell-screening handoff
pointer + LUT-done annotation). Everything else is committed (228418a, 91d8e14, c880e37, …).
Commit it (and this HANDOFF) before the new session if you want them on the branch — explicit
staging only (shared worktree; `investigations/utility/*` is another session's dirty WIP, do not
stage it).

## Files Created (the map a new session needs)

| File / folder | Purpose |
|---|---|
| `README.md` | The full investigation writeup: conventions (M=n_train; the `sample_lambda=False` methods note; figure conventions), "Why previous attempts failed", the file map. READ EARLY. |
| `lucent_param.py` | lucent grayscale Fourier param + sigmoid; affine to GP space; `fft_params_from_image01` (natural-image init). |
| `gp_models.py` | `train_default_gpy(cell, n_train, M=None→n_train, seed=42)` → `(model, lik, idx_train, test_r)`; `load_pool()`. |
| `optimize_image.py` | **Core**: the optimization loop + 8-panel diagnostic; `utility_mode='da' | 'standard' | 'standard_lut'`. |
| `useful_image_panels.py` | The per-cell panels: best-available start → DA-optimized, 3-row × n_train. The main pipeline to re-run on 3/13/36. |
| `make_figure.py`, `run_grid.py` | Grid runners + figure builders (the earlier 5-cell grids). |
| `lut/` | Vendored utility LUT + the torch-differentiable twin (`lut_utility.py`, `lut_standard_utility`) + `PROVENANCE.md`. |
| `standard_utility/` | The DA-vs-standard control, the LUT report (`LUT_IMPLEMENTATION_REPORT.md`), `FINDINGS.md`. |
| `cell_screening/` | The testbed screen (`screen_testbed.py`) + `FINDINGS.md` (cells 3, 13, 36). |
| `screen_cells.py`, `screen_ladder.py` | Cell-screening tools (M=n_train, seed 42). |
| `docs/lucent_explainer.md` | Plain-language "how lucent works". |
| `early_exploration_fixed_M50/` | Archived early M=50 figures (superseded). |
| `out/`, `cache/`, `refined/`, `lucent_repo/` | gitignored (figures regenerable; clone; caches). |

## If Someone Revisits This

**WHAT TO DO NEXT (most promising first):**
1. **Re-run the lucent DA-utility optimization on the testbed cells 3, 13, 36** across the
   n_train ladder (`useful_image_panels.py` or a sibling), natural-start, and LOOK: does the
   optimized image sharpen as test_r climbs? Expect: cells 3 and 36 show *progressive* sharpening
   (model improves at each step); cell 13 shows diffuse(n=50)→crisp(n≥100) then near-identical
   columns. This is the headline check on clean cells.
2. **DA vs `standard_lut` side-by-side** on these clean cells (now that standard is numerically
   fixed) — does DA still give the more sensible image when the model is well-fit?
3. **Add an STA-correlation metric** (Thing-Noticed 3) to objectively score "is this the RF".
4. **`sample_lambda=True`** (unbiased DA) + the DA-noise quantification (Things 1, 2).
5. **decay_power sensitivity** on a clean cell (with the sd coupling in mind).

**WHAT NOT TO TRY (dead ends / footguns):**
- Do NOT edit any GP engine file (`acquisition.py`, `utils.py`, `utility.py`, kernels, eigenspace,
  `gpy_*`, `run_single_mode.py`) — it breaks the byte-identity to the superrepo-pinned `75b207a`
  and silently changes what the analysis pipeline (run from this same checkout) imports.
- Do NOT `pip install` lucent (kornia pin) — use the `sys.path` clone.
- Do NOT use the raw `standard_utility` (`r_max=100` blow-up) — use `standard_lut`.
- Do NOT conflate PNAS cells (0–40) with the closed-loop analysis cells (different dataset/engine).
- Do NOT chase a perfectly-gradual PNAS cell — it does not exist at this grid/engine (Finding 6).
- Do NOT rebase onto `pietro/workingbranch` to "fix" the fit noise — its A-prior is default-OFF +
  partial, the high-n decline was closed as a known limit; rebasing won't make broken cells smooth
  and would break the byte-identity to the pin (cell_screening/FINDINGS.md "Stale-engine confound").

**PREREQUISITE for the clean demo:** the closed-loop data (deferred — "if lucent graduates").

## What to read first (ordered)

1. This `HANDOFF.md`.
2. `README.md` (conventions, the "why previous attempts failed" framing, the file map).
3. `cell_screening/FINDINGS.md` (the testbed cells 3, 13, 36 + the honest no-ideal-cell finding).
4. `standard_utility/FINDINGS.md` + `standard_utility/LUT_IMPLEMENTATION_REPORT.md` (DA-vs-standard,
   and the LUT — the `standard_lut` backend).
5. `docs/lucent_explainer.md` (how lucent works, if unfamiliar).
6. Code: `optimize_image.py` (the loop + `utility_mode`), `useful_image_panels.py` (the pipeline),
   `lucent_param.py`, `gp_models.py`, `lut/lut_utility.py`.
7. Project memories: `lucent-useful-images`, `default-gpy-low-ntrain-unstable`.
8. Theory (only if going into the math): `../utility/docs/da_utility_theory.md`;
   `~/IDV_code/Papers/latex_summaries/{distribution_aware_utility_pietro,why_sample_lambda}.tex`.

Env: `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python` (the Bash default base python
has no torch). GPU shared (RTX 4090) — `nvidia-smi` first.

---

## Continuation Prompt

```
Continue the (exploratory) lucent "most-useful image" investigation, now using the chosen
testbed cells 3, 13, 36. Goal: demonstrate that as a GP cell's model improves with more training
data (n_train), the synthesized utility-maximizing image sharpens from a diffuse probe into a
well-defined RF stimulus — and explore the lucent implementation further. You are on the
gaussian_processes submodule branch pietro/lucent-useful-images; confirm with
`git branch --show-current` and `git status` before doing anything.

READ FIRST (investigations/lucent_useful_images/):
  1. HANDOFF.md                         (this investigation's full context — start here)
  2. README.md                          (conventions, file map, "why previous attempts failed")
  3. cell_screening/FINDINGS.md         (testbed cells 3, 13, 36; the honest no-ideal-cell finding)
  4. standard_utility/FINDINGS.md + standard_utility/LUT_IMPLEMENTATION_REPORT.md  (DA vs standard; the LUT)
  5. docs/lucent_explainer.md           (how lucent works)
  6. optimize_image.py, useful_image_panels.py, lucent_param.py, gp_models.py, lut/lut_utility.py
  Plus the memories lucent-useful-images and default-gpy-low-ntrain-unstable.

KEY CONTEXT (do not re-derive):
  - lucent Fourier(1/f, decay_power=1.0)+sigmoid bounds images by construction; map (0,1) ->
    dataset range [-2.4013, 2.4780]. Start from the best-available natural image.
  - The utility intrinsically prefers UNnatural images; naturalness comes from the param + the
    natural start, NOT the objective (the standard-utility control proved it).
  - DA utility (utility_mode='da') needs default_gpy; M = n_train; all DA runs so far used
    sample_lambda=False (biased, unbiased not yet run). standard_lut is the numerically-fixed
    standard utility (the raw 'standard' blows up via r_max=100 — do not use it).
  - PNAS has no perfectly-gradual cell: cell 3 = gradual workhorse (0.18->0.83), cell 13 = high
    endpoint via one jump (0.37->0.96), cell 36 = gradual but weak endpoint (0.31->0.73); cell 33
    on standby. The clean demo ultimately wants the closed-loop data (deferred).

EXPLORATORY NEXT STEPS (see HANDOFF "If Someone Revisits"): re-run the panels on 3/13/36 and LOOK;
DA vs standard_lut on clean cells; an STA-correlation "is-this-the-RF" metric; sample_lambda=True;
decay_power sensitivity. This is still exploratory — look before concluding; keep the user in the loop.

HARD CONSTRAINTS:
  - ADDITIVE files under investigations/lucent_useful_images/ ONLY. NEVER edit a GP engine file
    (acquisition/utils/utility/kernels/eigenspace/gpy_*/run_single_mode) — the submodule must stay
    byte-identical to the pinned 75b207a. Verify
    `git diff --cached --name-only | grep -v 'investigations/lucent_useful_images/'` is EMPTY before
    every commit.
  - Shared worktree: other sessions' WIP is dirty (investigations/utility/*) — NEVER git add -A;
    stage explicit paths; clean commit messages (repo style, no coauthor footer).
  - nvidia-smi before GPU runs; free GPU memory between fits; env python
    /home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python.
```
