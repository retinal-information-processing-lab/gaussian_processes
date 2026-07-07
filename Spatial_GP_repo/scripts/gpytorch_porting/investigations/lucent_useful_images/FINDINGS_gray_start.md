# FINDINGS: gray-start crystallization on the testbed cells (3, 13, 36)

**Date**: 2026-07-04 · **Branch**: `pietro/lucent-useful-images` (gaussian_processes submodule)
**Engine**: pinned `75b207a`, `default_gpy`, M = n_train, seed 42, DA utility, **sample_lambda=False** (no engine edits).
**Scripts** (all in this folder): `useful_image_panels.py` (natural-start), `gray_panels.py` (gray-start),
`diff_panels.py` (difference view), `rf_localization.py` (concentration metric), `firing_diagnostic.py` (A / firing).
**Figures** (gitignored, regenerable): `out/panels/cell{3,13,36}_{panels,diff,gray}.png`, `out/panels/rf_localization.png`.

## Question

Does the synthesized most-useful image sharpen from a diffuse probe into a compact
receptive-field (RF) stimulus as the `default_gpy` model improves with training-set size
(n_train)? Tested on the three screened testbed cells (`cell_screening/FINDINGS.md`):
cell 3 (test_r 0.18→0.83, gradual), cell 13 (0.37→0.96, jump), cell 36 (0.31→0.73, weak).

## Findings

### 1. The narrative holds — but only the GRAY-START view shows it cleanly. CONFIRMED
Started from flat gray (same start every column), all three cells crystallize from a
diffuse/broad change into a compact center-surround RF as test_r climbs. The
crystallization tracks each cell's test_r jump:
- cell 13: sharpens at **n=75** (test_r 0.37→0.89), crisp dark-center RF thereafter.
- cell 3: broad bright blob → compact dark-center RF at **n=100** (test_r 0.19→0.74).
- cell 36: broad blob → softer compact RF at **n=150** (test_r 0.30→0.60); weakest (0.73 ceiling).

### 2. The natural-start difference view is CONFOUNDED; do not trust its concentration. CONFIRMED (correction)
The natural-start panels (`useful_image_panels.py`) start from the best pool image and
only retune the RF, so the change is buried in a busy scene. Worse, the change-energy
**concentration** metric computed on `final - natural_start` is confounded by the scene:
it *falls* for cell 3 (0.34→0.17) even as the RF sharpens. The gray-start concentration
(`final - gray`, flat reference) instead **rises** with test_r for all three — the correct
signal. Concentration = fraction of change-energy (Σ diff²) within 15 px of the peak;
uniform-ripple baseline = 0.06.

| cell | conc @ low n_train | conc @ high n_train (gray) | contrast (gray) low→high |
|---|---|---|---|
| 3  | 0.30 (n=50) | ~0.51 (n≥100) | 5 → 14–19 |
| 13 | 0.45 (n=50) | 0.64–0.71 (n=75–100) | 12 → 41–47 |
| 36 | 0.30 (n=50) | ~0.48–0.54 (n≥150) | 5 → 9–14 |

Contrast = max|diff| / median|diff| (RF peak above the diffuse floor).

### 3. "Only the RF changes" is too strong under the lucent FFT parameterization. CONFIRMED
lucent optimizes global Fourier coefficients, so every optimized image carries a diffuse
whole-frame ripple (43–71% of pixels exceed a 3%-of-range change threshold; a bounding-box
metric saturates to 108×108). The RF is where the largest *structured, high-contrast*
change concentrates, but the background is not untouched. (The README's "non-RF pixels get
no gradient" is true in pixel-gradient space but not in the FFT parameter space lucent
actually optimizes.) This is why the disk-concentration + contrast metrics are used instead
of pixel-count / bbox.

### 4. Two regimes, one mechanism — explained by the response gain A. CONFIRMED
The optimized RFs of cells 3 & 36 predict **~0 response**; cell 13's predict moderate
firing. `firing_diagnostic.py --n-train 300` shows this is a **cell property**, not an
optimizer failure:

| cell | test_r | A (gain) | λ₀ | pool firing min / median / **max** |
|---|---|---|---|---|
| 3  | 0.83 | **0.0868** | −0.969 | 0.08 / 0.33 / **1.55** |
| 13 | 0.96 | **0.2391** | −0.347 | 0.33 / 1.56 / **10.61** |
| 36 | 0.73 | **0.0866** | −1.040 | 0.33 / 0.38 / **2.09** |

Cells 3 & 36 have a ~3× smaller gain A (~0.087): the *entire* image pool drives them to at
most ~1.5–2 spikes, so no bounded image can make them fire hard. Their utility optimum is
therefore an **epistemic probe** (the model's most-uncertain RF direction), which the cell
barely responds to. Cell 13 (A ~0.24) responds up to ~10, so its optimum is a genuine
high-response stimulus. **The image crystallizes into a compact RF in both regimes; what
differs is whether that RF is a "make-it-fire" or a "reduce-my-uncertainty" stimulus.**

### 5. Bounding holds everywhere. CONFIRMED
Every optimized image (natural- and gray-start, all cells, all n_train) stays within the
physical range [GP_MIN, GP_MAX] = [−2.4013, 2.4780]; no OOB flagged. The lucent
sigmoid+affine bound is intact.

## Caveats / open

- **sample_lambda = False (biased).** Every result here uses mean-λ conditioning (Jensen
  bias, `why_sample_lambda.tex`). The unbiased `sample_lambda=True` has NOT been run — it is
  the next step, and adds MC noise that must be quantified (several realizations).
- **Single seed (42).** Cell 13 shows one-seed test_r dips at n=175 (0.77) and n=275 (0.78);
  the concentration/contrast at those points are correspondingly noisy.
- **Cell 36's weak ceiling (0.73)** makes its RF the least crisp; `cell 33` is the standby swap.

## Reproduce

```bash
PY=/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python
cd .../investigations/lucent_useful_images
$PY useful_image_panels.py --cells 3 13 36   # natural-start ladder (cells 3,13 cached; ~9 min for 36)
$PY diff_panels.py         --cells 3 13 36   # natural-start difference view (from cache, instant)
$PY rf_localization.py     --cells 3 13 36   # concentration vs test_r (natural cache, instant)
$PY gray_panels.py         --cells 3 13 36   # gray-start ladder (~26 min; the clean crystallization)
$PY firing_diagnostic.py   --cells 3 13 36 --n-train 300   # A / lambda0 / pool firing (~1 min)
```

Caches: `cache/panels_results.pkl` (natural), `cache/gray_panels_results.pkl` (gray; stores
per-(cell,n_train) final image, gray reference, σ²_g, concentration, contrast, and the full
per-step optimization trajectory). Both gitignored/regenerable. Committed = the scripts +
this doc.

## sample_lambda robustness (unbiased DA)

Re-ran the gray-start ladder with `sample_lambda=True` (unbiased: draw λ at each conditioning
image instead of using its posterior mean) and compared to the biased mean-λ result.
Scripts: `gray_panels.py --sample-lambda` (→ `gray_panels_sl_results.pkl`, `cell{N}_gray_sl.png`),
`compare_sl_panels.py` (Fig A: False/True/diff ladder + `sl_divergence.png`), `noise_probe.py`
(Fig B: seed spread vs n_mc at n_train=300 → `cell{N}_sl_noise.png`, `sl_noise_summary.png`).

- **Crystallization is robust.** All three cells still crystallize diffuse→compact RF and the
  concentration-vs-test_r story holds. True ≈ False *in the mean*; at most n_train the two
  optimized images are within 2–6% RMS.
- **A single unbiased realization is noisier.** Material divergence (RMS up to 0.29, RF polarity
  flips) at ~7 of 33 ladder points, concentrated on the low-gain cells 3 & 36 (flat utility
  landscape). Cell 13 stays ≤0.14.
- **n_mc=48 is cell-dependent, not blanket-too-small** (probe at n=300, seeds × n_mc {48,96,192}):
  cell 36 cross-seed spread 0.14 → 0.02 raising n_mc 48→96 (48 too small); cell 3 already stable
  (~0.018, flat in n_mc); cell 13 ~0.10 and does NOT shrink with n_mc (structural multi-optimum —
  the sharp RF has near-equivalent placements). The **mean over 3 realizations** is a clean RF
  matching the False image in every case.
- **Takeaway.** The biased mean-λ (`sample_lambda=False`) used elsewhere in this doc is a fine
  *deterministic proxy* (it matches the unbiased mean). If using the unbiased utility, n_mc≥96 is
  a cheap default and averaging a few realizations tames the residual. Caveat: the probe covered
  n_train=300 only (confident regime, best case); mid-ladder single realizations are noisier.
