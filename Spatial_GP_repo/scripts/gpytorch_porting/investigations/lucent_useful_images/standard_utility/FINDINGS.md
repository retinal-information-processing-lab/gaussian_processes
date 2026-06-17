# Standard utility — findings & handoff

For a new session: this folder is a **control** for the lucent image-optimization work in
the parent `lucent_useful_images/`. It re-runs the same per-cell panels with the STANDARD
utility instead of the distribution-aware (DA) one, to see whether the optimized images are
really different. Short answer: **yes, very** — and the standard utility hits a known
numerical landmine that a new session must understand before trusting its numbers.

## TL;DR
- Standard utility: `U = H_marg − E[H_noise]` — depends only on the marginal GP moments
  `(mu_g, sigma2_g)` at the candidate image; **no conditioning** on the natural-image pool.
- Our standard-utility optimization **blew up** (utility up to ~296,000 nats, firing up to
  ~26,600) at badly-fit n_train points. **This is NOT a meaningful result** — it is the
  documented `r_max=100` numerical breakdown of the *live* utility (see LUT section).
- Where the utility IS valid, standard ≠ DA: standard prefers higher-firing / higher-contrast
  and even **different-content** images (cell 3, n≥200: standard picks a high-contrast FACE,
  DA picks foliage). So the DA conditioning term is doing real work, not decoration.

## The lookup table (READ THIS — key context)
A numerically-stable precomputed LUT exists for *exactly this standard utility*:
- **`analysis/figures/utility_landscape/`** — `lut.npz` (+ `lut_meta.json`, `build_lut.py`,
  `lut_interp.py`). Start-here doc: `LUT_README.md`.
- **What:** a 2,501-node grid of `U(mu_g, sigma2_g)`; `mu_g ∈ [−6,6]` linear (61 pts),
  `sigma2_g ∈ [0,6]` log-spaced (41 pts). Built by **exact quadrature** (no Laplace, no
  r_max truncation); read via `scipy.RegularGridInterpolator` (bilinear). Interp error
  ≤ 0.003 nats, ranking exact.
- **Why it exists (the three traps, quoting `LUT_README.md`):** (1) `E[H_noise]`
  cancellation → negative `E[H_noise]`, `U > H_marg`; (2) **"the live experiment summed the
  Poisson to a fixed `r_max=100` … U blows up to tens of thousands of nats"**; (3) Laplace
  error grows with `sigma2_g`. The live utility is *"accurate only deep inside the gate (~4
  sigma of headroom) and blows up past it — exactly the high-uncertainty region active
  learning explores."* The LUT is the fix; the analysis **offline refits** read U from it.

## What we ran here
- `standard_panels.py` — same pipeline as `../useful_image_panels.py` (M = n_train; cells
  13, 3, 1, 11, 12; n_train = 50…300 step 25; **fresh fits, no warm start**; lucent
  Fourier+sigmoid param; start = best-available image) but `utility_mode='standard'`.
  Output: `panels/cell{N}_panels_std.png`.
- `compare_da_vs_std.py` — side-by-side DA (top) vs standard (bottom) optimized images per
  cell, firing annotated. Output: `panels/compare_cell{N}.png`.
- **The standard utility we used is `acquisition.standard_utility` → `utils.nd_utility_new`
  — i.e. the LIVE Laplace + fixed `r_max=100` utility, NOT the LUT.** The LUT is
  scipy/non-differentiable, and the lucent gradient-ascent needs a torch-differentiable
  objective, so the LUT could not be dropped in directly.

## Findings
1. **The blow-ups are the documented `r_max=100` trap, not a real utility.** Standard
   `U_opt` reached 28,640 (cell 13 n=50) and 295,800 (cell 13 n=275); firing up to 26,605.
   This is exactly `LUT_README`'s "blows up to tens of thousands of nats" past the ~4σ gate.
   The badly-fit n_train points (high epistemic `sigma2_g`) push the optimizer past the
   validity gate, where **both the value and the gradient are corrupted** — so the extreme
   images at the blow-up columns are partly a numerical artifact in the gradient, not purely
   the utility's preference.
2. **Where the utility is valid, standard ≠ DA — genuinely different images.** At well-fit,
   modest-firing points (cell 3 @ n≥200, firing ~1–2; firing well inside the gate) the
   standard utility's most-useful image is a high-contrast **face**, DA's is **foliage**;
   standard images are consistently higher-contrast. The two utilities even rank the image
   pool differently (different best-available start image).
3. **Correction to an earlier session claim.** Earlier we said "naturalness comes from the
   parameterization, not the utility." The standard control shows that's incomplete: lucent
   bounds the **pixels**, but the DA conditioning is what keeps the **firing / image**
   sensible. Both are load-bearing.

## Caveats / what a NEW session should do for a clean comparison
- **Use a numerically-stable, DIFFERENTIABLE standard utility.** Options, best first:
  (a) wrap the LUT as a torch `grid_sample` so it's differentiable AND matches the analysis
  convention; (b) add an `f_max` / σ-gate guard to the optimizer so it never leaves the ~4σ
  validity gate (`mu_g + 4·sqrt(sigma2_g) < log r_max`); (c) use the analytic high-rate
  asymptote `0.5·log(1 + e^{mu_g}·sigma2_g)` past the gate. Until one of these is done, the
  standard blow-up columns are confounded by the `r_max=100` instability.
- The DA run this is compared against used **`sample_lambda=False`** (biased; see parent
  README methods note). No warm-start, single seed (42), M = n_train — same as the DA run.
- The qualitative conclusion (standard ≠ DA; DA conditioning matters) is robust; the
  *quantitative* standard-utility values in the blow-up region are not.

## Pointers
- LUT + ground truth: `analysis/figures/utility_landscape/` (`LUT_README.md` first).
- DA work this controls for: `../` (parent `README.md`, `out/panels/`).
- This folder: `standard_panels.py`, `compare_da_vs_std.py`, `panels/`, `README.md`.
