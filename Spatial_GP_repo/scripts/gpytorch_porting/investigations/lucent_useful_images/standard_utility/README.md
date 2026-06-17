# standard_utility/ — control: the same panels with the STANDARD utility

A deliberate **control / ablation** for the distribution-aware (DA) utility. Same pipeline
as the parent `useful_image_panels.py` — M = n_train, the same 5 cells, the same
n_train = {50,75,…,300} grid, **fresh fits (no rank-1 warm start)**, lucent Fourier+sigmoid
parameterization, start from the best-available image — but the acquisition is the
**standard utility** `U = H_marg − E[H_noise]`: no distribution-aware conditioning, no
Monte-Carlo over images, no λ. It is **deterministic** (no MC noise).

**Why:** to see whether the optimized images are really different from the DA ones — i.e.
to isolate what the distribution-aware conditioning *adds* on top of what lucent already
provides (bounding + smoothness). The standard utility favors high firing / high contrast,
but lucent's sigmoid keeps every pixel in [GP_MIN, GP_MAX], so it **cannot blow up** —
worst case is high-contrast-but-in-range (the figure flags it).

Contents:
- `standard_panels.py` — runner (reuses the parent's `optimize_image` with
  `utility_mode='standard'` and the parent's `build_figure`).
- `panels/` — output figures `cell{N}_panels_std.png` (3 rows × 11 n_train, same layout
  as the DA panels).
- `cache/` — results pickle (gitignored).

Run: `python standard_panels.py --cells 13 3 1 11 12`

Note: like everything else this session, this uses no warm-start. The DA comparison it is a
control for used `sample_lambda=False` (see the parent README methods note).

## Finding (summary — full handoff in `FINDINGS.md`)
The images ARE substantially different: (1) standard-utility firing blows up at badly-fit
n_train points (cell 13: 3,115 @ n=50, 26,605 @ n=275) — but that blow-up is the documented
**`r_max=100` numerical trap**, not a meaningful value (a stable lookup table for this exact
utility lives at `analysis/figures/utility_landscape/`, `LUT_README.md`); (2) where the
utility IS valid, standard picks higher-contrast / different-content images (cell 3 @ n≥200:
a high-contrast face vs DA's foliage). **Takeaway:** the DA conditioning term is essential —
lucent bounds the pixels, DA keeps the firing/image sensible. See **`FINDINGS.md`** for the
full handoff + the LUT/r_max detail + how a new session should make the comparison clean.

