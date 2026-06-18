# Utility ground-truth LUT — start here

A precomputed lookup table (LUT) of the closed-loop selection utility
`U(mu_g, sigma2_g)`, plus the analysis that built and validated it. `mu_g`, `sigma2_g`
are the mean and variance of the GP posterior on the log firing rate
(`mu_g = A*lambda_mean + lambda0`, `sigma2_g = A^2*var(lambda)`); the spike count is
Poisson(`e^g`). The utility is the per-image information gain `U = H_marg - E[H_noise]`.

## Why this exists: three numerical traps in the utility

1. **`E[H_noise]` cancellation.** The form `E[H_noise] = -e^{mu+s/2}(mu+s-1) + sum_r p(r) log r!`
   subtracts two large (~10^3) terms that should cancel to ~3, and the `log r!` sum is
   tail-sensitive. Truncate it and you get **negative `E[H_noise]` and U > H_marg**
   (impossible). This broke an earlier ground-truth attempt and is present in the live code.
2. **`r_max` truncation blow-up.** The live experiment summed the Poisson to a fixed
   `r_max = 100`. Where the count distribution reaches `r_max`, the `log r!` term is cut
   and the cancellation fails -> **U blows up to tens of thousands of nats**.
3. **Laplace error.** The live `p(r)` is a Laplace approximation; its error grows with
   `sigma2_g` and is amplified by the `log r!` weighting.

## The fix (the chain in this folder)

- **Exact ground truth** (`utility_ground_truth.py` on top of `marginal_pr_reference.py`):
  `p(r)` by deterministic quadrature (no Laplace, no truncation); `E[H_noise]` by a
  **1-D integral over g** (no cancellation). Cross-checked by `validate_ground_truth.py`
  (physical bounds, monotonicity) and `mc_check_ehnoise.py` (an unbiased Monte-Carlo of
  `E[H_noise]`, agreeing to <= 0.001 nats).
- **LUT** (`build_lut.py` -> `lut.npz` + `lut_meta.json`): U on a grid, `mu in [-6,6]`,
  `sigma2 in [0,6]` (log-spaced sigma2). Quadrature where feasible; a validated log-normal
  in the high-mu/high-sigma2 corner. `scope_lognormal_error.py` maps where log-normal is
  accurate; `design_sigma2_grid.py` justifies the log-spaced grid.
- **Interpolation** (`lut_interp.py`): `ULUT()(mu, sigma2, method="linear"|"cubic")`,
  vectorised, returns NaN outside the domain (no extrapolation). **This is what the offline
  fit calls.** `validate_lut.py` / `test_interp.py`: interpolation error <= 0.003 typical,
  ranking exact.

## How to read `overview_rmax_vs_lut.png` (the starting picture)

3x2 grid over (`sigma2` x, `mu` y), linear axes:

| | left column = utility | right column = error vs LUT truth |
|---|---|---|
| row 1 | live `r_max = 100` | live100 - true |
| row 2 | live `r_max = 1000` | live1000 - true |
| row 3 | LUT (interpolated) | LUT node positions (basis) |

- Left utility panels share a `0 -> 4` colormap; **magenta = value off-scale (blow-up)**.
  The LUT panel (bottom-left) never goes magenta — that is the point.
- Right error panels (diverging symlog): white = agree, red = live over-estimates.
  **Lime line = where they differ by 0.05** (disagreement onset). Black `2/3/5-sigma`
  lines = `e^{mu+k*sigma} = r_max` (headroom). The lime line sits between the 3- and
  5-sigma lines: **the live utility needs ~4 sigma of headroom, not 3, to be trusted.**
- Bottom-right: blue = nodes done by quadrature, orange = log-normal corner, grey =
  the `sigma2 = 0` edge (U = 0). Dots bunch at small sigma2 because the grid is log-spaced.

## Takeaway

The live fixed-`r_max=100` utility is accurate only deep inside the gate (~4 sigma of
headroom) and **blows up past it** — exactly the high-uncertainty region active learning
explores. `r_max = 1000` shrinks but does not remove the problem. The LUT (stable
`E[H_noise]`, exact `p(r)`) is the fix; the offline fit reads U from it via `lut_interp`.

## File map

| file | role |
|---|---|
| `marginal_pr_reference.py` | exact marginal `p(r)` by quadrature |
| `utility_ground_truth.py` | `U_exact = H_marg - E[H_noise]` (the ground truth) |
| `validate_ground_truth.py`, `mc_check_ehnoise.py` | ground-truth validation |
| `build_lut.py` -> `lut.npz`, `lut_meta.json` | build + store the LUT |
| `lut_interp.py` | `ULUT` interpolator (offline fit's entry point) |
| `validate_lut.py`, `test_interp.py` | LUT bounds/monotonicity; interpolation accuracy |
| `design_sigma2_grid.py`, `scope_lognormal_error.py` | grid + method-split justification |
| `live_rmax_vs_truth.py`, `live_rmax_error_map.png`, `subgate_agreement_check.py` | "how wrong was the live r_max" |
| `overview_rmax_vs_lut.py` -> `overview_rmax_vs_lut.png` | the overview figure (start here) |
| `lut_validate.png`, `interp_error.png`, `lognormal_error_map.png` | validation figures |
