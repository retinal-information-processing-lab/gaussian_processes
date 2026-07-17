# Laplace approximation — validity & how to read the figure

**Figure:** `laplace_pointwise_error.png`   **Script:** `laplace_pointwise_error.py`
**Full method / validation:** `laplace_pointwise_error.md`

## The one question this answers
The utility builds `p(r)` (the predicted probability of `r` spikes) using a Laplace
approximation. For a prediction with mean log-firing-level `mu_g` and uncertainty
`sigma2_g`, **how wrong is that `p(r)`?** We measured it by comparing the Laplace
`p(r)` against an exact numerical integral, at fixed spike counts (no sum over `r`).
The separate truncation-sum question is handled elsewhere.

## Trust map  (mu_g from -10 to ~5, sigma2_g <= 20)

| if sigma2_g is below ... | Laplace error in p(r) is under ... |
|--------------------------|-------------------------------------|
| 1                        | 1%                                  |
| 6                        | 5%                                  |
| 20                       | 10%                                 |

- Below `sigma2_g ~ 1` the approximation is essentially exact (<1%) — this is where
  natural-image predictions normally sit.
- It degrades smoothly with the **uncertainty `sigma2_g`, not the firing level
  `mu_g`**: a confident prediction (small `sigma2_g`) is accurate at any firing level.
- The error is always largest at the **lowest counts (0, 1, 2)** — which become the
  most-likely count when uncertainty is high. The counts the cell actually fires most
  are the best-approximated.

## Where it breaks — do not use mu_g >~ 6
There the model predicts a physically impossible firing level (`mu_g=6` -> ~400
spikes, `mu_g=8` -> ~3000 spikes in the window). The per-count error explodes
(tens to hundreds of %), but only at spike counts the cell could never produce — a
sign the prediction itself is out of physical range, not a fixable error. The bottom
two panels (`mu_g=6, 8`) show this.

## Ranges: analysed vs encountered
- **Analysed & made safe (this investigation):** `sigma2_g` in [1e-8, 100], `mu_g`
  in [-10, 8]. Trustworthy (per-count error < 10%) for `mu_g <= 5` and
  `sigma2_g <= 20`; < 1% for `sigma2_g <~ 1`.
- **Actually encountered (experiments; `sigma2_g` at the chosen image, 16 cells):**
  median `sigma2_g` 0.05, 95th pct 2, 99th pct 4.2, **max ~15** (0.06% of points
  > 10, none > 20); `mu_g` from -15 to 6. So ~85% of points sit at `sigma2_g < 1`
  (< 1% error), 99% below 4.2 (a few % at worst), and nothing reaches the
  exact-Poisson gray band — the encountered range is comfortably inside the safe
  zone. The rare high-`sigma2_g` excursions (up to ~15) are handled by a separate
  mechanism (out of scope here).

## Two honest caveats
- **This is the Laplace error only.** The truncation-sum problem is separate and
  almost certainly shrinks the usable `sigma2_g` further. So "Laplace OK to
  sigma2_g=20" does NOT mean "utility OK to sigma2_g=20".
- **Precision.** The curves are computed in float64 (the pure approximation error).
  Running in float32 (the live precision) adds at most ~3e-4 nats on top — negligible
  — and the code's exact-Poisson switch keeps float32 safe at very small `sigma2_g`.

## How to read the figure
10 panels, 5 rows x 2 columns.

- **Each panel = one firing level `mu_g`**, from -10 (silent cell) to 8 (implausibly
  high), low to high.
- **x-axis = `sigma2_g`, the uncertainty** (log scale; left = confident, right = very
  uncertain).
- **y-axis = the error, in "nats."** Rule of thumb: **a value of `x` nats ~ `x`*100%
  error in `p(r)`** (so 0.01 = 1%, 0.05 = 5%, 0.1 = 10%). Above the zero line = Laplace
  over-estimates `p(r)`; below = under-estimates.
- **Each coloured line = one spike count `r`** (dark = low counts 0,1,2; yellow = high
  counts). The dark lines ride highest (biggest error).
- **Dashed grey lines = the 1% / 5% / 10% rulers.**
- **Numbers along the top of each panel = the most-likely spike count** at each
  uncertainty decade; it slides toward 0 as `sigma2_g` grows.
- **Grey shaded band (far left, `sigma2_g < 1e-6`) = exact-Poisson region**, not
  Laplace — ignore it for Laplace accuracy.

**To use it:** pick the panel for your `mu_g`, find your `sigma2_g` on the x-axis, and
read the height of the relevant count line (the most-likely count from the top axis is
the one that matters most). If `sigma2_g` is below ~1, you are under 1% everywhere.

## Reproduce
    <pytorch_gpytorch env>/python investigations/utility/laplace_pointwise_error.py
Deterministic (no random elements), ~10 s on CPU.
