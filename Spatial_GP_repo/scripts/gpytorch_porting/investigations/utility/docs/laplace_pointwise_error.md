# Pointwise Laplace-error measurement for log p(r)

**Script**: `laplace_pointwise_error.py`
**Figure**: `laplace_pointwise_error.png` (gitignored)
**Related audit**: `~/IDV_code/Papers/latex_summaries/utility_numerical_analysis.tex` §4.1

## What this measures

The predictive count distribution uses a Laplace (saddle-point) approximation of

    log p(r) = log INTEGRAL Poisson(r; e^g) N(g; mu_g, sigma2_g) dg

and entropies sum it over r up to r_max. Two independent errors live here:

- **(L) Laplace error** -- bias in each individual log p(r); present even at one r.
- **(T) truncation error** -- from cutting the sum at r_max.

Earlier work (`entropy_landscape.py`) only ever compared *entropies* (sums), which
mixes (L) and (T). This script isolates **(L) alone** by comparing the Laplace
log p(r) against a high-accuracy reference **at fixed single counts r, never
summing** -- so (T) is structurally absent and any discrepancy is pure (L).

## Method

- **Reference**: `scipy.integrate.quad` (adaptive QUADPACK). The integrand peak is
  located *independently* by `brentq` on the saddle equation
  `e^g + (g - mu)/sigma2 = r` (no reuse of the code under test), and quad
  integrates a peak-factored integrand on an informed finite window **split at the
  peak** (so a sharp spike is never missed). float64, CPU.
  No PyTorch reference: this is an offline, 1D, smooth, non-differentiated
  integral -- GPU/autograd buy nothing, and torchquad's only adaptive rule is
  Monte-Carlo VEGAS (rejected). The torch Simpson-grid "Fallback A" was tried and
  removed once scipy.quad proved sufficient.
- **Validated three ways**: (a) window-independence, 1x vs 2x half-width agree to
  2e-16 nats; (b) small sigma2 -> exact Poisson log-pmf, max diff 2.4e-6 nats
  (the genuine O(sigma2) correction); (c) sum_r p_ref(r) ~ 1 (true density
  normalised) while the Laplace sum drifts (+0.09% / -0.48% / -1.91% as sigma2
  grows -- that drift IS (L)).
- **Code under test**: `utils.py:_diff_laplace_log_probs`, called directly at the
  chosen r (no sum), in both float32 and float64. Note its sigma2 < 1e-6
  exact-Poisson branch (`utils.py:476`): the sweep crosses 1e-6 so the
  smallest-sigma2 points test that branch, not Laplace.
- **Design**: mu_g = 10 panels from -10 to 8 (-1.7 = production / natural
  images); sigma2_g log-spaced 1e-8..100; r per panel = {0,1,2,3,4,5} plus 8
  log-spaced up to r_hi = clip(round(e^{mu_g}), 30, 300).

## Result (measured: sigma2_g 1e-8 .. 100, mu_g from -10 to 8)

The authoritative trust map (1% / 5% / 10% bands, the mu_g<=5 clean range, the
mu_g>=6 breakdown) lives in **LAPLACE_VALIDITY_SUMMARY.md** -- kept in one place
so the numbers do not drift across docs. In one line:

- < 1% (0.01 nats) for sigma2_g <~ 1, at every mu_g and count;
- < 10% throughout sigma2_g <= 20 for mu_g <= 5 (degrading from <1% near sigma2_g=1);
- blows up at mu_g >= 6 (model predicts physically impossible firing, ~400-3000
  spikes; error up to ~300% / 3 nats at mu_g=8, but only at counts that never occur);
- error largest at low counts; float32 adds <= ~3.7e-4 nats.

**Gate trust on sigma2_g, not mu_g** -- a confident low-firing prediction is
accurate; the error is driven by the uncertainty. This is the Laplace step ONLY;
truncation (separate) likely tightens the usable sigma2_g further.

## Reproduce

    /home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python \
        investigations/utility/laplace_pointwise_error.py

Deterministic (no random elements). ~1.1 s on CPU. PNG is gitignored.

## Note on the operating range (verified 2026-05-26)

Verified from the saved utility-landscape trajectories (sigma2_g at the chosen
image; 16 cells, 8000 rep/n_images points): median sigma2_g 0.05, 95th pct 2,
99th pct 4.2, **max ~15** (0.06% of points > 10, none > 20); mu_g from -15 to 6.
So the encountered range sits almost entirely in the safe zone (~85% at
sigma2_g < 1 -> < 1% error; 99% < 4.2; nothing in the exact-Poisson gray band).
The rare high-sigma2_g cases (up to ~15) are handled separately (out of scope).
Earlier narrow "production band" quotes ([0.04, 0.19] or ~1e-4) were a single
cell's confident snapshot, not this range.
