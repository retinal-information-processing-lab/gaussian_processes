# GP Performance Metrics: Definitions and Results

This document defines the two performance metrics computed for our variational
GP fits and presents results across all tested configurations.

Both metrics attempt to answer the same question: how much of the neuron's
response can the model explain, after accounting for trial-to-trial noise?

---

## Shared Ingredients

Both metrics start from the same raw quantities, computed from 30 test images
each shown 30 times:

**r_even**: mean response over even-numbered trials (trials 0, 2, 4, ..., 28)
**r_odd**: mean response over odd-numbered trials (trials 1, 3, 5, ..., 29)
**f_pred**: model's predicted firing rate for each test image

**reliability** = corr(r_even, r_odd)

This measures how consistent the neuron is with itself.

**mean_accuracy** = 0.5 * (corr(f_pred, r_even) + corr(f_pred, r_odd))

This measures how well the model's predictions correlate with the actual neural
response. Averaging over even and odd halves reduces noise in the estimate.

### Why noise attenuates correlations (and why sqrt matters)

The neuron has a true underlying response r_true that we never observe directly.
We only see noisy versions: r_obs = r_true + noise. Noise attenuates any
correlation involving r_obs by a factor of sqrt(SNR), where
SNR = var(r_true) / var(r_obs).

When we correlate TWO noisy measurements (r_even vs r_odd), BOTH sides
contribute an attenuation factor:

    corr(r_even, r_odd) = 1 * sqrt(SNR) * sqrt(SNR) = SNR

The two sqrt(SNR) factors multiply into a plain SNR. This is why reliability
equals SNR directly -- both quantities are noisy.

When we correlate a CLEAN prediction with a NOISY measurement (f_pred vs r_obs),
only ONE side is noisy:

    corr(f_pred, r_obs) = corr(f_pred, r_true) * sqrt(SNR)

Just one sqrt(SNR) factor, because f_pred is deterministic (no noise).

This asymmetry is why the noise correction divides by sqrt(reliability) and
not by reliability: mean_accuracy was attenuated by only one sqrt(SNR), while
reliability absorbed two of them. To recover the true correlation:

    corr(f_pred, r_true) = mean_accuracy / sqrt(SNR) = mean_accuracy / sqrt(reliability)

---

## Metric 1: Adjusted R-squared

### Definition

    adjusted_r2 = mean_accuracy^2 / reliability

Equivalently:

    adjusted_r2 = (mean_accuracy / sqrt(reliability))^2

This is the Spearman-Brown noise correction applied to R-squared. The reasoning:

1. The observed correlation between prediction and response is ATTENUATED by
   noise. The true correlation with the noise-free signal is:
   corr_true = mean_accuracy / sqrt(reliability)

2. Squaring gives the fraction of TRUE SIGNAL VARIANCE explained:
   adjusted_r2 = corr_true^2

**Bounds**: Theoretically 0 to 1 under the additive noise model. A value of
1.0 means the model explains all signal variance. In practice CAN slightly
exceed 1.0 when mean_accuracy > sqrt(reliability), which happens if the
split-half reliability estimate is noisy or if the model's smooth predictions
correlate better with one half than the halves with each other. This is rarer
than for explained_var (see below) because the condition is stricter:
mean_accuracy > sqrt(reliability) vs mean_accuracy > reliability.

**Reference**: Goldin et al. 2023 PNAS, Eq. 5. Attributed to Keshishian et al.

### Our results (41 cells, 3 seeds averaged)

| Config | Mean | Median | n > 0.8 | n > 0.6 |
|--------|------|--------|---------|---------|
| our_defaults | 0.678 | 0.717 | 11/41 | 28/41 |
| our+paper_inner | 0.695 | 0.735 | 13/41 | 29/41 |
| paper_init | 0.700 | 0.759 | 13/41 | 30/41 |
| paper+interleave | 0.713 | 0.706 | 17/41 | 30/41 |
| **broad+interleave** | **0.730** | **0.751** | **15/41** | **34/41** |
| broad+A01+intl | 0.609 | 0.684 | 11/41 | 25/41 |

---

## Metric 2: Explained Variance (correlation ratio)

### Definition

    explained_var = mean_accuracy / reliability

Despite the name, this is NOT a variance quantity -- it is a correlation ratio.
It measures what fraction of the maximum achievable correlation the model
reaches: reliability is the ceiling, mean_accuracy is the observed value,
and dividing gives a normalized score.

The name "explained variance" is conventional in the neuroscience literature
(it appears as `explained_variance_fractions_bis` in the paper's CNN code)
but is a misnomer. A true explained variance metric would use variance
decomposition (e.g., (var_total - MSE) / (var_total - var_noise)), which the
paper's CNN code also computes separately as `explained_variance_fractions`.

**Bounds**: Theoretically 0 to 1, but in practice can exceed 1.0. This happens
when mean_accuracy > reliability, i.e., the model correlates with each half
better than the halves correlate with each other. This is a weaker condition
than what adjusted_r2 needs to exceed 1.0 (mean_accuracy > sqrt(reliability)),
so explained_var exceeds 1.0 more often. In our data, 6 cells have
explained_var > 1.0 while none have adjusted_r2 > 1.0.

**Relationship to adjusted_r2**:

    adjusted_r2 = explained_var^2 * reliability

For high-performing cells (explained_var near 1), the two metrics are similar.
For weaker cells, adjusted_r2 is much lower because squaring penalizes values
below 1.

Example: if mean_accuracy = 0.7 and reliability = 0.9:
  explained_var = 0.7 / 0.9 = 0.778
  adjusted_r2   = 0.7^2 / 0.9 = 0.544

The same model gets 0.778 on one metric and 0.544 on the other.

### Our results (41 cells, 3 seeds averaged)

| Config | Mean | Median | n > 0.8 | n > 0.6 |
|--------|------|--------|---------|---------|
| our_defaults | 0.833 | 0.891 | 30/41 | 39/41 |
| our+paper_inner | 0.847 | 0.903 | 31/41 | 40/41 |
| paper_init | 0.865 | 0.909 | 32/41 | 39/41 |
| paper+interleave | 0.878 | 0.896 | 33/41 | 41/41 |
| **broad+interleave** | **0.889** | **0.913** | **36/41** | **41/41** |
| broad+A01+intl | 0.744 | 0.854 | 27/41 | 35/41 |

---

## Metrics Found in the Paper's CNN Evaluation Code

Source: `regular_cnn.py:get_model_table()` in the paper's GitHub repo. This is
the CNN baseline evaluation -- the GP evaluation code is NOT in the repo (it
lives in the private `pyretina_systemidentification` package). We don't know
if the GP uses the same metrics, but this is the only evaluation code available.

The function computes 10 quantities per cell and saves them to a CSV. The
three "explained variance" metrics are:

### `explained_variance_fractions` (variance-based FVE)

    explained_variance_fractions = (total_var - MSE) / (total_var - noise_var)

Where `total_var = var(all_responses)`, `noise_var = mean(var per image across
trials)`, `MSE = mean((response - prediction)^2)`. This is the fraction of
explainable variance the model explains, computed via variance decomposition.
Not a correlation metric. We do NOT compute this in our code, and we did not
save per-trial MSE in the sweep, so we cannot check it post-hoc.

### `explained_variance_fractions_bis` (correlation ratio)

    explained_variance_fractions_bis = accuracy / reliability

Where `accuracy = corrcoef(mean_pred, r_odd)` (odd trials only, not averaged
with even) and `reliability = corrcoef(r_even, r_odd)`. This is our
`explained_var` formula (with the minor difference that accuracy uses odd
trials only, not the even+odd average).

### What we don't know

- Which of these two metrics gets plotted in Figure 2F (the plotting code is
  in the private package)
- Whether the GP evaluation uses the same metrics as the CNN evaluation
- Whether they used the variance-based or correlation-based version
- Both are unsquared and neither divides by sqrt(reliability), ruling out Eq. 5

---

## Why This Matters

The paper (Goldin et al. 2023) reports "adjusted R^2 > 0.8 for 36/41 cells."
The formula in Eq. 5 is the squared version (our adjusted_r2). However:

- Our best config gives **15/41 > 0.8 on adjusted_r2** but **36/41 > 0.8 on
  explained_var** -- the latter matches the paper's count exactly.
- The paper's Figure 2F caption calls the metric "explained variance" while the
  y-axis label says "adjusted r^2".
- Figure 2F shows data points above 1.0 for the GP model. While not strictly
  impossible for adjusted_r2, exceeding 1.0 is much easier for explained_var
  (requires mean_accuracy > reliability, a weaker condition than
  mean_accuracy > sqrt(reliability) needed for adjusted_r2).
- The only evaluation code in the paper's GitHub repo (`regular_cnn.py`)
  computes `accuracy / reliability` (our explained_var formula, unsquared).

---

## Glossary of Configurations

All configurations use: 108x108 images, M=250 inducing points, n_train=3160,
ground-truth RF centers, lambda0_init=-1, random inducing point selection,
fix_Amp=True, sigma_0 direct parameterization, no early stopping.

| Config | Description |
|--------|-------------|
| **our_defaults** | beta=0.1 (broad RF), A=0.01, no interleaving, 10 E-steps, 10 M-steps, 50 EM iterations. Our standard training schedule. |
| **our+paper_inner** | Same init as our_defaults, but with the paper's training schedule: 50 E-steps, 20 M-steps, 80 EM iterations. Tests effect of more inner iterations. |
| **paper_init** | Paper's initialization values (beta=0.0452 tight RF, A=1e-4 small gain), paper's training schedule (50/20/80), no interleaving. Tests effect of paper's init. |
| **paper+interleave** | Same as paper_init but with interleaved F-step: A and lambda0 are updated via damped Newton (alpha=0.25) at every E-step iteration, not just once per EM cycle. Matches the paper's training loop structure. |
| **broad+interleave** | Our broad beta=0.1 + paper's small A=1e-4 + interleaving + paper schedule (50/20/80). Best overall: combines our RF init with the paper's A bootstrapping. |
| **broad+A01+intl** | Same as broad+interleave but with A=0.01 instead of 1e-4. Tests interleaving with larger A init. Performs poorly because the damped Newton (alpha=0.25) is too conservative for A=0.01. |

**fix_Amp=True**: The Amp parameter (multiplies the C matrix in the kernel) is
frozen at 1.0. The paper's code has no Amp parameter at all. Free Amp absorbs
scale from A, confounding the optimization.

**interleave_fstep**: When enabled, A and lambda0 are updated via a damped
Newton method (2x2 Hessian, damping factor alpha=0.25) inside each E-step
Newton iteration. This keeps A in sync with the variational parameters (m, V).
When disabled, A is updated once per EM cycle via LBFGS after all E-step
iterations complete.

**sigma_0 direct parameterization**: sigma_0 is stored and optimized as itself
(identity transform), not in log-space (exp transform). The exp transform
caused sigma_0 to stagnate near its initial value due to LBFGS curvature
mismatch.
