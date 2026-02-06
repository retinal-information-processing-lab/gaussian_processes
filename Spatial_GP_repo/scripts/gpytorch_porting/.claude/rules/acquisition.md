---
paths:
  - "acquisition.py"
  - "tests/test_acquisition.py"
---

# Acquisition Functions Reference

**Auto-loads when**: Working with `acquisition.py` or `test_acquisition.py`. Also available via `/acquisition` skill.

## What exists

- `standard_utility(model, likelihood, x_candidates, r_max)` — H_marg - H_noise, works with any model mode
- `distribution_aware_utility(model, likelihood, x_candidates, x_samples, r_max, sample_lambda)` — H_marg - E[H_cond], default_gpy only (needs covariance_matrix)

Both return dicts. `distribution_aware_utility` returns `H_marg` and `H_cond` separately.

## Critical: A/lambda0 transform inconsistency

The imported functions have different conventions for the Poisson link `g = A*lambda + lambda0`:

- **`compute_H(mu, sigma2, a, lambda0)`** — takes RAW GP moments (lambda), transforms internally
- **`nd_utility_new(mu, sigma2)`** — takes LOG-FIRING RATE moments (already transformed)
- **`laplace_approximations_new(mu, sigma2)`** — takes log-firing rate moments

`distribution_aware_utility` uses `compute_H` (passes raw GP moments + A, lambda0).
`standard_utility` transforms manually before calling `nd_utility_new`.

Both deliver the same math to `laplace_approximations_new`. But mixing them up is a silent bug source.

## Import side effects

Importing from `gp_utility_playground.py` runs `torch.set_default_dtype(float64)`. We guard with save/restore in `acquisition.py`. Maintain this pattern for any new imports from playgrounds.

## Design decisions

- **No `torch.no_grad()` wrapper** — caller decides. Keeps door open for gradient-based x* optimization.
- **`sample_lambda=True`** — not a debug flag. When False, uses posterior mean (deterministic). Useful for sanity checks and reproducible comparisons.
- **p(x) as pre-drawn tensor** — `x_samples` passed by caller. PyTorch indexing returns views (no copy).
- **default_gpy only** — uses `model(X).covariance_matrix` for conditioning. vargp_direct deferred (needs augmented matrix approach).

## Math summary

U(x*) = H_marg(x*) - H_cond(x*)

- **H_marg**: Entropy of R at x* under current posterior. Via Laplace approximation of p(r|x*,D).
- **H_cond**: Expected entropy after hypothetically observing lambda at a natural image x~p(x). MC estimate: sample x_i, sample lambda_i, condition GP, compute entropy.
- Conditioning uses Gaussian identities on the joint covariance from GPyTorch.

Full derivations: `~/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex` and `predictive_distribution_conditioned_on_observation.tex`.
