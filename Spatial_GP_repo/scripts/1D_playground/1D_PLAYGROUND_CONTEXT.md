# 1D Playground Scripts

## gp_utility_playground.py
Central source of GP classes and utility functions. All other scripts import from here.
- `VariationalGP`, `PoissonLikelihood` - GP model classes
- `evaluate_nd_utility_new()` - standard utility
- `evaluate_distribution_aware_utility()` - accounts for p(x) distribution
- `get_marginal_moments()`, `get_conditional_moments()`, `compute_H()` - building blocks

## diagnose_1d_utility_w_fixed_ntrain.py
How much does conditioning on a sample x influence the GP predictive?
- Plot 1: mean/variance change for ONE sample
- Plots 2-3: average change in mean/variance over multiple samples
- Plot 4: utility averaged over samples
- `SAMPLE_X` flag: draw samples from p(x) or use fixed point?
- Uses fixed n=100 training points

## diagnose_1d_utility_w_active.py
Same idea but with active learning loop.
- Starts with 5 points, iteratively adds points at max utility
- `USE_DISTRIBUTION_AWARE` flag: which utility to use for selection?
- Plots selection history and utility snapshots at 6 iterations

## test_dirac_delta_collapse.py
Verifies: when x_samples = x_star (Dirac delta), distribution-aware utility collapses to nd_utility.
- Evaluates both utilities and compares them
- 3 subplots: utility comparison, GP fit, relative error
- Pass threshold: mean relative error < 10%
- **Note**: Only valid inside training domain (see diagnose_dirac_collapse/)

## diagnose_dirac_collapse/
Diagnostic scripts investigating why Dirac collapse fails outside training domain.

**Root cause found**: `nd_utility_new`'s analytical H_cond formula breaks for high σ²:
```
H_cond = -exp(μ+σ²/2)*(μ+σ²-1) + Σ p(r) log(r!)
```
When σ² > ~1.5, this produces negative entropy (impossible).

Scripts:
- `diagnose_covar_at_same_point.py` - GPyTorch covariance check (minor 1e-4 issue)
- `diagnose_H_cond_comparison.py` - Compares analytical vs MC H_cond
- `diagnose_utility_components.py` - Decomposes U = H_marg - H_cond
- `DIAGNOSIS_SUMMARY.md` - Full findings
