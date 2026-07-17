# How the Entropy Landscape Visualization is Built

**Audience**: A Claude Code session that has never seen this code before, and is here to **reuse the same plot pattern for a new quantity** (not to study truncation / r_max behavior).

**What the plot is**: a 2D heatmap of a scalar quantity Q(μ_λ, σ²_λ), evaluated on a grid in GP-posterior-moment space (mean and variance of the latent log-firing rate), with optional contour overlays from auxiliary diagnostics. Two side-by-side panels show the same quantity computed two different ways (Laplace sum vs Monte Carlo), so you can see where the two estimators agree.

This document explains the **mechanics** so you can write your own plot of `Q(μ_λ, σ²_λ)` with the same structure. The specific quantity here happens to be entropy `H(R | μ_λ, σ²_λ)` of the predictive spike-count distribution, but the recipe is generic.

---

## 1. The plot in one sentence

For each grid point `(μ_λ, σ²_λ)`, compute a scalar quantity Q, show it as a heatmap, and overlay contour lines from a related diagnostic (here: where the estimator is reliable). Do this twice with two different estimators of Q, side by side.

---

## 2. Files involved

| File | Role |
|------|------|
| `investigations/utility/entropy_landscape.py` | The script. Self-contained: builds grid, computes Q, plots, saves PNG. |
| `gpytorch_porting/utils.py:439 _diff_laplace_log_probs(mu, sigma2, r)` | Laplace approximation of `log p(r \| μ_g, σ²_g)`. Vectorized over query points and r values. Returns `(p_r, log_p_r)` of shape `(N, R)`. |
| `gpytorch_porting/utils.py:562 compute_H_MC(mu, sigma2, ...)` | Monte Carlo entropy estimator. Samples `g ~ N(μ_g, σ²_g)`, then `r ~ Poisson(exp(g))`, then averages `-log p(r)`. Returns `(H, clip_fraction)`. NOT differentiable. |
| `investigations/utility/docs/entropy_landscape.md` | Sister doc — discusses the numerical findings (truncation curve shape, etc). Not needed for plot mechanics. |

`entropy_landscape.py` imports the two utility functions via `importlib.util` (it lives two directories deep, hence the manual path setup at the top — see lines 28–41). If you write a new script in the same investigations folder, copy that path-setup block verbatim.

---

## 3. The math, in one paragraph (with pointers)

The latent log-firing rate is `g = A·λ + λ₀`, where `λ` is what the GP posterior is over. The posterior on `λ` at a query point is Gaussian with mean `μ_λ` and variance `σ²_λ`, so the posterior on `g` is Gaussian with `μ_g = A·μ_λ + λ₀` and `σ²_g = A²·σ²_λ`. The predictive distribution over spike counts is `p(r) = ∫ Poisson(r; exp(g)) · N(g; μ_g, σ²_g) dg`. The Laplace approximation of `log p(r)` is at `utils.py:439 _diff_laplace_log_probs` (vectorized, differentiable). The entropy `H(R | μ_λ, σ²_λ) = -Σᵣ p(r) log p(r)` is computed two ways:

- **Laplace sum**: truncate at some `r_max` and sum. `_diff_laplace_log_probs` produces the table.
- **Monte Carlo**: sample `r` from the model, average `-log p(r)`. `utils.py:562 compute_H_MC`.

In this script `A = 1` and `λ₀ = 1` (constants `A_PLOT`, `LAMBDA0_PLOT` in `entropy_landscape.py:54-56`), so the only difference between λ-space and g-space is a `+1` shift on the mean axis. If your new quantity Q lives in a different parameter space, change those constants — or drop them entirely if Q is computed directly in (μ_λ, σ²_λ).

---

## 4. Grid construction (the part you'll reuse)

`entropy_landscape.py:46-52`:

```python
MU_LAMBDA_MIN, MU_LAMBDA_MAX = -6.0, 11.0
SIGMA2_LAMBDA_MIN, SIGMA2_LAMBDA_MAX = 0.001, 15.0
N_MU, N_SIGMA2 = 400, 300

mu_lambda_range  = torch.linspace(MU_LAMBDA_MIN, MU_LAMBDA_MAX, N_MU)        # (N_MU,)
sigma2_lambda_range = torch.linspace(SIGMA2_LAMBDA_MIN, SIGMA2_LAMBDA_MAX, N_SIGMA2)  # (N_SIGMA2,)
```

**Convention**: in this script, μ_λ is the **row index** (Y axis) and σ²_λ is the **column index** (X axis). Every grid quantity has shape `(N_MU, N_SIGMA2)`. Keep this convention when you reuse the plot — `pcolormesh` is called as `pcolormesh(x_axis, y_axis, Z)` with `x_axis = sigma2_lambda_np`, `y_axis = mu_lambda_np`, `Z` shape `(N_MU, N_SIGMA2)` (`entropy_landscape.py:309`).

**Range choice rationale**: μ_g = μ_λ + 1 ranges over [-5, 12], which corresponds to expected spike counts `E[r] = exp(μ_g)` from ~0.007 up to ~163,000. σ²_λ up to 15 covers the variance regime where the predictive distribution is wide enough to be interesting. Adjust both for your quantity — large μ_g blows up Poisson rates, large σ²_λ makes the predictive distribution heavy-tailed.

---

## 5. Computing the quantity on the grid

### 5.1 Laplace estimator (Panel 1 backing data)

The reference compute is `compute_H_grid_fixed_rmax(mu_range, sigma2_range, rmax, device, batch_rows)` at `entropy_landscape.py:76-100`. The pattern you want to copy:

```python
H = np.full((N_mu, N_s2), np.nan, dtype=np.float32)
r = torch.arange(0, rmax, dtype=torch.float32, device=device)   # (R,)
sigma2_dev = sigma2_range.to(device)

for start in range(0, N_mu, batch_rows):
    end = min(start + batch_rows, N_mu)
    batch_size = end - start

    # Cartesian product of (batch of rows) × (all columns), flattened
    mu_flat = mu_range[start:end].repeat_interleave(N_s2).to(device)  # (batch * N_s2,)
    s2_flat = sigma2_dev.repeat(batch_size)                           # (batch * N_s2,)

    p_r, log_p_r = _diff_laplace_log_probs(mu_flat, s2_flat, r)       # (batch*N_s2, R)
    H_flat = -torch.sum(p_r * log_p_r, dim=1).cpu().numpy()           # (batch*N_s2,)
    H[start:end] = H_flat.reshape(batch_size, N_s2)
```

Key points:
- `_diff_laplace_log_probs` is fully vectorized over query points (axis 0) and r values (axis 1). It runs in one shot per row-batch.
- `batch_rows` is sized to keep the intermediate `(batch * N_s2, R)` tensor under ~2 GB (`entropy_landscape.py:183-184`). For a new quantity that consumes less memory, raise it.
- The grid takes ~0.3s on GPU for the reference adaptive computation. Don't preallocate sentinel-NaN if your quantity is dense and always defined.

### 5.2 Monte Carlo estimator (Panel 2 backing data)

`entropy_landscape.py:262-275`:

```python
H_mc       = torch.full((N_MU, N_SIGMA2), float('nan'))
clip_frac  = torch.full((N_MU, N_SIGMA2), float('nan'))
for i in range(N_MU):
    mu_vec = torch.full((N_SIGMA2,), mu_g_range[i].item(), dtype=torch.float32, device=device)
    H_row, clip_row = compute_H_MC(mu_vec, sigma2_g_range.to(device),
                                   n_samples=2000, a=1.0, lambda0=0.0,
                                   max_log_contrib=50.0,
                                   return_clip_fraction=True)
    H_mc[i]    = H_row
    clip_frac[i] = clip_row
```

Note `compute_H_MC` returns a second array (`clip_fraction`) used as a reliability diagnostic — that's what becomes the green/red overlay in Panel 2. **When you reuse this plot for a different Q, the reliability diagnostic in Panel 2 should be whatever your second estimator's "is this trustworthy?" indicator is.** If you don't have one, just plot Q twice with two different estimators and skip the contour overlay.

MC takes ~50s on GPU here because of the Python loop over rows + per-point `n_samples=2000` Poisson sampling. If that's too slow for your quantity, increase parallelism — but read the docstring at `utils.py:566` first ("HARD CODED VALUES. RAISE TO USER" — there's a known issue flagged inside `compute_H_MC`).

---

## 6. The plot itself (`entropy_landscape.py:287-501`)

### 6.1 Figure layout

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
```

Two side-by-side axes, shared color scale (`vmin=0`, `vmax=min(maxes, 8)` at line 304).

### 6.2 Heatmap call (same for both panels)

```python
im = ax.pcolormesh(sigma2_lambda_np, mu_lambda_np, Q_grid, cmap=cmap, shading='auto',
                   vmin=vmin, vmax=vmax)
cbar = plt.colorbar(im, ax=ax, pad=0.02)
```

`cmap` is `viridis` with NaN regions set to light gray (lines 301-302). That covers the "this estimator couldn't be computed here" case — useful when your quantity has a domain limit.

### 6.3 Axes & annotations

- `ax.set_xlabel(r'$\sigma^2_\lambda$ (GP variance)')`
- `ax.set_ylabel(r'$\mu_\lambda$ (GP mean)')`
- Corner annotations show `E[r] ≈ exp(μ_g)` at the top-left and bottom-left corners, and `σ_r` at the bottom-right (Poisson-lognormal std at the natural-image mean of `μ_λ = -2.5`). These are hand-placed `ax.text` calls at lines 379-402. **For a new quantity, replace these with whatever physical scale annotations make sense — or remove them.**

### 6.4 Contour overlays (THE part you'll reuse most)

Panel 1 overlays four pairs of curves, one per milestone `r_max` value. Each pair is:

1. **Solid colored line**: an empirical contour computed by `ax.contour(..., levels=[H_ERROR_THRESHOLD], colors=[color], linestyles=['-'])` on a precomputed `rel_err` grid (lines 331-334).
2. **Dotted same-color line**: an analytical curve plotted via `ax.plot(sigma2_lambda_smooth, mu_lambda_analytical, color=color, linestyle=':')` (lines 342-343).

The general pattern for **any** overlay:

```python
# To draw a contour from a precomputed grid G(μ, σ²):
G_masked = np.where(np.isnan(G), 999.0, G)        # mask NaN so contour doesn't go through them
sigma2_mesh, mu_mesh = np.meshgrid(sigma2_lambda_np, mu_lambda_np)
ax.contour(sigma2_mesh, mu_mesh, G_masked,
           levels=[threshold], colors=[color],
           linestyles=['-' or '--' or ':'], linewidths=[lw], zorder=6)

# To draw an analytical curve μ = f(σ²):
sigma2_smooth = np.linspace(SIGMA2_LAMBDA_MIN, SIGMA2_LAMBDA_MAX, 500)
mu_analytical = f(sigma2_smooth)
mu_analytical = np.clip(mu_analytical, MU_LAMBDA_MIN, MU_LAMBDA_MAX)  # keep in frame
ax.plot(sigma2_smooth, mu_analytical, color=color, linestyle=':', linewidth=1.0)
```

The `zorder=6` keeps overlays above the heatmap (which is `zorder=0` by default).

Panel 2's overlays are the green-dashed "reliable" contour (clip_fraction < 0.05) and the red-solid "unreliable" contour (clip_fraction > 0.10), drawn the same way (lines 462-472).

### 6.5 Legend pattern

The legend mixes real lines and pure text rows (used to add explanatory captions under the line entries). The trick is `Line2D([], [], color='none', label='caption text')` — invisible handle, but the label still renders. See lines 405-425 for the full pattern.

### 6.6 "No reference available" hatched region

Lines 354-366 fill the area above the adaptive-r_max boundary with gray hatching. This is **specific to the entropy investigation** (it marks where even adaptive truncation gave up). For a new quantity you probably don't want this — skip it unless you also have a "compute failed here" region to mark.

### 6.7 Saving

```python
plt.tight_layout()
save_path = _script_dir / 'entropy_landscape.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
```

`_script_dir` is the folder containing `entropy_landscape.py`. If you make a new script, the PNG lands next to it.

---

## 7. *Tangent: adaptive r_max and the milestone curves — IGNORE THIS UNLESS YOU ARE STUDYING TRUNCATION*

**You can skip this section entirely** if your task is not about how the Laplace sum truncates. Read it only if you're directly investigating `r_max` choice or the reliability of `_diff_laplace_log_probs`. The mechanics in §4–§6 are all you need to reuse the plot pattern.

The script computes the entropy heatmap multiple times: once with an **adaptive** per-row `r_max` (used as the reference, `compute_entropy_row_adaptive` at `entropy_landscape.py:103-131`), and once for each of four **fixed milestone** values `r_max ∈ {100, 500, 1000, 10000}` (`MILESTONE_RMAX` constant, line 64). It then computes a per-grid-point relative entropy error `|H_fixed - H_ref| / H_ref` and draws the contour where that error crosses 1% (`H_ERROR_THRESHOLD`, line 68). The dotted analytical curve `μ_λ = (log(r_max) − λ₀ − 3·A·√σ²_λ) / A` is the textbook 3σ truncation boundary in the same units. The comparison shows that the analytical formula is approximately right but biased by up to ~1 unit at small `r_max`. Panel 2's clip-fraction overlay is the MC analogue of the same reliability question.

If you're not investigating truncation, treat all of this as scaffolding — your new plot will likely have only one "compute Q on the grid" pass and zero or one overlay.

---

## 8. Checklist for reusing this pattern

To plot a new quantity `Q(μ_λ, σ²_λ)`:

1. Copy the path-setup block at `entropy_landscape.py:28-41` if your script lives at the same directory depth. Adjust `_diff_laplace_log_probs` / `compute_H_MC` imports to whatever helpers you need.
2. Set the grid constants (`MU_LAMBDA_MIN/MAX`, `SIGMA2_LAMBDA_MIN/MAX`, `N_MU`, `N_SIGMA2`). Keep μ_λ as Y, σ²_λ as X.
3. If your Q is naturally computed in g-space, also set `A_PLOT`, `LAMBDA0_PLOT` and transform.
4. Write a `compute_Q_grid(mu_range, sigma2_range, ..., device, batch_rows)` modeled on `compute_H_grid_fixed_rmax`. Make sure your inner kernel is vectorized over query points so you can batch full rows at a time.
5. If you have a second estimator, write `compute_Q_alt_grid` similarly (or call an existing function in a loop over rows like the MC version does).
6. Plot: two panels, `pcolormesh(sigma2_lambda_np, mu_lambda_np, Q_grid, ...)`, shared `vmin/vmax`, viridis cmap with NaN→gray.
7. If you have a reliability diagnostic, overlay it as a contour with the pattern in §6.4. Otherwise skip overlays.
8. Save PNG next to the script.

---

## 9. Known quirks to watch for

- **Import side effects**: the path setup at lines 28-41 uses `importlib.util` rather than `from utils import ...` because the script needs to import `utils.py` from two directories up. The whole project has a rule about not changing `torch.default_dtype` via imports (see `.claude/rules/critical_short_rules.md`), but `_diff_laplace_log_probs` doesn't do this, so it's safe.
- **Float32 everywhere**: the script uses `torch.float32`. If you switch to float64 you'll be ~10x slower for no accuracy gain at this scale.
- **`compute_H_MC` has a flagged hard-coded-values issue** in its docstring (`utils.py:566`). If your new plot relies on it, surface that to the user before calling.
- **`pcolormesh` shading**: `shading='auto'` is fine for this grid; if you change `N_MU`/`N_SIGMA2` to extremes (e.g. 5×5) you'll get aliasing warnings — set `shading='nearest'` or interpolate.

---

## 10. Quick reproduction

```bash
# from the gpytorch_porting/ directory:
python investigations/utility/entropy_landscape.py
# → writes investigations/utility/entropy_landscape.png (~50s on GPU, dominated by MC)
```

Output goes to stdout: timing for each phase, per-r_max error stats, analytical-vs-empirical boundary offsets, MC clip-fraction summary. The PNG is the deliverable.
