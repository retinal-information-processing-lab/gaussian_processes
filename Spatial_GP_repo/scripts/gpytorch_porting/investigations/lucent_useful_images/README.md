# Lucent useful-image generation

**Goal.** Synthesize the *most-useful image* for a trained GP cell by maximizing the
**distribution-aware (DA) utility** — and get **sensible, bounded images** instead of
the contrast/border blow-ups every previous attempt produced.

**One-line result.** Using [`lucent`](https://github.com/greentfrapp/lucent)'s Fourier
parameterization (spectral 1/f prior) + sigmoid bounding as the image
parameterization, DA-utility optimization stays **physically bounded by construction**
(no pixel clipping, no rescaling, no bounded-optimization solver) and yields
interpretable receptive-field-driven stimuli. Started from a natural image, the result
*is* a natural image with only its RF region retuned.

> **Convention: M = n_train AT ALL TIMES** (every training point is an inducing point →
> the full, non-sparse variational GP at each training size). `gp_models.train_default_gpy`
> defaults M to n_train; all scripts in this folder follow it. The FIRST exploration used a
> fixed low **M = 50** (a sparse approximation) — those figures are archived under
> `early_exploration_fixed_M50/` (superseded, kept for traceability). The current
> deliverable is `useful_image_panels.py` (per-cell 3-row × n_train panels → `out/panels/`).

> **⚠ Methods note — λ sampling (READ THIS):** EVERY optimization result produced so far in
> this session (all panels, all figures, DA and standard) used **`sample_lambda = False`** —
> the DA utility conditions on the posterior **mean** λ at each sampled image, not a random
> draw. This is deterministic and reproducible but **biased**: per
> `~/IDV_code/Papers/latex_summaries/why_sample_lambda.tex`, using the mean instead of
> sampling λ mis-estimates the conditional-entropy term (Jensen's inequality). The unbiased
> setting is `sample_lambda = True` (planned follow-up; it adds Monte-Carlo noise that must
> then be quantified by comparing different conditioning subsets). Nothing here has been run
> with `sample_lambda = True` yet.

> **Git:** this investigation lives on the submodule branch **`pietro/lucent-useful-images`**
> (first commit `569570b`, branched from the pinned commit `75b207a`). The superproject
> `analysis/april26` still pins `75b207a`; the branch is NOT merged into `pietro/workingbranch`
> (which has diverged). Committed content is code + docs only — images/caches are gitignored.

---

## Why previous attempts failed, and why this works

The utility **intrinsically prefers unnatural images** — it rewards high marginal
entropy H_marg, which grows with (a) firing rate and (b) epistemic variance (highest
*far from training data*). With the arc-cosine kernel K(x,x) ~ ‖x‖²_C, raw pixel-space
ascent drives ‖x‖ → ∞ (contrast runaway) and to the pixel borders (unconstrained).
**The DA utility does NOT fix this**: its conditional term penalizes off-distribution
*direction* (ρ², angular), but ρ² is scale-invariant, so utility still diverges ~‖x‖¹·⁹
(see `../utility/docs/da_utility_theory.md:93-110`). Past fixes (PCA / C-eigenspace
subspaces, normalized kernel, etc.) constrained *direction* but not *amplitude*, or
cost predictive accuracy.

**Naturalness has to come from the parameterization, not the objective** — the Lucid
feature-visualization philosophy. Lucent gives us exactly that:

1. **Sigmoid + affine bounding.** `img01 = sigmoid(...)` ∈ (0,1), mapped affinely to the
   dataset's physical range `[GP_MIN, GP_MAX] = [-2.4013, +2.4780]` (global min/max over
   train+val). Pixels can **never** exceed the display range → bounded ‖x‖ → bounded
   H_marg → bounded utility. No runaway, no borders. No clipping (the bound is smooth).
2. **Fourier 1/f^p spectral prior (`decay_power`).** The optimizable parameters are FFT
   coefficients scaled by 1/freq^p, biasing toward smooth, low-frequency (natural-spectrum)
   images and suppressing high-contrast checkerboards.
3. **DA utility ρ² term.** Adds angular alignment with the natural-image pool on top.

The kernel's C-matrix only "sees" RF pixels, so **non-RF pixels get no gradient and stay
at their init** — gray (gray-start) or the natural scene (natural-start). That is why the
natural-start result looks fully natural: only the RF is changed.

---

## Key findings

- **Bounded, no overblow.** Saturation 0–3% on every panel; pixel range always within
  `[GP_MIN, GP_MAX]`. The decades-old contrast/border problem is solved by construction.
- **Model-quality (n_train) controls the optimum's character** — the headline science:
  - **Low n_train (uncertain model, high σ²_g):** the optimum exploits **epistemic
    uncertainty** — a broad, smooth RF-region modification.
  - **High n_train (confident model, tiny σ²_g):** the optimum exploits **firing-rate**
    structure — a compact, well-localized RF (dark center / bright surround) emerges and
    sharpens. cell 13: σ²_g 3.25 → 1.94 → 0.014 → 0.009 as test_r 0.37 → 0.73 → 0.94 → 0.94.
- **Start point matters only when the model is uncertain.** At n_train≥200 the gray and
  natural starts converge to the *identical* optimum (the landscape is sharp). At
  n_train≤100 they differ (rich epistemic landscape, multiple optima).
- **Laplace validity holds.** σ²_g stays within the trustworthy zone (z_safe ≥ ~3, well
  above the "unreliable" <1.5 threshold from `../utility/docs/`); no entropy/truncation
  blow-up.
- **`decay_power` must be ≈1.0** with the default `sd=0.01`. p≥2 explodes the
  low-frequency amplitude (scale ~108^p) and **saturates the sigmoid into a binary blob
  before optimization starts** (vanishing gradients). Higher p needs a proportionally
  smaller `sd`.

---

## Files

| File | Role |
|------|------|
| `lucent_param.py` | lucent grayscale Fourier param + sigmoid; affine to GP space; FFT-from-image init (`fft_params_from_image01`) for natural-start. Imports lucent via `sys.path` (no install — avoids the `kornia<=0.4.1` pin that would damage the shared env). |
| `gp_models.py` | Train `default_gpy` model (DA utility needs `.covariance_matrix`); load image pool + global min/max. Dataset path points at the sibling repo (symlink absent here). |
| `optimize_image.py` | Core DA-utility optimization loop + 8-panel diagnostic. CLI: `--cell --n-train --decay-power --start {gray,natural}`. |
| `screen_cells.py` | Screen cells for default_gpy test_r at the top corner. |
| `run_grid.py` | Full cell × n_train grid, both starts; caches to `cache/grid_results.pkl` (incremental/resumable). |
| `make_figure.py` | Build the deliverable figures from the cache → `refined/`. |
| `lucent_repo/` | cloned lucent (gitignored). |
| `out/`, `cache/` | gitignored exploratory PNGs + result cache. |
| `refined/` | committed deliverable figures. |

**Env:** `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python` (the Bash tool
defaults to base python which lacks torch — always use this full path).

## Reproduce

```bash
PY=/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python
cd .../investigations/lucent_useful_images
$PY screen_ladder.py                       # find cells with a clean test_r ladder
$PY run_grid.py --cells 3 12 13 1 11       # ~45 min, caches incrementally (append-aware)
$PY make_figure.py --cells 13 3 1 11 12    # -> refined/fig_*.{png,svg}
# single run / inspect:
$PY optimize_image.py --cell 13 --n-train 300 --start natural --decay-power 1.0
```

## Cell roster + test_r ladders — EARLY M=50 SCREENING (seed=42)

> These ladders are from the early fixed-M=50 screening. Under the current **M=n_train**
> convention the numbers shift up a little (e.g. cell 13 @ n_train=300: 0.96 vs 0.94) and
> the low-n_train values are unchanged where n_train=50=M. The cell *selection* below still
> stands; the live `useful_image_panels.py` run reports the M=n_train ladders per cell.

Cells chosen for a **monotonically-increasing** default_gpy test_r ladder (the
model-quality axis must rise with n_train). Only ~2/13 screened cells were positive
throughout; most are unstable at low n_train (a known default_gpy issue,
`../../.claude/rules/debugging.md` 3.6/3.7), so "monotonic increase" (allowing a
negative/broken start) was the selection rule. Cell 8 (0.36→0.70→**0.27**→0.81,
non-monotonic) was screened out.

| cell | n=50 | n=100 | n=200 | n=300 | RF type | note |
|------|------|-------|-------|-------|---------|------|
| 13 | 0.37 | 0.73 | 0.94 | 0.94 | center-surround | cleanest gradual climb |
| 3  | 0.18 | 0.63 | 0.81 | 0.91 | center-surround | gradual climb |
| 1  | −0.15| 0.80 | 0.93 | 0.94 | oriented dipole | broken→great |
| 11 | −0.08| −0.05| 0.90 | 0.92 | oriented dipole×2 | broken→great |
| 12 | −0.07| −0.08| 0.85 | 0.91 | structured | broken→great |

σ²_g (Laplace log-firing variance) falls as the model gains confidence, e.g. cell 13:
3.25 → 1.94 → 0.014 → 0.009; z_safe stays ≥ ~3 (Laplace valid) at all points. Every
panel: saturation ≤ 7% (mostly < 2%), pixel range within [GP_MIN, GP_MAX] — **no
overblow anywhere**. Broken low-n_train models produce a broad diffuse blob (high
epistemic uncertainty over a wide region); confident models produce a compact RF.

## Deliverable figures (`refined/`)

- `fig_useful_natural.{png,svg}` — most-useful image started from a natural image
  (bounded, natural; the "not overblown" proof). Rows = cells, cols = n_train.
- `fig_useful_rf.{png,svg}` — gray-start: the preferred stimulus on neutral gray;
  crystallizes from diffuse (uncertain) to a compact RF (confident).
- `fig_rf_difference.{png,svg}` — (optimized − natural start): the RF-localized change;
  broad when uncertain, compact when confident.

## Caveats / open points

- **DA utility requires `default_gpy`** (covariance matrix). `vargp_direct` (higher test_r)
  would need the augmented-matrix DA port — deferred.
- The optimum's *naturalness* is delivered by the parameterization + natural start, NOT by
  the utility; the utility alone still favors the (bounded) high-firing / high-epistemic
  corner.
- Faint regularly-spaced background spots in gray-start panels are the unoptimized
  near-gray FFT init texture (non-RF pixels), not RF structure — the difference/natural
  figures avoid this.
- Reproducible: fixed seeds (train seed=42, lucent init=0, pool subset=0,
  `sample_lambda=False`). The DA MC λ-draw is the only stochastic op and is bypassed.
