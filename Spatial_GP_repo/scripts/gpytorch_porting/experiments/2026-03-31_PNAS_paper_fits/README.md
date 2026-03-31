# PNAS Paper Fits: Matching Goldin et al. 2023

These are our best GP fits on the PNAS retinal ganglion cell dataset,
configured to match the paper's architecture as closely as possible.

**Key result**: 36/41 cells achieve explained_var > 0.8, matching the
paper's reported "36 out of 41 cells" (see Metric Note below).

---

## Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Mode | vargp_direct | Eigenspace variational GP |
| Image resolution | 108x108 | Paper's resolution |
| M (inducing points) | 250 | Paper's M |
| n_train | 3160 | All available training images |
| beta init | 0.1 | Our broad RF init (paper uses 0.0452) |
| rho init | 0.1 | Our smoothness init (paper uses 0.0821) |
| A init | 1e-4 | Paper's gain init |
| lambda0 init | -1 | Paper's bias init |
| RF center init | ground_truth | From white noise/checkerboard ellipse fits |
| IP selection | random | Paper uses random + noise |
| fix_Amp | True | Paper has no Amp parameter |
| interleave_fstep | True | Damped Newton (alpha=0.25) A/lambda0 update inside each E-step Newton iteration, matching paper's `updateA()` |
| sigma_0 parameterization | direct (identity) | For this run; exp gives +0.008 mean adj_r2 |
| nEstep / nMstep / nIter | 50 / 20 / 80 | Paper's training schedule |
| Early stopping | off | Full convergence |
| Seeds | 1, 2, 3 | Averaged over 3 seeds |

### Interleaved F-step (interleave_fstep=True)

The paper updates A and lambda0 at EVERY E-step Newton iteration (50 times
per EM cycle) via a damped Newton method with alpha=0.25. This keeps the
firing rate parameters in sync with the variational parameters (m, V) at
every step. Without interleaving, A is updated once per EM cycle via LBFGS
after all E-step iterations, which is slower to bootstrap from A=1e-4.

Our implementation: `damped_newton_update_A_lambda0()` in `eigenspace_fstep.py`.
2x2 Hessian of the Poisson log-likelihood w.r.t. [A, lambda0], solved with
damping factor 0.25, convergence at sum(|gradient|) < 1e-6, max 100 iterations
per E-step step.

### fix_Amp=True

The paper's kernel computes C = alpha * C_smooth * alpha (no amplitude
parameter). Our code has an extra Amp multiplier: C = Amp * alpha * C_smooth * alpha.
Setting fix_Amp=True freezes Amp at 1.0, matching the paper's architecture.
Without this, Amp absorbs scale from A, confounding the optimization dynamics.

---

## Results

### Summary (mean over 3 seeds per cell)

| Metric | Mean | Median | n > 0.8 | n > 0.6 |
|--------|------|--------|---------|---------|
| adjusted_r2 | 0.730 | 0.751 | 15/41 | 34/41 |
| explained_var | 0.889 | 0.913 | **36/41** | 41/41 |
| test_r | 0.829 | 0.860 | -- | -- |

### Metric Note (IMPORTANT)

The paper reports "adjusted R^2 > 0.8 for 36/41 cells" (Eq. 5). However,
our investigation found strong evidence that the paper likely reports the
unsquared metric (our `explained_var = mean_accuracy / reliability`) despite
calling it "adjusted R^2":

- Our explained_var gives **36/41 > 0.8** (exact match)
- Our adjusted_r2 gives only 15/41 > 0.8
- Figure 2F caption says "explained variance" while y-axis says "adjusted r^2"
- Figure 2F shows data points above 1.0 (much easier with unsquared formula)
- The only evaluation code in the paper's GitHub repo computes accuracy/reliability (unsquared)

See `METRICS_COMPARISON.md` in this folder for full metric definitions,
formulas, and the evidence for this conclusion.

---

## Files

| File | Description |
|------|-------------|
| `results.jsonl` | 123 runs (41 cells x 3 seeds), one JSON per run |
| `METRICS_COMPARISON.md` | Metric definitions, formulas, and paper comparison |
| `README.md` | This file |

### Results fields (per run in results.jsonl)

Key fields: `cell`, `seed`, `adjusted_r2`, `explained_var`, `test_r`,
`reliability`, `final_A`, `final_sigma_0`, `final_beta`, `final_rho`,
`final_lambda0`, `train_time`, `n_iterations_run`.

---

## How to Reproduce

```bash
cd gpytorch_porting/

# Single cell
python run_single_mode.py --mode vargp_direct --cell 8 --seed 1 \
  --data-path datasets/PNAS_108x108_original.npz \
  --ntilde 250 --n-train 3160 \
  --beta 0.1 --rho 0.1 --A-init 1e-4 --lambda0-init -1 \
  --n-iterations 80 --n-estep 50 --n-mstep 20 \
  --ip-selection random --no-early-stopping

# Then set in config: fix_Amp=True, interleave_fstep=True, rf_init='ground_truth'
```

For the full sweep script that produced these results:
`investigations/paper_gap/run_sweep.py` (config `broad+interleave`).

---

## Investigation Context

These results come from a systematic investigation of the performance gap
between our GP implementation and the paper. Key findings:

1. **Our reimplementation is correct**: vargp_direct matches vargp_old
   (the original utils.py code) within 0.002 avg adj_r2 when confounds
   are controlled.

2. **The paper's code differs from both**: no Amp, interleaved F-step,
   no eigenspace projection, scipy L-BFGS-B, float64.

3. **The remaining gap is likely a metric definition mismatch**, not a
   model performance gap.

Full investigation: `investigations/paper_gap/INVESTIGATION_LOG.md` (21 findings).
