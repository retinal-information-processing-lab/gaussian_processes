# Vendored utility LUT — provenance

This `lut/` folder is a **standalone, vendored copy** of the precomputed closed-loop
selection-utility lookup table and its numpy reference. It exists so the LUT-backed
standard utility for the lucent investigation has **no runtime/test dependency on the
superrepo `analysis/` folder** (a fresh checkout of this submodule must work on its own).

## What was copied (verbatim, byte-identical)

Source folder (in the **superrepo**, NOT this submodule):
`analysis/figures/utility_landscape/`
(absolute at vendoring time:
`/home/idv-eqs8-pza/IDV_code/ClosedLoop-standalone-analysis_april26/analysis/figures/utility_landscape/`)

| file | role |
|------|------|
| `lut.npz`       | the table itself: `U`, `H_marg`, `E_Hnoise`, `method_code` (each (61,41)), `mu_axis` (61,), `s2_axis` (41,). THE standalone artifact. |
| `lut_meta.json` | build metadata (domain, grid, method legend, build commit). |
| `lut_select.py` | `LUTUtility` — the numpy/scipy selection utility with the high-rate OOB fallback. **This is the reference the torch twin mirrors and is validated against.** |
| `lut_interp.py` | `ULUT` — a plain bilinear/bicubic interpolator (returns NaN out-of-domain). Vendored for completeness; not used by the torch twin. |
| `LUT_README.md` | what the LUT is, the three numerical traps, the ~4-sigma gate. |

The five files are copied **unmodified**. Provenance lives only in this note (per the
plan: a citation is documentation, not a code dependency), so the vendored
`LUTUtility` is byte-identical to the one the offline analysis runs — which is exactly
what makes the validation test meaningful.

## Integrity

```
lut.npz  sha256 = 5f7e17298bb85ec8041bce53e94a88051dd88f93c7e2ffe191cc608f1f1fad2d
lut.npz  bytes  = 82342
```
(verify: `sha256sum lut/lut.npz`)

## Source commits

- **LUT build commit** (stamped inside `lut_meta.json` `git_commit`): `851b49d`, built
  `2026-05-28 17:21` — the `build_lut.py` run that produced this `lut.npz`.
- **Superrepo commit that currently owns the file**: `853edaf`
  ("extend utility LUT mu domain to -6; clamp U>=0 at the U~0 floor") — the most recent
  superrepo commit touching `analysis/figures/utility_landscape/lut.npz`. The two differ
  because the file was re-committed after the build stamp; both are recorded for honesty.
- **Superrepo HEAD when vendored**: `08ef1f5`
  ("selection_mechanism: v3 full-pool marginals panel").
- This submodule branch (`pietro/lucent-useful-images`) is based off the pinned
  submodule commit `75b207a` ("Backport d771c90: detach variational params + no_grad
  E-step"); the vendored LUT comes from the **parent** repo, not from this submodule's
  history.

## LUT structure (for the torch twin)

- `mu_axis`: 61 points, **uniform** on [-6, 6] (step 0.2).
- `s2_axis`: 41 points, **log-spaced**: `[0] + geomspace(0.001, 6, 40)`, range [0, 6].
- `U`: (61, 41), finite everywhere, range [0, 3.899]; the `s2=0` edge column is exactly 0.
- `mu_g = A*lambda_mean + lambda0`, `sigma2_g = A^2*var(lambda)` (current-iteration
  PoissonLikelihood params). Spike count ~ Poisson(e^g). `U = H_marg - E[H_noise]`.

## Stale-docstring caveat (mirror the CODE, not the prose)

`lut_select.py`'s module docstring says the LUT is built on "`mu_g in [-2, 6]`" and clamps
`mu_g` to `[-2, 6]`. That text is **stale**: the actual `mu_axis.min()` in this `lut.npz`
is **-6** (61 nodes, -6..6 step 0.2; the owning commit `853edaf` is literally
"extend utility LUT mu domain to -6"), and `LUTUtility.__call__` clamps to
`self.mu_axis.min()/max()` = **[-6, 6]**, not the docstring's [-2, 6]. The torch twin
(`lut_utility.py`) mirrors the **code** (clamp to [-6, 6]); the file is left verbatim so
the vendored reference stays byte-identical. Flagged here so the next reader is not misled.

## Regenerate (from the source folder, superrepo)

```bash
cd analysis/figures/utility_landscape
python build_lut.py        # rebuilds lut.npz + lut_meta.json (exact quadrature ground truth)
```
See `analysis/figures/utility_landscape/LUT_README.md` (the "File map" section) for the
full ground-truth → LUT → interpolation chain and its validation scripts.
