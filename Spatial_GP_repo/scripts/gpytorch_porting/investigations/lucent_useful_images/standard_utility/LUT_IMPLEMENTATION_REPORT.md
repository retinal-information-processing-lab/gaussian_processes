# LUT-backed standard utility — implementation report

**Branch**: `pietro/lucent-useful-images` (gaussian_processes submodule)
**Date**: 2026-06-18
**Status**: DONE — plan steps 1–6 implemented; all acceptance criteria met.
**Plan**: `PLAN_lut_standard_utility.md` · **Handoff**: `HANDOFF_lut_standard_utility.md`

Replaced the numerically-unstable live `r_max=100` standard utility with the precomputed,
exact LUT in a **torch-differentiable** form, usable inside the lucent optimization. The
standard-utility blow-up (cell 13: **U=295,800 nats / firing 26,605** at n=275) is gone:
LUT-standard U is **finite (≤ 3.73 nats) and monotone everywhere**, so the standard-vs-DA
comparison is now fair (numerics removed as a confound). Standard utility only; DA deferred.

---

## 1. Files created / edited

All under `investigations/lucent_useful_images/`. **No engine file touched.**

**New — vendored LUT (standalone artifact), `lut/`:**
| file | role |
|------|------|
| `lut/lut.npz` | the table (vendored, byte-identical; sha256 5f7e1729…fad2d). **Committed** despite `*.npz` ignore. |
| `lut/lut_meta.json`, `lut/LUT_README.md` | metadata + start-here doc (vendored). |
| `lut/lut_select.py`, `lut/lut_interp.py` | numpy references `LUTUtility` / `ULUT` (vendored). |
| `lut/PROVENANCE.md` | source path, sha256, build commit `851b49d` / owning commit `853edaf`, stale-docstring caveat. |
| `lut/__init__.py` | makes `lut` an importable package. |
| `lut/lut_utility.py` | **the only real new code**: `lut_U(mu_g,sigma2_g)` (torch-differentiable twin of `LUTUtility`) + `lut_standard_utility(model,likelihood,x)`. |
| `lut/test_lut_torch.py` | validation vs the vendored numpy `LUTUtility` + gradcheck. |

**New — step-6 runners, `standard_utility/`:**
| file | role |
|------|------|
| `standard_utility/lut_panels.py` | LUT-backed re-run of `standard_panels.py` (own cache/output; single-sources constants from `standard_panels`). |
| `standard_utility/compare_three_way.py` | DA vs Laplace-standard vs LUT-standard figures. |
| `standard_utility/LUT_IMPLEMENTATION_REPORT.md` | this report. |

**Edited — exactly one existing file (the plan's allowed exception):**
| file | change |
|------|--------|
| `optimize_image.py` | `from lut.lut_utility import lut_standard_utility` + a `utility_mode=='standard_lut'` branch (mirrors the existing `'standard'` branch) + docstring. |
| `.gitignore` | added `!lut/lut.npz` so the vendored table is committed despite `*.npz`. |

---

## 2. Validation — torch `lut_U` vs vendored numpy `LUTUtility`

`lut/test_lut_torch.py` (standalone: numpy + torch only, no engine/GPU; deterministic, seed=0).
Gate = ≤ 1e-5 in-domain AND out-of-domain. **All PASS, by ~10 orders of margin:**

| check | max\|err\| | gate |
|-------|-----------|------|
| node reproduction (all 2,501 grid nodes) | **0.0** | — |
| in-domain (mu∈[-6,6], sigma2∈[0,6]) | **8.88e-16** | 1e-5 |
| sigma2-OOB (sigma2∈(6,50], high-rate fallback) | **1.78e-15** | 1e-5 |
| mu-clamp (\|mu\|>6) | **0.0** | — |
| combined mu & sigma2 OOB | **1.78e-15** | 1e-5 |
| float32 (production dtype, informational) | **5.78e-7** | (≪1e-5) |
| `torch.autograd.gradcheck` (float64, mid-cell + OOB) | **PASS** | — |

The float64 twin matches the numpy reference to **machine precision** (the interpolation,
the mu-clamp, and the continuity-corrected `sigma2>6` fallback `U(mu,6)+[h(mu,s2)-h(mu,6)]`
are all reproduced exactly), and gradients are correct (gradcheck). Reproduce:
`<pytorch_gpytorch python> lut/test_lut_torch.py`.

---

## 3. Before / after — the blow-up is removed (acceptance criterion #4)

Same models (default_gpy, M=n_train, seed=42), same lucent param, same start-from-best-pool
rule; only the utility BACKEND differs. `lut_panels.py --cells 13 3`, 1.8 min.

**Cell 13 (the blow-up cell), optimized utility `U_opt` (nats) and firing `fr`:**

| n_train | Laplace-standard U / fr | **LUT-standard U / fr** | sigma2 (LUT) |
|--------:|------------------------:|------------------------:|-------------:|
| 50  | **28,640.2** / 3,115.7 | **3.72** / 3,389.8 | 4.22 |
| 100 | 383.4 / 116.0 | 2.05 / 113.7 | 0.75 |
| 175 | 1,629.5 / 310.2 | 2.55 / 233.5 | 1.48 |
| 275 | **295,800.7** / 26,604.6 | **3.72** / 17,430.5 | 4.27 |
| 300 | 27.8 / 39.9 | 1.23 / 26.3 | 0.51 |
| **max over all n** | **295,800.7** | **3.724** | — |

LUT-standard `U_opt` is finite and ≤ 3.73 nats at EVERY n (the LUT's in-domain ceiling is
3.899); the live-Laplace 28,640 / 295,800-nat explosions are gone. **`oob=0%` at every
point** — the optimizer stayed inside the verified LUT domain (sigma2 ≤ 4.27 < 6), so these
are the table's TRUE entropies, not the `sigma2>6` ranking-proxy fallback.

**Cell 3 (content-difference cell, well-fit / low sigma2):** LUT ≈ Laplace everywhere
(both U ≤ 0.5 nats, fr ≤ 1.9) — the LUT MATCHES the live utility where the live utility is
valid, and only diverges (by fixing it) where the live utility breaks. Full per-n tables for
both cells are printed by the dump in §6.

---

## 4. The 3-way comparison (`compare_three_way.py`)

`panels/compare_cell{13,3}_3way.png` (image grids, DA / Laplace-std / LUT-std rows; Laplace
blow-up columns flagged red) and `panels/compare_3way_curves.png` (the decisive picture):

- **Top (utility vs n_train, log y):** Laplace spikes to ~10⁴–10⁵ nats at the badly-fit
  points (n=50, 275); LUT stays flat at ~1–4 nats; DA ~0.01–0.05. → the LUT fixes the
  **numerics**.
- **Bottom (firing vs n_train, log y):** BOTH standard variants reach high firing
  (10³–10⁴) at badly-fit points; DA stays low (~1–20). → the high-firing preference is the
  standard utility's **real taste**, NOT the numerical artifact; the DA conditioning term is
  what keeps firing/image sensible. This confirms `FINDINGS.md`'s qualitative conclusion with
  the numerics now removed as a confound.

**Expected, not a bug:** LUT-standard images are still high-contrast and firing is still high
at badly-fit points (n=50: fr 3,390; n=275: fr 17,430). The LUT fixes the utility VALUE and
GRADIENT, not the standard utility's intrinsic preference for high-firing images (see the
plan / `FINDINGS.md`). Firing is finite because lucent's sigmoid bounds the pixels.

---

## 5. Confirmations

- **NO engine file touched.** Every commit's
  `git diff --cached --name-only | grep -v 'investigations/lucent_useful_images/'`
  printed NOTHING (verified before each of the 4 commits). The only edited existing file is
  `optimize_image.py` (the plan's allowed exception) + the in-folder `.gitignore`.
- **`lut/lut.npz` is committed:** `git ls-files lut/lut.npz` → `…/lut/lut.npz` (un-ignored
  via `!lut/lut.npz`; sha256 5f7e1729…fad2d, byte-identical to source).
- **Standalone:** no runtime/test path reads the superrepo `analysis/` folder — `lut_U`,
  `lut_standard_utility`, and `test_lut_torch.py` all read the vendored `lut/lut.npz` /
  `lut/lut_select.py`. (Provenance citations in `PROVENANCE.md` are documentation, not deps.)

---

## 6. Reproduce

```bash
PY=/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python
cd .../investigations/lucent_useful_images
$PY lut/test_lut_torch.py                      # validation gate (no GPU)
cd standard_utility
$PY lut_panels.py --cells 13 3                 # ~2 min; -> cache/panels_results_lut.pkl + panels/
$PY compare_three_way.py                       # -> panels/compare_*_3way.png, compare_3way_curves.png
```

---

## 7. Deviations from the plan, surprises, open questions

**Deviations (all minor, none affecting correctness):**
- **`lut_standard_utility` lives in `lut/lut_utility.py`** (the plan offered "or sibling") and
  was committed in commit 2 with the interpolator, not commit 3 — the import of `acquisition`
  is lazy (inside the function) so the module + its test stay engine/GPU-free.
- **Step 6 uses a sibling `lut_panels.py`** rather than adding `--backend lut` to
  `standard_panels.py`. The plan offered both; the sibling keeps the change purely additive
  (honors "the only existing file you may edit is optimize_image.py") and avoids
  duplication-drift by single-sourcing the experimental constants from `standard_panels`.
- **Cells run: 13 and 3** (the plan's required minimum). Extending to 1/11/12 is a one-line
  rerun (`lut_panels.py --cells 1 11 12`); not required for the deliverable.

**Surprises (worth knowing):**
- The vendored `lut_select.py` **docstring is stale**: it says the LUT is on `mu∈[-2,6]` and
  clamps mu to `[-2,6]`, but the actual `mu_axis.min()` is **-6** (owning commit `853edaf`
  = "extend utility LUT mu domain to -6") and the CODE clamps to `mu_axis.min()/max()` =
  `[-6,6]`. The plan already specified [-6,6]; the torch twin mirrors the CODE. Flagged in
  `lut/PROVENANCE.md` and kept verbatim (so the vendored reference stays byte-identical).
- At cell 13 n=275 the model dips to test_r=0.78 (from 0.96 at n=250) — a badly-fit point with
  high epistemic sigma2 (4.27); that is why a LATE n_train still blew up under Laplace.
- At well-fit points the LUT and Laplace `U_opt` are both small but **not identical** (e.g.
  cell 13 n=200: Laplace 1.4 vs LUT 0.95). This is expected: they reach different optima from
  different best-pool starts, and the LUT value is the trustworthy one (the live Laplace
  mildly over-estimates even inside the gate, per `LUT_README.md`). Only the LUT value is a
  true entropy.

**Open questions / explicitly out of scope (unchanged from the handoff):**
- **DA-via-LUT** is deferred (needs a separate `H_marg` fallback; the `sigma2>6` asymptote is
  the asymptote of `U`, not of `H_marg`). Not attempted.
- **Promotion into `acquisition.py`** (a `backend='lut'` switch in the live utility) is the
  separate, later task on a branch off `pietro/workingbranch`. Not done here.
- **`FINDINGS.md`'s "what a NEW session should do" (option a)** is now implemented, but I did
  NOT edit `FINDINGS.md` (the scope rule restricts existing-file edits to `optimize_image.py`).
  Recommend a one-line pointer there in a follow-up so that TODO isn't read as still-open.
- The dataset path in `gp_models.py` still points at the sibling ClosedLoopProject repo —
  a pre-existing non-standalone reference, flagged in the handoff, out of scope here.
