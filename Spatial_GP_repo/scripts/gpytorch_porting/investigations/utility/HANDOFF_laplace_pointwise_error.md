# Handoff: Pointwise Laplace-error measurement for log p(r)

> **STATUS (2026-05-26): DONE** — built, validated, and documented. See
> `laplace_pointwise_error.py`, `docs/LAPLACE_VALIDITY_SUMMARY.md` (user-facing),
> and `docs/laplace_pointwise_error.md`. Kept below as the original plan / record.

**Written:** 2026-05-25
**Author:** previous Claude session ("ONGOING: Utility and Entropy visualization plots")
**For:** a fresh Claude Code session that will build the experiment described below.

---

## 0. TL;DR for the impatient

Build a new investigation that measures the **error of the Laplace approximation
of `log p(r)`** at **individual, fixed spike counts `r`** — with **no sum over `r`
anywhere** — by comparing it against a high-accuracy **adaptive-quadrature**
reference. The point of fixing `r` and never summing is to **isolate Laplace
approximation error from sum-truncation error**: those are two independent failure
modes that the existing investigation tangled together. We have already decided
**not to use Monte Carlo** as the reference. The exact set of `r` values is still
**to be finalized with the user** (a relative-to-mode proposal is below).

---

## 1. Goal and scope

**Goal (plain words).** The active-learning utility is computed from the
predictive spike-count distribution `p(r)`. Computing `p(r)` uses a Laplace
(saddle-point) approximation, and computing entropies from it uses a finite sum
truncated at `r_max`. These introduce **two independent errors**:

- **(L) Laplace approximation error** in each `log p(r)` value — present even at a
  single `r`; grows as the GP log-rate variance `σ²_g` grows.
- **(T) Truncation error** — from cutting the sum `Σ_r` at `r_max`; present even if
  every `log p(r)` were exact.

The previous session re-derived the whole chain and rewrote a LaTeX audit
(`utility_numerical_analysis.tex`, see §6). One conclusion stood out: nobody has
ever **measured** where (L) turns on, because every existing comparison compares
*entropies* (sums), which mixes (L) and (T). 

**This session's job:** measure (L) alone. Compare Laplace `log p(r)` against an
adaptive-quadrature ground truth **at fixed single `r`** (no sum → (T) is
structurally absent), swept over `σ²_g` (and a few `μ_g`). Output: a quantitative
answer to "at what `σ²_g` does the Laplace approximation of `log p(r)` start to
bend, and by how much, as a function of the count `r`?"

**In scope:**
- A new script under `investigations/utility/` that computes per-`r` Laplace error
  vs. an adaptive-quadrature reference.
- 1D slices of error vs. `σ²_g` at a few fixed `μ_g`, for several `r` values.
- Feeding the measured threshold back into `utility_numerical_analysis.tex` §4.1
  and the summary (replacing the "this is dimensional reasoning, not a measurement"
  caveat with a real number).

**Out of scope (unless the user asks):**
- Monte Carlo (explicitly dropped — see §3).
- Anything involving the *sum* over `r` / entropy / utility values (that is the
  truncation question, already visualized by `entropy_landscape.py`; not what we
  are isolating here).
- The distribution-aware utility, conditioning, gradient ascent, kernels.
- Production code changes. This is a measurement/investigation, not a fix.

---

## 2. Current status, honestly split

**Done and verified (this session):**
- Mapped the full utility computation chain (file:line in §10).
- Confirmed, by reading the code, that **all** existing non-Laplace methods
  (`nd_utility_MC`, `nd_utility_NUMERICAL`) still truncate the `r`-sum at
  `r_max=500` — they only change *how `p(r)` is computed*, not the sum
  (verified: `utility.py:445` and `utility.py:744` both build
  `r = arange(0, r_max)`). So they do **not** isolate (L) from (T) either.
- Confirmed the historic "exp(μ) overflow" bug was fixed by a log-space Lambert-W
  substitution: buggy `argmax_g_old` (`utility.py:168`, forms
  `exp(rsigma2+mu).clamp(max=85)`) and fixed `argmax_g` (`utility.py:223`, uses
  `lambertw0_log`) both still exist in the file.
- Re-derived and verified the `E[H_noise]` closed form against `utils.py:719`.
- Rewrote `utility_numerical_analysis.tex` (see §6 for exact changes).

**Done but UNVERIFIED (assumed, flagged honestly):**
- The "large σ² ≈ order 1" scale stated in the .tex is **dimensional reasoning**
  (σ²_g is the variance of the *log* rate; σ²_g≈1 ⇒ rate uncertain by factor ~e),
  **not a measurement**. *Measuring it is precisely this session's job.*
- Natural-image operating point (μ_g ∈ [-1.8,-1.3], σ²_g ∈ [0.04,0.19]) is taken
  from `investigations/utility/docs/da_utility_theory.md:181`, not re-measured.
- Whether `torchquad` or any PyTorch adaptive quadrature actually exists/works for
  this integrand is **unconfirmed** — the new session must check (see §5, §9).

**Not started:**
- The pointwise-error experiment itself. No script written yet.
- The final `r`-value grid (pending user discussion).
- The .tex update with measured numbers.

---

## 3. Decisions made, with rationale

1. **Compare at fixed single `r`, never sum.** This is the whole trick. Truncation
   error lives only in the outer sum `Σ_r`. If we evaluate `log p(r)` at one `r`
   and compare to ground truth, truncation is *structurally absent* and any
   discrepancy is pure Laplace error (L). This is the one idea that makes the
   isolation clean.

2. **Reference = adaptive quadrature; NOT Monte Carlo.** The user asked directly
   "isn't MC the best?" and we worked through it: MC is unbiased and
   assumption-free, but as a *precision* reference it is the weakest — it is
   stochastic (`1/√S` convergence vs. exponential for quadrature), and `log` of a
   noisy `p(r)` is *biased* (Jensen), and it is worst in the tails. Deterministic
   quadrature converges far faster and self-certifies. **Decision: adaptive
   quadrature is the primary (and only) reference. MC is dropped entirely.**

3. **Look for a PyTorch implementation of adaptive quadrature** (user's explicit
   instruction), with documented fallbacks (see §5) because a true *adaptive,
   error-controlled deterministic* quadrature may not exist as a clean PyTorch
   drop-in.

4. **`r` grid relative to the mode** `r* ≈ e^{μ_g}` (the peak of `p(r)`, in *count*
   space — NOT `μ_g` itself). Rationale: Laplace error at a given count depends on
   whether the count is in the bulk, shoulder, or tail; anchoring to `r*` keeps
   "small/bulk/tail" meaningful as `μ_g` sweeps. Proposed set: `0, 1, 2`, then
   `~r*/2, r*, 2·r*`, plus one far-tail point. **The exact set is still to be
   confirmed with the user.**

5. **Primary output = 1D slices: error vs. `σ²_g`** at a few fixed `μ_g`, swept
   from ~1e-4 up to ~10. Rationale: directly answers "where does Laplace bend?".
   A 2D `(μ_g, σ²_g)` heatmap per `r` (parallel to `entropy_landscape.py`) is an
   optional later add, not the primary deliverable.

6. **Precision: float64 reference; run Laplace in BOTH float32 and float64.**
   Rationale: float64 reference so the ground truth is not itself
   precision-limited; running the Laplace path in both precisions separates
   *approximation error* (the math) from *float32 rounding* (the hardware). 

7. **Feed results back into the .tex** §4.1 + summary, replacing the reasoning-only
   caveat with the measured threshold.

---

## 4. Dead-ends / what NOT to do

- **Do not use MC as the reference.** Already litigated and rejected (§3.2). If you
  feel tempted, re-read the reasoning first.
- **Do not reuse `nd_utility_MC` / `nd_utility_NUMERICAL` as-is for the reference.**
  They internally sum over an `r`-grid (`r_max=500`) — that reintroduces exactly
  the truncation we are trying to exclude. If you borrow from
  `nd_utility_NUMERICAL`, extract the **per-`r` `p(r)`** value *before* the sum
  (the Gauss-Hermite machinery at `utility.py:748-790` computes `p(r)` per `r`;
  the sum happens afterward).
- **Do not compute entropies** to answer this question. Entropy is a sum →
  reintroduces (T). The deliverable is per-`r` `log p(r)` error.
- **Do not trust a Gauss-Hermite reference without a convergence check.** For large
  `r` the integrand peaks at `g ≈ log r`, which can be far in the tail of the
  `N(μ_g,σ²_g)` weight where GH nodes are sparse → under-resolved. Always compare
  node counts (e.g. 100 vs 200 vs 500) before trusting it. This is why adaptive
  (peak-refining) quadrature is preferred.

---

## 5. The PyTorch-adaptive-quadrature situation (READ THIS)

The user wants a PyTorch implementation of adaptive quadrature for the reference.
**Search first**, but be aware:

- `torchquad` exists and is GPU-native, but its *fixed* rules are
  Trapezoid/Simpson/Boole and its *adaptive* method is **VEGAS — which is
  Monte-Carlo-based** (the thing we dropped). So torchquad may not give you a
  deterministic adaptive rule. Verify before committing.

**Fallbacks, in order of preference:**

- **Fallback A (likely best, pure PyTorch): informed fixed grid.** We *know* the
  integrand's peak location `g_bar(r)` exactly from the Lambert-W solve
  (`_diff_argmax_g`, `utils.py:418`) and its width from the local curvature
  `φ''(g_bar) = e^{g_bar} + 1/σ²_g`. So center a dense Simpson/Boole grid on
  `g_bar`, width a few local σ, and **refine by doubling points until `log p(r)`
  stops moving** (convergence self-certifies). For a smooth 1D unimodal integrand
  this is as trustworthy as true adaptivity, runs on GPU, and is easy to
  implement. This sidesteps the "does PyTorch adaptive quadrature exist" question
  entirely.
- **Fallback B: `scipy.integrate.quad` on CPU.** Gold-standard, self-reports error.
  The reference is computed *offline* (not in the real-time loop), so a CPU
  round-trip is fine for modest point counts (1D slices = thousands of integrals,
  trivial). Only a concern if you later go to a dense 2D grid × many `r`.

**Recommendation:** if no clean PyTorch deterministic-adaptive quadrature turns up
quickly, go with **Fallback A** — it exploits the exact peak we already have and
keeps everything on the GPU in torch. Cross-check a handful of points against
scipy.quad (Fallback B) to confirm A is correct.

---

## 6. Files created / modified this session

| Path | State | Purpose |
|------|-------|---------|
| `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/utility_numerical_analysis.tex` | **MODIFIED, uncommitted** (Papers is NOT a git repo — see §5 gotcha below) | The audit doc. Heavily revised: Section 4 restructured into 4 independent failure modes (added §4.1 Laplace error; rewrote §4.2 truncation dropping the unjustified `p(r) peaks at e^μ` + wrong width formula; added two-flavor catastrophic cancellation in §4.4); fixed audit table, summary, abstract. |
| `investigations/utility/docs/entropy_landscape_visualization.md` | **UNTRACKED** (`??` in git status) | Doc written earlier this session explaining how to *reuse* the entropy_landscape 2-panel plot pattern for a new quantity. Not directly needed for the new experiment, but a useful template reference. |
| `investigations/utility/entropy_landscape.png` | **Regenerated, gitignored** | Re-ran `entropy_landscape.py` this session. PNG is gitignored by design (won't show in status). Not part of the new work. |

(All paths under `investigations/` are relative to
`gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/`.)

Nothing was committed this session.

---

## 7. Verified-vs-assumed boundary

**Verified by reading code (trust these):**
- `standard_utility` → `nd_utility_new` → `_diff_laplace_log_probs` → `_diff_argmax_g`
  call chain and all file:line numbers in §10.
- `_diff_laplace_log_probs` has an **exact-Poisson branch for `σ²<1e-6`**
  (`utils.py:476`). ⇒ At the *very small* `σ²_g` end of your sweep you will be
  testing the exact-Poisson path, not Laplace. Know which branch is active.
- `nd_utility_MC` (`utility.py:445`) and `nd_utility_NUMERICAL` (`utility.py:744`)
  both truncate at `r_max=500`.
- `E[H_noise] = -e^{μ_g+σ²_g/2}(μ_g+σ²_g-1) + Σ p(r) log r!` matches `utils.py:719`.
- Production: `utility_r_max=100`, `adaptive_r_max=False` (`config.py:234`), called
  from `standalone_linux/main_loop.py:1005`.

**Verified by running:**
- `entropy_landscape.py` runs under `pytorch_gpytorch` and produces the 2-panel
  figure. Its reported numbers (70% of grid points >1% entropy error at r_max=100;
  2.4% at r_max=10000) are from an actual run this session.

**Derived (standard math, not code-checked — re-derive if in doubt):**
- `E[R] = e^{μ_g+σ²_g/2}`; `Var(R) = e^{μ_g+σ²_g/2} + e^{2μ_g+σ²_g}(e^{σ²_g}-1)`.
- Laplace formula: `log p(r) ≈ r·g_bar - e^{g_bar} - (g_bar-μ)²/(2σ²) - 0.5·log(1+σ²·e^{g_bar}) - log r!`.
- `g_bar(r) = rσ² + μ - W_0(σ²·e^{rσ²+μ})`.
- Mode of `p(r)` ≈ `e^{μ_g}` for small `σ²_g`, shifts up as `σ²_g` grows.
- "Large σ² ≈ order 1" and "cancellation kills float32 precision at μ_g≈15" are
  dimensional/order-of-magnitude reasoning, **not measured**.

**Assumed / unconfirmed:**
- Natural-image moment ranges (from a doc, not re-measured).
- Existence/quality of any PyTorch adaptive quadrature.
- Whether the existing comparison scripts were ever run in the current parameter
  regime (small `A≈0.0265`, `λ₀≈-1.686`).

---

## 8. Open questions / pending user decisions

1. **The exact `r` grid** — user flagged this explicitly as "to be discussed."
   Start from the relative-to-mode proposal (§3.4) but confirm: how many counts,
   how far into the tail, absolute vs. mode-relative.
2. **Which `μ_g` values** for the 1D slices (proposal: a production-like one ≈ -1.7,
   plus a few higher ones like 0, 2, 5 to walk into the bad regime). Confirm.
3. **`σ²_g` sweep range and spacing** (proposal: log-spaced ~1e-4 → ~10). Confirm,
   and note the `σ²<1e-6` exact-Poisson-branch boundary (§7).
4. **Error metric**: signed `log p` error in nats is primary; also report relative
   error in `p(r)`? Confirm which the user wants plotted.

---

## 9. Concrete next steps, in order

1. **Re-orient.** Read this handoff, then skim `utility_numerical_analysis.tex`
   §3–§4 (the Laplace fix + the four failure modes) and the bottom of the previous
   chat if available. Confirm you understand *why fixed-`r`/no-sum isolates (L)*.
2. **Finalize the `r` grid and sweep with the user** (§8). Don't start coding the
   grid until this is settled — it's cheap to ask, expensive to redo.
3. **Search for a PyTorch adaptive quadrature** (§5). Time-box it. If nothing clean
   turns up, commit to Fallback A (informed fixed grid on the Lambert-W peak).
4. **Build the reference.** Implement `log p_ref(r | μ_g, σ²_g)` via the chosen
   quadrature, in float64. Validate it: (a) self-convergence (double the
   resolution, error < tol), (b) cross-check ~5 points against `scipy.quad`,
   (c) sanity: at small `σ²_g` it must agree with the exact Poisson `p(r)`.
5. **Wire up the Laplace path** to return `log p(r)` at chosen `r` without summing:
   call `_diff_laplace_log_probs(mu_g, sigma2_g, r_tensor)` directly with `r_tensor`
   = your chosen counts. Run it in both float32 and float64.
6. **Compute and plot** signed error `log p_Laplace(r) - log p_ref(r)` vs `σ²_g`,
   one line per `r`, one panel per `μ_g`. Mark the σ²_g where |error| crosses, say,
   0.01 and 0.1 nats. Save PNG next to the script (PNGs are gitignored — that's
   fine).
7. **Read off the threshold** and report it to the user. This is the deliverable:
   "Laplace `log p(r)` error stays below X nats for σ²_g < Y (at count r=...)".
8. **Update the .tex** §4.1 and the summary with the measured number, replacing the
   "dimensional reasoning, not measured" caveat. Keep the structure.
9. **Document** findings in a short `docs/` markdown next to `entropy_landscape.md`,
   and (gently, the user is new to git) offer to stage the untracked files — but
   see the detached-HEAD gotcha (§11) first.

---

## 10. References to read (in order)

1. **This handoff.**
2. `Papers/latex_summaries/utility_numerical_analysis.tex` — the audit. §1-2 (setup
   + `E[H_noise]` derivation), §3 (Laplace + the log-Lambert-W fix), §4 (the four
   failure modes — §4.1 Laplace error and §4.2 truncation are the two you're
   separating).
3. `investigations/utility/docs/entropy_landscape.md` — the prior investigation's
   findings (truncation boundary). Read critically; it conflates (L) and (T).
4. `investigations/utility/docs/da_utility_theory.md:117-181` — the `z_safe`
   metric (which is a *truncation* metric, not a Laplace one) and the natural-image
   operating point.

**Code (all under `gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/`):**
- `utils.py:439 _diff_laplace_log_probs` — the Laplace `log p(r)` you're testing.
  Note the `σ²<1e-6` exact-Poisson branch at line 476.
- `utils.py:418 _diff_argmax_g` — Lambert-W peak `g_bar(r)` (gives you the peak for
  Fallback A).
- `utils.py:533 compute_H`, `utils.py:688 nd_utility_new` — where the sum lives
  (what you're NOT doing).
- `utils.py:482 compute_adaptive_rmax` — the two-layer truncation heuristic.
- `acquisition.py:41 standard_utility` — production entry point.
- `../../../utility.py:709 nd_utility_NUMERICAL` — Gauss-Hermite per-`r` `p(r)`
  machinery you can borrow from (extract `p(r)` *before* its sum). GH substitution
  `z=(g-μ)/(σ√2)` at lines 718-760.
- `../../../utility.py:223 argmax_g` / `:168 argmax_g_old` — fixed vs. historic-buggy
  saddle solver (context for the .tex §3 story).

---

## 11. How to run things (operational)

- **Repo layout (work entirely in this worktree):** this handoff and all the
  utility code live in the **april26** worktree, which is your cwd:
  `/home/idv-eqs8-pza/IDV_code/ClosedLoop-standalone-analysis_april26/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/`.
  Do all the new work here — no cross-checkout `cd` needed. The `.tex` is the one
  exception: it lives at `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/utility_numerical_analysis.tex`
  (a separate, non-git location). **Worktree note:** this repo has several sibling
  worktrees (`...-analysis_1`, `...-analysis_2`, `...standalone`, etc.) — all at the
  **same** Spatial_GP_repo submodule commit `75b207a`, so the code is identical
  across them. The *previous* session accidentally created its artifacts in the
  `analysis_2` worktree (absolute-path mixup, likely from a resume); this handoff
  has been moved here to april26 so you don't have to chase them. Ignore the other
  worktrees and work in april26.
- **Python env:** verify with `which python`. In the session that wrote this
  handoff, `python` resolved to **base anaconda (NO torch)**, so commands used the
  full path: `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python`
  (torch 2.5.1, has gpytorch). The project docs claim the env is "already active";
  if `which python` shows the env, plain `python` is fine — if it shows base
  anaconda, use the full path.
- **Run a script** (from the `gpytorch_porting/` dir):
  `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python investigations/utility/<your_script>.py`
- **GPU** is available (`cuda`). Quadrature/Laplace at these sizes is trivially
  fast; GPU only matters if you go to dense 2D grids.
- **Git gotchas:**
  - `Spatial_GP_repo` is in **DETACHED HEAD at `75b207a`** (the submodule pin from
    CLAUDE.md). Be very careful with any git operations here — do NOT create
    branches/commits without checking with the user; you can lose work in detached
    HEAD. New investigation files are fine to leave untracked.
  - `Papers/` is **NOT a git repo** (no `.git`). The `.tex` edits are just on disk —
    there is no version control and no commit to make there. Mention this to the
    user if they ask to "commit the doc."
  - PNGs under `investigations/utility/` are **gitignored** — regenerating figures
    won't show in `git status`; that's by design.

---

## 12. Ready-to-paste continuation prompt

```
Continue the utility numerical-analysis investigation. You are in the april26
worktree (ClosedLoop-standalone-analysis_april26); do all work here. Read this
handoff first:
gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/investigations/utility/HANDOFF_laplace_pointwise_error.md

Goal: measure the Laplace approximation error of log p(r) at FIXED single spike
counts r, with NO sum over r, by comparing against an adaptive-quadrature reference.
The point is to isolate Laplace error from sum-truncation error (two independent
failure modes). We already decided: no Monte Carlo; use adaptive quadrature; you
should first search for a PyTorch adaptive-quadrature implementation, and if none
is clean, use the "informed fixed grid centered on the Lambert-W peak g_bar" fallback
(Fallback A in the handoff).

Before writing code, do step 2 in the handoff's "next steps": finalize with me the
exact r-value grid (I want to discuss this), the μ_g values, and the σ²_g sweep
range. The handoff has a relative-to-mode proposal — start there but ask me.

Env: use /home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python (plain
`python` has no torch). Note _diff_laplace_log_probs has an exact-Poisson branch
for σ²<1e-6, so the very-small-σ² end of the sweep tests that branch, not Laplace.
```
