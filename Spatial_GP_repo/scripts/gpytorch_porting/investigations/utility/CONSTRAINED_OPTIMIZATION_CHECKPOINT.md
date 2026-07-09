# Checkpoint: the "most-useful image" constrained-optimization problem

**Date**: 2026-07-08 · **Scope**: the WHOLE image-optimization pipeline (not just lucent).

**Evidence rule**: every claim carries a tag and a citation. Tags: **[PROVEN]** (analytic proof
in a `.tex`), **[NUM]** (numerically confirmed, no analytic proof), **[MEASURED]** (a reported
number, reproducing script exists), **[ASSERTED]** (a stated number with no reproducing script).
**File paths are all relative to `scripts/gpytorch_porting/`** (this doc lives in
`investigations/utility/`). The `.tex` proofs are in `investigations/utility/docs/`, currently
under active edit by another session — line numbers may drift, section names are stable.

### Notation (all per the utility docs)
- **standard (marginal) utility** `U_std = H_marg − E[H_noise]` (no pool conditioning);
  **DA (distribution-aware) utility** `U_DA = H_marg − E[H_cond]` (conditions on the natural pool).
- `H_marg`, `H_cond` = marginal / conditional predictive entropy (nats).
- `c` = radial scale applied to an image (`x → c·x`); `‖x‖_C = √(xᵀCx + σ₀²)` = C-weighted image
  norm; `C` = receptive-field structure matrix; `σ₀²` = kernel bias variance.
- `μ(x)`, `σ²(x)` = GP latent posterior mean / variance. `μ_g = A·μ(x)+λ₀` = log-firing-rate mean;
  `σ²_g = A²·σ²(x)` = log-firing-rate variance; **firing rate** = `exp(μ_g + σ²_g/2)`.
- `ρ²` = fraction of variance the conditioning step removes (angular → scale-invariant).
- `test_r` = Pearson r of model vs held-out responses (accuracy). `LUT` = exact-quadrature utility
  lookup table. `OOB` = pixels outside the physical display range. `sat` = pixels pinned at the
  sigmoid rail.

---

## 1. Goal

Synthesize the image that maximizes the GP active-learning utility (standard or DA), subject to
being physically displayable (pixels in the dataset range `[-2.4013, 2.4780]`).

## 2. The core problem — UNSOLVED — and its proof

**The utility drives the optimizer to high-norm ‖x‖ (max-contrast) images, because the kernel's
self-variance is quadratic in the norm.** The chain:

- **Self-variance is quadratic in norm: `K(x,x) = ‖x‖²_C`.** **[PROVEN]**
  `investigations/utility/docs/proof_moments_and_conditioning.tex:30-36`; stated
  `investigations/utility/docs/da_utility_theory.md:32`; code `kernels.py:594-600`.
- **Scaling `x → c·x`: latent mean `μ(x) ∝ c` (linear), latent variance `σ²(x) ∝ c²` (quadratic).**
  **[PROVEN]** Corollary 2, `investigations/utility/docs/proof_divergence_theorems.tex:121-127`
  (proof-status "Proved" at `:370-372`). So `μ_g ∝ c` but `σ²_g ∝ c²`.
- **Conditioning cannot cap the norm.** The conditioned variance is `σ²_g(1−ρ²)`; `ρ²` is
  scale-invariant **[PROVEN]** (Corollary 1) while `σ²_g ∝ c²` **[PROVEN]** (Corollary 2), so the
  conditioned variance still grows `∝ c²`. `investigations/utility/docs/proof_moments_and_conditioning.tex:224-225, 246-249`.
- **Therefore the utility grows with the norm** (this last step, entropy growing with variance, is a
  **heuristic** bound `U ≳ ¼σ²_g` — "a rigorous proof of this bound is not provided here",
  `investigations/utility/docs/proof_divergence_theorems.tex:248-252`). Measured exponent:
  **`U_DA ~ c^1.9`** over the Laplace-valid range (`c∈{2,5,10}`), **consistent with the `c²`
  structural prediction**. **[NUM]** `investigations/utility/docs/proof_divergence_theorems.tex:269-272, 378`
  (explicitly "Not proved, numerically confirmed").
- **Root cause = kernel homogeneity**: `K(cx,y) = c·K(x,y)` for `σ₀²=0`; norm and direction are
  decoupled, so ascent inflates the norm to grow variance→entropy→utility for free.
  `investigations/utility/docs/da_utility_theory.md:106-110` ("**This is NOT a bug** — intrinsic to
  the DA utility with homogeneous kernels").
- **Direct evidence the optimized images inflate** (trained model, seed 42 cell 8 M=50): the
  DA-optimized image reaches **latent mean μ(x*)=101** (~13× the target's μ=7.86; its C-norm ‖x*‖_C
  is ~7.8× larger); the standard-optimized image reaches **μ(x*)=68,497** with **U_std→∞**. **[NUM]**
  `investigations/utility/docs/proof_divergence_theorems.tex:299-324`.

**Precise form of the informal "firing grows with ‖x‖²":** it is the prior *variance*
`K(x,x)=‖x‖²_C` that is quadratic; that drives `σ²_g ∝ c²` and hence `H_marg` and utility `~c^1.9`.
The log-firing *mean* `μ_g` is only *linear* in ‖x‖ (the firing rate `exp(μ_g+σ²_g/2)` is then
dominated by the `σ²_g ∝ c²` term). The intuition is right; this is the exact chain.

## 3. A SEPARATE failure mode — numerical, and already solved

Do not conflate with §2. The **standard** utility also suffered a purely numerical blow-up: the
Laplace entropy summed the Poisson to a fixed `r_max=100`; past the ~4σ gate both the utility value
AND its gradient corrupt. **[MEASURED]** the standard utility reached **28,640 nats** (cell 13 n=50)
and **295,800 nats** (n=275), with firing rate up to **26,605**
(`investigations/lucent_useful_images/standard_utility/FINDINGS.md:57-64`).
**Fixed** by a vendored exact-quadrature lookup table (LUT), now torch-differentiable: max utility
**295,800 → 3.724 nats**, validated to machine precision (`|err|≤8.9e-16`, gradcheck pass)
(`investigations/lucent_useful_images/standard_utility/LUT_IMPLEMENTATION_REPORT.md:9-12, 49-59, 73-82`).
**Key control:** after this numerical fix, the standard utility STILL drives the firing rate to
~17,000 at bad-fit points — so the extreme-image preference of §2 is **real**, not the numerical
artifact (`investigations/lucent_useful_images/standard_utility/LUT_IMPLEMENTATION_REPORT.md:110-113`;
`investigations/lucent_useful_images/standard_utility/README.md:14-15`).

## 4. Everything tried (constraint / bound / kernel), with evidence

Paths in this table are under `investigations/utility/docs/` unless prefixed otherwise.

| # | Approach | Mechanism | Result / why | Evidence | Status |
|---|----------|-----------|--------------|----------|--------|
| a | Raw pixel-space gradient ascent | none | norm→∞, borders, max contrast | `subspace_operations.md:84`; `da_utility_theory.md:149-152` | fails (baseline) |
| b | PCA subspace | restrict optimization directions | constrains **direction, not amplitude**; still diverges along allowed axes | `subspace_theory.md:74-78, 230-233` | insufficient |
| c | C-eigenspace subspace | optimize in kernel eigenbasis | "a **convergence aid, not a naturalness constraint**" | `subspace_theory.md:127, 230-233` | insufficient |
| d | Normalized arc-cosine `K̄(x,x)=1` | remove norm from kernel | **eliminates** norm divergence (**[PROVEN]** `K̄(cx,cx)=1`); but **test_r 0.79→0.59** (~25% drop, PNAS cell 8 M=100) — norm is genuinely informative | proof: `proof_kernel_solutions.tex:129-141`; test_r: `da_utility_theory.md:157`; "5× scaling keeps utility stable" is **[ASSERTED]** (folder deleted, no script) | accuracy cost |
| e | Arc-sine kernel `K_sat` | erf activation, self-variance saturates at 1 | **[PROVEN]** saturates; but "**limits, does NOT eliminate** norm-driven growth"; on PNAS `K_sat≈0.91–0.97` → behaves ~like the normalized kernel; **test_r never measured** (predicted ~0.59); trains poorly at M=50 | proof: `proof_kernel_solutions.tex:162-167`; `da_utility_theory.md:161-166`; `.claude/handoffs/HANDOFF_2026-02-10_arcsine-kernel-implementation.md:71-73,112` | untested / likely same cost |
| f | LocalRBF `K(x,x)=1` | stationary kernel | multi-image conditioning cancels gradients (utility barely moves); **test_r 0.25 vs 0.78** (M=50) | `da_utility_theory.md:168-171` | poor accuracy |
| g | `f_max` firing guard (=100) | clamp rate exploitation | a clamp on the symptom, effective for arc-cosine; not a fix of the divergence | `da_utility_theory.md:176` | mitigation only |
| h | `r_max` LUT | exact-quadrature entropy table | fixes the **numerical** `r_max=100` blow-up ONLY (§3); not the kernel-shape issue | `investigations/lucent_useful_images/standard_utility/LUT_IMPLEMENTATION_REPORT.md` | solved (numerical) |
| i | **lucent Fourier(1/f)+sigmoid** (current) | bound pixels by construction | pixels always in range (sat ≤7%, no OOB); enables **generation**; but bounds the **symptom** — see §5 | `investigations/lucent_useful_images/FINDINGS_gray_start.md`; `investigations/lucent_useful_images/README.md:38-64` | partial (this session) |

## 5. Where lucent stands (NOT the solution)

- **What it buys:** images stay in physical pixel bounds by construction (sigmoid+affine), and it
  makes actual image *generation* possible
  (`investigations/lucent_useful_images/README.md:38-64`;
  `investigations/lucent_useful_images/FINDINGS_gray_start.md` finding 5).
- **What it does NOT fix:** the §2 kernel-shape preference. lucent caps ‖x‖ at the sigmoid rail; it
  does not change *where the optimum sits* — the utility still prefers the highest-contrast /
  highest-firing image inside the bounded box. **[MEASURED]** under the SAME lucent parameterization
  the standard utility still picks a high-contrast **face** where DA picks foliage, and standard
  images are consistently higher-contrast
  (`investigations/lucent_useful_images/standard_utility/FINDINGS.md:65-69`).
- **Where "naturalness" comes from:** **both** the parameterization+natural-start (bounds the
  *pixels*) **and** the DA conditioning term (keeps the *firing/content* sensible) are load-bearing —
  NOT the objective's preference, which still favors the bounded high-firing/high-epistemic corner.
  `investigations/lucent_useful_images/standard_utility/FINDINGS.md:70-73` (the load-bearing
  correction); `investigations/lucent_useful_images/README.md:162-163`.
- **sample_lambda / n_mc (this session, secondary, orthogonal to §2):** the biased mean-λ
  conditioning (`sample_lambda=False`) ≈ the unbiased sampled-λ (`sample_lambda=True`) in the mean,
  so the cheap deterministic proxy suffices; the MC conditioning-set size `n_mc=48` is adequate for
  some cells, too small for others (≥96 safer).
  `investigations/lucent_useful_images/FINDINGS_gray_start.md` ("sample_lambda robustness").

## 6. The most promising lever: change the kernel — and the exact open sub-problem

The documented cross-kernel conclusion is the crux
(`investigations/utility/docs/da_utility_theory.md:173-176`):
> "The core issue is shared across all kernels: DA utility rewards high marginal entropy H_marg,
> which correlates with predicted firing rate… **Only the normalized kernel truly eliminates norm
> dependence, but it sacrifices predictive accuracy.**"

So the evidence defines a **tension**, not yet a solution:
- Kernels with `K(x,x) = ‖x‖²_C` (arc-cosine) → utility diverges with norm (§2). **[PROVEN]**
- Kernels with `K(x,x) = const` (normalized, RBF) or saturating (arc-sine) → divergence removed, but
  **test_r drops** because the image norm carries real encoding information (rows d, e, f).
  **[MEASURED for normalized & RBF; invariance [PROVEN]; arc-sine test_r UNMEASURED].**

**Open sub-problem (the target for a kernel fix):** bound the *utility's* dependence on ‖x‖ WITHOUT
discarding the norm information the neuron uses. The norm has **two coupled roles** — (A) it scales
the prior variance `K(x,x)` → drives the utility divergence (bad), and (B) it scales the predictive
mean `μ(x)` → carries encoding signal (good). Both currently run through the same `‖x‖_C`. The kernel
change that works must **decouple A from B**.

## 7. Open directions (PROPOSALS — unproven, for future expansion)

Flagged as proposals, not results:
1. **Decouple variance-norm from mean-norm.** A model where the predictive *mean* still uses ‖x‖
   (keep test_r) but the prior *variance* `K(x,x)` is bounded (kill the utility divergence). Direct
   read of §6's tension; no implementation exists.
2. **Arc-sine, actually measured.** Its `K(x,x)` saturation is [PROVEN] but its test_r was never
   measured (only predicted ~0.59, row e). One measured test_r would confirm/deny whether the
   saturating kernel is a dead end like the normalized one.
3. **Input warping / bounded-domain kernel.** A prior investigation exists
   (`investigations/input_warping/`, scaled-tanh, deferred per `SESSION_LOG.md` 2026-02-24); warp
   inputs so ‖x‖_C is bounded pre-kernel.
4. **Constrain to the natural-image manifold explicitly** (a learned density / diffusion prior — the
   repo has a diffusion-generator thread) rather than only bounding the box; the utility would then
   be maximized *on the manifold*, where the ‖x‖²_C blow-up cannot be reached.
5. **Rank-not-ascend.** Use the utility only to rank a bounded/natural candidate set (the current
   active-natural loop + lucent natural-start already approximate this); sidesteps §2 but forfeits
   free-form generation.

## 8. Evidence index

| Topic | File (under `scripts/gpytorch_porting/`) |
|-------|------|
| Kernel structure, scaling table, cross-kernel results | `investigations/utility/docs/da_utility_theory.md` |
| Moment/conditioning derivations; `K(x,x)=‖x‖²_C`; normalized-kernel derivation | `investigations/utility/docs/proof_moments_and_conditioning.tex` |
| Theorem 1, Corollaries 1-2, divergence exponent, numerical evidence table | `investigations/utility/docs/proof_divergence_theorems.tex` |
| Utility decomposition; normalized (Sol.1) + arc-sine (Sol.2) kernels | `investigations/utility/docs/proof_kernel_solutions.tex` |
| Subspace approaches (direction vs amplitude) | `investigations/utility/docs/subspace_theory.md`, `.../subspace_operations.md` |
| `r_max=100` numerical blow-up + LUT fix | `investigations/lucent_useful_images/standard_utility/{FINDINGS.md,LUT_IMPLEMENTATION_REPORT.md}`, `investigations/lucent_useful_images/lut/LUT_README.md` |
| Standard-vs-DA control (extreme preference is real) | `investigations/lucent_useful_images/standard_utility/FINDINGS.md` |
| lucent bounding + gray-start results (this session) | `investigations/lucent_useful_images/{README.md,FINDINGS_gray_start.md}` |
