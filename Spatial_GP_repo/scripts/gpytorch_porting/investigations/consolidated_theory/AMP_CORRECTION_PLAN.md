# Amp correction plan — promote Amp to a genuine kernel hyperparameter

**Why.** The original consolidation brief instructed excluding the `Amp`
parameter as `vargp_old` baggage. That instruction was wrong: in the CURRENT
code, `Amp` is a live, **optimized-by-default** kernel hyperparameter. The
document currently (incorrectly) presents it as excluded / frozen / a no-op. This
plan promotes `Amp` to a first-class 6th kernel hyperparameter, systematically and
coherently, so the correction does not become a whack-a-mole of half-fixes.

## Ground truth (verified in code)

- `kernels.py:216` registers `raw_Amp` as an `nn.Parameter` ⇒ `requires_grad=True`
  by default (trainable).
- `kernels.py:219` `Positive()` constraint = **softplus**; `kernels.py:285`
  transform; `AMP_MAX = 1000`.
- `kernels.py:554` `C = Amp * alpha[:,None] * C_smooth * alpha[None,:]`.
- `eigenspace_training.py:181` `fix_Amp: bool = False` (the default).
- `eigenspace_training.py:239-242` freezes Amp **only if** `fix_Amp` is True
  (comment: "Freeze Amp if requested (paper's code has no Amp parameter)").
- `eigenspace_mstep.py:324` `kernel.raw_Amp.grad = dL['Amp'] * sigmoid(raw_Amp)`
  — the gradient is computed and applied.
- `eigenspace_training.py:284,444` log `param_Amp_curve` per iteration — only
  meaningful because Amp is being optimized.

**Conclusion:** by default Amp is optimized (a 6th kernel hyperparameter);
`fix_Amp=True` freezes `Amp=1` to reproduce the paper's Amp-free model.

## The single invariant to enforce everywhere

> `C = Amp · α ⊙ C_smooth ⊙ αᵀ`, with `Amp` a softplus-constrained kernel
> hyperparameter (bounds (0, 1000], default 1.0, **optimized by default**). The
> kernel/M-step hyperparameter set has **SIX** members
> `{σ₀, Amp, β, ρ, ε₀ₓ, ε₀ᵧ}`. No "frozen", "no-op", or "excluded" framing for
> Amp anywhere. `fix_Amp=True` is the opt-in that reproduces the paper.

Because `C` is used **abstractly** in Parts III–IV (self-kernel `K(x,x)=xᵀCx+σ₀²`,
magnitude `v_x`, `∇ₓK`, the divergence/subspace proofs), defining `C` with `Amp`
at `eq:Cprior` propagates automatically — **Parts III and IV need NO edits**. The
divergence proofs are unchanged (Amp is a positive constant w.r.t. `x`, so
`K(x,x)` stays homogeneous of degree 2 in `x`).

## Touchpoints (exhaustive) — file : line : current → intended

### Part I — `sections/part1_model_kernel.tex`
- **T1. `eq:Cprior` (256–263) + Definition preamble (254–255).** Add the `Amp`
  prefactor: `C_ij = Amp · α_i [C_smooth]_ij α_j = Amp · exp(...)·exp(...)·exp(...)`.
  Introduce `Amp` in the preamble ("an amplitude `Amp`, a per-pixel locality
  weight α_i, and a pairwise smoothing factor C_smooth").
- **T2. Physical-meaning text (250, 286+).** Add `Amp`: it scales the overall
  prior variance of the latent (a gain on the structured covariance); a
  hyperparameter alongside β (RF width), ρ (smoothness), ξ₀ (RF center).
- **T3. `∂C/∂θ` Proposition `eq:dCtheta` (~445–470).** (a) Note `C = Amp·(…)`, so
  the shape-parameter gradients ∂C/∂β, ∂C/∂ρ, ∇_{ξ₀}C each carry the factor `Amp`;
  (b) ADD `∂C/∂Amp = α ⊙ C_smooth ⊙ αᵀ = C/Amp`. Update any "five" language here.
- **T4. Bounds table (334–347), Amp row (344).** Remove "(footnote; excluded)".
  Present `Amp`: constraint `Amp = softplus(raw_Amp)`, bounds natural `(0,1000]`,
  optimized. Keep the σ₀/β/ρ/ξ₀ rows.
- **T5. Amp footnote (273–285).** DISSOLVE. Amp now lives in the main C definition
  and the hyperparameter set; the "frozen/no-op/excluded" footnote is deleted. Its
  only surviving content — that the code multiplies C by Amp — is now in `eq:Cprior`.
- **T6. Code-comment remark (305, `C = Amp*alpha*...`).** Already correct — keep.

### Part II training — `sections/part2b_training.tex`
- **T7. Parameter set (312–317).** `{σ₀, β, ρ, ε₀ₓ, ε₀ᵧ}` → `{σ₀, Amp, β, ρ,
  ε₀ₓ, ε₀ᵧ}` (SIX); add `raw_Amp` to the raw-form list; "genuine free
  hyperparameters" text (310) updated accordingly.
- **T8. Count language.** "the free block has five scalars" (61) → six; "its five
  gradients ∂_θC" (359) → six; "materialising five ∂_θK" (471) → six. Overview
  (24, 78) consistent.
- **T9. Amp footnote (319–326).** REMOVE the "frozen / held fixed / no-op /
  Amp-free" footnote. Replace with a Remark near the M-step: *Amp is optimized by
  default; `fix_Amp` (default False) can freeze `Amp=1` to reproduce the paper's
  Amp-free model.*
- **T10. Constraint-transform caveat (328–331).** Keep the correct fact (σ₀=exp,
  Amp=softplus, the handoff's "softplus for sigma_0" is wrong), but it now
  supports Amp's REAL gradient rather than a frozen one — reword if it implies
  frozen.
- **T11. Line 425 "Only σ₀ (and the frozen Amp)".** → both σ₀ (`exp`) and Amp
  (`softplus`) carry a non-identity constraint Jacobian; drop "frozen".
- **T12. Gradient formulas (465–466).** Already carry ∂J/∂Amp and the Amp factor
  in ∂J/∂α — keep; they now correctly describe an optimized parameter.

### Appendix — `sections/appendix.tex`
- **T13. Notation row (34).** "amplitude, default 1.0 (frozen; a no-op)" →
  "amplitude — kernel hyperparameter, softplus-constrained, bounds (0,1000],
  optimized by default".
- **T14. Constraint-transforms item (96–98).** Correct — keep.
- **T15. "Frozen amplitude" reconciled-discrepancy item (99–103).** REMOVE. It was
  the mistake, not a code-vs-doc discrepancy. (The `fix_Amp` note now lives in
  Part II per T9.) Optionally add, elsewhere in App. B open items, a one-liner that
  `fix_Amp=True` reproduces the paper.

### Intermediate specs (secondary but part of "systematic")
- **T16. `00_understanding_memo.md`** §4.5 item 18 and §5 (scope exclusions):
  correct the Amp entry — it is a genuine optimized hyperparameter, not frozen.
- **T17. `01_outline.md`** any Amp "excluded/footnote" note → 6th hyperparameter.

## Coherence checks the executor MUST run before declaring done
1. `grep -rin "frozen\|no-op\|excluded" sections/ | grep -i amp` → **empty**
   (except a deliberate `fix_Amp` reproduction note).
2. `grep -rn "five" sections/part2b_training.tex` → no "five" referring to the
   hyperparameter/gradient count (all six).
3. The hyperparameter set reads `{σ₀, Amp, β, ρ, ε₀ₓ, ε₀ᵧ}` identically wherever
   it is listed.
4. `eq:Cprior` shows the `Amp` prefactor; `eq:dCtheta` includes `∂C/∂Amp`.
5. Parts III (`part3b`) and IV (`part4`) **unchanged** (verify `git diff`/no edits;
   the "frozen" tokens there refer to β during synthesis and the E-step inputs —
   do NOT touch them).
6. `tectonic unified_theory.tex` compiles clean (0 errors, 0 undefined refs).

## Execution model
A single executor teammate applies T1–T17 in one coherent pass (it owns the whole
Amp thread, so the count and framing stay consistent). A verifier teammate then
independently runs the 6 coherence checks and reads the diff for any missed
"frozen/excluded/five" residue or broken logic. Orchestrator compiles and does a
final visual check.
