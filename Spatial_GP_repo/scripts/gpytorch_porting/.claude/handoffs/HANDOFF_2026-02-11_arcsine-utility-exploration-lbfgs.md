# Handoff: Arc-Sine Kernel Utility Exploration + LBFGS Gradient Ascent

**Branch**: `pietro/arcsine-kernel`
**Date**: 2026-02-11
**Status**: Partial implementation — explore/gradient scripts created and verified, LBFGS upgrade planned but not yet implemented
**Plan file**: `.claude/plans/merry-nibbling-sutton.md`

---

## Motivation

The arc-sine kernel (Williams 1998) was implemented in previous sessions as an alternative to the arc-cosine kernel. Its key property is that K(x,x) saturates at 1 for large inputs, preventing the quadratic growth of the arc-cosine kernel. This session created utility exploration and gradient ascent scripts for the arc-sine kernel (analogous to existing scripts for the normalized and unnormalized arc-cosine kernels) and began planning an LBFGS optimizer upgrade to replace the slow plain gradient ascent.

The LBFGS upgrade is the pending work. The current `gradient_arcsine.py` uses plain steepest ascent (`x += lr * grad`) which took 5000 steps to reach only 92% of initial distance in 11,664 dimensions. LBFGS with strong Wolfe line search should converge in 20-50 steps by approximating curvature from gradient history.

## Decisions and Rationale

### 1. Template choice for explore script: arc-cosine (not normalized)

The `explore_utility_arcsine.py` was based on `investigations/understanding_utility/explore_utility.py` (unnormalized arc-cosine), NOT on `explore_utility_normalized.py`. The user directed this because:
- Arc-sine kernel has varying norms (like arc-cosine), unlike the normalized kernel where ||x||_C = 1 always
- The arc-cosine explore script has "U_DA vs norm" as its right panel, which is the informative plot for arc-sine
- The normalized version has more complex image grouping (synthetic, special markers) that isn't needed here

### 2. Template choice for gradient script: normalized kernel

The `gradient_arcsine.py` was based on `gradient_normalized.py`. The user confirmed this template works well — it has the interpolation sweep, gradient ascent, and multi-panel visualization.

### 3. Import chain: use run_arcsine.py (same directory)

Both new scripts import from `run_arcsine.py` which lives in the same `investigations/arcsine_kernel/` directory. This works because Python adds the script's directory to `sys.path[0]` automatically. The key import is `from run_arcsine import run_single_config, build_config_from_defaults, load_pnas_data` — `run_arcsine.py`'s `run_single_config` uses `ArcSineKernel` internally (the kernel swap).

### 4. M=50, N_TRAIN initially 50 then changed to 300

The scripts were created with M=50, N_TRAIN=50 matching the arc-cosine template. The user then manually changed `explore_utility_arcsine.py` to N_TRAIN=300, which improved results significantly (the utility landscape was more informative with a better-trained model). The gradient script imports `setup()` from the explore script, so it automatically inherits N_TRAIN=300.

### 5. LBFGS over Adam for gradient ascent

Discussed three options:
- **Plain GD** (current): Too slow — 5000 steps, 92% distance remaining
- **Adam**: Adaptive per-pixel LR, more robust to noise, but no curvature information
- **LBFGS**: Approximates Hessian from gradient history (10 vectors of size 11,664 = ~2MB), strong Wolfe line search. Best fit because the utility is smooth and differentiable.

The user chose LBFGS with strong Wolfe and asked to expose `max_iter` and `max_eval` as tunable parameters.

### 6. Output image naming convention

User specified: scripts must produce images with the same name as themselves. So:
- `explore_utility_arcsine.py` → `explore_utility_arcsine.png`
- `gradient_arcsine.py` → `gradient_arcsine.png`

### 7. Deferred investigation: fractional distance paradox

With N_TRAIN=300, the user observed that the optimized image VISUALLY gets closer to the target, but the fractional distance metric INCREASES. This is marked as a deferred investigation — the metric may not capture what matters (e.g., the optimizer improves pixels within the RF while the overall L2 distance grows due to changes outside it, or the utility landscape has structure that doesn't align with pixel distance).

## Critical Subtleties

### LBFGS closure must negate utility

LBFGS minimizes. To maximize utility, the closure must return `-utility` and call `.backward()` on that. If you forget the negation, LBFGS will minimize utility (move away from target).

### Post-step evaluation must be separate from closure

The closure is called multiple times per LBFGS step (line search). Values captured inside the closure may reflect intermediate evaluations, not the final accepted step. Track metrics by evaluating utility in a separate `torch.no_grad()` block AFTER `optimizer.step(closure)`.

### The explore script's N_TRAIN was changed by the user

`explore_utility_arcsine.py` line 88: `N_TRAIN = 300` (was 50 when created). The gradient script imports `setup()` from it and inherits this value. Don't revert to 50.

### gradient_arcsine.py has NOT been modified yet

The LBFGS changes described in the plan file have NOT been applied. The script still uses plain gradient ascent. The plan is approved and ready to implement.

## Uncommitted Changes

```
Modified (binary, from previous sessions):
  imgs/default_gpy_M100.png          — regenerated during earlier verification
  imgs/default_gpy_M50.png           — regenerated with STA+RF visualization
  imgs/vargp_direct_M100.png         — regenerated during earlier verification
  imgs/vargp_direct_M50.png          — regenerated with STA+RF visualization
  investigations/arcsine_kernel/imgs/arcsine_default_gpy_M100.png — updated

Untracked (this session):
  investigations/arcsine_kernel/explore_utility_arcsine.py   — NEW: utility exploration for arc-sine
  investigations/arcsine_kernel/explore_utility_arcsine.png  — output from above
  investigations/arcsine_kernel/gradient_arcsine.py          — NEW: gradient ascent for arc-sine (LBFGS NOT yet applied)
  investigations/arcsine_kernel/gradient_arcsine.png         — output from above

Untracked (from previous sessions, not this work):
  experiments/exploratory/2026-02-07_*                        — jitter investigation experiments
  investigations/arcsine_kernel/imgs/arcsine_vargp_direct_M100.png
  investigations/normalized_kernel/*.png                      — normalized kernel investigation images
```

## Files to Read First

1. **`.claude/plans/merry-nibbling-sutton.md`** — The approved LBFGS plan. Contains exact code for the new `gradient_ascent()` function, parameter values, and call site changes. This is what needs to be implemented.

2. **`investigations/arcsine_kernel/gradient_arcsine.py`** — The file to modify. Currently has plain gradient ascent at lines 95-146 and call site at line 271-272. The LBFGS plan replaces these.

3. **`investigations/arcsine_kernel/explore_utility_arcsine.py`** — The companion explore script. Note N_TRAIN=300 at line 88 (user changed from 50). The gradient script imports `setup()` from here.

4. **`investigations/arcsine_kernel/HANDOFF.md`** — Previous handoff covering arc-sine kernel implementation, LBFGS stability fixes, beta/center issues. Provides broader context.

## Caveats and Open Questions

1. **LBFGS parameter defaults are educated guesses**: LR=1.0, max_iter=20, max_eval=25, history_size=10 are standard quasi-Newton values but may need tuning for this specific utility landscape. The user explicitly asked for these to be exposed as tunable parameters.

2. **Each LBFGS closure evaluation is expensive**: It computes the full DA utility (GP conditional moments + Laplace approximation + entropy). With max_eval=25 per step and 50 steps, that's up to 1250 utility evaluations. Compare with 5000 for plain GD — similar total cost but much better convergence expected.

3. **Fractional distance paradox (deferred)**: With N_TRAIN=300, the optimized image looks closer to the target visually but frac_dist increases. Needs investigation — could be that the optimizer improves RF-relevant pixels while L2 distance grows from changes elsewhere.

4. **M=50 arc-sine still trains poorly**: LBFGS for model training gets stuck after ~3 productive iterations (known issue from previous sessions). The utility exploration works despite this, but results may improve with M=100.

---

## Continuation Prompt

```
I'm continuing work on the arc-sine kernel investigation, branch pietro/arcsine-kernel.

Read these files first:
1. `.claude/plans/merry-nibbling-sutton.md` — APPROVED plan for LBFGS gradient ascent
2. `.claude/handoffs/HANDOFF_2026-02-11_arcsine-utility-exploration-lbfgs.md` — full context
3. `investigations/arcsine_kernel/gradient_arcsine.py` — file to modify (plain GD → LBFGS)

The plan is approved and ready to implement. Key points:
- Replace plain gradient ascent with torch.optim.LBFGS + strong_wolfe
- Expose max_iter, max_eval, history_size as constants at top of file
- Closure returns -utility (minimize negative = maximize)
- Track metrics AFTER optimizer.step(), not inside closure
- explore_utility_arcsine.py has N_TRAIN=300 (user changed from 50)

Check git status and git branch before starting.
After implementing, run the script to verify it works.
```
