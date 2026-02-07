# Critical Short Rules

**Purpose**: Project-specific gotchas that have caused bugs in past sessions. Always loaded. Each rule exists because of a real mistake.

---

## Coordinates
- Always use **normalized coordinates** (0-1 range), not pixel coordinates, unless the user explicitly says otherwise.

## Variational Parameters
- Variational parameters (inducing points, variational distribution m/V) are **critical model state** — never omit them when reproducing or comparing models.

## Import Side Effects
- Importing from old codebase (1D/2D playgrounds, utility.py) can change `torch.default_dtype` and `sys.path`. Always save and restore both before/after import. See `acquisition.py` for the pattern.

## Eigenspace Conventions
- `K_tilde_b` is DIAGONAL in eigenspace — use element-wise operations, not matrix solves.
- `V_b` is NOT diagonal — it is the full projected covariance.

## Float32
- All training and evaluation uses `--float32`. Float64 is 10x slower and the reference code used float32.

## Numerical Debugging Checklist
When diagnosing numerical issues, always check these in order:
1. Kernel initialization (bounds, lengthscale)
2. Coordinate normalization (pixel vs normalized)
3. Parameter bounds (A > 0, lambda0 constraints)
4. Jitter consistency (must match model.jitter everywhere)
5. Gradient flow (no accidental detach or no_grad)
