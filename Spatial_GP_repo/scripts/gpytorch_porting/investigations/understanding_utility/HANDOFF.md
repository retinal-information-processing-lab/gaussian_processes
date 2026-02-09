# Understanding Utility Functions — Session Handoff

**Location**: `investigations/understanding_utility/`
**Branch**: `pietro/acquisition-functions`
**Purpose**: Pedagogical exploration of standard and distribution-aware utility behavior

---

## What This Folder Is

A clean, self-contained workspace for building intuition about how the two
acquisition functions (standard utility and distribution-aware utility) behave
with a trained variational GP using the arc-cosine kernel on neural data.

This is NOT a debugging or investigation folder. It is a learning workspace.

## Files

| File | Purpose |
|------|---------|
| `key_facts.md` | Distilled reference: what the two utilities are, how the kernel factorizes, what scaling does, Laplace limits. Read this FIRST. |
| `norm_scaling_analysis.tex` | Mathematical proofs: kernel homogeneity theorem, posterior moment scaling, conditioning consequences. Detailed reference — consult when the math in key_facts.md isn't enough. |
| `explore_utility.py` | Workbench script. `setup()` trains a GP model and returns everything needed. Helper functions: `kernel_angle()`, `kernel_norm()`, `gp_moments()`, `logfiring_moments()`, `describe()`. Uses `standard_utility()` and `distribution_aware_utility()` from `acquisition.py` with adaptive r_max. `eval_da_conditioned()` evaluates DA utility of all training images conditioned on one image. Run directly for a demo, or import functions for custom exploration. |

## What the User Wants to Understand

The user is a scientist building active learning systems for neural data. They
have two acquisition functions for selecting which image to show to a neuron
next. They want to develop intuition about:

1. **How the two utilities compare**: When do they agree? When do they disagree?
   What does each one actually measure?

2. **The role of "distance" in image space**: The arc-cosine kernel decomposes
   image similarity into two independent axes:
   - **Angle** (direction in C-space): captures structural similarity
   - **Amplitude** (norm in C-space): captures overall intensity/contrast

   These affect the GP posterior — and therefore the utilities — in very
   different ways. Angle determines cross-covariance (conditioning effectiveness),
   while amplitude determines variance magnitude.

3. **Why standard utility doesn't care about angle**: It only looks at marginal
   GP uncertainty, which grows with amplitude regardless of direction.

4. **Why DA utility depends strongly on angle**: The conditioning step (which
   makes DA "distribution-aware") is only effective when the query image and
   the conditioning images are directionally aligned. At large angles,
   conditioning does nothing and DA utility collapses to ~0.

5. **Practical implications**: In a pool of natural images (bounded amplitudes,
   diverse angles), how different are the two utility rankings?

## Key Concepts to Build On

The arc-cosine kernel has the form:

    K(x, y) = (1/pi) * ||x||_C * ||y||_C * J(angle)

This means:
- Scaling an image x by c multiplies all kernel values by ~c (linear homogeneity)
- The GP posterior mean scales as c, variance as c^2
- The angle between scaled and unscaled versions stays constant
- Conditioning effectiveness depends on angle, not amplitude

The DA utility conditions on "what would we learn about the response to x_sample
if we observed a spike count at x_star?" This turns into Gaussian conditioning
on the GP posterior joint [x_star, x_sample], where the effectiveness is
controlled by the posterior correlation rho^2, which depends on the kernel angle.

## Technical Notes for the Session

- **Model training**: `setup()` in `explore_utility.py` trains a default_gpy
  model (M=50, n_train=50, seed=42, cell=8). Takes ~3 seconds on GPU.
  All params from `default_params.json`.

- **Adaptive r_max**: All utility computations use adaptive r_max from
  `compute_adaptive_rmax()` in `utils.py`. Raises ValueError if the needed
  r_max exceeds max_rmax. No more fixed r_max=100 truncation issues.

- **The `describe()` function**: Prints a complete summary of any image's GP
  properties, utility values, and (optionally) relationship to a reference image.
  Use this liberally for exploration.

---

## Continuation Prompt

```
I want to build intuition about how utility functions work for active learning
with neural data. I have a workspace at:

  investigations/understanding_utility/

Read these files in order:
1. HANDOFF.md (this file — context and goals)
2. key_facts.md (distilled reference)
3. explore_utility.py (workbench script with helpers and a demo)

Then start a pedagogical exploration session. Run the demo first:

  python investigations/understanding_utility/explore_utility.py

After seeing the demo output, walk me through what the numbers mean. Then
let's explore specific scenarios together to build intuition:

- Compare natural images at different angles from a reference
- Show what scaling does to each utility (and why)
- Find cases where standard and DA utility disagree about which image is best
- Show when conditioning actually matters (and when it doesn't)

Work interactively: run code, show me the numbers, explain what they mean.
Ask me questions to check my understanding. Keep it pedagogical.
```
