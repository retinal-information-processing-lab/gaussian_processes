# Investigation: generated-input active learning (and the kernel that blocks it)

You are a mathematics + machine-learning research agent on an autonomous, overnight
run. You are trusted to decide how to attack this — what follows is the goal and
the few things you need to know, not a recipe.

## The goal
An **active-learning** loop: from a small training set, each iteration you choose
one input, acquire its target, add the pair to the training set, refit, repeat.
Two ways to choose the input — **select** the highest-utility one from a fixed pool
(the current baseline), or **generate** a new one by optimizing an input to
maximize the acquisition utility (not restricted to the pool). The goal is a loop
where **generate learns faster than select**: the model reaches good held-out
predictive fit with fewer queries when the inputs are synthesized to be maximally
informative rather than pool-picked. Demonstrate that with an honest evaluation —
or rigorously characterize why it cannot hold.

## Why it doesn't work yet — and where to look
Generation currently fails because the **acquisition utility diverges**. The
arc-cosine kernel is positively homogeneous in the input: `K(x,x)=xᵀCx+σ₀²` grows
with `‖x‖`, so under scaling `x*=κ x_t` you get `μ∝κ`, `σ²∝κ²`, the utility blows
up (`∝κ²`), and optimizing an input to maximize it simply drives it to the boundary
of the bounded input box (saturation) — a degenerate query. This is a *structural*
property of the arc-cosine kernel, so solving it most likely means **changing the
kernel itself** — a different kernel, or a reformulation of the prior/utility — not
a small tweak. The two obvious kernel fixes both fail: the normalized arc-cosine
kernel (`K̄=J(θ)/π`) is scale-invariant, which kills the divergence, but it fits
poorly; the arc-sin saturation kernel is likewise unsatisfactory. Scale-invariance
is the easy half; keeping good predictive fit is the hard half. A genuinely new
kernel (or idea) is probably what's needed.

## What you have
- **Reference:** `reference/unified_theory_math.pdf` — a self-contained account of
  the model (Poisson–Gaussian-process, arc-cosine kernel over a structured prior
  `C`, sparse variational inference, the entropy-reduction utility that scores
  inputs, and a guided-diffusion route to synthesizing high-utility inputs). The
  divergence is set up in the front matter and proved in Part III §8.4; the two
  failed fixes are §8.5.
- **Data:** `inputs/data.npz` (see `inputs/README.md`) — real inputs and
  integer-count targets, with held-out inputs and repeated held-out observations
  for a **noise ceiling**; judge predictive fit against that ceiling.
- **Tools:** gpytorch and torch (with a GPU) for building and fitting the model;
  tectonic if you want to compile a writeup.

## Worth knowing
- A *generated* input has no target in the data, so to score generation and run the
  loop you will need an **oracle** — a reference model fit on the full data, used as
  a simulated ground truth that returns a target for any input. A fair
  pool-vs-generate comparison hinges on setting this up honestly.
- Don't take the reference on faith: re-derive anything you rely on and reproduce
  its numbers before building on them.
- Use subagents/teammates freely, work anywhere in the container, and commit as you
  go (git is set up, no remote). You will likely iterate several times — a reusable
  test pipeline built early will pay off.
