# How lucent works (and how we use it)

A plain-language explainer. lucent is the only "new" tool in this investigation; the
rest is our existing GP code.

## What lucent is

`lucent` is a PyTorch library for **feature visualization** — the technique behind the
"what does this neuron see" images from Distill/OpenAI. Its original job: find the input
image that maximally excites a chosen neuron in a deep network. **We borrow only its
image-generation machinery, not the neural-network part** — we swap in our GP utility as
the thing to maximize. We don't pip-install it (its old `kornia` pin would damage the env);
we just import three small, pure-PyTorch files from a local clone.

## The core idea: optimize a *recipe* for the image, not the pixels

The naive approach treats the 108×108 = 11,664 pixels as free variables and does gradient
ascent on them directly. That is exactly what failed before: with nothing holding them
back, the pixels run to extreme values (max contrast) and to the borders, because that is
what the utility rewards. It is like letting an optimizer draw with a pen that can press
infinitely hard — it scribbles noise.

lucent instead optimizes a **parameterization**: a recipe that turns a set of hidden
parameters into an image, with good behavior baked into the recipe so that *whatever* the
optimizer does, the image it produces is well-behaved. Two ingredients:

## Ingredient 1 — build the image from frequencies (Fourier parameterization)

Instead of storing pixel values, lucent stores the image's **Fourier coefficients** (how
much of each spatial frequency — smooth gradients vs fine checkerboards — the image
contains). To get the actual image it runs an inverse FFT.

The trick: before the inverse FFT, each frequency is multiplied by `1 / f^decay_power`.
Low frequencies (smooth, large-scale structure) get a big weight; high frequencies (fine
noise) get a tiny weight. This is the **"1/f" spectrum that natural images actually have**.
The effect: the optimizer finds it *cheap* to add smooth structure and *expensive* to add
high-frequency noise, so it naturally produces smooth, natural-looking images instead of
static. `decay_power` is the knob: 1.0 = the natural 1/f; higher = even smoother (but too
high saturates the start — see README).

(This is why the old "optimize the PCs" / "optimize the top eigenvectors of C" attempts
were on the right track — they also restricted the image to smooth / low-dimensional
structure. The Fourier prior does the same thing, but gently and without a hard cutoff.)

## Ingredient 2 — a guardrail on pixel values (sigmoid)

After the inverse FFT, lucent passes every pixel through a **sigmoid**, which squashes any
number into the open range (0,1). No matter how large the Fourier coefficients grow, the
output pixels are always strictly between 0 and 1 — they can never blow up or sit exactly
on a border. We then linearly map (0,1) onto our physical display range
`[GP_MIN, GP_MAX] = [-2.40, +2.48]` (the dataset's true min/max).

This is the part that solves the overblow problem **by construction**: the image is
bounded not because we clip it (which would hide the problem) but because the sigmoid
mathematically *cannot* output an out-of-range value. And because the sigmoid is smooth
(not a hard clamp), gradients still flow nicely — no dead zones.

## How we plug in our utility

We never touch lucent's neural-network code. We:

1. Ask lucent for the image recipe: `params, image_fn = fft_image(...)` then
   `to_valid_rgb(...)` (the sigmoid).
2. In our own loop: `img = image_fn()` → affine-map to our pixel range → feed to our GP's
   **distribution-aware utility** → `loss = -utility` → `loss.backward()`.
3. The gradient flows all the way back: utility → pixels → sigmoid → inverse-FFT → Fourier
   coefficients. Adam updates the coefficients. Repeat.

So the optimizer searches over "which mix of smooth frequency components, kept in a
bounded range, maximizes the neuron's information utility" — instead of "which 11,664 raw
pixel values." That single change turns scribbled high-contrast noise into bounded,
natural-looking stimuli.

## The honest caveat

lucent's prior makes the image **smooth and bounded**; it does not by itself make it
**belong to the natural-image distribution**. The remaining naturalness in our figures
comes from combining lucent's prior with (a) the distribution-aware utility's angular term
and (b) starting the optimization from the best available natural image, so the parts of
the image *outside* the receptive field stay as a real scene (the kernel gives them no
gradient). lucent is the piece that makes the search bounded and smooth; the start image
and the DA utility make it natural.
