"""
Lucent-based image parameterization for GP utility optimization.

We use ONLY lucent's parameterization layer (pure torch+numpy, no install needed):
  - fft_image: Fourier/spectral parameterization. The optimizable parameters are
    the real+imag FFT coefficients, each scaled by 1/max(freq, .)**decay_power
    before the inverse FFT. This biases the optimizer toward LOW-frequency
    (smooth, natural-spectrum) images and suppresses high-frequency checkerboard
    noise. decay_power=1 -> the natural-image 1/f spectral prior.
  - to_valid_rgb(..., decorrelate=False): maps the image through a sigmoid, so the
    output lives strictly in (0,1) -- bounded WITHOUT any pixel clipping.

The (0,1) lucent image is then affine-mapped into the GP's normalized input space
[GP_MIN, GP_MAX] (the dataset global min/max). Because sigmoid guarantees (0,1),
the affine guarantees (GP_MIN, GP_MAX): the synthesized image can NEVER exceed the
physical display range. No clipping, no rescaling, no bounded-optimization solver.

Math-to-code:
  img01 = sigmoid( irfft( scale(decay_power) * fft_params ) / magic )   in (0,1)
  img_gp = GP_MIN + img01 * (GP_MAX - GP_MIN)                            in (GP_MIN, GP_MAX)

Everything is differentiable w.r.t. fft_params, so gradients flow from the GP
utility all the way back to the spectral coefficients.
"""
import os
import sys
import numpy as np
import torch

# Make the cloned lucent importable without installing it (avoids the kornia<=0.4.1
# pin that would damage the shared conda env). Import only the leaf param modules,
# which depend on torch+numpy alone.
_LUCENT_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lucent_repo")
if _LUCENT_REPO not in sys.path:
    sys.path.insert(0, _LUCENT_REPO)

from lucent.optvis.param.spatial import fft_image, pixel_image, rfft2d_freqs  # noqa: E402
from lucent.optvis.param.color import to_valid_rgb                            # noqa: E402

W_DEFAULT = 108  # PNAS images are 108x108
MAGIC = 4.0      # lucent's fft_image saturation-reducing divisor (spatial.py)


def fft_params_from_image01(target01, decay_power=1.0, device="cuda", eps=1e-4):
    """Invert lucent's fft_image to find spectral coefficients reconstructing target01.

    lucent forward:  img = sigmoid( irfftn(scale * spectrum, norm='ortho') / MAGIC )
    so to start AT a given image t in (0,1):
        pre      = logit(t) * MAGIC                       # the irfftn output
        scaled   = rfftn(pre, norm='ortho')               # complex (H, W//2+1)
        spectrum = scaled / scale                          # undo the 1/f^p scaling
    Returns a (1,1,H,W//2+1,2) real tensor to assign into params[0].data.
    """
    t = torch.as_tensor(target01, dtype=torch.float32, device=device)
    t = t.clamp(eps, 1.0 - eps)                      # keep logit finite at the rails
    H, W = t.shape
    pre = torch.logit(t) * MAGIC                      # (H, W)
    scaled = torch.fft.rfftn(pre, s=(H, W), norm="ortho")   # (H, W//2+1) complex
    freqs = rfft2d_freqs(H, W)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(W, H)) ** decay_power
    scale_t = torch.tensor(scale, dtype=torch.float32, device=device)
    spectrum = scaled / scale_t                       # (H, W//2+1) complex
    return torch.view_as_real(spectrum)[None, None]   # (1,1,H,W//2+1,2)


def make_grayscale_param(W=W_DEFAULT, decay_power=1.0, sd=0.01, seed=0, fft=True,
                         start_image01=None):
    """Build a lucent grayscale image parameterization.

    Args:
        W: image side length (108).
        decay_power: spectral 1/f^p prior strength. 1.0 = natural 1/f; higher =
            smoother, but couples with `sd` (high p needs small sd or the init
            saturates). Only used when fft=True.
        sd: init std of the spectral coefficients (lucent default 0.01 -> near-gray
            start at decay_power=1).
        seed: torch RNG seed for the random init (reproducibility).
        fft: True -> Fourier parameterization (spectral prior). False -> raw pixel.
        start_image01: optional (W,W) array in (0,1). If given (fft only), the
            spectral coefficients are initialized to reconstruct this image, so
            optimization starts FROM it (e.g. the most-useful natural image)
            instead of from near-gray noise.

    Returns:
        (params, image01_fn): params is a list with one leaf tensor (requires_grad);
        image01_fn() -> (1, 1, W, W) tensor in (0,1), differentiable w.r.t. params.
    """
    torch.manual_seed(seed)
    shape = [1, 1, W, W]
    if fft:
        params, raw = fft_image(shape, sd=sd, decay_power=decay_power)
        if start_image01 is not None:
            init = fft_params_from_image01(start_image01, decay_power=decay_power,
                                           device=params[0].device)
            with torch.no_grad():
                params[0].copy_(init)
    else:
        params, raw = pixel_image(shape, sd=sd)
    image01_fn = to_valid_rgb(raw, decorrelate=False)  # sigmoid -> (0,1)
    return params, image01_fn


def make_gp_image_fn(image01_fn, gp_min, gp_max):
    """Wrap a (0,1) image fn into one that returns a (1, W*W) image in GP space.

    The affine map [0,1] -> [gp_min, gp_max] is linear, so gradients flow cleanly.
    """
    span = float(gp_max) - float(gp_min)
    gp_min = float(gp_min)

    def gp_fn():
        img01 = image01_fn()                 # (1,1,W,W) in (0,1)
        img_gp = gp_min + img01 * span       # affine -> (gp_min, gp_max)
        return img_gp.reshape(1, -1)         # (1, W*W)

    return gp_fn


if __name__ == "__main__":
    # Self-test: build a param, map to GP space, check ranges + gradient flow.
    GP_MIN, GP_MAX = -2.401252, 2.478047
    params, img01_fn = make_grayscale_param(seed=0, decay_power=1.0)
    gp_fn = make_gp_image_fn(img01_fn, GP_MIN, GP_MAX)
    x = gp_fn()
    loss = -(x ** 2).mean()
    loss.backward()
    print("img_gp shape", tuple(x.shape))
    print("img_gp range  [%.4f, %.4f]  (bounds [%.4f, %.4f])"
          % (float(x.min()), float(x.max()), GP_MIN, GP_MAX))
    print("param grad ok:", params[0].grad is not None,
          "grad norm: %.4e" % float(params[0].grad.norm()))

    # Round-trip test: init params from a target image, image01_fn() should match it.
    torch.manual_seed(1)
    target01 = torch.rand(108, 108, device="cuda")  # arbitrary (0,1) image
    # smooth it a bit so logit is well-conditioned (natural images are not white noise)
    k = torch.ones(1, 1, 5, 5, device="cuda") / 25.0
    target01 = torch.conv2d(target01[None, None], k, padding=2)[0, 0].clamp(0.02, 0.98)
    for dp in (1.0, 1.5):
        p2, fn2 = make_grayscale_param(decay_power=dp, start_image01=target01)
        recon = fn2().detach()[0, 0]
        err = float((recon - target01).abs().max())
        print(f"round-trip dp={dp}: max|recon-target|={err:.2e}")
