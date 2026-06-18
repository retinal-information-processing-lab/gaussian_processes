"""Torch-differentiable twin of the LUT selection utility (lut_select.LUTUtility).

WHY THIS EXISTS
---------------
The lucent optimization gradient-ascends the standard utility U = H_marg - E[H_noise]
w.r.t. the image. The LIVE utility (acquisition.standard_utility -> utils.nd_utility_new)
sums the Poisson count to a fixed r_max=100; past the ~4-sigma gate the `log r!`
cancellation fails and U blows up to tens of thousands of nats (LUT_README.md, trap 2),
corrupting BOTH the value and the gradient. A precomputed, numerically-stable LUT of this
exact utility already exists (lut.npz, vendored here); its shipped reader
`lut_select.LUTUtility` is numpy/scipy (RegularGridInterpolator) and is NOT differentiable.

This module ports that reader to torch so the same numerically-stable U is usable inside
the optimization with correct gradients. It MIRRORS `lut_select.LUTUtility` exactly
(validated to <=1e-5 by test_lut_torch.py) — same clamp, same bilinear, same out-of-domain
high-rate fallback — and adds nothing new mathematically.

MATH / CODE MAP (U on a grid U[i,j] over mu_axis[i] x s2_axis[j])
-----------------------------------------------------------------
  mu_g     = A*lambda_mean + lambda0     log-firing-rate posterior MEAN  (clamped to [-6, 6])
  sigma2_g = A^2 * var(lambda)           log-firing-rate posterior VAR   (in-domain [0, 6])
  U(mu_g, sigma2_g) = H_marg - E[H_noise]   per-image information gain (nats)

  In-domain: bilinear interpolation of the 4 surrounding grid nodes. The gradient flows
  through the fractional weights tx (mu) and ty (sigma2); the node values are constants.
    tx = (mu_c - mu0) / (mu1 - mu0),  ty = (s2_c - s20) / (s21 - s20)
    U = (1-tx)(1-ty)U00 + tx(1-ty)U10 + (1-tx)ty U01 + tx ty U11
  mu_axis is UNIFORM, s2_axis is LOG-SPACED; torch.searchsorted brackets both.

  Out-of-domain sigma2 > 6 (the active-learning frontier): CONTINUITY-CORRECTED high-rate
  fallback, mirroring LUTUtility:
    U = U(mu_c, 6)  +  [ h(mu_c, sigma2_g) - h(mu_c, 6) ],   h(mu, s2) = 0.5*log(1 + e^mu*s2)
  h() is the true high-rate asymptote of U; anchoring at the s2=6 LUT edge and adding only
  the asymptote INCREMENT makes it continuous (increment=0 at the seam) and monotone in
  both mu and sigma2 -> finite, well-defined gradient everywhere (never NaN/Inf). Past
  sigma2=6 this is a RANKING proxy: its absolute value is not a trusted entropy
  (lut_select.py), but it gives a sane gradient instead of the r_max=100 blow-up.

This is STANDARD utility only. DA-via-LUT is deferred (needs a separate H_marg fallback;
the sigma2>6 asymptote above is the asymptote of U, not of H_marg). See PROVENANCE.md and
the parent HANDOFF for scope.
"""
import numpy as np
import torch
from pathlib import Path

# --------------------------------------------------------------------------- #
# Load the vendored LUT once (numpy float64). The torch tensors are built lazily
# per (device, dtype) and cached, so the 2,501-node table is copied to a device
# at most once even though the optimization calls lut_U hundreds of times.
# --------------------------------------------------------------------------- #
_LUT_NPZ = Path(__file__).resolve().parent / "lut.npz"
_z = np.load(_LUT_NPZ)
_MU_AXIS_NP = np.ascontiguousarray(_z["mu_axis"], dtype=np.float64)   # (61,) uniform [-6, 6]
_S2_AXIS_NP = np.ascontiguousarray(_z["s2_axis"], dtype=np.float64)   # (41,) log-spaced [0, 6]
_U_NP = np.ascontiguousarray(_z["U"], dtype=np.float64)               # (61, 41) finite, U>=0

# Domain endpoints as python floats (mirror LUTUtility.mu_lo/mu_hi/s2_lo/s2_hi). Clamp to
# these so the interpolator never reads out of bounds; the *code* range is [-6, 6] x [0, 6]
# (the lut_select.py docstring's "[-2, 6]" is stale -- see PROVENANCE.md).
_MU_LO, _MU_HI = float(_MU_AXIS_NP[0]), float(_MU_AXIS_NP[-1])        # -6.0, 6.0
_S2_LO, _S2_HI = float(_S2_AXIS_NP[0]), float(_S2_AXIS_NP[-1])        #  0.0, 6.0

_cache = {}  # (device, dtype) -> (mu_axis, s2_axis, U) torch tensors on that device/dtype


def _get_lut(device, dtype):
    key = (device, dtype)
    t = _cache.get(key)
    if t is None:
        t = (torch.as_tensor(_MU_AXIS_NP, device=device, dtype=dtype),
             torch.as_tensor(_S2_AXIS_NP, device=device, dtype=dtype),
             torch.as_tensor(_U_NP, device=device, dtype=dtype))
        _cache[key] = t
    return t


def high_rate_U(mu_g, sigma2_g):
    """True high-rate asymptote of U = H_marg - E[H_noise]:  0.5*log(1 + e^mu * s2).

    Monotone in mu_g and sigma2_g. The sigma2>6 ranking proxy is built from its increment.
    Mirrors LUTUtility.high_rate_U (numpy log1p -> torch.log1p)."""
    return 0.5 * torch.log1p(torch.exp(mu_g) * sigma2_g)


def oob_fraction(sigma2_g):
    """Fraction of queries served by the high-rate fallback (sigma2 > LUT cap = 6).

    A high fraction means the run operates largely outside the verified LUT (the absolute
    utility is then a ranking proxy, not a trusted entropy). Report it, per lut_select.py."""
    s2 = torch.as_tensor(sigma2_g)
    return float((s2 > _S2_HI).float().mean())


def lut_U(mu_g, sigma2_g):
    """Differentiable LUT utility U(mu_g, sigma2_g). Torch in -> torch out; NEVER NaN/Inf.

    Mirrors lut_select.LUTUtility.__call__ exactly: clamp mu_g to [-6, 6]; bilinear in
    [0, 6]; continuity-corrected high-rate fallback for sigma2_g > 6. Differentiable w.r.t.
    BOTH inputs (and hence w.r.t. the image, since the moments are). mu_g, sigma2_g must be
    broadcastable; the result has their broadcast shape."""
    mu_g, sigma2_g = torch.broadcast_tensors(mu_g, sigma2_g)
    shp = mu_g.shape
    mu = mu_g.reshape(-1)
    s2 = sigma2_g.reshape(-1)
    mu_axis, s2_axis, U = _get_lut(mu.device, mu.dtype)
    n_mu, n_s2 = mu_axis.numel(), s2_axis.numel()

    # Clamp to the LUT domain (mu>6 unphysical / mu<-6 off-grid -> flat = zero grad past edge,
    # matching RegularGridInterpolator on pre-clamped points). s2 clamp = the interp domain;
    # the s2>6 increment below is added back separately.
    mu_c = mu.clamp(_MU_LO, _MU_HI)
    s2_c = s2.clamp(_S2_LO, _S2_HI)

    # Bracket each query into its grid cell. searchsorted(..., right=True) - 1 gives j with
    # axis[j] <= value < axis[j+1]; clamp keeps j+1 in range (and yields weight 1 at the top
    # edge). Works for the uniform mu axis and the log-spaced s2 axis identically.
    iu = (torch.searchsorted(mu_axis, mu_c, right=True) - 1).clamp(0, n_mu - 2)
    js = (torch.searchsorted(s2_axis, s2_c, right=True) - 1).clamp(0, n_s2 - 2)

    mu0, mu1 = mu_axis[iu], mu_axis[iu + 1]
    s20, s21 = s2_axis[js], s2_axis[js + 1]
    tx = (mu_c - mu0) / (mu1 - mu0)
    ty = (s2_c - s20) / (s21 - s20)

    U00, U10 = U[iu, js], U[iu + 1, js]
    U01, U11 = U[iu, js + 1], U[iu + 1, js + 1]
    U_in = ((1 - tx) * (1 - ty) * U00 + tx * (1 - ty) * U10
            + (1 - tx) * ty * U01 + tx * ty * U11)

    # Continuity-corrected high-rate fallback for sigma2 > 6. For these points s2_c is
    # clamped to 6, so U_in == LUT(mu_c, 6) == the edge anchor; we add only the asymptote
    # increment (= 0 at sigma2=6, so continuous and monotone). Uses the ORIGINAL (unclamped)
    # s2, matching LUTUtility's `s2[oob]`.
    oob = s2 > _S2_HI
    incr = high_rate_U(mu_c, s2) - high_rate_U(mu_c, _S2_HI)
    U_out = U_in + torch.where(oob, incr, torch.zeros_like(incr))
    return U_out.reshape(shp)


def lut_standard_utility(model, likelihood, x_candidates):
    """LUT-backed standard utility, shaped like acquisition.standard_utility's output.

    Drop-in for acquisition.standard_utility(...) inside the lucent optimization, but reads
    U from the numerically-stable LUT (torch-differentiable) instead of the live r_max=100
    Laplace utility. Computes the SAME log-firing moments acquisition.standard_utility does
    (lines 79-86), then swaps nd_utility_new(mu_g, sigma2_g, r_max) -> lut_U(mu_g, sigma2_g).

    Returns dict: {'utility': (N,), 'mu_g': (N,), 'sigma2_g': (N,)}. Differentiable w.r.t.
    x_candidates (the moment helper is; lut_U is). NO engine file is edited -- this only
    CALLS the read-only helper acquisition.get_gp_marginal_moments.

    Args:
        model: trained default_gpy GP model in eval mode.
        likelihood: PoissonLikelihood with .A and .lambda0.
        x_candidates: (N, n_pix) query images in GP space.
    """
    # Lazy import: keeps this module (and test_lut_torch.py) importable with only
    # numpy + torch -- no GP engine / GPU / dataset chain needed to test the interpolator.
    import acquisition
    lambda_mean, lambda_var = acquisition.get_gp_marginal_moments(model, x_candidates)
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    mu_g = A * lambda_mean + lambda0          # log-firing-rate posterior mean
    sigma2_g = A ** 2 * lambda_var            # log-firing-rate posterior variance
    utility = lut_U(mu_g, sigma2_g)
    return {"utility": utility, "mu_g": mu_g, "sigma2_g": sigma2_g}
