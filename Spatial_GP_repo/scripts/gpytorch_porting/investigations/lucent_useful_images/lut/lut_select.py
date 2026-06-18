"""Universal LUT-based active-selection utility, with a documented high-rate fallback.

This is the UNIVERSAL selection-utility mechanism for offline active refits — it is
engine- and env-agnostic (pure numpy + scipy) and is intended to be reused across
August (varGP), April (gpytorch-port), and future online experiments. The caller's
engine supplies the LOG-RATE posterior moments per candidate image:

    mu_g     = A * lambda_mean + lambda0          (A, lambda0 = CURRENT-iteration
    sigma2_g = A**2 * var(lambda)                  PoissonLikelihood params; they drift)

and this returns U for ranking / argmax. The utility is the ground-truth per-image
information gain U = H_marg - E[H_noise], read from the precomputed LUT (lut.npz);
see LUT_README.md for what the LUT is and why it replaces the online r_max=100 utility.

OUT-OF-DOMAIN POLICY — HIGH-RATE ANALYTIC FALLBACK (this is THE design choice; stated
loudly because it determines what an active loop picks at its uncertainty frontier):

  The LUT is built only on  mu_g in [-2, 6],  sigma2_g in [0, 6]  (H_marg quadrature is
  infeasible past sigma2=6; mu>6 is unphysical, lambda>400 spikes). An active learner
  spends much of its time AT sigma2>6 (the most-uncertain images = highest info gain),
  so we must NOT return NaN/drop them (that would be anti-active). Instead:

    * sigma2_g > 6  ->  CONTINUITY-CORRECTED high-rate fallback:
          U = LUT(mu_g, 6)  +  [ h(mu_g, sigma2_g) - h(mu_g, 6) ],
          h(mu, s2) = 0.5 * log(1 + e^{mu} * s2)   (the high-rate asymptote of U).
        I.e. anchor at the LUT's sigma2=6 edge value and add only the asymptote's
        INCREMENT past it. The raw asymptote h() alone is the true high-rate limit of
        U=H_marg-E[H_noise] (HANDOFF...), but it does NOT meet the LUT at sigma2=6 for
        moderate mu (a ~0.29 jump, and it dips -> non-monotone at the seam, which would
        mis-rank frontier images). The edge-anchored increment is continuous at the seam
        (increment=0 there) and monotone in BOTH mu_g and sigma2_g, so it preserves the
        ranking of high-uncertainty images across the frontier -- all the argmax needs.
        Used ONLY as a RANKING proxy; absolute value is not trustworthy past sigma2=6.
    * mu_g outside [-2, 6]  ->  clamped to the edge (mu_g>6 unphysical; mu_g<-2 rare).

  => U is defined EVERYWHERE (never NaN). The active argmax never silently drops the
  frontier images. Callers should still COUNT how many queries used the fallback
  (`oob_mask`) and report it, since a high fallback fraction means the run is operating
  largely outside the verified LUT (expected early in a refit; rarer for well-trained
  anchors).

Interpolation: LINEAR (validated one-sided/ranking-preserving, no overshoot). The LUT
node values are systematically LOW by <=0.011 (quad) / <=0.024 (corner) but smooth and
one-signed -> argmax/ranking UNAFFECTED (ground_truth_precision/REPORT.md).

Usage:
    from lut_select import LUTUtility
    lut = LUTUtility()
    U = lut(mu_g, sigma2_g)                 # vectorised; never NaN
    n_oob = int(lut.oob_mask(sigma2_g).sum())   # frontier (sigma2>6) usage to report
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

_DEFAULT_LUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lut.npz")


class LUTUtility:
    def __init__(self, npz_path=_DEFAULT_LUT, method="linear"):
        z = np.load(npz_path)
        self.mu_axis = z["mu_axis"]
        self.s2_axis = z["s2_axis"]
        self.U = z["U"]
        self.method = method
        self.mu_lo, self.mu_hi = float(self.mu_axis.min()), float(self.mu_axis.max())
        self.s2_lo, self.s2_hi = float(self.s2_axis.min()), float(self.s2_axis.max())
        self._interp = RegularGridInterpolator(
            (self.mu_axis, self.s2_axis), self.U,
            method=method, bounds_error=False, fill_value=np.nan)

    @staticmethod
    def high_rate_U(mu_g, sigma2_g):
        """True high-rate asymptote of U = H_marg - E[H_noise]: 0.5*log(1 + e^mu * s2).
        Monotone in mu_g and sigma2_g. Used as the sigma2>6 ranking proxy."""
        return 0.5 * np.log1p(np.exp(mu_g) * sigma2_g)

    def oob_mask(self, sigma2_g):
        """Boolean mask of queries served by the high-rate fallback (sigma2 > LUT cap)."""
        return np.atleast_1d(np.asarray(sigma2_g, float)) > self.s2_hi

    def __call__(self, mu_g, sigma2_g):
        """U(mu_g, sigma2_g): linear LUT in-domain; high-rate fallback for sigma2>6;
        mu_g clamped to [-2,6]. NEVER NaN. Scalar in -> float out; arrays vectorised."""
        mu = np.atleast_1d(np.asarray(mu_g, dtype=float))
        s2 = np.atleast_1d(np.asarray(sigma2_g, dtype=float))
        mu_c = np.clip(mu, self.mu_lo, self.mu_hi)            # mu>6 unphysical, mu<-2 rare
        s2_c = np.clip(s2, self.s2_lo, self.s2_hi)
        U = self._interp(np.column_stack([mu_c, s2_c]))       # LUT (mu, sigma2 clamped)
        oob = s2 > self.s2_hi                                 # sigma2 beyond the LUT
        if oob.any():
            U = U.copy()
            mu_o = mu_c[oob]
            # continuity-corrected high-rate fallback: LUT sigma2=6 edge + asymptote increment
            U_edge = self._interp(np.column_stack([mu_o, np.full(mu_o.shape, self.s2_hi)]))
            incr = self.high_rate_U(mu_o, s2[oob]) - self.high_rate_U(mu_o, self.s2_hi)
            U[oob] = U_edge + incr
        return float(U[0]) if U.size == 1 else U
