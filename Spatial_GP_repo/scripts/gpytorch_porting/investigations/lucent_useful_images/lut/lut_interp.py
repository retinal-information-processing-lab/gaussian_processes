"""Lookup-table interpolator for the utility U(mu_g, sigma2_g).

Loads lut.npz (built by build_lut.py) and exposes fast interpolation:

    ulut = ULUT()
    U = ulut(mu, sigma2, method="linear")    # "linear" (bilinear) or "cubic" (bicubic)

mu, sigma2 are LOG-RATE moments:  mu_g = A*lambda_mean + lambda0,  sigma2_g = A^2*var.
Vectorised (pass arrays for a whole pool). Out-of-domain queries return NaN -- NO
extrapolation -- so the caller decides what to do (e.g. gate-reject the image).

Domain (from the LUT): mu in [-6, 6], sigma2 in [0, 6].
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lut.npz")


class ULUT:
    def __init__(self, npz_path=_DEFAULT):
        z = np.load(npz_path)
        self.mu = z["mu_axis"]
        self.s2 = z["s2_axis"]
        self.U = z["U"]
        self.domain = ((float(self.mu.min()), float(self.mu.max())),
                       (float(self.s2.min()), float(self.s2.max())))
        self._lin = RegularGridInterpolator((self.mu, self.s2), self.U,
                                            method="linear", bounds_error=False, fill_value=np.nan)
        self._cub = RegularGridInterpolator((self.mu, self.s2), self.U,
                                            method="cubic", bounds_error=False, fill_value=np.nan)

    def __call__(self, mu, sigma2, method="linear"):
        """U at (mu, sigma2); NaN where out of domain. Scalar in -> float out."""
        m = np.atleast_1d(np.asarray(mu, dtype=float))
        s = np.atleast_1d(np.asarray(sigma2, dtype=float))
        pts = np.column_stack([m, s])
        out = (self._lin if method == "linear" else self._cub)(pts)
        return float(out[0]) if out.size == 1 else out


if __name__ == "__main__":
    import time
    u = ULUT()
    print("domain:", u.domain, " grid:", u.U.shape)
    rng = np.random.default_rng(0)
    ii = rng.integers(0, len(u.mu), 300)
    jj = rng.integers(0, len(u.s2), 300)
    for m in ("linear", "cubic"):
        got = u(u.mu[ii], u.s2[jj], method=m)
        print(f"  {m}: max node-reproduction error = {np.max(np.abs(got - u.U[ii, jj])):.2e}")
    print("  midpoint (mu=1.1, s2=2.35): linear", round(u(1.1, 2.35), 4),
          " cubic", round(u(1.1, 2.35, 'cubic'), 4))
    print("  out-of-domain (mu=8, s2=12):", u(8.0, 12.0))
    M = rng.uniform(-2, 6, 3000)
    S = rng.uniform(0, 6, 3000)
    t = time.time(); u(M, S, "linear"); tl = time.time() - t
    t = time.time(); u(M, S, "cubic"); tc = time.time() - t
    print(f"  3000-pt query: linear {tl*1e3:.2f} ms, cubic {tc*1e3:.2f} ms")
