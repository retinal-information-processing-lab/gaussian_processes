"""NGD-compatible variational GP model.

SUPERSEDED (Phase 3D, 2026-04-22): the production equivalent is
`VariationalGPModel(..., variational_distribution_cls='tril_natural')`
in `gpy_model.py`. See run_single_mode.py `mode='ngd'` for the wiring.
This file is preserved as the Phase 3 investigation artifact; do not
use it in new code.

Thin wrapper around gpytorch.models.ApproximateGP that swaps in
TrilNaturalVariationalDistribution so gpytorch.optim.NGD can update
the variational parameters natively (rather than via whitened Cholesky).

Design:
- Same whitened VariationalStrategy as the production default_gpy model.
  The only swap is the variational distribution class.
- Takes a pre-built kernel and inducing_points as input, so it can be
  dropped into the production setup flow (kernel/inducing built by
  run_single_mode.run_single_config) without reinitialising either.

This file lives inside an investigation folder; it does NOT edit
production code (gpy_model.py is untouched).
"""
from __future__ import annotations

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import TrilNaturalVariationalDistribution, VariationalStrategy


class NGDVariationalGPModel(ApproximateGP):
    """Sparse variational GP with Tril-natural variational distribution.

    Parameters
    ----------
    inducing_points : Tensor, shape (M, D)
    kernel : gpytorch.kernels.Kernel
    jitter : float
        Cholesky jitter (added to K_uu before Cholesky by VariationalStrategy).
    learn_inducing_locations : bool, default False
    """

    def __init__(self, inducing_points, kernel, jitter, learn_inducing_locations=False):
        variational_distribution = TrilNaturalVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
            jitter_val=jitter,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.jitter = jitter

        # Attribute required by gpy_training.train_gpy_default (we don't call
        # that path here, but downstream code in run_single_config inspects it
        # — kept for interop robustness).
        self.standard_variational_distribution = True

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
