"""NGD+Adam training loop for non-conjugate variational GP (Poisson likelihood).

Pattern follows GPyTorch's NGD tutorial
(https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.html):

  - gpytorch.optim.NGD updates the variational parameters (natural_vec,
    natural_tril_mat).
  - torch.optim.Adam updates the kernel + likelihood hyperparameters.
  - Both optimizers step from a single backward() pass per iteration, using
    gpytorch.mlls.VariationalELBO as the loss so that NGD's internal
    num_data multiplier scales the per-point gradient to a full-dataset
    natural gradient step. See the "scaling contract" note at line 64.

Clamps (kernel.clamp_hyperparameters + likelihood.clamp_params) run after
every Adam step, mirroring gpy_training.train_gpy_default. Variational
natural parameters are NEVER clamped — the PSD constraint is maintained by
NGD.step() alone, and external writes would break it (see
gpytorch/variational/natural_variational_distribution.py:103-107).

ELBO-based early stopping uses the same Lightning-style decoupled
best-tracking / patience-counter as gpy_training.py and
eigenspace_training.py. Defaults come from `_constants.py`:
NGD_ES_PATIENCE=200, NGD_ES_MIN_DELTA_REL=1e-2, NGD_ES_MIN_ITERATIONS=50.
These are ~10x vargp_direct's (15, 1e-3, 10) on both axes — NGD's per-iter
step is ~1 grad eval vs vargp LBFGS's ~7 grad evals per outer iter.

See investigations/default_gpy_gap_v2/SCRAPBOOK.md Phase 3/3B for the
derivation and validation (16 cells x 3 seeds, |mean_Δ vs vargp_direct| =
0.012 < 0.02 threshold, 40% wall-time reduction via ES).

Logging convention (post 2026-04-22 audit): `train_loss`, `train_log_lik`,
and `train_kl` are all logged as PRE-step full-dataset values at each
iteration i — i.e. computed from `(output_i, θ_i, q_i)` before the
optimizer steps that produce `(θ_{i+1}, q_{i+1})`. This matches the
convention used by gpy_training.py / eigenspace_training.py and makes
`best_iteration` point at the true pre-step ELBO maximum (the quantity
`restore_best` actually wants). Phase 3B and Phase 3C sweep JSONLs were
produced BEFORE this fix and use a mixed convention; their loss curves
carry a systematic upward bias that grows with the per-iter KL change.
See SCRAPBOOK.md §47 for the full derivation and the stamped results
notes in the affected folders.
"""
from __future__ import annotations

import time

import gpytorch
import torch
from linear_operator import settings as lo_settings

from _constants import (
    JITTER,
    CHOLESKY_MAX_TRIES,
    NGD_ES_PATIENCE,
    NGD_ES_MIN_DELTA_REL,
    NGD_ES_MIN_ITERATIONS,
    NGD_ES_RESTORE_BEST,
    NGD_LR,
    NGD_ADAM_LR,
)


def train_ngd(
    model,
    likelihood,
    X_train,
    r_train,
    *,
    n_iterations,
    ngd_lr=NGD_LR,
    adam_lr=NGD_ADAM_LR,
    jitter=JITTER,
    cholesky_max_tries=CHOLESKY_MAX_TRIES,
    device=None,
    print_every=50,
    early_stop=True,
    patience=NGD_ES_PATIENCE,
    min_delta_rel=NGD_ES_MIN_DELTA_REL,
    min_iterations=NGD_ES_MIN_ITERATIONS,
    restore_best=NGD_ES_RESTORE_BEST,
    test_r_probe=None,
    test_r_every=25,
):
    """Train a variational GP model via NGD (variational params) + Adam (hyperparams).

    Scaling contract: `gpytorch.optim.NGD.step()` does
    `p += -lr * num_data * p.grad`. The tutorial achieves a one full-dataset
    natural gradient step by using `gpytorch.mlls.VariationalELBO` (per-point
    loss). Feeding a sum-scaled loss here would produce an O(num_data^2)
    step and diverge within a handful of iters. We therefore wrap the loss
    in VariationalELBO below and do NOT use `likelihood.expected_log_prob(...)`
    as the loss directly.

    Parameters
    ----------
    model : gpy_model.VariationalGPModel
        Must be constructed with `variational_distribution_cls='tril_natural'`.
    likelihood : PoissonLikelihood
    X_train, r_train : Tensors (full batch)
    n_iterations : int
        Max number of NGD+Adam steps.
    ngd_lr : float
    adam_lr : float
    jitter, cholesky_max_tries : float, int
        Passed to `lo_settings.cholesky_jitter` / `.cholesky_max_tries`.
    early_stop : bool
        ELBO-based patience ES (see module docstring).
    patience, min_delta_rel, min_iterations, restore_best : ES parameters.
    test_r_probe : callable(model, likelihood) -> float, optional
        If provided, called every `test_r_every` iterations to record test_r
        during training — diagnostic only; NEVER used as an ES signal.
        Default None = no probe (keeps test data out of the training loop).

    Returns
    -------
    dict with keys:
      losses (list[float])
      stopped_early (bool)
      best_iteration (int)
      final_iteration (int)
      diverged (bool)
      diverged_at (int | None)
      curves (dict[str, list]) — per-iter trajectories including:
        train_loss (= -ELBO, full-dataset scale), train_log_lik, train_kl,
        A, lambda0, beta, rho, eps_0x, eps_0y, iter_time,
        nat_vec_norm, nat_tril_offdiag_norm,
        test_r_iter, test_r (populated iff test_r_probe is not None).
    """
    if device is None:
        device = X_train.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    X_train = X_train.to(device)
    r_train = r_train.to(device)

    if getattr(model, 'variational_distribution_kind', None) != 'tril_natural':
        raise ValueError(
            "train_ngd requires a VariationalGPModel built with "
            "variational_distribution_cls='tril_natural'. "
            f"Got variational_distribution_kind="
            f"{getattr(model, 'variational_distribution_kind', 'MISSING')!r}."
        )

    model.train()
    likelihood.train()

    n_data = r_train.size(0)
    variational_ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(), num_data=n_data, lr=ngd_lr,
    )
    hyperparameter_optimizer = torch.optim.Adam(
        [
            {'params': model.hyperparameters()},
            {'params': likelihood.parameters()},
        ],
        lr=adam_lr,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_data)

    # Curve buffers.
    losses = []
    curves = {
        'train_loss': [],
        'train_log_lik': [],
        'train_kl': [],
        'A': [],
        'lambda0': [],
        'beta': [],
        'rho': [],
        'eps_0x': [],
        'eps_0y': [],
        'iter_time': [],
        'nat_vec_norm': [],
        'nat_tril_offdiag_norm': [],
        'test_r_iter': [],
        'test_r': [],
    }

    # Hoist hasattr checks once (kernel params are class-level attributes,
    # so these never change during training).
    kernel = model.covar_module
    has_beta = hasattr(kernel, 'beta')
    has_rho = hasattr(kernel, 'rho')
    has_eps_0x = hasattr(kernel, 'eps_0x')
    has_eps_0y = hasattr(kernel, 'eps_0y')
    has_clamp_kernel = hasattr(kernel, 'clamp_hyperparameters')
    has_clamp_like = hasattr(likelihood, 'clamp_params')

    diverged = False
    diverged_at = None
    final_iteration = 0

    # ELBO-based early stopping. Decoupled best-tracking (updates on any
    # improvement) + patience-counter (resets only on rel_improvement >
    # min_delta_rel). Mirror of gpy_training.py:545-582.
    best_es_value = float('-inf')
    patience_reference = float('-inf')
    patience_counter = 0
    best_state = None
    best_iteration = 0
    stopped_early = False

    with lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):

        for i in range(n_iterations):
            iter_start = time.time()

            variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()

            output = model(X_train)
            loss = -mll(output, r_train)  # per-point -ELBO, PRE-step

            if not torch.isfinite(loss):
                diverged = True
                diverged_at = i + 1
                break

            # Capture PRE-step full-dataset ELL and KL for the diagnostic
            # curves, BEFORE backward/step. If we did this after the step
            # (as the code did prior to the 2026-04-22 audit), `output.mean`
            # and `output.variance` would still be pre-step (cached) but
            # `likelihood.A`, `likelihood.lambda0`, and the variational
            # params would already be POST-step — producing a logged loss
            # that mixed pre-step q with post-step θ. The backward pass's
            # gradients used the pre-step state, so the optimizer trajectory
            # is unaffected, but the logged curves (and by extension the
            # `best_iteration` argmax) were semantically incoherent and not
            # comparable to gpy_training / eigenspace_training curves.
            # See SCRAPBOOK.md §47 for the full derivation.
            with torch.no_grad():
                ell_full_pre = likelihood.expected_log_prob(r_train, output).item()
                kl_full_pre = model.variational_strategy.kl_divergence().item()

            loss.backward()

            variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()

            if has_clamp_kernel:
                kernel.clamp_hyperparameters()
            if has_clamp_like:
                likelihood.clamp_params()

            # loss.item() * n_data is exactly -ELL_sum_pre + KL_full_pre
            # (algebraic identity with VariationalELBO; see
            # _ApproximateMarginalLogLikelihood.forward:60-76). We compute
            # it from `loss` rather than from ell_full_pre/kl_full_pre above
            # to preserve a single source of truth for the logged loss.
            current_loss = loss.item() * n_data

            losses.append(current_loss)
            curves['train_loss'].append(current_loss)
            curves['train_log_lik'].append(ell_full_pre)
            curves['train_kl'].append(kl_full_pre)
            curves['A'].append(likelihood.A.item())
            curves['lambda0'].append(likelihood.lambda0.item())
            if has_beta:
                curves['beta'].append(kernel.beta.item())
            if has_rho:
                curves['rho'].append(kernel.rho.item())
            if has_eps_0x:
                curves['eps_0x'].append(kernel.eps_0x.item())
            if has_eps_0y:
                curves['eps_0y'].append(kernel.eps_0y.item())

            vd = model.variational_strategy._variational_distribution
            with torch.no_grad():
                curves['nat_vec_norm'].append(vd.natural_vec.norm().item())
                ntm = vd.natural_tril_mat
                offdiag = ntm - torch.diag_embed(torch.diagonal(ntm, dim1=-2, dim2=-1))
                curves['nat_tril_offdiag_norm'].append(offdiag.norm().item())

            curves['iter_time'].append(time.time() - iter_start)
            final_iteration = i + 1

            # Optional test_r probe — diagnostic only.
            if test_r_probe is not None and (i + 1) % test_r_every == 0:
                model.eval()
                likelihood.eval()
                with torch.no_grad():
                    tr = float(test_r_probe(model, likelihood))
                model.train()
                likelihood.train()
                curves['test_r_iter'].append(i + 1)
                curves['test_r'].append(tr)

            # ---- ELBO ES ----
            es_value = -current_loss
            is_first = (best_es_value == float('-inf'))

            if is_first or es_value > best_es_value:
                best_es_value = es_value
                best_iteration = i + 1
                if restore_best:
                    best_state = {
                        'model_state': {k: v.clone() for k, v in model.state_dict().items()},
                        'likelihood_state': {k: v.clone() for k, v in likelihood.state_dict().items()},
                    }

            if is_first:
                patience_reference = es_value
                patience_counter = 0
            else:
                rel_improvement = (es_value - patience_reference) / max(abs(patience_reference), 1e-8)
                if rel_improvement > min_delta_rel:
                    patience_reference = es_value
                    patience_counter = 0
                else:
                    patience_counter += 1

            if early_stop and patience_counter >= patience and (i + 1) >= min_iterations:
                if restore_best and best_state is not None:
                    model.load_state_dict(best_state['model_state'])
                    likelihood.load_state_dict(best_state['likelihood_state'])
                stopped_early = True
                if print_every > 0:
                    print(
                        f"  Early stopping at iter {i + 1}: no ELBO improvement "
                        f"for {patience} iters (best={best_es_value:.3f} "
                        f"at iter {best_iteration}"
                        f"{', restored best' if restore_best else ''})",
                        flush=True,
                    )
                break

            if print_every > 0 and (i + 1) % print_every == 0:
                beta_now = curves['beta'][-1] if has_beta else float('nan')
                print(
                    f"  Iter {i + 1:4d}/{n_iterations}  "
                    f"loss={current_loss:.3f}  "
                    f"A={likelihood.A.item():.4g}  "
                    f"beta={beta_now:.4g}  "
                    f"iter_time={curves['iter_time'][-1]:.3f}s",
                    flush=True,
                )

    return {
        'losses': losses,
        'stopped_early': stopped_early,
        'best_iteration': best_iteration,
        'diverged': diverged,
        'diverged_at': diverged_at,
        'final_iteration': final_iteration,
        'curves': curves,
    }
