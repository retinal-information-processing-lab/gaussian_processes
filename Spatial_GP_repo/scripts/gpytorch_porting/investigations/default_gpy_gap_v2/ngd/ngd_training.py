"""NGD+Adam training loop for non-conjugate SVGP (Poisson likelihood).

SUPERSEDED (Phase 3D, 2026-04-22): the production equivalent is
`ngd_training.train_ngd` at the project root. It is called by
`run_single_mode.py` when `mode='ngd'`. This file is preserved as the
Phase 3 investigation artifact with narrative comments; do not use it
in new code.


Pattern follows the GPyTorch NGD tutorial:
  https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/
      Natural_Gradient_Descent.html

Two optimizers per step:
  - gpytorch.optim.NGD on model.variational_parameters() (lr 0.1, per tutorial)
  - torch.optim.Adam   on model.hyperparameters() + likelihood.parameters()

Clamps after each Adam step mirror gpy_training.train_gpy_default:
  kernel.clamp_hyperparameters() + likelihood.clamp_params().
Variational natural parameters are NOT clamped — the NGD optimizer is the
only path that is allowed to touch them (see
natural_variational_distribution.py:103-107 for the runtime check).

Per-iteration curves logged:
  train_loss (= -ELBO), train_log_lik, train_kl
  A, lambda0, beta, rho, eps_0x, eps_0y
  iter_time

The f_mean guard (f_mean.max > F_MEAN_MAX_THRESHOLD, .mean > F_MEAN_MEAN_THRESHOLD)
from gpy_training is NOT duplicated here: Adam's small steps + post-step
clamps keep A bounded below A_MAX=10, and divergence is tracked via NaN/Inf
guards on the loss itself. If this becomes a problem during the prototype,
we'll flag and add the guard.

Deliberately:
  - No early stopping (ES off during prototype; full trajectory is the point).
  - Full-batch (n_train=1500 fits comfortably on GPU).
  - No validation carving; uses entire training pool.
"""
from __future__ import annotations

import time

import gpytorch
import torch
from linear_operator import settings as lo_settings


def train_ngd(
    model,
    likelihood,
    X_train,
    r_train,
    *,
    n_iterations,
    ngd_lr,
    adam_lr,
    jitter,
    cholesky_max_tries,
    device=None,
    print_every=50,
    early_stop=True,
    patience=100,
    min_delta_rel=1e-3,
    min_iterations=50,
    restore_best=True,
    test_r_probe=None,
    test_r_every=25,
):
    """
    test_r_probe : callable(model, likelihood) -> float, optional
        If provided, called every `test_r_every` iterations to log test_r
        during training (for overfitting diagnostics, NOT as an ES signal).
    """
    """Train an NGDVariationalGPModel + PoissonLikelihood via NGD+Adam.

    Returns a dict with:
      losses : list[float]      — per-iteration train_loss (= -ELBO)
      curves : dict[str, list]  — per-iteration trajectories
      final_iteration : int
      stopped_early : bool      — always False (no ES)
      diverged : bool           — True if loss became NaN/Inf at some iter
      diverged_at : int or None
    """
    if device is None:
        device = X_train.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    X_train = X_train.to(device)
    r_train = r_train.to(device)

    model.train()
    likelihood.train()

    # Two optimizers (tutorial pattern).
    variational_ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(),
        num_data=r_train.size(0),
        lr=ngd_lr,
    )
    hyperparameter_optimizer = torch.optim.Adam(
        [
            {'params': model.hyperparameters()},
            {'params': likelihood.parameters()},
        ],
        lr=adam_lr,
    )

    # Use gpytorch.mlls.VariationalELBO for the loss — it averages the ELL
    # over the (mini)batch and divides KL by num_data, producing a per-point
    # ELBO. gpytorch.optim.NGD internally multiplies the gradient by num_data,
    # so its effective step is one "full-dataset" natural gradient step.
    # Using sum-scaled loss here makes the NGD step num_data² too large
    # (see `_ApproximateMarginalLogLikelihood.forward` in GPyTorch for the
    # scaling contract).
    n_data = r_train.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_data)

    losses = []
    train_loss_curve = []
    train_ll_curve = []
    train_kl_curve = []
    A_curve = []
    lambda0_curve = []
    beta_curve = []
    rho_curve = []
    eps_0x_curve = []
    eps_0y_curve = []
    iter_time_curve = []
    # Variational-distribution diagnostics: useful for spotting
    # ill-conditioning / divergence of the Tril-natural parameters.
    nat_vec_norm_curve = []
    nat_tril_offdiag_norm_curve = []
    # Optional test_r curve (diagnostic only — NOT used for any decision).
    test_r_iter_curve = []
    test_r_curve = []

    diverged = False
    diverged_at = None
    final_iteration = 0

    # ELBO-based early stopping (mirrors gpy_training.train_gpy_default and
    # eigenspace_training, but with patience tuned for NGD's first-order
    # step size — LBFGS's patience=15 corresponds to ~15 big jumps; NGD's
    # equivalent is ~patience*max_per_LBFGS_inner ≈ 100 first-order steps).
    # Still argmax-based for restore_best (see gpy_training line 550 pattern).
    best_es_value = float('-inf')
    patience_reference = float('-inf')
    patience_counter = 0
    best_state = None
    best_iteration = 0
    stopped_early = False

    # Cholesky jitter overrides — same pattern as gpy_training.train_gpy_default.
    with lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):

        for i in range(n_iterations):
            iter_start = time.time()

            variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()

            # Per-point ELBO via VariationalELBO (see scaling note above).
            output = model(X_train)
            loss = -mll(output, r_train)

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                diverged_at = i + 1
                break

            # For logging: also compute the full-dataset ELL and KL (undo the
            # num_data scaling) so the curves are comparable to gpy_training's.
            with torch.no_grad():
                ell_full = likelihood.expected_log_prob(r_train, output)
                kl_full = model.variational_strategy.kl_divergence()
                loss_full = (-ell_full + kl_full).item()

            loss.backward()

            variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()

            # Clamp hyperparameters (kernel + likelihood). Variational
            # natural parameters are deliberately NOT clamped.
            kernel = model.covar_module
            if hasattr(kernel, 'clamp_hyperparameters'):
                kernel.clamp_hyperparameters()
            if hasattr(likelihood, 'clamp_params'):
                likelihood.clamp_params()

            # ----- log curves -----
            # We log the FULL-dataset ELBO/ELL/KL (not the per-point mll loss)
            # so numbers are directly comparable to gpy_training's curves.
            current_loss = loss_full
            losses.append(current_loss)
            train_loss_curve.append(current_loss)
            train_ll_curve.append(ell_full.item())
            train_kl_curve.append(kl_full.item())
            A_curve.append(likelihood.A.item())
            lambda0_curve.append(likelihood.lambda0.item())
            if hasattr(kernel, 'beta'):
                beta_curve.append(kernel.beta.item())
            if hasattr(kernel, 'rho'):
                rho_curve.append(kernel.rho.item())
            if hasattr(kernel, 'eps_0x'):
                eps_0x_curve.append(kernel.eps_0x.item())
            if hasattr(kernel, 'eps_0y'):
                eps_0y_curve.append(kernel.eps_0y.item())

            vd = model.variational_strategy._variational_distribution
            with torch.no_grad():
                nat_vec_norm_curve.append(vd.natural_vec.norm().item())
                # offdiag norm of the Tril-natural matrix: the diagonal is ~1
                # at init, so the offdiag magnitude tells us whether the
                # posterior covariance has moved away from the prior.
                ntm = vd.natural_tril_mat
                M_dim = ntm.size(-1)
                offdiag = ntm - torch.diag_embed(torch.diagonal(ntm, dim1=-2, dim2=-1))
                nat_tril_offdiag_norm_curve.append(offdiag.norm().item())

            iter_time_curve.append(time.time() - iter_start)
            final_iteration = i + 1

            # Optional test_r probe (diagnostic): never used to decide ES.
            if test_r_probe is not None and (i + 1) % test_r_every == 0:
                model.eval()
                likelihood.eval()
                with torch.no_grad():
                    tr = float(test_r_probe(model, likelihood))
                model.train()
                likelihood.train()
                test_r_iter_curve.append(i + 1)
                test_r_curve.append(tr)

            # ---- ELBO-based early stopping ----
            # es_value = -train_loss (higher is better = ELBO maximisation).
            # Decoupled best-tracking + patience-counter, matching
            # gpy_training.train_gpy_default's Lightning-style pattern.
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
                print(
                    f"  Iter {i + 1:4d}/{n_iterations}  "
                    f"loss={current_loss:.3f}  "
                    f"A={likelihood.A.item():.4g}  "
                    f"beta={beta_curve[-1] if beta_curve else float('nan'):.4g}  "
                    f"iter_time={iter_time_curve[-1]:.3f}s",
                    flush=True,
                )

    return {
        'losses': losses,
        'stopped_early': stopped_early,
        'best_iteration': best_iteration,
        'diverged': diverged,
        'diverged_at': diverged_at,
        'final_iteration': final_iteration,
        'curves': {
            'train_loss': train_loss_curve,
            'train_log_lik': train_ll_curve,
            'train_kl': train_kl_curve,
            'A': A_curve,
            'lambda0': lambda0_curve,
            'beta': beta_curve,
            'rho': rho_curve,
            'eps_0x': eps_0x_curve,
            'eps_0y': eps_0y_curve,
            'iter_time': iter_time_curve,
            'nat_vec_norm': nat_vec_norm_curve,
            'nat_tril_offdiag_norm': nat_tril_offdiag_norm_curve,
            'test_r_iter': test_r_iter_curve,
            'test_r': test_r_curve,
        },
    }
