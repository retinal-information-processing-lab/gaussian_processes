"""
Training Functions for Standard GPyTorch Variational GP

This module contains training and prediction functions for the default_gpy mode,
which uses standard GPyTorch variational inference (VariationalStrategy).

Key functions:
- train_gpy_default: Standard ELBO optimization with LBFGS or Adam
- predict: Prediction at test points using GPyTorch model

Extracted from train.py during codebase reorganization (2025-02).
"""

import time
import warnings

import torch
from linear_operator import settings as lo_settings

from metrics import compute_pearson_correlation, compute_spearman_correlation
from _constants import (
    LAMBDA_VAR_CLAMP, JITTER, CHOLESKY_MAX_TRIES,
    GPY_LBFGS_MAX_ITER, F_MEAN_MAX_THRESHOLD, F_MEAN_MEAN_THRESHOLD,
    ES_ENABLED, ES_PATIENCE, ES_MIN_DELTA_REL, ES_MIN_ITERATIONS,
    ES_RESTORE_BEST, ES_METRIC,
)


def _compute_val_metrics_gpy(model, likelihood, X_val, r_val,
                              jitter=JITTER, cholesky_max_tries=CHOLESKY_MAX_TRIES,
                              lambda_var_clamp=LAMBDA_VAR_CLAMP):
    """Compute validation log-likelihood, Pearson r, and Spearman rho for default_gpy mode.

    Val log-lik formula (same as ELBO's log-lik term on held-out data):
      val_ll = sum(r_val * (A*mu + lambda0) - f_mean)
    where f_mean = exp(A*mu + 0.5*A^2*var + lambda0)

    Val Pearson r: corr(f_pred, r_val) — same metric as test_r.
    Val Spearman rho: rank correlation between f_pred and r_val.

    Returns:
        Tuple (val_ll, val_r, val_rho)
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), \
         lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):
        posterior = model(X_val)
        mu = posterior.mean
        var = torch.clamp(posterior.variance, min=lambda_var_clamp)
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * mu + 0.5 * A**2 * var + lambda0)
        val_ll = (r_val * (A * mu + lambda0) - f_pred).sum().item()
        val_r = compute_pearson_correlation(r_val, f_pred)
        val_rho = compute_spearman_correlation(r_val, f_pred)
    model.train()
    likelihood.train()
    return val_ll, val_r, val_rho


def _compute_train_metrics_gpy(model, likelihood, train_x, train_y,
                                jitter=JITTER, cholesky_max_tries=CHOLESKY_MAX_TRIES,
                                lambda_var_clamp=LAMBDA_VAR_CLAMP):
    """Compute training Pearson r and Spearman rho for default_gpy mode."""
    model.eval()
    likelihood.eval()
    with torch.no_grad(), \
         lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):
        posterior = model(train_x)
        mu = posterior.mean
        var = torch.clamp(posterior.variance, min=lambda_var_clamp)
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * mu + 0.5 * A**2 * var + lambda0)
        train_r = compute_pearson_correlation(train_y, f_pred)
        train_rho = compute_spearman_correlation(train_y, f_pred)
    model.train()
    likelihood.train()
    return train_r, train_rho


class AlternatingFstepError(Exception):
    """Correctness failure in the alternating F-step code path.

    Raised by the first-iteration grad-leak check when the slim
    closure_likelihood modifies a gradient it shouldn't, or fails to produce
    a gradient on A/lambda0. Distinct from LBFGS-crash `RuntimeError`s
    (which are caught and warned-and-stopped) — this must propagate.
    """


def _precompute_likelihood_constants(model, train_x):
    """Snapshot the quantities that are invariant during an alternating F-step.

    During `optimizer_likelihood.step(...)` only A and lambda0 change; kernel
    and variational params are frozen. So K_uu, K_uf, the posterior (mu, var),
    and the KL divergence are all constant across the inner LBFGS closure
    evaluations. Computing them once avoids re-running GPyTorch's full forward
    (with a fresh K_uu Cholesky) on every closure call.

    Returns (mu_const, var_const, kl_const), all detached float tensors on
    train_x's device/dtype.
    """
    with torch.no_grad():
        out = model(train_x)
        mu_const = out.mean.detach()
        var_const = out.variance.detach()
        kl_const = model.variational_strategy.kl_divergence().detach()
    return mu_const, var_const, kl_const


def _compute_ell_slim(A, lambda0, train_y, mu_const, var_const):
    """Poisson expected log-likelihood using precomputed (mu, var).

    Implements the same formula as `PoissonLikelihood.expected_log_prob`
    (likelihoods.py:138-158), which omits the lgamma(y+1) term (constant
    w.r.t. the parameters). We intentionally match that convention so
    `last_ell[0]` stays numerically consistent whether produced by the
    full closure (closure_model) or the slim one (closure_likelihood).

    Returns (ell, f_mean).
    """
    log_fmean = A * mu_const + 0.5 * A**2 * var_const + lambda0
    f_mean = torch.exp(log_fmean)
    ell = (train_y * (A * mu_const + lambda0) - f_mean).sum()
    return ell, f_mean


def train_gpy_default(model, likelihood, train_x, train_y, optimizer_name, lr, n_iterations,
                       print_every=100, device=None,
                       early_stop=ES_ENABLED, patience=ES_PATIENCE,
                       min_delta_rel=ES_MIN_DELTA_REL, min_iterations=ES_MIN_ITERATIONS,
                       restore_best=ES_RESTORE_BEST,
                       lbfgs_max_iter=GPY_LBFGS_MAX_ITER,
                       jitter=JITTER, cholesky_max_tries=CHOLESKY_MAX_TRIES,
                       f_mean_max_threshold=F_MEAN_MAX_THRESHOLD,
                       f_mean_mean_threshold=F_MEAN_MEAN_THRESHOLD,
                       lambda_var_clamp=LAMBDA_VAR_CLAMP,
                       X_val=None, r_val=None,
                       es_metric=ES_METRIC,
                       alternating_fstep=False):
    """Train using GPyTorch's standard variational inference (no custom E-step).

    Maximizes the ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        train_x: Training inputs, shape (n_train, n_features)
        train_y: Training targets (spike counts), shape (n_train,)
        optimizer_name: Optimizer to use ('adam' or 'lbfgs')
        lr: Learning rate
        n_iterations: Maximum number of optimization iterations
        print_every: Print loss every N iterations (0 to disable)
        device: Device to use (defaults to train_x.device)
        early_stop: Enable early stopping based on validation log-lik (default: True)
        patience: Iterations without sufficient validation improvement before stopping
        min_delta_rel: Minimum relative improvement to reset patience counter
        min_iterations: Minimum iterations before early stopping can trigger (default: 10)
        restore_best: Restore model to best-validation-iteration on early stop
        lbfgs_max_iter: Max inner iterations for LBFGS per outer step (default: 20)
        jitter: Jitter value for Cholesky retry schedule starting point (default: 1e-4)
        cholesky_max_tries: Number of Cholesky retry attempts (default: 3)
        f_mean_max_threshold: Max f_mean.max() before step rejection (default: 500)
        f_mean_mean_threshold: Max f_mean.mean() before step rejection (default: 100)
        lambda_var_clamp: Minimum posterior variance clamp (default: 1e-6)
        X_val: Validation images, shape (N_val, n_features). None = no validation.
        r_val: Validation spike counts, shape (N_val,). None = no validation.
        alternating_fstep: If True and optimizer_name='lbfgs', use separate LBFGS
            optimizers for (model=variational+kernel) and (likelihood=A,lambda0),
            stepping them in sequence each outer iteration. Mirrors vargp_direct's
            F-step isolation to prevent A from coupling with kernel/variational
            optimization. Default: False (joint optimization, current behavior).

    Returns:
        dict with:
            'losses': list of training losses
            'stopped_early': bool
            'final_iteration': int
            'best_iteration': int (iteration with highest ES metric value, ELBO by default)
            'curves': dict of per-iteration curves
    """
    if device is None:
        device = train_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Validate model has required attribute
    if not hasattr(model, 'standard_variational_distribution'):
        raise AttributeError(
            "Model does not have 'standard_variational_distribution' attribute. "
            "Use VariationalGPModel which defines this attribute."
        )

    model.train()
    likelihood.train()

    # Collect all parameters
    all_params = list(model.parameters()) + list(likelihood.parameters())

    # Create optimizer(s)
    use_alternating = alternating_fstep and optimizer_name == 'lbfgs'

    if optimizer_name == 'lbfgs':
        if use_alternating:
            # Two separate LBFGS optimizers: one for model (variational+kernel),
            # one for likelihood (A, lambda0). Stepped in sequence each outer
            # iteration to prevent A from coupling with kernel/variational updates.
            optimizer_model = torch.optim.LBFGS(
                list(model.parameters()),
                lr=lr, max_iter=lbfgs_max_iter, line_search_fn='strong_wolfe'
            )
            optimizer_likelihood = torch.optim.LBFGS(
                list(likelihood.parameters()),
                lr=lr, max_iter=lbfgs_max_iter, line_search_fn='strong_wolfe'
            )
            optimizer = None  # unused in alternating mode
        else:
            optimizer = torch.optim.LBFGS(
                all_params,
                lr=lr,
                max_iter=lbfgs_max_iter,
                line_search_fn='strong_wolfe'
            )
            optimizer_model = None
            optimizer_likelihood = None
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=lr)
        optimizer_model = None
        optimizer_likelihood = None
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: 'lbfgs', 'adam'.")

    losses = []

    # For LBFGS, we need a closure that computes loss and gradients
    # We store the last computed values for logging
    last_output = [None]
    last_loss = [None]
    last_ell = [None]
    last_kl = [None]

    # Precomputed F-step invariants (populated each outer iter, immediately
    # before optimizer_likelihood.step in alternating mode). Allow the slim
    # closure_likelihood below to read them without redoing the GPyTorch
    # forward pass every call. Same mutable-slot pattern as last_loss above.
    mu_const = [None]
    var_const = [None]
    kl_const = [None]

    def _compute_elbo_and_backward(zero_grad_fn):
        """Shared ELBO computation for all closure types.

        Calls zero_grad_fn(), then computes the ELBO and calls backward().
        Returns loss (inf if any guard triggers, real value otherwise).
        Also updates last_output/loss/ell/kl for logging.
        """
        zero_grad_fn()
        # Reject trial step if parameters are out of bounds
        kernel = model.covar_module
        if hasattr(kernel, 'params_in_bounds') and not kernel.params_in_bounds():
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        if hasattr(likelihood, 'params_in_bounds') and not likelihood.params_in_bounds():
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        # Guard: catch kernel NaN/errors during LBFGS line search.
        try:
            output = model(train_x)
        except Exception:
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        # Firing rate stability check
        A_val = likelihood.A.squeeze()
        lambda0_val = likelihood.lambda0.squeeze()
        f_mean = torch.exp(A_val * output.mean + 0.5 * A_val**2 * output.variance + lambda0_val)
        if (f_mean.mean().item() > f_mean_mean_threshold
                or f_mean.max().item() > f_mean_max_threshold
                or torch.any(torch.isnan(f_mean))):
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        ell = likelihood.expected_log_prob(train_y, output)
        kl = model.variational_strategy.kl_divergence()
        loss = -ell + kl
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        loss.backward()
        last_output[0] = output
        last_loss[0] = loss
        last_ell[0] = ell
        last_kl[0] = kl
        return loss

    def closure():
        return _compute_elbo_and_backward(optimizer.zero_grad)

    def closure_model():
        return _compute_elbo_and_backward(optimizer_model.zero_grad)

    def closure_likelihood():
        # Slim F-step closure: only A and lambda0 are being optimized here, so
        # the posterior (mu, var) and the KL divergence are invariant across
        # inner LBFGS evaluations. The precompute block in the outer loop has
        # populated mu_const / var_const / kl_const for this iteration. See
        # _precompute_likelihood_constants() for the rationale.
        #
        # Kernel params are frozen in this optimizer's param list, so no
        # kernel bounds check is needed. Only the likelihood bounds + f_mean
        # explosion guard remain.
        optimizer_likelihood.zero_grad()
        if hasattr(likelihood, 'params_in_bounds') and not likelihood.params_in_bounds():
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        A_val = likelihood.A.squeeze()
        lambda0_val = likelihood.lambda0.squeeze()
        ell, f_mean = _compute_ell_slim(A_val, lambda0_val, train_y,
                                         mu_const[0], var_const[0])
        if (f_mean.mean().item() > f_mean_mean_threshold
                or f_mean.max().item() > f_mean_max_threshold
                or torch.any(torch.isnan(f_mean))):
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        loss = -ell + kl_const[0]
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        loss.backward()
        last_loss[0] = loss  # update for ES tracking
        last_ell[0] = ell    # fresh ELL after A/lambda0 update
        # last_kl[0] intentionally untouched: KL is constant during the F-step,
        # so whatever closure_model wrote is still the correct current value.
        return loss

    # Early stopping state
    # `best_es_value`: true argmax of the metric (for restore_best). Updates
    # on ANY improvement. `patience_reference`: value at last meaningful
    # reset (for patience counter). Decoupled so slow steady growth can
    # accumulate and still reset patience. See eigenspace_training.py
    # for the full rationale.
    stopped_early = False
    final_iteration = 0
    has_val = X_val is not None and r_val is not None
    best_es_value = float('-inf')
    patience_reference = float('-inf')
    patience_counter = 0
    best_state = None
    best_iteration = 0

    # Curve storage
    train_loss_curve = []
    train_ll_curve = []
    train_kl_curve = []
    val_ll_curve = []
    train_r_curve = []
    val_r_curve = []
    train_rho_curve = []
    val_rho_curve = []
    param_A_curve = []
    param_lambda0_curve = []
    iter_time_curve = []

    if train_x.dtype == torch.float64 and jitter >= 1e-4:
        warnings.warn(
            f"jitter={jitter} is high for float64 (GPyTorch default is 1e-6). "
            f"Consider reducing jitter for float64 training."
        )

    # Cholesky stability: GPyTorch adds jitter_val (our 1e-4) to K_uu before
    # Cholesky, then promotes to float64. If Cholesky still fails,
    # psd_safe_cholesky retries with escalating jitter. We override:
    #   - cholesky_jitter: retry starting jitter = our jitter value (not 1e-8)
    #   - cholesky_max_tries: number of retries (each adds 10x more jitter)
    # With jitter=1e-4, max_tries=3: retries at 1e-4, 1e-3, 1e-2.
    with torch.enable_grad(), \
         lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):
        for i in range(n_iterations):
            iter_start = time.time()

            if optimizer_name == 'lbfgs':
                try:
                    if use_alternating:
                        # Step 1: update variational distribution + kernel
                        optimizer_model.step(closure_model)
                        # Precompute posterior moments + KL once for the
                        # F-step: kernel / variational params don't change
                        # during optimizer_likelihood.step, so every
                        # closure_likelihood call would otherwise redo the
                        # full GPyTorch forward (incl. K_uu Cholesky) for
                        # bit-identical results. See
                        # _precompute_likelihood_constants() docstring.
                        mu_c, var_c, kl_c = _precompute_likelihood_constants(model, train_x)
                        mu_const[0] = mu_c
                        var_const[0] = var_c
                        kl_const[0] = kl_c
                        # First-iter correctness check: verify the slim
                        # closure's backward only populates gradients on
                        # likelihood params (A, lambda0), never on kernel
                        # or variational params. A silent grad leak would
                        # violate the mathematical intent of the
                        # alternating F-step, so hard-fail here instead of
                        # warning. Runs once per training run → negligible.
                        if i == 0:
                            _snap_model_grads = {
                                id(p): (p.grad.detach().clone() if p.grad is not None else None)
                                for p in model.parameters()
                            }
                            _ = closure_likelihood()
                            for p in model.parameters():
                                before = _snap_model_grads[id(p)]
                                after = p.grad
                                if before is None:
                                    leaked = (after is not None
                                              and after.abs().sum().item() > 0)
                                else:
                                    leaked = (after is None
                                              or not torch.equal(before, after))
                                if leaked:
                                    raise AlternatingFstepError(
                                        "Gradient leak into model param during "
                                        "alternating F-step: closure_likelihood "
                                        "modified a kernel or variational param's "
                                        ".grad. The precompute block must detach "
                                        "mu/var/kl."
                                    )
                            # PoissonLikelihood exposes A as a non-leaf view
                            # (A = exp(raw_A)); the actual leaf is raw_A, so
                            # that's where the gradient lands.
                            if (likelihood.raw_A.grad is None
                                    or likelihood.raw_A.grad.abs().sum().item() == 0):
                                raise AlternatingFstepError(
                                    "closure_likelihood did not produce a gradient "
                                    "for likelihood.raw_A (A's underlying leaf)."
                                )
                            if (likelihood.lambda0.grad is None
                                    or likelihood.lambda0.grad.abs().sum().item() == 0):
                                raise AlternatingFstepError(
                                    "closure_likelihood did not produce a gradient "
                                    "for likelihood.lambda0."
                                )
                        # Step 2: update A and lambda0 (isolated from model updates)
                        optimizer_likelihood.step(closure_likelihood) 
                    else:
                        optimizer.step(closure)
                except (IndexError, RuntimeError) as e:
                    warnings.warn(f"LBFGS crashed at iteration {i+1}: {e}. Stopping training.")
                    stopped_early = True
                    break
                loss = last_loss[0]
                expected_log_lik = last_ell[0]
                kl_div = last_kl[0]
            else:
                optimizer.zero_grad()
                output = model(train_x)
                expected_log_lik = likelihood.expected_log_prob(train_y, output)
                kl_div = model.variational_strategy.kl_divergence()
                loss = -expected_log_lik + kl_div
                loss.backward()
                optimizer.step()

            # Clamp parameters to physical bounds after each step
            # (projected gradient descent — matches eigenspace_mstep.py pattern)
            kernel = model.covar_module
            if hasattr(kernel, 'clamp_hyperparameters'):
                kernel.clamp_hyperparameters()
            if hasattr(likelihood, 'clamp_params'):
                likelihood.clamp_params()

            # Detect divergence: loss is None (all LBFGS evals rejected) or NaN/inf
            if loss is None:
                print(f"Training diverged at iteration {i+1}: all LBFGS evaluations rejected")
                stopped_early = True
                break
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Training diverged at iteration {i+1}: loss={loss.item()}")
                stopped_early = True
                break
            current_loss = loss.item()
            losses.append(current_loss)
            final_iteration = i + 1

            # Log training curves
            ell_val = expected_log_lik.item() if expected_log_lik is not None else None
            kl_val = kl_div.item() if kl_div is not None else None
            train_loss_curve.append(current_loss)
            train_ll_curve.append(ell_val)
            train_kl_curve.append(kl_val)
            param_A_curve.append(likelihood.A.item())
            param_lambda0_curve.append(likelihood.lambda0.item())

            # Training correlations
            train_r, train_rho = _compute_train_metrics_gpy(
                model, likelihood, train_x, train_y,
                jitter=jitter, cholesky_max_tries=cholesky_max_tries,
                lambda_var_clamp=lambda_var_clamp)

            # Validation evaluation
            val_ll = None
            val_r = None
            val_rho = None
            if has_val:
                val_ll, val_r, val_rho = _compute_val_metrics_gpy(
                    model, likelihood, X_val, r_val,
                    jitter=jitter, cholesky_max_tries=cholesky_max_tries,
                    lambda_var_clamp=lambda_var_clamp)
                # Back to train mode after validation
                model.train()
                likelihood.train()
            val_ll_curve.append(val_ll)
            train_r_curve.append(train_r)
            val_r_curve.append(val_r)
            train_rho_curve.append(train_rho)
            val_rho_curve.append(val_rho)

            iter_time_curve.append(time.time() - iter_start)

            if print_every > 0 and (i + 1) % print_every == 0:
                val_str = f", val_ll={val_ll:.2f}" if val_ll is not None else ""
                val_r_str = f", val_r={val_r:.4f}" if val_r is not None else ""
                val_rho_str = f", val_rho={val_rho:.4f}" if val_rho is not None else ""
                print(f"Iter {i+1}/{n_iterations}, Loss: {current_loss:.2f}{val_str}{val_r_str}{val_rho_str}, "
                      f"ELL: {ell_val:.2f}, KL: {kl_val:.2f}")

            # Patience-based early stopping on ELBO (only supported metric).
            # See eigenspace_training.py for rationale and references.
            es_value = None
            if es_metric == 'elbo':
                es_value = -current_loss  # ELBO = -train_loss (higher is better)
            elif es_metric == 'none':
                es_value = None  # ES disabled
            else:
                raise ValueError(
                    f"Unknown es_metric: {es_metric!r}. "
                    f"Valid choices are 'elbo' (default) or 'none' (disabled). "
                    f"val_ll/val_r/val_rho were removed in April 2026 — see "
                    f"investigations/optimization/possible_optimizations.md "
                    f"Investigation 2 for the rationale."
                )

            if es_value is not None:
                # See eigenspace_training.py for the full rationale.
                # (1) Best tracking (updates on ANY improvement) and
                # (2) Patience counting (resets only on meaningful cumulative
                # gain vs patience_reference) are decoupled.
                is_first = (best_es_value == float('-inf'))

                # (1) Best tracking
                if is_first or es_value > best_es_value:
                    best_es_value = es_value
                    best_iteration = i + 1
                    if restore_best:
                        best_state = {
                            'model_state': {k: v.clone() for k, v in model.state_dict().items()},
                            'likelihood_state': {k: v.clone() for k, v in likelihood.state_dict().items()},
                        }

                # (2) Patience counter
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
                        print(f"Early stopping at iteration {i+1} (metric={es_metric}): "
                              f"no improvement for {patience} iters "
                              f"(best={best_es_value:.4f} at iter {best_iteration})"
                              f"{', restored best' if restore_best else ''}")
                    break

    # Fallback: if ES was disabled (es_metric='none') or never tracked a
    # best iteration, report the final iteration as "best". Covers the
    # es_metric='none' + n_val_split=0 combo where best_iteration would
    # otherwise stay at 0.
    if not stopped_early and best_iteration == 0:
        best_iteration = final_iteration

    return {
        'losses': losses,
        'stopped_early': stopped_early,
        'final_iteration': final_iteration,
        'best_iteration': best_iteration,
        'curves': {
            'train_loss': train_loss_curve,
            'train_log_lik': train_ll_curve,
            'train_kl': train_kl_curve,
            'val_log_lik': val_ll_curve,
            'train_r': train_r_curve,
            'val_r': val_r_curve,
            'train_rho': train_rho_curve,
            'val_rho': val_rho_curve,
            'A': param_A_curve,
            'lambda0': param_lambda0_curve,
            'iter_time': iter_time_curve,
        },
    }


def predict(model, likelihood, test_x, device=None,
            jitter=JITTER, cholesky_max_tries=CHOLESKY_MAX_TRIES, lambda_var_clamp=LAMBDA_VAR_CLAMP):
    """Make predictions on test data.

    Args:
        model: Trained VariationalGPModel
        likelihood: Trained PoissonLikelihood
        test_x: Test inputs, shape (n_test, n_features)
        device: Device to use (defaults to test_x.device)
        jitter: Jitter value for Cholesky retry schedule starting point (default: 1e-4)
        cholesky_max_tries: Number of Cholesky retry attempts (default: 3)
        lambda_var_clamp: Minimum posterior variance clamp (default: 1e-6)

    Returns:
        dict with:
        - 'f_pred': Predicted firing rates E[exp(A·λ + λ₀)]
        - 'lambda_mean': Posterior mean of λ
        - 'lambda_var': Posterior variance of λ
    """
    if device is None:
        device = test_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    test_x = test_x.to(device)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), \
         lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):
        # Get posterior q(λ*) at test points
        posterior = model(test_x)
        lambda_mean = posterior.mean
        lambda_var = posterior.variance
        lambda_var = torch.clamp(lambda_var, min=lambda_var_clamp)

        # Predicted firing rate: E[exp(A·λ + λ₀)] = exp(A·μ + A²σ²/2 + λ₀)
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * lambda_mean + 0.5 * A**2 * lambda_var + lambda0)

    return {
        'f_pred': f_pred,
        'lambda_mean': lambda_mean,
        'lambda_var': lambda_var
    }
