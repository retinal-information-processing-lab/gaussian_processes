"""NGD + LBFGS(hyperparams) training loop — Phase 3E investigation fork.

Forked from project-root `ngd_training.train_ngd` (NGD+Adam), with the
hyperparameter optimizer swapped from Adam to LBFGS. Keeps NGD on the
variational natural parameters.

Spec: investigations/default_gpy_gap_v2/SCRAPBOOK.md §39-46 (Phase 3E).

Locked design choices (user-approved before this file was written):

  Q1 — ES patience   : patience=15, min_delta_rel=1e-3, min_iterations=10
                       (= vargp_direct's values; per §43, NGD+LBFGS outer
                       iter cost is similar to vargp's outer iter).
  Q2 — Warm-up       : none. LBFGS starts at outer iter 1. If iter-1
                       strong-wolfe rejection is a problem we'll see it in
                       the prototype and add 10 NGD-only warm-up iters.
  Q3 — Loss scale    : per-point inside the LBFGS closure, via
                       `mll(output, r_train)` reused from the outer
                       VariationalELBO. This matches the scaling contract
                       used by NGD (see `ngd_training.py:86-93`).
  Q4 — Cadence       : LBFGS step on every outer iteration (spec §40).
                       Alternative cadences (e.g. LBFGS every N iters)
                       flagged in SCRAPBOOK §46 as possible-future-work.
  Q5 — fix_Amp       : honoured by filtering the LBFGS param list to
                       `p for p in ... if p.requires_grad`. Adam did this
                       implicitly via zero-grad; LBFGS needs the filter
                       explicit because it reads the param list once at
                       construction and rebuilds the state vector from it.

Closure guards are reused verbatim from `gpy_training.py:260-298`:
  - `kernel.params_in_bounds()` before the forward
  - `likelihood.params_in_bounds()` before the forward
  - try/except around `model(X_train)` for Cholesky failures during the
    strong_wolfe line search
  - `f_mean` explosion guard (F_MEAN_MAX_THRESHOLD / F_MEAN_MEAN_THRESHOLD)
  - NaN/Inf guard on the final loss

Logging uses the post-2026-04-22 pre-step convention (see SCRAPBOOK §47):
`train_loss`, `train_log_lik`, `train_kl` are all pre-step full-dataset
values at each outer iter i.
"""
from __future__ import annotations

import time

import gpytorch
import torch
from linear_operator import settings as lo_settings

# Investigation-local file → reach the project root imports.
# (sys.path is extended by the runner before import.)
from _constants import (
    JITTER,
    CHOLESKY_MAX_TRIES,
    GPY_LBFGS_MAX_ITER,
    F_MEAN_MAX_THRESHOLD,
    F_MEAN_MEAN_THRESHOLD,
    NGD_LR,
)


def train_ngd_lbfgs(
    model,
    likelihood,
    X_train,
    r_train,
    *,
    n_iterations,
    ngd_lr=NGD_LR,
    adam_lr=None,  # ignored — accepted so this is a drop-in monkey-patch for train_ngd
    lbfgs_lr=1.0,
    # V5: damp the inner LBFGS loop. Default was GPY_LBFGS_MAX_ITER (20 from
    # _constants.py — what default_gpy and V0..V3 used). Setting to 1 means
    # a single line-search step per outer iter, which approximates damped
    # Newton with strong_wolfe rather than a full LBFGS inner trajectory.
    lbfgs_max_iter=1,
    jitter=JITTER,
    cholesky_max_tries=CHOLESKY_MAX_TRIES,
    f_mean_max_threshold=F_MEAN_MAX_THRESHOLD,
    f_mean_mean_threshold=F_MEAN_MEAN_THRESHOLD,
    device=None,
    print_every=5,
    early_stop=True,
    # Phase 3E Q1(a): vargp-scale ES defaults.
    patience=15,
    min_delta_rel=1e-3,
    min_iterations=10,
    restore_best=True,
    # V5 default: V2 (joint + warmup) + damped LBFGS inner loop.
    # History of the defaults flag pair (separate_likelihood, n_warmup):
    #   V0 (False, 0):  joint LBFGS fails on cell 30 (2/3 seeds).
    #   V1 (True,  0):  F-step LBFGS aggressively hits A→0; cell 30 3/3, cell 40 2/3.
    #   V2 (False, 50): warm-up fixes 4/5 cells; cell 30 still disasters.
    #   V3 (True,  50): warm-up wasted; cell 30 A-collapse on 3/3 seeds.
    #   V5 (False, 50) + `lbfgs_max_iter=1`:  damp the inner LBFGS loop.
    # V5 hypothesis: the A→0 attractor is reached by strong_wolfe line search
    # taking one large step to the basin. Limiting to 1 inner LBFGS iter
    # (≈ damped-Newton step with line search) prevents the full jump.
    separate_likelihood=False,
    # Warm-up: run `n_warmup` NGD-only iters with kernel + likelihood frozen
    # at init before enabling LBFGS. This lets q(λ̃) acquire structure that
    # correlates with the data, so at the first LBFGS iter ∂ell/∂A has a
    # non-vanishing signal term that counteracts the A→0 attractor. Without
    # warm-up (n_warmup=0), the iter-1 LBFGS line search sees a zero-signal
    # gradient dominated by -A·σ²·f_mean, pushing A to 0. See SCRAPBOOK
    # §40 Phase 3E debrief for the full mechanism.
    n_warmup=50,
    test_r_probe=None,
    test_r_every=5,
):
    """Train a variational GP via NGD (variational params) + LBFGS (hyperparams).

    Parameters
    ----------
    model : VariationalGPModel
        Must be constructed with `variational_distribution_cls='tril_natural'`
        (same precondition as `train_ngd`).
    likelihood : PoissonLikelihood
    X_train, r_train : Tensors (full batch)
    n_iterations : int
        Max outer iterations (NGD + LBFGS pair). Each outer iter triggers
        up to `lbfgs_max_iter` inner LBFGS steps (via strong_wolfe line
        search), so one outer iter here is comparable in compute to one
        vargp outer iter.
    ngd_lr : float
    lbfgs_lr : float
        LBFGS initial step size. With strong_wolfe, line search may deviate.
        1.0 matches `gpy_training.train_gpy_default`.
    lbfgs_max_iter : int
        Max inner LBFGS iterations per outer step. Default from
        `_constants.GPY_LBFGS_MAX_ITER`, matches default_gpy.
    f_mean_max_threshold, f_mean_mean_threshold : float
        Firing-rate explosion guards inside the LBFGS closure.
    patience, min_delta_rel, min_iterations, restore_best : ES params.
        Phase 3E Q1(a): defaults are (15, 1e-3, 10, True) — vargp's values.

    Returns
    -------
    dict with the same schema as `train_ngd`:
      losses, stopped_early, best_iteration, final_iteration, diverged,
      diverged_at, curves (includes per-iter LBFGS diagnostics:
      lbfgs_n_func_evals_curve, lbfgs_closure_infs_curve).
    """
    if device is None:
        device = X_train.device
    model = model.to(device)
    likelihood = likelihood.to(device)
    X_train = X_train.to(device)
    r_train = r_train.to(device)

    if getattr(model, 'variational_distribution_kind', None) != 'tril_natural':
        raise ValueError(
            "train_ngd_lbfgs requires a VariationalGPModel built with "
            "variational_distribution_cls='tril_natural'. Got "
            f"variational_distribution_kind="
            f"{getattr(model, 'variational_distribution_kind', 'MISSING')!r}."
        )

    model.train()
    likelihood.train()
    n_data = r_train.size(0)

    # NGD optimizer — variational natural params only.
    ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(), num_data=n_data, lr=ngd_lr,
    )

    # LBFGS optimizer(s). Filtered by requires_grad (Q5a) so `fix_Amp` works.
    kernel_params = [p for p in model.hyperparameters() if p.requires_grad]
    likelihood_params = [p for p in likelihood.parameters() if p.requires_grad]

    if separate_likelihood:
        # Variant A — vargp-style E/M/F separation at the LBFGS level.
        # One LBFGS for kernel hyperparams, one for A/lambda0. The coupling
        # that drove cell-30 seed-789's A-collapse (A=8.7e-4) in the joint
        # baseline cannot happen here because A is not in the kernel LBFGS's
        # param list.
        if not kernel_params:
            raise ValueError("No trainable kernel hyperparameters for LBFGS.")
        if not likelihood_params:
            raise ValueError("No trainable likelihood parameters for LBFGS.")
        lbfgs_kernel = torch.optim.LBFGS(
            kernel_params, lr=lbfgs_lr, max_iter=lbfgs_max_iter,
            line_search_fn='strong_wolfe',
        )
        lbfgs_likelihood = torch.optim.LBFGS(
            likelihood_params, lr=lbfgs_lr, max_iter=lbfgs_max_iter,
            line_search_fn='strong_wolfe',
        )
        lbfgs_optimizer = None
    else:
        # Joint baseline (the failing configuration; kept for direct
        # comparison).
        hyp_params = kernel_params + likelihood_params
        if not hyp_params:
            raise ValueError(
                "No trainable hyperparameters found for LBFGS. Check that "
                "requires_grad is True on at least one kernel/likelihood param."
            )
        lbfgs_optimizer = torch.optim.LBFGS(
            hyp_params,
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            line_search_fn='strong_wolfe',
        )
        lbfgs_kernel = None
        lbfgs_likelihood = None

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_data)

    # Curve buffers (match train_ngd schema + LBFGS diagnostics).
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
        # LBFGS-specific diagnostics.
        'lbfgs_n_func_evals': [],          # total closure calls this outer iter
        'lbfgs_closure_infs': [],          # any closure call returned +inf
        # Variant A splits the counter between kernel- and likelihood-steps
        # so we can see if the two inner loops have different characteristics.
        'lbfgs_kernel_evals': [],
        'lbfgs_likelihood_evals': [],
    }

    kernel = model.covar_module
    has_beta = hasattr(kernel, 'beta')
    has_rho = hasattr(kernel, 'rho')
    has_eps_0x = hasattr(kernel, 'eps_0x')
    has_eps_0y = hasattr(kernel, 'eps_0y')
    has_clamp_kernel = hasattr(kernel, 'clamp_hyperparameters')
    has_clamp_like = hasattr(likelihood, 'clamp_params')
    has_kernel_bounds = hasattr(kernel, 'params_in_bounds')
    has_like_bounds = hasattr(likelihood, 'params_in_bounds')

    diverged = False
    diverged_at = None
    final_iteration = 0

    # Closure-scoped counters. Mutable containers so the closures can write
    # to them without declaring nonlocal for every outer iter.
    closure_state = {'n_evals_k': 0, 'n_evals_f': 0, 'any_inf': False}

    dtype = X_train.dtype
    inf_loss = torch.tensor(float('inf'), device=device, dtype=dtype)

    def _zero_all_grads():
        """Zero grads across every optimizer's param list.

        Each LBFGS closure produces a backward that accumulates grads on
        every parameter (kernel + likelihood + variational), because
        `-mll(out, r)` depends on all of them. Without zeroing everything,
        a stale gradient from a prior backward would bias the next LBFGS
        step. Zeroing the NGD optimizer's params too is belt-and-suspenders
        — NGD's own zero_grad at the top of each outer iter already handles
        them, but redundant zeroing is free.
        """
        ngd_optimizer.zero_grad()
        if separate_likelihood:
            lbfgs_kernel.zero_grad()
            lbfgs_likelihood.zero_grad()
        else:
            lbfgs_optimizer.zero_grad()

    def _elbo_forward_backward():
        """Full-ELBO forward + backward with guards.

        Guards (reused from gpy_training.py:260-298):
          1. kernel.params_in_bounds()
          2. likelihood.params_in_bounds()
          3. try/except around model(X_train) — Cholesky may fail in line search
          4. f_mean explosion guard
          5. NaN/Inf on the final loss

        Returns the per-point -ELBO scalar (same scale NGD saw on its
        outer backward), or +inf if any guard triggers.
        """
        if has_kernel_bounds and not kernel.params_in_bounds():
            closure_state['any_inf'] = True
            return inf_loss
        if has_like_bounds and not likelihood.params_in_bounds():
            closure_state['any_inf'] = True
            return inf_loss
        try:
            out = model(X_train)
        except Exception:
            closure_state['any_inf'] = True
            return inf_loss

        A_val = likelihood.A.squeeze()
        lambda0_val = likelihood.lambda0.squeeze()
        f_mean = torch.exp(
            A_val * out.mean + 0.5 * A_val ** 2 * out.variance + lambda0_val
        )
        if (torch.any(torch.isnan(f_mean))
                or f_mean.max().item() > f_mean_max_threshold
                or f_mean.mean().item() > f_mean_mean_threshold):
            closure_state['any_inf'] = True
            return inf_loss

        loss = -mll(out, r_train)
        if not torch.isfinite(loss):
            closure_state['any_inf'] = True
            return inf_loss

        loss.backward()
        return loss

    def closure_kernel():
        """LBFGS_kernel / joint-LBFGS closure.

        Used by `lbfgs_kernel` in separate_likelihood mode, and by
        `lbfgs_optimizer` in the joint baseline. Same body either way — the
        LBFGS instance itself decides which params to update from its
        param_groups.
        """
        closure_state['n_evals_k'] += 1
        _zero_all_grads()
        return _elbo_forward_backward()

    def closure_likelihood():
        """LBFGS_likelihood closure (separate_likelihood mode only).

        Separate counter so we can see whether the kernel and likelihood
        inner loops have different line-search behaviour (e.g. does the
        F-step converge in 1-2 evals while the M-step takes 20+?).
        """
        closure_state['n_evals_f'] += 1
        _zero_all_grads()
        return _elbo_forward_backward()

    # ELBO ES state (decoupled best-tracking / patience counter, matches
    # ngd_training.py and gpy_training.py).
    best_es_value = float('-inf')
    patience_reference = float('-inf')
    patience_counter = 0
    best_state = None
    best_iteration = 0
    stopped_early = False

    # --- Warm-up: NGD-only iters with kernel + likelihood frozen.
    # The `requires_grad` state is saved and restored so that anything the
    # caller froze BEFORE calling us (e.g. `fix_Amp` freezing raw_Amp) stays
    # frozen afterwards. Only params the caller left trainable get the
    # freeze-then-unfreeze treatment.
    warmup_curves = {'train_loss': [], 'iter_time': []}
    if n_warmup > 0:
        saved_req_grad = []
        for p in list(model.hyperparameters()) + list(likelihood.parameters()):
            saved_req_grad.append((p, p.requires_grad))
            p.requires_grad_(False)
        if print_every > 0:
            print(
                f"  Warm-up: {n_warmup} NGD-only iters (kernel + likelihood "
                f"frozen at init).",
                flush=True,
            )
        with lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
             lo_settings.cholesky_max_tries(cholesky_max_tries):
            for w in range(n_warmup):
                t0 = time.time()
                ngd_optimizer.zero_grad()
                output = model(X_train)
                loss_w = -mll(output, r_train)
                if not torch.isfinite(loss_w):
                    # Warm-up shouldn't diverge (hyperparams frozen at safe init),
                    # but guard anyway.
                    if print_every > 0:
                        print(f"  Warm-up diverged at iter {w + 1}; stopping warm-up.",
                              flush=True)
                    break
                loss_w.backward()
                ngd_optimizer.step()
                warmup_curves['train_loss'].append(loss_w.item() * n_data)
                warmup_curves['iter_time'].append(time.time() - t0)
        # Restore requires_grad — the main loop needs these to be trainable
        # again so LBFGS can step them.
        for p, rg in saved_req_grad:
            p.requires_grad_(rg)
        if print_every > 0 and warmup_curves['train_loss']:
            print(
                f"  Warm-up done: loss {warmup_curves['train_loss'][0]:.1f} → "
                f"{warmup_curves['train_loss'][-1]:.1f} "
                f"over {len(warmup_curves['train_loss'])} iters.",
                flush=True,
            )

    with lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):

        for i in range(n_iterations):
            iter_start = time.time()

            # --- PRE-step forward for logging + NGD backward ---
            output = model(X_train)
            loss = -mll(output, r_train)  # per-point -ELBO, PRE-step

            if not torch.isfinite(loss):
                diverged = True
                diverged_at = i + 1
                break

            with torch.no_grad():
                ell_full_pre = likelihood.expected_log_prob(r_train, output).item()
                kl_full_pre = model.variational_strategy.kl_divergence().item()

            # --- NGD step on variational natural params ---
            ngd_optimizer.zero_grad()
            loss.backward()
            ngd_optimizer.step()  # q_i -> q_{i+1}

            # --- LBFGS step on hyperparams. In separate_likelihood mode
            #     (Variant A) this is two sequential LBFGS.step() calls, one
            #     for the kernel and one for A/lambda0, mirroring vargp's
            #     E/M/F separation. In joint mode it's a single step over all
            #     hyperparams (the Phase 3E-baseline that failed on cell 30).
            closure_state['n_evals_k'] = 0
            closure_state['n_evals_f'] = 0
            closure_state['any_inf'] = False
            lbfgs_crashed = False
            try:
                if separate_likelihood:
                    # M-step: kernel hyperparams only.
                    lbfgs_kernel.step(closure_kernel)
                    # F-step: A/lambda0. Runs with the POST-M-step kernel +
                    # POST-NGD-step variational posterior.
                    lbfgs_likelihood.step(closure_likelihood)
                else:
                    lbfgs_optimizer.step(closure_kernel)
            except (IndexError, RuntimeError) as e:
                lbfgs_crashed = True
                if print_every > 0:
                    print(
                        f"  LBFGS crashed at outer iter {i + 1}: {e!r}. "
                        f"Stopping training.",
                        flush=True,
                    )

            if has_clamp_kernel:
                kernel.clamp_hyperparameters()
            if has_clamp_like:
                likelihood.clamp_params()

            # loss.item() * n_data is exactly -ELL_sum_pre + KL_full_pre
            # (algebraic identity with VariationalELBO; matches ngd_training).
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

            curves['lbfgs_kernel_evals'].append(closure_state['n_evals_k'])
            curves['lbfgs_likelihood_evals'].append(closure_state['n_evals_f'])
            curves['lbfgs_n_func_evals'].append(
                closure_state['n_evals_k'] + closure_state['n_evals_f']
            )
            curves['lbfgs_closure_infs'].append(bool(closure_state['any_inf']))
            curves['iter_time'].append(time.time() - iter_start)
            final_iteration = i + 1

            if lbfgs_crashed:
                diverged = True
                diverged_at = i + 1
                break

            if test_r_probe is not None and (i + 1) % test_r_every == 0:
                model.eval()
                likelihood.eval()
                with torch.no_grad():
                    tr = float(test_r_probe(model, likelihood))
                model.train()
                likelihood.train()
                curves['test_r_iter'].append(i + 1)
                curves['test_r'].append(tr)

            # --- ELBO ES (decoupled best + patience, matches ngd_training) ---
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
                rel_improvement = (
                    (es_value - patience_reference) / max(abs(patience_reference), 1e-8)
                )
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
                    f"lbfgs_evals={closure_state['n_evals_k']}+{closure_state['n_evals_f']}  "
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
        'warmup_curves': warmup_curves,
        'n_warmup_run': len(warmup_curves['train_loss']),
    }
