This document collects next steps or important but not urgent tasks to be completed regarding the porting to gpytorch

Things to remember having claude check:

[X] Masking of C -> works
[ ] Is it a good idea to detach it so that mask does not change during backprop?
[X] Visualization -> first version in tes_fit prints in imgs/
[ ] Comparison with one_cell_fit script
[ ] I think hyperparameters are still parametrized as weird log expressions ( like beta for example) and they do not use the gpytorch link functions. to change
[ ] Documentation refers too many times to "PNAS data" , we should remove it.
[ ] has_lengthscale parameter in acosker
[ ] for efficiency, we should check that we are not computing the C matrix again when its not used
[ ] Clamping of acosker ( cos_theta = torch.clamp(C12 / M, -1.0 + eps, 1.0 - eps)) in ArcCosineKernel
[ ] Maybe enlarge the testing suite for the kernel
[ ] Is the loop in train_model standard of gpytorch?             
            output = model(train_x)

            # ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))
            expected_log_lik = likelihood.expected_log_prob(train_y, output)
            kl_div = model.variational_strategy.kl_divergence()

            # Minimize negative ELBO
            loss = -expected_log_lik + kl_div

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

[ ] Check that the beta/rho conversion from the log-space parameters is correct. we can use the methods defined by pietro
[ ] Still isnt clear if the r_cutoff has an influence on stability ( is 100 enough?)
[ ] Are we including the ntilde in the ntrain?
[ ] To check if we can condense the mathematical foundation file into the claude.md file to not dispere too much
[ ] No need to create compute_moments() function
[X] old code seems to have 0 mstep as defauls BUG
- Notes on Estep failures:
    - M=50 leads to constant prediction and rtrain=0, no overfitting. maybe A is going to 0?

[X]   from gpytorch.constraints import Positive IMPLEMENTED
  Positive(transform=torch.exp, inv_transform=torch.log)    
-> htis might be a way to define a likelihood that optimized logA instead of A, globally.