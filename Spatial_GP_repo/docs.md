

### When are matrices positive definite in the code and when are they not

Matrix K_tilde and V in the code should always be positive definite ( simmetric, with eigenvalues greater than zero )

In the **E step** V ( V_b ) gets updated, a version for updating V_inv exists but is not to be used for not ( not tested )

With alpha = 1 the current value of V_b is not actually used, and the updated value is guaranteed to be positive definite:

From Estep:

```    
    G = A*A * KKtilde_inv.T@(KKtilde_inv*f_mean[:,None]) 
    # G = f_params['A']*f_params['A'] * torch.einsum('ij,jk->ikj', KKtilde_inv.T, KKtilde_inv) @ f_mean 
   
    # Updates on V can be performed on V or V_inv. Results are the same but the former is more stable cause it does not ivert directly V, but rather solves a linear system.
    # The update on V also allows for a smaller step size regulated by alpha. ( Not yet stable )
    # Results are still the best with alpha=1 (static images) but if the E step was to give problems the stable implementation of alpha!=1 should be the first thing
    if update_V_inv == False and K_tilde is not None:
        if alpha==1:
            V_new = torch.linalg.solve( torch.eye(K_tilde.shape[0]) +  K_tilde@G, K_tilde) #-> current V_b is not used
```

Values of alpha!=1 could be useful if the model is not converging cause the newton update actually relies on a quadratic approximation of the loss ( the -logmarginal ), which could be wrong. The problem is that they dont ensure positive definiteness of the output V_b.

Because of the positive definiteness assumption , in the case update_V_inv = False and alpha = 1 , V_b will be positive definite in the whole code, except when being reprojected to the eighenspace from V_b_old when the hyperparameters have changed ( and hence the supbspace ). For this reason the **warning in the log_det function is disabled for the V function**. The only place in which it could execute is inded in the case of reprojection onto an eigenspace of higher dimension, but **for alpha=1 and updata V_b = True this does not present a problem cause V_b is not being used**.

### When are matrices non simmetric?

In the same lines of code as the above, when V_b gets reprojected there are numerical errors that prevent it from being simmetric.

The reprojected matrix ```V_b_new = B.T@(B_old@V_b@B_old.T)@B``` is not even simmetric most of the time, with an error that can go up to 1.e-11 ( for certain cells ). Precision should be 1.e-15 for float64 but to avoid errors MIN_TOL was set to 1.e-11.

### Why does the LBFGS method sometimes fail

The LBFGS method is a quasi Newton method, this means that it is using an approximate of the inverse of the Hessian, therefore an approximation of the curvature, to choose how long of a step to take in the update direction in the parameters space. This direction is dependent on the inverse of the Hessian and the gradient at the current point. 

The first approximate inverse of the Hessian is chosen as the identity.

A history of the previous gradients is kept ( in ```old_dirs = state.get('old_dirs') old_stps = state.get('old_stps') ``` line 373 lbfgs.py ), and is used both to compute the next search direction, but also to modify dinamically the step size ( basically, change the learning rate.). Therefore the torch.optim.LBFGS has to be defined outside the loops it's used in, otherwise this history gets flushed at each new instantiation.

The use of a line search

#### How does LBFGS work

It uses a line search ( wolfe conditions ) to decide the step size

[i] Armijo rule: If satisfied, the objective function has decreased enough with the chosen step size aplha. If the step size wal greater, we could end up at a higher loss value. Upper bound on alpha

[ii] Curvature condition: If satisfied, the slope has decreased sufficently. If we stepped closer, we might not get enough decrease. Lower bound on alpha.

The lbfgs used function checks right away if the first is satisfies, if its not, the chosen t is the higher bound of the braket in which value of f(x) satisfies the condition, therefore we hav a braket containning an optimal point.


