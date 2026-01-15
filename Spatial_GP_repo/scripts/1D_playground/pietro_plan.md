
The 1d playground scope was to :

[OK] confirm intuition that standard utility (nd_utility) leads to extremes values explorations in the active loop, due to increase in uncertainty in those areas. 

This is true, and the distribution aware one does indeed cluster utility on the distribution p(x)  [gp_utility_playground_result.png]

It remains unclear why the real gp algorithm for predicting neural responses to natural images does not converge to the sampled image x->lambda(x) when this is the only sample used in the monte carlo sampling used for estimating the conditional entropy.

BIG QUESTION still remains:
[ ] understand why in the high dimensional case image optimization for utility still leads to extreme pixel values ( even outside bounds )

Step by step:

I need to check in the 1D case :
[OK] Sampling X samples from a given SET instead of a distribution does make the utility point to the right area of the domain. this can be decoupled from the optimization loop of the point above if I just take a set of x instead of a distribution. Utility should peak on that set.
[OK] Optimizaiton of X . Does x_opt converge to x_sample in 1D? it should absolutely

    [OK] It does, but its using an approximation for the E[H_cond] by using directly E[lambda(x_sample)] in the formula instead of sampling with monte carlo. 
    [OK] Numerical problems. For a starting point x far awat from tha sampled x, utility is zero and gradients completely unreliable
    [ ] Investigate how much the error this gives ( MC vs <lambda_sample>)

[ ] One more thing to try would be to use an arcosin kernel and a true lambda that comes from the prior. This ensures correct model choice 

# In Parallel : Gpytorch porting

[ ] Remember to have claude compare the lambda_moments() function and other simple function to the ones we are now using in the utility related scripts. there might be some cleaner alternatives.