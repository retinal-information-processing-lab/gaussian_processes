import numpy as np
import scipy.io

from tqdm import tqdm
import pickle
import torch
import math
import os
import torch.optim as optim
import warnings

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732


# Warnings
warnings.filterwarnings("ignore", "The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated")


## This file was the Spatial_GP.py file in the original code.


TORCH_DTYPE = torch.float64
# Set the default dtype to float64
torch.set_default_dtype(TORCH_DTYPE)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

print(f'Using device: {DEVICE} (from utils.py)')
##################   Utility functions  ##################

def h_r_given_xD ( p_response_tensor):
    # Computes the first term of the utility function [eq 28 Paper PNAS]

    sum    = torch.sum( p_response_tensor*safe_log(p_response_tensor) )
    return -sum

def mean_noise_entropy(p_response_tensor, r, sigma2, mu):
    # Computes the conditional noise entropy < H( r|f,x ) >_p(f|D) [eq 33 Paper PNAS]
    # INPUTS:
    # p_response_tensor: set of probabilities p(r|x,D) for r that goes from 0 to a low number, set in utility(). Should go to infninty but the mean responses are lwo
    
    # argument of the sum, remember gamma(r+1) = r!
    argument = p_response_tensor*torch.lgamma(r+1)
    #argument = p_response_tensor*safe_log( math.factorial(p_response_tensor))
    H_mean = torch.exp(mu + 0.5*sigma2)*(mu + sigma2 - 1) + torch.sum( argument )
    return H_mean

def lambda_r_mean(r, sigma2, mu):
    # Computes the  argmax of the first row of the laplace approxmated logp(r|x,D) [eq 32 Paper PNAS] . Eq 33,34

    # r is a tensor of values from 0 to r_cutoff, its the max numver for the sum in eq 29 Paper PNAS

    prod = r*sigma2
    z    = sigma2 * torch.exp( prod + mu)

    sum_mask = z != torch.inf
    z    = z[sum_mask] # Avoids overflow in the exponential
    prod = prod[sum_mask]


    lamb    = prod + mu - torch.real(scipy.special.lambertw( z=z.cpu(), k=0, tol=1.e-8).to(DEVICE)) # Take only the real part

    # print(f' Kept {z.shape[0]} values for the summation in the Utility function')
    return lamb, sum_mask 

def p_r_given_xD(r, sigma2, mu):
    # Computes p( r|x,D ) [eq 31 Paper PNAS]. It's the first term of I(r;f|x,D) [eq:27] 

    # Callculating lambda mean for different values of r, keeping only the ones for which an exponential is not infinite
    lambda_mean, sum_mask = lambda_r_mean(r, sigma2, mu)
    ex_lambda_mean = torch.exp(lambda_mean)

    r = r[sum_mask]  

    # To calculate log(r!) we use the fact that the Gamma function of an integer is the factorial of that integer -1
    # G(r+1) = r! and torch. provides exactly the logarithm of this
    # torch.lgamma(r+1) = log( G(r+1) ) = log( r! )
    log_p = lambda_mean*r - ex_lambda_mean - ((lambda_mean-mu)**2)/(2*sigma2) - 0.5*safe_log( sigma2*ex_lambda_mean + 1) - torch.lgamma(r+1) # TODO: is this factorial too slow?

    return torch.exp(log_p), r

def utility( sigma2, mu ):
    # Computes the utility function [eq 27 Paper PNAS]

    # Generates the tensor of the probability of the responses p(r|x,D)
    # r_cutoff: for r that goes from 0 to a low number set by r_cutoff, 
    # r_tensor_masked: this will as well be reduced if the exponential in the lambda_r_mean function goes to infinity

    # sigma2 and mu are the mean and variance over lambda 
    r_cutoff = 100 
    r_tensor = torch.arange(0, r_cutoff, dtype=TORCH_DTYPE)

    # Returns the p(r|x,D) and the masked r tensor to use in the sum in mean_noise entropy
    p_response_tensor, r_tensor_masked = p_r_given_xD( r=r_tensor, sigma2=sigma2, mu=mu) 

    a = h_r_given_xD( p_response_tensor )
    b = mean_noise_entropy( p_response_tensor, r_tensor_masked, sigma2, mu )
    U = a - b

    # print(f'Sigma2 = {sigma2.item():.4f}, H(r|x,D) = {a.item():.4f}, <H(r|f,x)> = {b.item():.4f}, U = {U.item():.4f}')
    return U

def get_utility(xstar, xtilde, C, mask, theta, m, V, K_tilde, K_tilde_inv, kernfun):
            
    if K_tilde_inv is None:
        K_tilde_inv = torch.linalg.solve(K_tilde, torch.eye(K_tilde.shape[0]))

    xstar = xstar.unsqueeze(0)

    # Inference on new input(s)
    mu_star, sigma2_star = mu_sigma2_xstar(xstar[:,mask], xtilde[:,mask], C, theta, K_tilde_inv , m, V, kernfun=kernfun)

    return utility( sigma2=sigma2_star, mu=mu_star )
##########################################################

def safe_log(x):
    # Wrapper function to log that checks for negative or zero input
    if torch.any(x <= 0):
        raise ValueError("Negative or zero input to log detected")
        # print("Warning: Negative or zero input to log detected")
    if torch.any(x < 1e-10):
        raise ValueError("Very small input to log detected")
        # print("Warning: Very small input to log detected")
    return torch.log(x)

def safe_acos(x):
    # Wrapper function to acos that checks for input out of range
    # if torch.any(x < -1 - 1.e-7) or torch.any(x > 1 + 1.e-7):
        # raise ValueError("Input out of range for acos")

    if torch.any( x > 1 - 1e-6) or torch.any( x < -1 + 1e-6):
        # warnings.warn(" Input very close to the edge of the range for acos, clamping it to the range")
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)

    acos = torch.acos(x)
    return acos
##################   Initialization   ####################

def save_pickle(filename, **kwargs):

    # To use call as:  save_pickle('pietro_data', **{'K_pietro':K, 'K_tilde_pietro':K_tilde})    
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))


    # Create the path for the pickle file
    pickle_path = os.path.join(script_dir, f'data/{filename}.pkl')
        # Save the file
    with open(pickle_path, 'wb') as f:
        pickle.dump(kwargs, f)

    return 0

def generate_xtilde(ntilde, x):

    tilde_idx = torch.randperm(ntilde)
    xtilde_first = x[ tilde_idx, :]
    epsi = torch.finfo(TORCH_DTYPE).eps*10*torch.randn(xtilde_first.shape).to(DEVICE) # TODO is randn missing arguments? this is different than the implementation of smauele
    xtilde = xtilde_first + epsi
    return xtilde

def logbetaexpr_to_beta(theta):

    # Go from the logarithmic expression of beta used in the learning algorithm to beta of the PNAS Paper 
    beta_paper = torch.exp(-0.5*theta['-2log2beta']) * torch.tensor(0.5)
    return beta_paper

def logrhoexpr_to_rho(theta):
    # Go from the logarithmic expression of rho used in the learning algorithm to rho of the PNAS Paper
    rho_paper = torch.exp(-0.5 * theta['-log2rho2']) /  torch.sqrt( torch.tensor(2)) 
    return rho_paper

def fromlogbetasam_to_logbetaexpr( logbetasam ):
    # Go from the value of logbeta_sam used in his code to the logbetaexpr used in this code
    logbetaexpr = logbetasam - torch.log(torch.tensor(2.0))
    return logbetaexpr

def fromlogrhosam_to_logrhoexpr( logrhosam ):
    # Go from the value of logrho_sam used in his code to the logrhoexpr used in this code
    logrhoexpr = logrhosam - torch.log(torch.tensor(2.0)) 
    return logrhoexpr

def get_sta( x, r, n_px_side):   
    
    # INPUTS
    # x, r Dataset with wich to calculate sta. Shape x = (nt,nx), r = (nt)

    img_mean   = torch.matmul( torch.t(x), torch.ones_like(r))/ r.shape[0] 
    sta        = torch.matmul( torch.t(x), r)/r.shape[0]-img_mean # Is it without average? #TODO check
    sta_matrix = torch.reshape(sta, (n_px_side, n_px_side))
    # find max index
    max_idx = torch.argmax(torch.abs(sta_matrix))
    
    row_idx =  max_idx // sta_matrix.shape[1]    
    col_idx =  max_idx %  sta_matrix.shape[1] 

    # Manually chosen a width of the RF. TODO calculate it in a meaningful way
    sta_variance = torch.tensor(10)  

    return sta, sta_variance, (row_idx, col_idx)

def generate_theta(x, r, n_px_side, display=False, **kwargs):
        
        # Initializes the values of the Kernel and C hyperparameters and their limits
        # Some of the hyperaparameters are learnt as their log with some factors.
        # These factors where different in Matthew's code and Samuele's code. See hyperparameters_conversion.txt for details. 

        # sigma_0 : Acoskern specific hyperparameter.
        # Amp :     Amplitude of the localker 
       
        # eps_0 = (eps_0x, eps0y) : Center of the receptive field as real numbers from -1 to 1. 
            # They are obtained from the STA that gives them bas initially as integers (pixel position)
       
        # logbetaexpr : a logarithmic expression of beta, which is is the scale of local filter of C in the paper.    
        # logrhoexpr  : a logarithmic expression of rho, which is the scale of the smoothness filter part of C.

        # The hyperparameters will be used in: 
        # C_smooth : (nx, nx) gaussian shaped exponential of distance between x. Can be seen as the covariance of the gaussian for the weights (in weights view)


        # Choose how to rescale the receptive field:
        up_lim  =  1
        low_lim = -1
        # half_n_px_side = n_px_side/(up_lim-low_lim) not used

        # _____ Sigma_0 and A _____
        sigma_0 = torch.tensor(0., requires_grad=True) # This makes sigma_0 = 0
        Amp     = torch.tensor(1., requires_grad=True) # Amplitude of the receptive field Amp = 1, NOT PRESENT IN SAMUELE'S CODE

        # Center and size of the receptive field RF as sta and its variance (rf_width_pxl2). eps are indeces from 0 to 107
        sta, rf_width_pxl2, (eps_0x, eps_0y) = get_sta(x, r, n_px_side)  # rf_width_pxl is manually chosen for now TODO


        # eps_0 go from 0 to 107, n_px_side = 108. I then bring it to [-1,1] multiplying by 2 and shifting
        eps_0x_rescaled = ( eps_0x / (n_px_side - 1))*2 - 1
        eps_0y_rescaled = ( eps_0y / (n_px_side - 1))*2 - 1       

        # _____ Temporary _____
        rf_width_pxl2   = torch.tensor(10, dtype=TORCH_DTYPE)  
        # eps_0x = torch.tensor(2, dtype=TORCH_DTYPE) # Center of the RF in pixels
        # eps_0y = torch.tensor(3, dtype=TORCH_DTYPE) # Center of the RF in pixels
        eps_0x_rescaled = torch.tensor(0.0, requires_grad=True)
        eps_0y_rescaled = torch.tensor(0.0, requires_grad=True)

        # Make them learnable
        eps_0x_rescaled.requires_grad = True
        eps_0y_rescaled.requires_grad = True

        if not (low_lim <= eps_0x_rescaled <= up_lim) or not (low_lim <= eps_0y_rescaled <= up_lim):
            raise ValueError(f"eps_0x_rescaled and eps_0y_rescaled must be within the range of {low_lim} and {up_lim}.")
        
        # _____ Beta and Rho _____
        # Here the caracteristic lenght is Dict of the sqrt the  variance of the sta = receptive field pixel size squared.  TODO check
        # It is chosen in pixels but Hransfered to [0,2]. It is also stored as theta[i]=-2log(2*beta) in the dict
        # so that e^(theta[i]) = 1/(4beta2) can be multiplied in the exponent of a_local of the kernel
        rf_width_pxl = torch.sqrt(rf_width_pxl2)
        
        beta         = (rf_width_pxl / n_px_side) * (up_lim-low_lim) #sqrt of variance of sta brought to [0,2]
        # _____ Temp ___
        # beta         = rf_width_pxl 
        # beta = torch.tensor(0.0580, dtype=TORCH_DTYPE)
        # ______________

        logbetaexpr      = -2*safe_log(2*beta) # we call it logbetaexpr cause of the factors in the expression (see hyperparameters_conversion.txt)
        logbetaexpr.requires_grad = True
        
        # Smoothness of the localkernel. Dict of half of caracteristic lenght of the RF
        # It is chosen in Hut then transfered to [0,2]. It is also stored as theta[i]=-log2rho2 in the dict
        rho    = beta/2
        logrhoexpr = -safe_log(torch.tensor(2.0)*(rho*rho)) # we call it logrhoexpr cause of some factors in the expression (see hyperparameters_conversion.txt)
        logrhoexpr.requires_grad = True

        theta = {'sigma_0':sigma_0, 'eps_0x':eps_0x_rescaled, 'eps_0y':eps_0y_rescaled, '-2log2beta': logbetaexpr, '-log2rho2': logrhoexpr, 'Amp': Amp }

        # If theta is passed as a keyword argument, update the values of the learnable hyperparameters
        for key, value in kwargs.items():
            if key in theta:
                theta[key] = value
                print(f'updated {key} to {value}')


        # Print the learnable hyperparameters
        if display:
            print(' Before overloading')
            print(f' Hyperparameters have been SET as  : beta = {beta:.8f}, rho = {rho:.8f}')
            print(f' Samuele hyperparameters           : logbetasam = {-torch.log(2*beta*beta):.4f}, logrhosam = {-2*safe_log(rho):.4f}')
            
            print('\n After overloading')
            print(f' Dict of learnable hyperparameters : {", ".join(f"{key} = {value.item():.8f}" for key, value in theta.items())}')
            print(f' Hyperparameters from the logexpr  : beta = {logbetaexpr_to_beta(theta):.8f}, rho = {logrhoexpr_to_rho(theta):.8f}')
            beta = logbetaexpr_to_beta(theta)
            rho  = logrhoexpr_to_rho(theta)
            print(f' Samuele hyperparameters           : logbetasam = {-torch.log(2*beta*beta):.4f}, logrhosam = {-2*safe_log(rho):.4f}')


            

        # Lower bounds for these hyperparameters, considering that:
        # rho > 0 -> log(rho) > -inf
        # sigma_b > 0 -> log(sigma_b) > -inf
        # beta > e^4 (??) -> log(beta) > 4
        theta_lower_lims  = {'sigma_0': 0           , 'eps_0x':low_lim, 'eps_0y':low_lim, '-log2rho2':-float('inf'), '-2log2beta': -float('inf'), 'Amp': 0. }
        theta_higher_lims = {'sigma_0': float('inf'), 'eps_0x':up_lim,  'eps_0y':up_lim,  '-log2rho2': float('inf'), '-2log2beta':  float('inf'), 'Amp': float('inf') }
        
        return ( theta, theta_lower_lims, theta_higher_lims )

##################   Kernel related functions   ####################

def localker(nx, theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False):
    # Compute C, the part of the kernel responsible for implementing the receptive field and smoothness

    eps_0 = torch.stack([theta['eps_0x'], theta['eps_0y']]) # Do not create new tensor, just stack the two elements to preserve the gradient graph
    xcord = torch.linspace(theta_lower_lims['eps_0y'], theta_higher_lims['eps_0x'], n_px_side)

    # ______ Samuele's code ______

     # spatial localised prior
    ycord, xcord = torch.meshgrid( torch.linspace(-1, 1, n_px_side), torch.linspace(-1, 1, n_px_side), indexing='xy') #a grid of 108x108 points between -1 and 1
    xcord = xcord.flatten()
    ycord = ycord.flatten()
    logalpha    = -torch.exp(theta['-2log2beta'])*((xcord - eps_0[0])**2+(ycord - eps_0[1])**2)
    alpha_local =  torch.exp(logalpha)    #aplha_local in the paper

    mask        = alpha_local >= 0.001
    alpha_local = alpha_local[mask] # cropped alpha local
    logalpha    = logalpha[mask]
    xcord       = xcord[mask]
    ycord       = ycord[mask]

    # smooth prior
    logCsmooth = -torch.exp(theta['-log2rho2'])*((xcord - xcord[:, None])**2+(ycord - ycord[:, None])**2)
    
    C_smooth = torch.exp(logCsmooth)

    # multiply smoothness prior and spatial localised prior
    C = theta['Amp']*alpha_local[:, None]*C_smooth*alpha_local[None, :]

    # make sure its simmetric
    C = (C + C.T)/2# + torch.eye(C.shape[0])*1.e-7

    if grad:
        # derivative with repect to different hyperparameters 
        dC_Amp = C/theta['Amp']
        # derivative with respect to theta['eps_0x'] and theta['eps_0y']
        dC_eps0x = torch.tensor(2)*torch.exp(theta['-2log2beta'])*C*(xcord[:, None]+xcord[None, :]- 2*eps_0[0])
        dC_eps0y = torch.tensor(2)*torch.exp(theta['-2log2beta'])*C*(ycord[:, None]+ycord[None, :]- 2*eps_0[1])
        # derivative with respect to theta['-2log2beta']
        dC_logbetaexpr = C*(logalpha[:, None] + logalpha[None, :])
        # derivative with respect to theta['-log2rho2']
        dC_logrhoexpr  = C*logCsmooth
        dC = {'Amp':dC_Amp, '-2log2beta':dC_logbetaexpr, '-log2rho2':dC_logrhoexpr, 'eps_0x':dC_eps0x, 'eps_0y':dC_eps0y}
        return C, mask, dC # shape of C: (nx, nx)
    
    else:
        return C, mask

def linker(x1, x2, C, theta, xtilde_case, scalar_case=False):
    # For the moment this does not work, it needs implementation as acosker_samu
    # Compute and return the kernel function given by marix elements k(xi, xj) = xi * C * xj

    #x1 shape  (nt/ntilde, nx) 
    #x2 shape  (nt/ntilde, nx) 

    # Calculating the product C*x2.T to be used as C*Xtilde.T

    Cx2 = torch.matmul(C, x2.T) #(nx, nt/ntilde)

    if scalar_case:
        # We take the diagonal of XCX
        K = torch.einsum('ij,ji->i', x1, Cx2)
        return  K
    
    K = torch.matmul(x1,  Cx2) #  (n, ntilde)
    
    if xtilde_case:
        return  (K + K.T)/2 + 1.e-9*torch.eye(K.shape[0]) # make sure it's simmetric 
    if not xtilde_case:
        return  K 


def acosker_samu(theta, x1, x2=None, C=None, dC=None, diag=False):
    """
    arc cosine covariance function

    Parameters
    ----------
    theta : dictionary of hyperparameters

    x1 : array-like
        [n1, nx] matrix, first input   #(500,ntilde or r.shape)
    x2 : array-like
        [n2, nx] matrix, second input  #(500, ntilde or r.shape)
    C : array-like, optional
        [nx, nx] matrix, smooth local covariance matrix
    dC : array-like, optional
        [nx, nx, ntheta], derivative of C with respect to hyperparameters
    diag : bool, optional
        if diag ==1, then just return diagonal elements of covariance matrix

    Returns
    -------
    K : array-like
        [n1, n2] kernel 
    dK : array-like
        [n1, n2, ntheta], derivative of kernel with respect to theta
    """

    # I am using a direct translation of Samuele's code so I need to transpose the inputs. 

    x1 = x1.T
    if x2 is not None : x2 = x2.T
    n1 = x1.shape[1]
    sigma_0 = theta['sigma_0']

    if C is None: C = torch.eye(n1)

    if not diag:
        n2 = x2.shape[1]
        
        X1 = torch.sqrt(torch.sum(x1*(C @ x1), dim=0) + sigma_0 ** 2) # torch.sum(x1*(C@x1), dim=0) is the same as Diag(x1.T @ C @ x1) #shape(n1)
        X2 = torch.sqrt(torch.sum(x2*(C @ x2), dim=0) + sigma_0 ** 2) # shape(n2,)
        #print((C @ x2)[19,0])
        X1X2 = torch.outer(X1,X2)               #shape(n1,n2)
        x1x2 = x1.T @ C @ x2 + sigma_0 ** 2   
        
        cosdelta = torch.clip(x1x2 / (X1X2 + 1e-7), -1, 1) #  TODO remove this clip (making it more numerically stable)

        delta = torch.arccos(cosdelta)  # angle theta in the paper and Samuele's code.
        # Angle dependend ezpression in eq (8) of the paper
        J = (torch.sqrt(1 - cosdelta ** 2) + torch.pi * cosdelta - delta * cosdelta) / torch.pi   #shape(n1,n2)
        #TODO check if this is not missing a *0.5 as in the paper (there it is /2pi)
        K = X1X2 * J       #shape(n1, n2) ( in case of x1=x and x2=xtilde -> shape (nt,ntilde) )

        if dC is not None:
            # Define the gradient of K with respect to the hyperparameters ( including sigma_0 )
            dK = {}

            dX1X2 = sigma_0 ** 2 * (X2 / X1[:, None] + X1[:, None] / X2) #shape same as X1X2

            dcosdelta = (2 * sigma_0 ** 2 - cosdelta * dX1X2) / X1X2    #shape same as cosdelta

            dJ = -(delta - torch.pi) * dcosdelta / torch.pi           #shapesame as J
            # Add gradient for sigma_0. In Samuele's code what is learned is logsigma_0 so the expression is different
            dK['sigma_0'] =  (X1X2 * dJ + dX1X2 * J)      #shape same as K
                                   
            # for j in range(1, dC.shape[2] + 1):
            for key, value in dC.items():
                if key == 'sigma_0':
                    continue

                dX1 = 0.5*torch.sum(x1*torch.matmul(  dC[key]  , x1), dim=0)/X1  #shape(n1,)
                dX2 = 0.5*torch.sum(x2*torch.matmul(  dC[key]  , x2), dim=0)/X2  #shape(n2,)
                
                dX1X2 = dX1[:, None]*X2 + X1[:, None]*dX2

                dcosdelta = (torch.matmul(x1.T, torch.matmul(dC[key], x2)) - cosdelta*dX1X2)/X1X2

                dJ =  -(delta-torch.pi)*dcosdelta/torch.pi

                dK[key] = X1X2*dJ + dX1X2*J

        # make sure that K is positive definite
        if n1==n2:
            K = (K+K.T)/2 #+ 1e-7*torch.eye(n1)


    else: # In the diagonal case onle the complete dataset passed as x1 is considered
        # return just diagonal of kernel
        K = torch.sum(x1*torch.matmul(C, x1), dim=0)[:, None]+sigma_0**2
        K = K.squeeze() # To return a vector of shape (n1,)
        # Gradient
        if dC is not None:
            dK = {}

            dK['sigma_0'] = (2*sigma_0**2*torch.ones((n1, 1))).squeeze()
            
            for key in dC.keys():
                if key == 'sigma_0':
                    continue
                dK[key] = torch.sum(x1 * torch.matmul(dC[key], x1), dim=0)

            #K += 1e-7 * torch.eye(n1, 1)

    # Returns
    if dC is not None: 
        return K, dK    #shape(n1,n2), shape(n1,n2,6) 
    else: 
        return K    #shape (n1,n2)


def acosker(x1, x2, C, theta, xtilde_case, scalar_case=False, dC=None):
    # NB
    #####################################################
    # Removed the 1/2 from the formula cause its like that in samueles code
    #####################################################

    # Compute and return the kernel function k(x1, x2) using the formula in the paper
    # note that it is reported or x and xprime as vectors, so the result there is a scalar, and all operations
    # must be considered element wise for their matrix equivalent

    # In case inputs are input matrices -> return matrix. scalar_case must be False
    # In case of two vectors            -> return scalar. scalar_case must be True

    # INPUTS before transposition
    # xtilde_case = True switch to choose to calculate K(Xtilde, Xtilde). where Xtilde is the MATRIX of xtildes.
    # The input shapes will be:
    # x1: shape ( ntilde, nx )
    # x2: shape ( ntilde, nx ) 

    # xtilde_case = False. Switch to choose to calculare K(x_i, Xtilde) for every x_i of the dataset. This means K(X. Xtilde) where X is the complete dataset.
    # Will return a matrix of shape (nt, ntilde) where every row is the vector K(x_i, Xtilde)
    # x1: shape ( nt, nx)
    # x2: shape ( ntilde, nx )

    # scalar_case = True -> return a vector Kvec of shape (nt)
    # Before any transposition:
    # x1: shape (nt, nx)
    # x2: shape (nt, nx) they are the same

    # C part of the kernel encoding locality and smoothnes of the Receptive Field (Hyperparameter dependence)
    # C: shape (nx, nx)

    # theta: dict of hyperparameters of the kernel

    # RETURNS
    # K: shape (1) if scalar_case = True
    # else
    # K: shape (ntilde, ntilde) if xtilde_case = True
    # K: shape (n, ntilde) if xtilde_case = False. Matrix whose rows are vectors K(x_i, Xtilde)
    
    sigma_sq = torch.square(theta['sigma_0'])

    if scalar_case:
        # We only transpose x2 case otherwise Id have to traspose x1 again in the einsum
        x2 = x2.T #(nx, nt)
        Cx2 = torch.matmul(C, x2) 
        diagx1Cx2 = torch.einsum('ij,ji->i', x1, Cx2)

        # Kvec = (diagx1Cx2 + sigma_sq)

        
        # Removed the 1/2 from the formula cause its like that in samueles code
        #####################################################
        Kvec = (diagx1Cx2 + sigma_sq)
        #####################################################
        return Kvec
    
    elif xtilde_case:
        
        x2  = x2.T  #(nx, ntilde) 
        Cx2 = torch.matmul(C, x2) #(nx, ntilde)

        x1Cx2 = (torch.matmul( x1, Cx2 ) + sigma_sq) #+ 1e-9*torch.eye(Cx2.shape[1])# (ntilde, ntilde)
        # compute the diagonal
        diagx1Cx2 = torch.diag(x1Cx2) #(ntilde)

        # Create matrix in which each i-th column is the same i-th element of diagx1Cx2 repeated ntilde times
        # all elements of a coumn are the same
        B = diagx1Cx2.repeat(x1Cx2.shape[0], 1) # (ntilde, ntilde)

        # Create the first two factors of the formula K
        product = torch.sqrt(B.T * B) # (ntilde, ntilde)

        cosdelta = torch.div( x1Cx2, product ) # (ntilde, ntilde)
        delta    = safe_acos( cosdelta ) # (ntilde, ntilde
        sindelta = torch.sin( delta )   # (ntilde, ntilde
        angle = sindelta + (-delta+torch.pi)*cosdelta # shape (ntilde, ntilde)

        # the following multiplies every column of angle by the same element of sqrt_diag
        # K = (product * angle) / (2*torch.pi) # shape (ntilde, ntilde)

        # Removing the 1/2 from the formula cause its like that in samueles code
        #######################################################
        K = (product * angle) / (torch.pi) # shape (ntilde, ntilde)
        #######################################################
        K = (K+K.T)/2 + 1.e-6*torch.eye(K.shape[0])    
        

        if dC is not None:
            
            dC = torch.stack(list(dC.values()), dim=2)

            # remember that in the xtilde case    # x1: shape ( ntilde, nx )    # x2: shape ( ntilde, nx )
            # dC is a dictionary of the derivatives of C with respect to the hyperparameters
            n_theta = dC.shape[2]
            dK = torch.zeros((x1.shape[0], x2.shape[0], n_theta + 1)) # ( ntilde, ntilde, n_theta + 1)  derivative of kernel with respect to theta
#     dK : array-like

            # in samueles code 
            # X1 = torch.sqrt(torch.sum(x1*(C @ x1), axis=0) + sigmab ** 2) # np.sum(x1*(C@x1), axis=0) is the same as x1.T @ C @ x1 #shape(n1)
            # X2 = torch.sqrt(torch.sum(x2*(C @ x2), axis=0) + sigmab ** 2) # shape(n2,)

            # if i trust that sum(x1.T*(C@x1.T), axis=0) is the same as x1 @ C @ x1.T #shape(n1)
            # than X1 of samu is is sqrt( x1Cx2 ) of Pietro

            # np.sum(x1*(C@x1), axis=0) is the same as x1.T @ C @ x1 #shape(n1)
            X1 = torch.sqrt(x1Cx2) #shape(ntilde,)
            X2 = X1 # shape(ntilde,)
            X1X2 = torch.outer(X1, X2) #shape(ntilde, ntilde)

            # Samueles arg is my cosdelta

            dX1X2 = sigma_sq * (X2 / X1[:, None] + X1[:, None] / X2)

            darg = (2 * sigma_sq - cosdelta * dX1X2) / X1X2    

            dJ = - (delta - torch.pi) * darg / torch.pi # adding a factor 0.5 that is apparently missing in samuele's code         

            dK[:, :, 0] =  (X1X2 * dJ + dX1X2 * J)      # shape same as K
                                   
            for j in range(1, n_theta + 1):

                dX1 = 0.5*torch.sum(x1.T*(dC[:, :, j-1]@x1.T), axis=0)/X1  #shape(n1,)
                dX2 = dX1
                
                dX1X2 = dX1[:, None]*X2 + X1[:, None]*dX2

                darg = (x1@(dC[:, :, j-1]@x2.T) - cosdelta*dX1X2)/X1X2

                # In samuele's code J    
                J = angle / (2*torch.pi)

                dJ =  -(delta-torch.pi)*darg/torch.pi 

                dK[:, :, j] = X1X2*dJ + dX1X2*J
            
            #print('dX1X2:{}, darg:{}, dJ:{}, dK[:,:,1]:{}'.format(dX1X2.shape,darg.shape, dJ.shape, (A * (X1X2 * dJ + dX1X2 * J)).shape))
            # make sure that K is positive definite    

            return K, dK
        
    elif not xtilde_case: 
        #This is the case there i creating a matrix o vectors K(x_i, Xtilde), so xtilde is actually here
        # Will return a matrix of shape (nt, ntilde) where every row is the vector K(x_i, Xtilde)

        # x1  ( nt, nx)
        X      = x1
        Xtilde = x2 # (ntilde, nx)

        #_______ First two factors
        CX = torch.matmul(C, X.T) # (nx, nt)
        diagXCX = torch.einsum('ij,ji->i', X, CX) + sigma_sq # (nt)

        CXtilde = torch.matmul(C, Xtilde.T) # (nx, nt)
        diagXtildeCXtilde = torch.einsum('ij,ji->i', Xtilde, CXtilde) + sigma_sq # (ntilde)    
        
        # Create matrix in which each i-th column is the same i-th element of diagXCX repeated ntilde times
        # all elements of a coumn are the same        
        BX      = diagXCX.repeat(Xtilde.shape[0], 1) # (ntilde, nt)
        # Create matrix in which each i-th column is the same i-th element of diagXtildeCXtilde repeated nt times
        # all elements of a coumn are the same        
        BXtilde = diagXtildeCXtilde.repeat(X.shape[0], 1) # (nt, ntilde )
        
        product = torch.sqrt(BX.T * BXtilde) # ( ntilde, nx )
        
        #________ Angular factor
        XCXtilde = torch.matmul(X, CXtilde) # (nt, ntilde)
        cosdelta = torch.div(  XCXtilde, product ) # (nt, ntilde)
        delta    = safe_acos( cosdelta )           # (nt, ntilde
        sindelta = torch.sin(  delta )             # (nt, ntilde

        angle = sindelta + (-delta+torch.pi)*cosdelta # shape (nt, ntilde)

        # the following multiplies every column of angle by the same element of sqrt_diag
        # K = (product * angle) / (2*torch.pi) # shape (n, ntilde)

        #######################################################
        K = (product * angle) / (torch.pi) # shape (ntilde, ntilde)
        #######################################################


    return K

##################   Other functions   ####################

def lambda_moments1( x, K, K_tilde, K_tilde_inv, C, m, V, theta, kernfun ):
            # Calculate the mean and variance of lambda over the given training points used to calculate K
            # Formulas for <lambda_i> and Variance(lambda_i) in notes for Pietro are the case of a single training point x_i, therefore a K matrix of shape (1, ntilde)

            # INPUTS
            # x training points x_i over wich we are calculating mean and variance of lambda_i, shape ( nt_or_less, nx )
            # K : matrix of kernel values K(x_i, X_tilde) for every training point x_i in x, shape (nt_or_less, ntilde) 
            # K_tilde matrix of shape (ntilde, ntilde)
            # m : mean of the variational distribution q(lambda) = N(m, V), shape (ntilde, 1)
            # V : variance of the variational distribution q(lambda) = N(m, V), shape (ntilde, ntilde)
            
            #TODO: pass K_tilde inverse ar argument
            #a    = torch.matmul(K, torch.inverse(K_tilde)) # shape (nt, ntilde)
            a   = torch.einsum( 'ij,jk->ik', K, K_tilde_inv ) # shape (nt, ntilde)

            # vector of mean target function for every training point
            lambda_m = torch.matmul( a, m ) # shape (nt, 1)

            # Vector of kernel values kii for every training point
            Kvec = kernfun(x, x, C, theta, xtilde_case=False, scalar_case=True) # shape (nt_or_less, 1)
            # vector of variances of the target function for every training point
            lambda_var = Kvec + torch.einsum( 'ij,ji->i', a, torch.matmul(V-K_tilde, a.T) )

            return lambda_m, lambda_var

def lambda_moments( x, K_tilde, KKtilde_inv, Kvec, C, m, V, theta, kernfun, dK=None , dK_tilde=None, dK_vec=None, K_tilde_inv=None, K=None):
            # Calculate the mean and variance (diagonal of covariance matrix ) of (vec)lambda(of the training points) over the distribution given by:
            # p_cond(lambda|lambda_tilde,X,theta)*(N/q)_posterior(lambda_tilde|m,V) as ini eq (56)(57) of Notes for Pietro

            # Formulas for <lambda_i> and Variance(lambda_i) in notes for Pietro are the case of a single training point x_i, here we are calculating the whole vector of lambda_i ( nt, )

            # INPUTS
            # x training points x_i over wich we are calculating mean and variance of lambda_i, shape ( nt_or_less, nx )
            # K : matrix of kernel values K(x_i, X_tilde) for every training point x_i in x, shape (nt_or_less, ntilde) 
            # K_tilde matrix of shape (ntilde, ntilde)
            # m : mean of the variational distribution q(lambda) = N(m, V), shape (ntilde, 1)
            # V : variance of the variational distribution q(lambda) = N(m, V), shape (ntilde, ntilde)
            
            # KKtilde_inv is calculated outside as:
            # KKtilde_inv = torch.linalg.solve(K_tilde, K.T).T # shape (nt, ntilde) 
            # vector of mean target function for every training point
            a = KKtilde_inv  # shape (nt, ntilde)
            lambda_m = torch.matmul( a, m ) # shape (nt, 1)

            # Vector of kernel values kii for every training point (k_ii in notes for Pietro )
            if Kvec is None:
                Kvec = kernfun(theta, x, x2=None, C=C, dC=None, diag=True)    # shape (nt_or_less)

            # vector of variances of the target function for every training point

            lambda_var = Kvec + torch.einsum( 'ij,ji->i', a, torch.matmul(V-K_tilde, a.T) ) # This is the same as doing Diag(a.T @ (V-K_tilde) @ a
            # lambda_var = Kvec + torch.sum( -(K*a).T + a.T*(  torch.linalg.solve(V_inv, a.T)), dim=0)

            # TODO check that this method with einsum is actually faster than torch.sum(a*(V-K_tilde)@a, dim=1)

            if dK is not None and dK_tilde is not None and dK_vec is not None and K_tilde_inv is not None:
                # Calculate the derivatives of the moments of lambda with respect to the hyperparameters
                # dK, dK_tilde, dK_vec are dictionaries of the derivatives of the kernel with respect to the hyperparameters
                # dK_vec is the derivative of the diagonal of the kernel with respect to the hyperparameters

                da          = {}
                dlambda_m   = {}
                dlambda_var = {}
                for key in dK.keys():
                    da[key] = (dK[key] - a@dK_tilde[key])@K_tilde_inv # TODO check if it can be made more efficient puling dK out of the parenthesis
                    # Derivative of the mean of lambda with respect to the hyperparameters
                    dlambda_m[key] = da[key]@m
                    # Derivative of the variance of lambda with respect to the hyperparameters
                    # dlambda_var[key] = dK_vec[key] + torch.einsum( 'ij,ji->i', 2*da[key], torch.linalg.solve(V_inv, a.T)) - torch.einsum( 'ij,ij->i', dK[key],a ) - torch.einsum( 'ij,ij->i', K, da[key] )
                    dlambda_var[key] = dK_vec[key] + torch.einsum( 'ij,ji->i', 2*da[key], V@a.T) - torch.einsum( 'ij,ij->i', dK[key],a ) - torch.einsum( 'ij,ij->i', K, da[key] )
                return lambda_m, lambda_var, dlambda_m, dlambda_var

            else :
                return lambda_m, lambda_var

def mean_f_given_lambda_moments( f_params, lambda_m, lambda_var):
        # The expectation value of the vector of firing rates for every training point: 
        # <f> = exp(A*<lambda> + 0.5*A^2*Var(lambda) + lambda0)
        # as shown (between other things) in (34)-(37) of Notes for Pietro
        return torch.exp(f_params['A']*lambda_m + 0.5*f_params['A']*f_params['A']*lambda_var + f_params['lambda0'] )
        
def mean_f( f_params, calculate_moments, lambda_m=None, lambda_var=None,  x=None, K_tilde=None, KKtilde_inv=None, 
           Kvec=None, C=None, m=None, V=None, V_inv=None, theta=None, kernfun=None, dK=None, dK_tilde=None, dK_vec=None, K_tilde_inv=None, K=None):
        
        # Compute the mean of the firing rate f for every training point (a vector) as in (52) Notes for Pietro, 
        # It calls lambda_moments to calculate mean and variance of lambda [ eq (56)(57) of Notes for Pietro ] if they are not known.
        # In this case it needs parameters from x to kernfun to calculate the moments of lambda

        # In case the moments of lambda are known ( like in the updateA of f_params ) they are expected as argument

        # mean_f_given_lambda_moments() is used once the moments are calculated to compute the actual mean of the firing rate

        # RETURNS:
        # f_mean     : shape (nt_or_less, 1) the mean of the firing rate for every training point\
        # if the moments are calculated here it returns also:
        # lambda_m   : shape (nt_or_less, 1) the mean of lambda for every training point
        # lambda_var : shape (nt_or_less, 1) the variance of lambda for every training point


        # INPUTS
        # f_params : shape (2) vector of parameters of the firing rate (A, lambda0)

        # lambda_m, lambda_var: mean and covariance matrix of lambda for every training point, shape (nt_or_less)

        # x      :  shape (nt_or_less, nx), datapoints of which we are calculating the mean of
        # xtilde : inducing datapoints shape (ntilde, nx)
        # C      : calculated localker shape (nx, nx)
        # m      : (vec) mean of (vec) lambda for the variational distribution q(lambda) = N(m, V), shape (ntilde)
        # V      : covariance matrix of the variational distribution q(lambda) = N(m, V), shape (ntilde, ntilde)

        # DEBUGGING :
        # - Should m be of shape ntilde,1 ?

        if calculate_moments and (lambda_m is None or lambda_var is None):
            # Calculate the moments

            # Do we need the gradients of the moments with respect to the hyperparameters?
            if dK is not None and dK_tilde is not None and dK_vec is not None and K_tilde_inv is not None:
                lambda_m, lambda_var, dlambda_m, dlambda_var = lambda_moments( x, K_tilde, KKtilde_inv, Kvec, C, m, V, theta, kernfun=kernfun, dK=dK, dK_tilde=dK_tilde, dK_vec=dK_vec, K_tilde_inv=K_tilde_inv, K=K)
                # Calculate the actual mean of the firing rate
                f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)
                return f_mean, lambda_m, lambda_var, dlambda_m, dlambda_var
            
            else:
                lambda_m, lambda_var = lambda_moments( x, K_tilde, KKtilde_inv, Kvec, C, m, V, theta, kernfun=kernfun)
                # Calculate the actual mean of the firing rate
                f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)
                return f_mean, lambda_m, lambda_var
        
        else:
            # Calculate the actual mean of the firing rate
            f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)
            return f_mean

def compute_g( f_params, K, r, f_mean):
    g = f_params[0] * torch.einsum('ji,j->i', K, (r - f_mean)) 
    return g

def compute_g_assamu( f_params, KKtilde_inv, r, f_mean):
    # KKtilde_inv is calculated outside as:
    # KKtilde_inv = torch.linalg.solve(K_tilde, K.T) # shape (ntilde, nt) 
    a = KKtilde_inv
    g = f_params['A'] * a.T @ (r - f_mean)
    return g

def compute_G_assamu( f_params, KKtilde_inv, f_mean):
    # Compute G_samu as K_tilde_inv @ G @ K_tilde_inv , where the G on the rhs is the one in the paper and notes
    # KKtilde_inv is calculated outside as:
    # KKtilde_inv = torch.linalg.solve(K_tilde, K.T) # shape (ntilde, nt) 
    a = KKtilde_inv
    G = f_params['A']*f_params['A'] * torch.einsum('ij,jk->ikj', a.T, a) @ f_mean # Shape (ntilde, ntilde)
    # G = f_params[0]*f_params[0]*a@(a.T*f_mean)
    return G

def compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=False, dlambda_m=None, dlambda_var=None):
    # Returns the Sum of <loglikelihood> terms of the logmarginal likelihood loss as in (51) Notes for Pietro   
    # TODO handle the thwo cases better (with and without gradients). Its returning a tuple in one case and and not int he other


    # NB: A here is the firing rate parameter, not the receptive field Amplitude one
    A = f_params['A']
    lambda0 = f_params['lambda0']
    rlambda_m = r@lambda_m  
    sum_r     = torch.sum(r)

    logLK     = A*rlambda_m + lambda0*sum_r - torch.sum(f_mean)

    if compute_grad_for_f_params:
        # This is used when -loglikelihood is used loss for optimizer of f_params
        # Derivative of the loglikelihood with respect to the parameters of the firing rate
        dlogLK = {}
        dlogLK['A'] = rlambda_m - torch.dot(lambda_m + A*lambda_var, f_mean)
        dlogLK['lambda0'] = sum_r - sum(f_mean)

        return logLK , dlogLK

    if dlambda_m is not None and dlambda_var is not None:
        dlogLK = {}
        for key in dlambda_m.keys():
            dlogLK[key] = r@dlambda_m[key] - A*torch.dot(f_mean, dlambda_m[key]) - 0.5*A*A*torch.dot(f_mean, dlambda_var[key]) 
        return logLK , dlogLK    
    else:
        return logLK, rlambda_m, sum_r # Only here i return these two values cause i need them in updateA

def log_det(M):

    try:
        # Try a Cholesky decomposition, L is the matrix that multiplitd by its (cunjugate) transpose gives M
        L=torch.linalg.cholesky(M, upper=True)
        # The determinant of M is then the product (the * 2 ) of the product of the diagonal elements of L and L.T (the same if real) 
        # Its log is 2*torch.log(torch.product(torch.diag(L))) which corresponds to
        return 2*torch.sum(safe_log(torch.diag(L))) 
    except:
        warnings.warn("The matrix is not positive definite, using eigendecomposition to calculate the log determinant")
        eigenvalues, eigenvectors = torch.linalg.eig(M) # Need to use .eig() cause .eigh() is only for simmetrix matrices
        # Matrix M should always be positive semidefinite
        # count how many eigenvalues are negative, complex or zero
        #print(f'n eigvals<0: {torch.sum(torch.real(eigenvalues)<0)}', f'n eigvals imag>1.e-6: {torch.sum(torch.abs(eigenvalues.imag) > 1e-6)}', 
        #      f'n small eigvals ( < 1.e-6 ): {torch.sum(torch.real(eigenvalues)<1e-6)}')

        # eigenvectors = eigenvectors[ :,  torch.real(eigenvalues)>1e-6  ]
        large_real_eigenvals = torch.real(eigenvalues[torch.real(eigenvalues)>1.e-6])
        
        return torch.sum(safe_log( large_real_eigenvals )) 

def compute_KL_div( m, V_inv, K_tilde, K_tilde_inv, dK_tilde=None):
    
    # Computes the Kullback Leiber divergence term of the complete loss function (marginal likelihoo)
    # D_KL( q(lambda_tilde) || p(lambda_tilde) 
    # c = K_tilde_inv @ V
    c_inv = V_inv @ K_tilde
    b = K_tilde_inv @ m
    
    # c = torch.linalg.solve(K_tilde, V)       # Shape (ntilde, ntilde)# This is " C " in samuele code written as V @ K_tilde_inv
    # b = torch.linalg.solve(K_tilde, m)       # Shape (ntilde, 1)
    # derivative with respect to theta
    
    KL = -0.5*(1/log_det(V_inv) - log_det(K_tilde)) + 0.5*torch.matmul(m.T, b) + 0.5*torch.sum(1./torch.linalg.eig(c_inv)[0])      #0.5*torch.trace(c)

    if dK_tilde is not None:
        dKL = {}
        for key in dK_tilde.keys():
            B = dK_tilde[key]@K_tilde_inv # Shape (ntilde, ntilde

            dKL[key] = 0.5*torch.trace(B) - 0.5*torch.trace(torch.linalg.pinv(c_inv)@B) - 0.5*b.T@(B@m)

        return KL, dKL
    else:
        return KL

def updateA(f_params, r, lambda_m, lambda_var, nit=1000, eta=0.25, tol=1e-6, Print=False, i_step=0):

            # Update f_params ( the A and lambda0 of the paper, parameters of the firing rate )
            
            # INPUTS
            # tol: tolerance for the convergence of the gradient descent
            # eta: learning rate

            # RETURNS
            # f_par: updated f_par
            # likelihood[-1]: likelihood reached by the last iteration

            count = 0 
            flag = True

            #L = np.zeros((int(nit), 1))
            # L = torch.tensor.zeros((nit, 1))
            likelihood = torch.zeros((nit, 1))

            while flag:
                f_mean    =  mean_f( f_params=f_params, calculate_moments=False, lambda_m=lambda_m, lambda_var=lambda_var )

                # f_mean    = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)
                likelihood[count], rlambda_m, sum_r  = compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params)
                
                # Derivarive of the exponential of f_mean
                d_exp = (lambda_m + f_params[0]*lambda_var)
                f_star = d_exp * f_mean # shape (f_mean.shape). This vector appears often in the computation
                sum_f_star = torch.sum(f_star)
                sum_f_mean = torch.sum(f_mean)
                R = torch.tensor( [ rlambda_m - sum_f_star  , sum_r - sum_f_mean ] ) # In samu's code ther is minus sign
                H = -torch.tensor( [[ lambda_var@f_mean + d_exp@f_star , sum_f_star ], # Here also, the minus that i have is not there
                                    [ sum_f_star                       , sum_f_mean ]] )
                # ____ TEMP ____
                # Check concavity of the likelyhood in that point
                eigvals  = torch.linalg.eigh(H)[0]
                they_negative = torch.all(eigvals < 0)
                if not they_negative:
                    print(f'All eigenval of hessian are nonpositive? {they_negative}, the likelihood is not concave in this point. Iteration: {count}')

                #  ____ TEMP ___
                # Gradient descent update:
                # f_params  = f_params - 0.001*eta*R
                
                # Newton update
                a = torch.linalg.solve(H, R)
                f_params = f_params - eta*a        # In Samu's code it's the same, it is subtracting from f_params
                                             # NB it should not be in place -=, cause it gives problem with the gradient computation
                
                count += 1  
                if torch.sum(torch.abs(R)) < tol:
                        if Print:
                            print(f'  GD converged at iteration: {count}')
                        flag = False
                        continue
                if count >= nit:
                        if Print:
                            print(f'  GD reached max iterations: {count}')
                        flag = False
                        continue
                
            return f_params, likelihood[count-1], mean_f_given_lambda_moments( f_params, lambda_m, lambda_var )

def Estep( r, K_tilde, K_tilde_inv, KKtilde_inv, m, f_params, f_mean):

    # Does not use kernfun


    # print(f' \n **In Estep:')
    # print(f' m.mean() before E step inside function: {m.mean()}')
    # print(f' V.mean() before E step: {V.mean()}')

    # TODO find a better way to preserve the value KKtilde_inv, aka 'a' in the notes for Pietro and samuele's code

    # g = f_params[0] * torch.sum( K.T * (r - f_mean), dim=1) # Shape (n_tilde)
    # g = compute_g( f_params, K, r, f_mean)
    
    # Add a small perturbation to the diagonal of K_tilde
    # epsilon = 1e-9
    # K_tilde_perturbed = K_tilde + epsilon * torch.eye(K_tilde.shape[0])
    # KtildeG_inv        = torch.inverse(K_tilde_perturbed + G)
    # Ktilde_KtildeG_inv = torch.matmul(K_tilde, KtildeG_inv)

    # I redid the calculations for the update on m  and V on the 31 01 24, these should be correct but they 
    # are very different from Samuele's
    # if needed you have to passs V_inv as well
    # m     = m + K_tilde @ torch.inverse( K_tilde_perturbed - G ) @ (g-m)
    # V_inv = 2*V_inv - K_tilde_inv @ (K_tilde_perturbed + G) @ K_tilde_inv        
    # V     = torch.inverse(V_inv)

    #_____ TEMP - SAMUELE ____
    # No need for the above
    g = compute_g_assamu(f_params=f_params, KKtilde_inv=KKtilde_inv, r=r, f_mean=f_mean)
    G = compute_G_assamu(f_params=f_params, KKtilde_inv=KKtilde_inv, f_mean=f_mean)
    V_new = torch.linalg.solve(torch.eye(K_tilde.shape[0])+K_tilde @ G, K_tilde)
    V_new = (V_new + V_new.T) / 2 + 1e-5 * torch.eye(K_tilde.shape[0]) # make sure positive definite!
    m_new = V_new @ (G @ m + g)  #shape(250,1)


    # Matthew's update
    # In the Matlab code a = K_tilde_inv @ K which means a = KKtilde_inv.T for this code. 
    # Since I use instead a = KKtilde_inv in other parts of the code Im using KKtilde_inv diretly here.

    # g = f_params['A'] * KKtilde_inv.T @ (r - f_mean)
    # G1 = f_params['A']*f_params['A'] * KKtilde_inv.T@(KKtilde_inv*f_mean[:,None]) # f_mean is a vector

    # V_inv_new = ( K_tilde_inv + G ) # shape (ntilde, ntilde)
    # V_inv_new = (V_inv_new + V_inv_new.T) / 2  # making sure it is symmetric
    # V_new     = torch.linalg.inv(V_inv_new) # shape (ntilde, ntilde)

    # m_new = torch.linalg.solve(V_inv_new , (G @ m + g))  #shape(ntilde)

    return m_new, V_new

def print_hyp( theta ):
        for key in theta.keys():

            if key == '-2log2beta':
                print(f' beta: {logbeta_to_beta(theta):.8f}')
                continue
            if key == '-log2rho2':
                print(f' rho: {logrho_to_rho(theta):.8f}')
                continue

            print(f' {key}: {theta[ key ]:.8f}')       


##################   Inference   ########################
def test(X_test, R_test, xtilde, **kwargs):

    maxiter = kwargs.get('Maxiter', 0)
    nestep  = kwargs.get('Nestep', 0)
    nmstep  = kwargs.get('Nmstep', 0)
    mask    = kwargs.get('mask')
    cellid  = kwargs.get('cellid')
    theta   = kwargs.get('theta')
    C       = kwargs.get('C')
    m       = kwargs.get('m')
    V       = kwargs.get('V')
    K_tilde = kwargs.get('K_tilde')
    K_tilde_inv = kwargs.get('K_tilde_inv')
    f_params = kwargs.get('f_params')
    kernfun = kwargs.get('kernfun')

    R_predicted = torch.zeros(X_test.shape[0])

    for i in range(X_test.shape[0]):
        xstar = X_test[i,:,:,:]
        xstar = torch.reshape(xstar, (1, xstar.shape[0]*xstar.shape[1]))

        mu_star, sigma_star2 = mu_sigma2_xstar(xstar[:,mask], xtilde[:,mask], C, theta, K_tilde_inv, m, V, kernfun)

        # Compute rate
        rate_star = torch.exp( f_params[0]*mu_star + 0.5*f_params[0]*f_params[0]*sigma_star2 + f_params[1] )

        R_predicted[i] = rate_star #ends up being of shape 30
        # print(f'rate_star: {rate_star.item():.4f}')


    r2, sigma_r2 = explained_variance( R_test[:,:,cellid], R_predicted, sigma=True)

    # for i in range(R_predicted.shape[0]):
        # print(f' {R_predicted[i].item():4f}')

    # Print the results

    rtst = R_test[:,:,cellid].to('cpu').numpy()
    R_predicted = R_predicted.to('cpu').numpy()

    
    print(f"\n\n Pietro's model: R2 = {r2:.2f} ± {sigma_r2:.2f} Cell: {cellid} Maxiter = {maxiter}, Nestep = {nestep}, nMstep = {nmstep} \n")

    return rtst, R_predicted, r2, sigma_r2

def mu_sigma2_xstar( xstar, xtilde, C, theta, K_tilde_inv, m, V, kernfun):
    # Computes the mean of the gaussian posterior distribution over lambda(x^*)

    k_vec = kernfun(xstar, xtilde, C, theta, xtilde_case=False, scalar_case=False)

    KKtilde_inv = k_vec @ K_tilde_inv
    # Lambda(x^*) medio.
    mu_star =  KKtilde_inv @ m #
 
    # Scalar covariance between input xstar and itself
    kstarstar = kernfun(xstar, xstar, C, theta, xtilde_case=False, scalar_case=True)

    A = - KKtilde_inv @ k_vec.T
    
    B =  KKtilde_inv @ V @ K_tilde_inv @ k_vec.T 

    sigma_star2 = kstarstar + A + B

    return mu_star, torch.reshape(sigma_star2, (1,))

def explained_variance(rtst, f_pred, sigma=True):

    # Compute the observed r2 for the sequence of images
    # rtst = ( repetitions, nimages )
    # f_pred = ( nimages )

    # Even and odd repetitions of the same image, mean response. First index is repetitions
    reven = torch.mean(rtst[0::2,:], axis=0)
    rodd  = torch.mean(rtst[1::2,:], axis=0)

    # stacked_R = torch.stack( (r, f ) )
   
    reliability = torch.abs(torch.corrcoef( torch.stack((reven,rodd))))[0,1]
    accuracy_o = torch.corrcoef(torch.stack((f_pred, rodd)))[0,1]
    accuracy_e = torch.corrcoef(torch.stack((f_pred, reven)))[0,1]
    r2 = 0.5 * (accuracy_o + accuracy_e) / reliability

    if sigma:
        nbootstrap = 1000  # Number of bootstrap iterations
        r2 = torch.zeros(nbootstrap)
        n = rtst.shape[0]
       
        for i in range(0, nbootstrap):
            ilist = torch.randperm(n)
            ieven = ilist[0::2]
            iodd  = ilist[1::2]

            reven = torch.mean(rtst[ieven,:], axis=0)
            rodd  = torch.mean(rtst[iodd,:] , axis=0)
           
            reliability = torch.abs(torch.corrcoef(torch.stack((reven, rodd)))[0, 1])
            accuracy_o = torch.corrcoef(torch.stack((f_pred, rodd )))[0, 1]
            accuracy_e = torch.corrcoef(torch.stack((f_pred, reven)))[0, 1]
            r2[i] = 0.5 * (accuracy_o + accuracy_e) / reliability

        sigma_r2 = torch.std(r2)
        r2 = torch.mean(r2)
        return r2, sigma_r2
    else:
        return r2, None

def plot_fit(R_predicted, rtst, r2, sigma_r2, cellid):
# Plot results
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(5, 5,
                left=0.1, right=0.9, bottom=0.1, top=0.9,
                wspace=0.3, hspace=0.7)
    dt = 0.05
    time_values = dt * np.arange( len(R_predicted) )
    ax = fig.add_subplot(gs[3:, :])
    ax.plot(time_values, np.mean(rtst, axis=0) , 'k', linewidth=1)

    ax.plot(time_values, R_predicted , color='red', label='GP')
    
    # ax.errorbar(time_values, R_predicted / 0.05, yerr=np.sqrt(sigma2_f[:,0].cpu()) / 0.05, color='red')
    # ax.legend(['data', 'GP'], loc='upper right', fontsize=14)
    txt = f'Pietro adjusted r^2 = {r2:.2f} ± {sigma_r2:.2f} Cell: {cellid}'
    ax.set_title(f'{txt}')
    # ax.set_ylabel('Firing rate (Hz)')
    # plt.show()
    # plt.close()
    return fig


def compute_loglikelihood_(f_params, lambda_m, lambda_var, r):

    A       = f_params['A']
    lambda0 = f_params['lambda0']

    f = torch.exp(A*lambda_m + 0.5*lambda_var*A*A+lambda0);

    loss_A = torch.sum(f) - torch.dot(r,(lambda_m*A+lambda0))

    dloss_A = {}

    dlambda_A          = lambda_m + lambda_var*A
    dloss_A['A']       = torch.sum(f-r)
    dloss_A['lambda0'] = torch.dot(f,dlambda_A) - torch.dot(r,lambda_m)

    return loss_A, dloss_A

##########################################################

def varGP(x, r, **kwargs):
    #region
    # Learn the hyperparameters of the GP model

    # INPUTS:

    # x = [nt, nx], stimulus
    # r = [1, n],  spike counts
    
    # OPTIONAL INPUTS:
    
    # ntilde: number of inducing data points if they are not provided as argument(increses approximation accuracy )
    # xtilde: inducing points [nx, ntilde]
    
    # Nestep:  number of  of iterations in E-step (updating m, and V)
    # Nmstep:  number of steps in M-step (updating theta )
    # Display_hyper: Bool to display the initial hyperparameters
    # display_prog: Bool to display the progress bar
    
    # kernfun: kernel function ( acosker is default)
    # theta: dict of initial hyperparameters of the kernel
    # lb, ub: lower and upper bound for theta
    # m, V: initial mean and variance of variational distribution, q(lambda) = N(m, V). Requires_gradietn is set to False cause in the M step i keep them fixed
     
    # RETURNS
    
    # theta,
    # f_par, 2 Parameters of the firing rate (A and lambda_0) in the paper
    # m, V (tensors tracking the values of all of the above)
    # xtilde: set of inducing datapoints
    # L, loss function during learning

    #endregion

    #region
    # number of pixels, number of training points 
    nt, nx = x.shape 

    ntilde  = kwargs.get('ntilde', 100 if nt>100 else nt) # if no ntilde is provided try with 100, otherwise inducing points=x   
    xtilde  = kwargs.get('xtilde', generate_xtilde(ntilde, x)) 
    if ntilde != xtilde.shape[0]: raise Exception('Number of inducing points does not match ntilde')
    
    Maxiter       = kwargs.get('Maxiter', 200)
    Nestep        = kwargs.get('Nestep',  50) 
    Nmstep        = kwargs.get('Nmstep',  20)
    display_hyper = kwargs.get('display_hyper', True)
    display_prog  = kwargs.get('display_prog', True)

    kernfun = kwargs.get('kernfun', acosker_samu)

    n_px_side = kwargs.get('n_px_side', torch.tensor(np.sqrt(nx), dtype=torch.int))
    
    # Initialize hyperparameters of Kernel and parameters of the firing rate
    if 'hyperparams_tuple' not in kwargs: hyperparams_tuple = generate_theta(x, r, n_px_side=n_px_side, display=display_hyper)
    else: hyperparams_tuple = kwargs.get('hyperparams_tuple')

    theta             = kwargs.get( 'theta', hyperparams_tuple[0]) 
    theta_lower_lims  = kwargs.get('theta_lower_lims', hyperparams_tuple[1] )
    theta_higher_lims = kwargs.get('theta_higher_lims', hyperparams_tuple[2] )
    f_params          = kwargs.get( 'f_params', {'A': torch.tensor(0.0001), 'lambda0':torch.tensor(-1)} ) # Parameters of the firing rate (A and lambda_0) in the paper
    # f_params[0]       = f_params[0].clone().detach().requires_grad_(True)
    # f_params[1]       = f_params[1].clone().detach().requires_grad_(True)

    # Calculate the part of the kernel responsible for implementing smoothness and the receptive field

    # calculating dC in debug but its not necessary
    C, mask = localker(nx=x.shape[1], theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=False)
    #TODO Calculate it only close to the RF (for now it's every pixel)

    m = kwargs.get('m', torch.zeros( (ntilde) )).detach()
    V = kwargs.get('V', kernfun(theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=None, diag=False) ).detach()
    V_inv = torch.linalg.inv(V)
    # Pietro's implementation of acoskern, faster but not stable.
    # V = kwargs.get('V', kernfun( xtilde[:,mask], xtilde[:,mask], C, theta, xtilde_case=True, dC=None )).detach()  # shape (ntilde, ntilde)
    
    # Track 
    loss            = {'marginal'  : torch.zeros((Maxiter)),
                       'likelihood': torch.zeros((Maxiter)),
                       'KL'        : torch.zeros((Maxiter)),
                       } # Loss to  maximise: Log Likelihood - KL
    theta_track     = {key : torch.zeros((Maxiter)) for key in theta.keys()}
    f_par_track     = {'A': torch.zeros((Maxiter)), 'lambda0': torch.zeros((Maxiter))} # track hyperparamers
    values_track    = {'loss': loss, 'theta_track': theta_track, 'f_par_track': f_par_track}
    #endregion


    #_________ Main Loop ___________
    # progbar = tqdm( range(Maxiter), disable = not display_prog, desc=f"GP step loss", position=0 )   
    for iteration in range(Maxiter):

        #________________ Compute the KERNELS after M-Step and the inverse of V _____________
        # calculating dC in debug but its not necessary
        C, mask    = localker(nx=x.shape[1], theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=False)

        K_tilde    = kernfun(theta, xtilde[:,mask], xtilde[:,mask], C=C, diag=False) # shape (ntilde, ntilde)
        # K_tilde and dK_tilde checked. They return the same as Samuele up to the 1e-5/6
        K                = kernfun(theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)      # shape (nt, ntilde) set of row vectors K_i for every input 
        # K checked, returns the same as Samuele up to the 1e-5/6
        Kvec             = kernfun(theta, x[:,mask], x2=None, C=C, dC=None, diag=True)              # shape (nt)

        try:
            L=torch.linalg.cholesky(K_tilde, upper=True)
        except:
            warnings.warn('   K_tilde not positive definite') 

        #  Stabilization
        eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')#calculates the eigenvalues for an assumed symmetric matrix, eigenvalues are returned in ascending order. Uplo=L uses the lower triangular part of the matrix
        ikeep = eigvals > max(eigvals.max() / 1e4, 1e-4)       # Keep only the largest eigenvectors
        B = eigvecs[:, ikeep]                                  # shape (ntilde, n_eigen)            


        # Vlambda, Vvec = torch.linalg.eigh(V) 
        # # find the eigenvalues of V hthat are 0 or less:
        # Vkeep = Vlambda < 1e-5
        # print(f'\n Iteration {iteration} Number of  small or negative eigvals of V: {torch.sum(Vkeep)}, for the V used in previous iteration')
        # print(' Projecting onto updated eigenspace' )

        K_tilde_b = B.T@K_tilde@B                # Projection of K_tilde into eigenspace (n_eigen,n_eigen) 
        K_b = K @ B                              # Project K into eigenspace, shape (3190, n_eigen)
        m_b = B.T @ m                            # Project m into eigenspace, shape (n_eigen, 1) 
        V_b = (B.T@V)@B                          # shape (n_eigen, n_eigen). There is no guarantee that V is positive definite. It has been obtained by expanding the V_b updated in the previous iteration 
                                                 # back to the original space. We are now projecting in to the new eigenspace of the updated K_tilde.

        # V_blambda, V_bvec = torch.linalg.eigh(V_b) 
        # # find the eigenvalues of V hthat are 0 or less:
        # V_bkeep = V_blambda < 1e-5
        # print(f'  Iteration {iteration} Number of small or negative eigvals of V_b: {torch.sum(V_bkeep)}. V_b will be used in the this iteration')

        K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep]) # shape (n_eigen, n_eigen)
        KKtilde_inv_b = K_b @ K_tilde_inv_b # shape (nt, n_eigen)

        #region  _______________ Control over possible Nans ______
        for tensor in [C, K_tilde_b, K_b, KKtilde_inv_b, V_b, m_b, f_params['A'], f_params['lambda0']]:
            if torch.any(torch.isnan(tensor)):
                variable_name = [k for k, v in locals().items() if v is tensor][0]
                raise ValueError(f'NaN in {variable_name}')
            if torch.any(torch.isinf(tensor)):
                variable_name = [k for k, v in locals().items() if v is tensor][0]
                raise ValueError(f'Inf in {variable_name}')
        #endregion
        
        f_mean, lambda_m, lambda_var  = mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, K=K_b, C=C, m=m_b, V=V_b, 
                                                theta=theta, kernfun=kernfun, lambda_m=None, lambda_var=None )# calling  the function without dK and Ktilde_inv cause i dont need the gradients of lambda from this call 
           
        #region Paused _______________ Update the trakinf dictionaries _______________   
        # Computing the KL divergence here might give problems with the V_b not being positive definite. 
        # Even though it should if the matrices V_b and K_tilde_b live in the same space
        values_track['loss']['likelihood'][iteration] = compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params)[0]
        values_track['loss']['KL'][iteration]         = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None )
        values_track['loss']['marginal'][iteration]   = values_track['loss']['likelihood'][iteration] - values_track['loss']['KL'][iteration]
        
        for key in theta.keys():
            values_track['theta_track'][key][iteration] = theta[key]

        values_track['f_par_track']['A'][iteration]       = f_params['A']
        values_track['f_par_track']['lambda0'][iteration] = f_params['lambda0']
        #endregion

        #region _______________ Display _______________
        # if iteration == 0:
        #     print(f' Starting    : Likelihood : {likelihood.item():.4f}, -KL: {-KL.item():.4f}, Marginal: {marginal_likelihood.item():.8f}' )        
        # else:
        #     print(f' After M-step: Likelihood : {likelihood.item():.4f}, -KL: {-KL.item():.4f}, Marginal: {marginal_likelihood.item():.8f}' )
        # # print(f' Iteration: {iteration}:  A: {f_params[0]:.4f}, lambda0: {f_params[1]:.4f}')
        # # print_hyp( theta, iteration)  

        print(f'\n**Iteration: {iteration}**')
        #endregion

        #region _____________ _ E-Step : Update on m & V and f(lambda) parameters  ________________
        with torch.no_grad():  # Temporarily set all requires_grad flag to false          
            # progbar2 = tqdm( range(Nestep), disable=not display_prog, position=1, leave=True )
            for i_estep in range(Nestep):
                
                if i_estep == 0:
                    lambda_m, lambda_var = lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, C, m_b, V_b, theta, kernfun=kernfun)

                #region ____________ Update f_params ______________
                print(f"  Before closure2:                      f_params['A'] = {f_params['A'].item():.3f}, f_params['lambda0'] = {f_params['lambda0'].item():.3f}, firing rate Nan or inf: { torch.any( torch.isnan(f_mean) | torch.isinf(f_mean)  ) }")
                learning_rate2 = 0.1
                optimizer2 = torch.optim.LBFGS(f_params.values(), lr=learning_rate2, max_iter=10)
                def closure2( ):
                    optimizer2.zero_grad()
                    f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)   
                    logLK, dlogLK = compute_loglikelihood(  r,  f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=True )
                    
                    # Update gradients of the loss with respect to the firing rate parameters
                    f_params['A'].grad       = -dlogLK['A']
                    f_params['lambda0'].grad = -dlogLK['lambda0']
                    # lossA.backward(retain_graph=True)
                    return -logLK
                # the closure has to return the loss, that's why the minus sign after the return in closure2.
                # since what we are interested in is the likelihood we take the minus.
                logLK = -optimizer2.step(closure2) 
                print(f"  After closure2: loglikelihood = {logLK:.3}, f_params['A'] = {f_params['A'].item():.3f}, f_params['lambda0'] = {f_params['lambda0'].item():.3f}, firing rate Nan or inf: { torch.any( torch.isnan(f_mean) | torch.isinf(f_mean)  ) }")
            #endregion
                
                f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)

                # print( f' A = {f_params["A"].item():.4f}, lambda0 = {f_params["lambda0"].item():.4f} f_mean_mean: {f_mean.mean():.4f} \n lambda_m_mean: {lambda_m.mean():.4f} lambda_var_mean: {lambda_var.mean():.4f}')

                m_b, V_b = Estep( r=r, K_tilde=K_tilde_b, K_tilde_inv=K_tilde_inv_b, KKtilde_inv=KKtilde_inv_b, m=m_b, f_params=f_params, f_mean=f_mean )
                
                # V_blambda, V_bvec = torch.linalg.eigh(V_b) 
                # # find the eigenvalues of V hthat are 0 or less:     
                # V_bkeep = V_blambda < 1e-5
                # print(f'   Iteration {iteration} Number of small or negative eigvals of V_b: {torch.sum(V_bkeep)}. Just after E-step')
                try:
                    L=torch.linalg.cholesky(V_b, upper=True)
                except:
                    warnings.warn('   V_b not positive definite')
                try:
                    L=torch.linalg.cholesky(K_tilde_b, upper=True)
                except:
                    warnings.warn('   K_tilde_b not positive definite')                    

                f_mean, lambda_m, lambda_var  =  mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, K=K_b, C=C, m=m_b, V=V_b, 
                                                 theta=theta, kernfun=kernfun, lambda_m=None, lambda_var=None  )
                
                likelihood          = compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params)[0]
                # Computing this KL divergence might give problems in the LogDet if V_b is not positive definite
                KL                  = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None )
                marginal_likelihood = likelihood - KL            
                # print(f' After E-Step: Likelihood : {likelihood.item():.4f}, -KL: {-KL.item():.4f}, Marginal: {marginal_likelihood.item():.8f}' )
        
                #region _______________ Display ____________________
        #         if display_prog:
        #             likelihood          = compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params)[0]
        #             # Computing this KL divergence might give problems in the LogDet if V_b is not positive definite
        #             KL                  = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None )
        #             marginal_likelihood = likelihood - KL
        #             print(f' After E-Step: Likelihood : {likelihood.item():.4f}, -KL: {-KL.item():.4f}, Marginal: {marginal_likelihood.item():.8f}' )
        #             # progbar2.set_description(f"EE step loss: {-marginal_likelihood.item():.4f}")
        #             progbar2.update(1)
        #         #endregion   
        # progbar2.close()
        #endregion            

        #region _______________ Restoring m and V to the original space ________________

        # Look at eigenvalues of V_b are they positive definite? 
        # I have to reproject before the m step is I change and eigenbasis
        # m = B@m_b
        # V = (B@V_b)@B.T 
        
        #endregion

            #region _______________ M-Step : Update on theta  ________________
            if Nmstep > 0:

                learning_rate = 0.1
                # optimizer = torch.optim.Adam(theta.values(), lr=learning_rate)
                optimizer = torch.optim.LBFGS(theta.values(), lr=learning_rate, max_iter=Nmstep)

                def closure( ):
            
                    optimizer.zero_grad()
                    C, mask, dC     = localker(nx=x.shape[1], theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=True)
                    K_tilde, dK_tilde     = kernfun(theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=dC, diag=False)
                    K, dK                 = kernfun(theta, x[:,mask], xtilde[:,mask], C=C, dC=dC, diag=False)
                    Kvec, dKvec           = kernfun(theta, x[:,mask], x2=None, C=C, dC=dC, diag=True) 

                    #region ____________Stabilization____________________
                    # In the first version of Samuele's code with stabilization, the eigenvector matrix is not recalculated during the M-step. 
                    # This is not entirely precise because a change in hyperparameters could change the eigenvalues  over the threshold (and therefore change the dimension of the subspace I'm projecting onto)
                    # But this most likely has a minimal effect. And it saves Nmstep eigenvalue decompositions per iteration.

                    try:
                        L=torch.linalg.cholesky(K_tilde, upper=True)
                    except:
                        warnings.warn('   K_tilde not positive definite') 
                    try:
                        eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')  # calculates the eigenvalues for a symmetric matrix
                    except:
                        warnings.warn('   Problem with K tilde')
        
                    # ikeep = eigvals > max(eigvals.max() / 1e4, 1e-4)         # Keep only the largest eigenvectors
                    # B = eigvecs[:, ikeep]                                    # shape (ntilde, n_eigen)            

                    # Projecting the Kernel into the same eigenspace used in the E-step (its not changing with the changing hyperparameters)
                    K_tilde_b = B.T@K_tilde@B                # Projection of K_tilde into eigenspace (n_eigen,n_eigen) 
                    K_b  = K @ B                             # Project K into eigenspace, shape (3190, n_eigen)

                    # m_b = B.T @ m                            # Project m into eigenspace, shape (n_eigen, 1) 
                    # V_b = (B.T@V)@B                          # shape (n_eigen, n_eigen) 

                    dK_tilde_b = {}
                    dK_b = {}
                    for key in dK_tilde.keys():
                        dK_tilde_b[key] = B.T@dK_tilde[key]@B
                        dK_b[key]       = dK[key] @ B                            # Project dK into eigenspace, shape (3190, n_eigen)
                    #endregion

                    # K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep]) # shape (n_eigen, n_eigen) To use if I have recalculated the eigenspace of K_tilde
                    K_tilde_inv_b = torch.linalg.solve(K_tilde_b, torch.eye(K_tilde_b.shape[0]))
                    KKtilde_inv_b = K_b @ K_tilde_inv_b # shape (nt, n_eigen)                

                    f_mean, lambda_m, lambda_var, dlambda_m, dlambda_var  =  mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, 
                                                                                    Kvec=Kvec, C=C, m=m_b, V=V_b, theta=theta, kernfun=kernfun, lambda_m=None, lambda_var=None, dK=dK_b,
                                                                                    dK_tilde=dK_tilde_b, dK_vec=dKvec, K_tilde_inv=K_tilde_inv_b, K=K_b ) # Shape (nt
                    
                    loglikelihood, dloglikelihood = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, f_params, dlambda_m=dlambda_m, dlambda_var=dlambda_var )
                    # Computing this KL divergence might give problems in the LogDet if V_b is not positive definite. If it hasnt given problems in the E step, it shouldnt here
                    # try:
                    #     L=torch.linalg.cholesky(V_b, upper=True)
                    # except:
                    #     warnings.warn('   V_b not positive definite in m step')
                    # try:
                    #     L=torch.linalg.cholesky(K_tilde_b, upper=True)
                    # except:
                    #     warnings.warn('   K_tilde_b not positive definite in m step')                     

                    KL, dKL                       = compute_KL_div(m_b, V_b, K_tilde_b, K_tilde_inv=K_tilde_inv_b, dK_tilde=dK_tilde_b)
                    logmarginal                   = loglikelihood - KL
                    loss                          = -logmarginal

                    # print(f' After E-Step: Likelihood : {likelihood.item():.4f}, -KL: {-KL.item():.4f}, Marginal: {marginal_likelihood.item():.8f}' )

                    # Dictionary of gradients of the -loss with respect to the hyperparameters, to be assigned to the gradients of the parameters
                    dlogmarginal = {}
                    for key in dK_tilde.keys():
                        dlogmarginal[key] = dloglikelihood[key] - dKL[key]

                    # loss.backward(retain_graph=True)
                    # Update the gradients of the loss with respect to the hyperparameters ( its minus the gradeints of the logmarginal)
                    for key in theta.keys():
                        if theta[key].requires_grad:
                            theta[key].grad = -dlogmarginal[key]

                    return loss
                
                loss = optimizer.step(closure) 
            
                # progbar.set_description(f"GP step loss: {loss.item():.4f}")
            # progbar.update(1)
            #endregion

            #region _______________ Restoring m and V to the original space ________________
            # After having updated m and V in the E-step in their projected version m_b and V_b, we restore them to the original space.
            # These updated & restored m and V are not used in the M-step ( there, we still use the projected version), but will be used in the next E-step after being projected in the
            # NEW eigenspace, which depends on the updated hyperparameters coming from the M-step.
            # If K_tilde is positive definite than B is orthogonal and B.T = B^-1, therefore

            m = B@m_b
            V = (B@V_b)@B.T 
            
        #endregion


    #region Utility function
    # Test on utility function #
    # avoid keeping track of the gradeints
    # with torch.no_grad():  # Temporarily set all requires_grad flag to false
    #     # take the last x as test to avoid taking an xtilde
    #     xstar    = x[-3:-1,:] # shape (1, nx)
    #     #K_for_u    = kernfun(x_for_u, xtilde[:,mask], C, theta, xtilde_case=False, scalar_case=False) #set of row vectors K_i for every input # shape (nt, ntilde)
    #     # Take the prior moments estimated for this new x

    #     # Inference on new input
    #     mu_star, sigma2_star = mu_sigma2_xstar(xstar[:,mask], xtilde[:,mask], C, theta, torch.inverse(K_tilde), m, V, kernfun=kernfun)

    #     #mu, sigma2 = lambda_moments( mu_star, sigma_star, K_tilde, C, m, V, theta, kernfun=kernfun )
    #     u = utility( sigma2=sigma2_star, mu=mu_star )
    #     print(u)
    #endregion

    # C, mask     = localker(nx=x.shape[1], theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=False)
    # K_tilde     = kernfun(theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)
    # K_tilde_inv = torch.inverse(K_tilde)
    

    # return theta, f_params, m, V, C, mask, K_tilde_inv, K_tilde, values_track

