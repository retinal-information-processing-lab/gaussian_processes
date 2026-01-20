import torch
torch.set_grad_enabled(False)
import numpy as np
import scipy.io
from torchlambertw.special import lambertw as torch_lambertw

from tqdm import tqdm
import pickle
import torch
import math
import os
import traceback
import torch.optim as optim
import warnings
import time 
from datetime import datetime
import json
import copy
import itertools
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

import logging

from config.config import *

from gaussian_processes.Spatial_GP_repo.model import GPModel

# Import new clean kernel implementations from kernels/ subpackage
# These are re-exported here for backward compatibility
from gaussian_processes.Spatial_GP_repo.kernels import (
    C_gradients_hyp,
    LocalkerCleanFunction,
    localker_clean,
    acosker_clean,
    AcoskerCleanFunction,
    acosker_with_grad,
)

# Import visualization utilities from visualization/ subpackage
# These are re-exported here for backward compatibility
from gaussian_processes.Spatial_GP_repo.visualization import (
    update_optimization_comparison,
    single_image_sampling_plot,
)

# torch.zeros(1)

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1420927410125732

print(f"In GP Utils: Default device: {torch.empty(1).device}")  # Verify default device
print(f"In GP Utils: Default dtype: {torch.get_default_dtype()}")    #

# Warnings
warnings.filterwarnings("ignore", "The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated")

## This file was the Spatial_GP.py file in the original code.
# TORCH_DTYPE = torch.float64
# TORCH_DTYPE = torch.float32
# torch.set_default_dtype(TORCH_DTYPE)

# The minimum tolerance for float64 should be 1.e-15 but there are matrices that dont appear to be simmetric up to more than 1.e-13 precision, 
# even if they should ( see V_b reprojection after M step )
MIN_TOLERANCE = 1.e-11 
# Minimum tolerance for the eigenvalues of a matrix to be considered positive definite            
EIGVAL_TOL    = 1.e-4

LOSS_STOP_TOL = 1.e-4

# ################## Expeptions ##################
class LossStagnationError(Exception):
    """Exception raised when the loss has not changed significantly over recent iterations."""
    pass


class LossInfError(Exception):
    """Exception raised when the loss is infinite."""
    pass


################## Miscellaneous ##################
class NullLock:
    """A dummy lock that does nothing, for simplifying code that uses optional locks."""
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass



def upload_natural_image_dataset( dataset_path, astensor=True, zscore=True ):
    '''
    Uploads the natural image dataset. Its basically a copy of load_stimuli_responses 
    from GP_utils.py but without the responses.    
    '''

    X_train = np.load( dataset_path / train_img_dataset_name )
    X_test  = np.load( dataset_path / test_img_dataset_name)

    if zscore: # Each pixel distribution is now mean 0 and std 1
        X_train = scipy.stats.zscore(  X_train, axis=0)
        X_test  = scipy.stats.zscore(  X_test,  axis=0)

    if astensor:

        X_train = torch.from_numpy(X_train).to(DEVICE, dtype=TORCH_DTYPE)
        X_test  = torch.from_numpy(X_test).to(DEVICE, dtype=TORCH_DTYPE)

        X_train = torch.reshape(X_train, ( X_train.shape[0], X_train.shape[1]*X_train.shape[2])) 
        X_test  = torch.reshape(X_test, ( X_test.shape[0], X_test.shape[1]*X_test.shape[2])) 

    else:
        X_train = np.reshape(X_train, ( X_train.shape[0], X_train.shape[1]*X_train.shape[2])) 
        X_test  = np.reshape(X_test,  ( X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

    return  X_train, X_test 



################## ClosedloopProject ##################

def set_global_seed(seed: int):
    import os, random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_idx_for_active_training( n_tot_img_dataset, ntrain, ntilde, ntest_lk, n_reshuffles=0):
    
    '''
    Generate training and testing indices to be used to generate the datasets 
    given to the model.

    Copy of get_idx_for_training_testing_validation() without return of the datasets themselves

    Args:
    ntrain : int
        Number of training points
    ntilde : int
        Number of inducing points
    ntest_lk : int
        Number of test points to exclude from training, to keep for the test loglikelihood estimation
    
    '''

    # set all seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0) # not needed if DEVICE is cpu
    
    all_idx       = torch.arange(0, n_tot_img_dataset )      # Indices of the whole dataset 
    if ntest_lk > 0:
        test_lk_idx   = all_idx[-ntest_lk:]                       # These will be the indices of the test_lk set
    else:
        test_lk_idx   = torch.empty(0, dtype=torch.int64)            # No test set indices
        
    all_idx_perm  = all_idx[~torch.isin( all_idx, test_lk_idx )] # Remove the test set indices from the permutation
    
    for _ in range(n_reshuffles):
        all_idx_perm  = torch.randperm(all_idx_perm.shape[0] )    # Random permutation of the indices

    rndm_idx      = all_idx_perm[:]                                            # These will be the indices of the training. 

    # Choose the indices of the training set. This is overkill here, but in the active learning these indices are constantly changing
    in_use_idx    = rndm_idx[:ntrain]
    xtilde_idx    = in_use_idx[:ntilde] 
    remaining_idx = all_idx_perm[~torch.isin( all_idx_perm, in_use_idx )]

    idx_tuple = (xtilde_idx, in_use_idx, remaining_idx, test_lk_idx, all_idx_perm)

    return idx_tuple

def threaded_train_GP_phase1(init_model, img_train, spike_counts, threadict):

    '''
    Used in phase1 initial_listener_linux(). Fits the model using the initial training set.

    Args:
    init_model (dict): Dictionary containing the initial model infos in the form of a dictionary:
        fit_parameters = {'ntilde':    ntilde_init,
                        'maxiter':     maxiter_init,
                        'nMstep':      nMstep_init,
                        'nEstep':      nEstep_init,
                        'nFparamstep': nFparamstep_init,
                        'kernfun':     GP_utils.acosker,
                        'cellid':      cellid_init,
                        'n_px_side':   n_px_side_init,
                        'in_use_idx':  in_use_idx,     # Used idx for generating xtilde, referred to the whole X dataset
                        'xtilde_idx':  xtilde_idx,     # Used idx for generating the complete set, referred to the whole X dataset
                        'start_idx':   in_use_idx,     # Indexes used to generate the initial training set
                        'lr_Mstep':      lr_Mstep_init, 
                        'lr_Fparamstep': lr_Fparamstep_init
        }

        init_model = {
                'fit_parameters':    fit_parameters,
                'xtilde':            xtilde,
                'hyperparams_tuple': hyperparams_tuple,     # Contains also the upper and lower bounds for the hyperparameters
                'f_params':          f_params,
            }

    spike_counts (torch.tensor): Tensor containing the spike counts for each image in the initial training set

    threadict (dict): Dictionary containing the threads, their variables and events

    '''

    try:
        print('Starting GP fit with collected spikes...')

        X_in_use = img_train[init_model['fit_parameters']['in_use_idx']]
        R_in_use = spike_counts

        assert X_in_use.shape[0] == R_in_use.shape[0]

        # Fit the model
        fit_model, err_dict = varGP(X_in_use, R_in_use, **init_model)

        fit_model['model_type']   = init_model['model_type']

        if err_dict['is_error']:
            if isinstance( err_dict['error'] , LossStagnationError):
                with threadict['print_lock']:
                    print(f'\n...GP Thread: Loss stagnation detected. Stopping the training')
                threadict['model_queue'].put(fit_model)
                return
        
            threadict['global_stop_event'].set()
            threadict['exceptions_q'].put(err_dict['error'])
            with threadict['print_lock']:
                print(f'\n...GP Thread: Error in fitting the model: {err_dict["error"]}')
            return
            # loginfo(f"\n...RCV Thread: Unexpected ERROR: {e}")

        threadict['model_queue'].put(fit_model)
        return
    
    except Exception as e:
        threadict['global_stop_event'].set()
        threadict['exceptions_q'].put(e)
        with threadict['print_lock']:
            print(f'\n...GP Thread: Unexpected ERROR: {e}')
        return


def threaded_train_GP_phase2(model, img_train, spike_counts, threadict):

    '''
    Used in phase1 initial_listener_linux(). Fits the model using the initial training set.

    Args:
    model (GPModel)
            
    spike_counts (torch.tensor): Tensor containing the spike counts for each image in the initial training set

    threadict (dict): Dictionary containing the threads, their variables and events

    '''

    try:
        model_dict = model.to_dict() # very important, the " init kernel values " had not been set out of just the kernel_values attribute"

        print('Starting GP fit with collected spikes...')

        X_in_use = img_train[model_dict['fit_parameters']['in_use_idx']]
        R_in_use = spike_counts

        assert X_in_use.shape[0] == R_in_use.shape[0]

        fit_model_dict, err_dict = varGP(X_in_use, R_in_use, **model_dict, 
                                         verbose=False, silent=True)

        # time.sleep(5)  


        if fit_model_dict is not None:
            fit_model_dict['all_idx_perm'] = model_dict['fit_parameters']['all_idx_perm']
            fit_model_dict['test_lk_idx']  = model_dict['fit_parameters']['test_lk_idx']
            fit_model_dict['in_use_idx']   = model_dict['fit_parameters']['in_use_idx']
            fit_model_dict['model_type']   = model.model_type

            fit_model = GPModel(model_dict=fit_model_dict)

        # else:
        #     with threadict['print_lock']:
        #         print(f'\n...GP Thread: fit_model_dict returned None')
        #     with threadict['dict_lock']:
        #         threadict['global_stop_event'].set()
        #         threadict['exceptions_q'].put(Exception('fit_model_dict returned None'))
        #     return

        if err_dict['is_error']:
            if isinstance( err_dict['error'] , LossStagnationError):
                with threadict['print_lock']:
                    print(f'\n...GP Thread: Loss stagnation detected. Stopping the training')
                threadict['model_queue'].put(fit_model)
                return
            if isinstance( err_dict['error'] , LossInfError):
                with threadict['print_lock']:
                    print(f'\n...GP Thread: Loss is infinite. Adding except to queue. ')
                # threadict['exceptions_q'].put(err_dict['error'])
                return
            
            # threadict['global_stop_event'].set()

            # threadict['exceptions_q'].put(err_dict['error'])
            with threadict['print_lock']:
                print(f'\n...GP Thread: Error in fitting the model: {err_dict["error"]}')
            return
            

        threadict['model_queue'].put(fit_model)
        return
    
    except Exception as e:
        threadict['global_stop_event'].set()
        threadict['exceptions_q'].put(e)
        with threadict['print_lock']:
            print(f'\n...GP Thread: Unexpected ERROR: {e}')
        return


def set_new_model_idxs( current_model, new_model, x_idx_chosen, img_train,):
    '''
    Sets the model image indexes based on the index new image added

    Returns:
        X_in_use_new:  The new training set with the best image added
        xtilde_new:    The new xtilde with the best image added
        new_img:       The best image added to the training set
    
    Args:
        current_model (GPModel): The current active model
        new_model (GPModel): The new active model ( empty at the beginning )
        x_idx_chosen (int): The index of the best image in the total dataset
        img_train (torch.Tensor): The full image dataset

    Sets:
        new_model.in_use_idx (torch.Tensor): The new in_use_idx with the best image added
        new_model.nt (int): The new number of training images
        
        new_model.xtilde (torch.Tensor): The xtilde images #TODO remove?
        new_model.xtilde_idx (torch.Tensor): The new xtilde_idx with the best image added
        new_model.ntilde (int): The new number of xtilde images
    
    '''
    # idxs
    in_use_idx_new = torch.cat((current_model.in_use_idx, x_idx_chosen))

    # images
    new_img        = img_train[x_idx_chosen]
    X_in_use_new   = torch.cat( (img_train[current_model.in_use_idx], new_img), axis=0 )

    xtilde_idx_new = in_use_idx_new
    xtilde_new     = X_in_use_new
    assert in_use_idx_new.shape[0] == xtilde_idx_new.shape[0], 'In use and xtilde indexes must have the same length'

    assert new_img.mean() == X_in_use_new[-1].mean(), 'The new image is not the last in the X_in_use_new array'

    nt_new              = in_use_idx_new.shape[0]
    ntilde_new          = xtilde_idx_new.shape[0]

    new_model.in_use_idx = in_use_idx_new
    new_model.nt         = nt_new

    new_model.xtilde_idx = xtilde_idx_new
    new_model.xtilde     = xtilde_new
    new_model.ntilde     = ntilde_new


    assert img_train[new_model.in_use_idx[-1]].mean() == X_in_use_new[-1].mean(),\
        'The last image in the in_use_idx is not the last in X_in_use_new'
    
    return X_in_use_new, xtilde_new, new_img

def set_new_model_variational_params( current_model, new_model ):
    '''
    Sets variational parameters m and V of the new model, based on the old model. 
    Used in the closed loop case when adding one image

    To update the variational parameters to the new dimensionality we need to pass through the original space. 
    V and m will be projected onto the right eigenspace in varGP using the last used B.
    '''

    # Previous variational parameters
    B   = current_model.B
    V_b = current_model.V_b
    m_b = current_model.m_b

    # On the complete space
    V = B @ V_b @ B.T    # shape (ntilde-1, ntilde-1)
    V = 0.5*(V + V.T)    # Ensure symmetry
    m = B @ m_b          # shape (ntilde-1,)

    # On the complete space +1 dimension
    V_new = torch.eye(new_model.ntilde)
    V_new[:new_model.ntilde-1, :new_model.ntilde-1] = V       

    # New variational parameters
    new_model.V = V_new 
    new_model.m = torch.cat( (m, m.mean()[None]) )

    assert V_new.device.type == DEVICE.type, 'The new V is not on the right device'
    assert new_model.m.shape[0] == new_model.ntilde, 'The new m has the wrong shape'
    assert new_model.V.shape[0] == new_model.ntilde, 'The new V has the wrong shape'

    return 

def project_kernel_matrices(K_tilde, K):
    '''
    Projects the kernel matrices K_tilde and K onto the subspace of the largest eigenvectors of K_tilde
    '''
    
    # eigenvalues  are returned in ascending order. 
    # Uplo=L uses the lower triangular part of the matrix. 
    # Eigenvectors are columns
    eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L') 
    ikeep            = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)                          # Keep only the largest eigenvectors
    B                = eigvecs[:, ikeep]                                     
    # make K_tilde_b and K_b a projection of K_tilde and K into the eigenspace of the largest eigenvectors
    K_tilde_b        = torch.diag(eigvals[ikeep])                    
    K_b              = K @ B                                         
    
    K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep])        
    # if K_tilde.shape[0]==K.shape[0]: 
        # KKtilde_inv_b = B 
    # else K_b @ K_tilde_inv_b:

    assert K_tilde.shape[0]==K.shape[0], 'ntilde and nt must be the same'

    KKtilde_inv_b = B

    return B, K_tilde_b, K_b, K_tilde_inv_b, KKtilde_inv_b

def add_one_img_to_kernel( current_model, X_in_use_new, xtilde_new, new_img ):
    '''
    Calculates the kernel matrices K_tilde, K and Kvec for the new image based on a current current_model kernel

    Adds one line to kernel matrices

    '''
    if current_model.ntilde!=current_model.in_use_idx.shape[0]:
        raise NotImplementedError('Fast calculation of K_tilde not implemented for ntilde != ntrain')
    theta              = current_model.theta
    C                  = current_model.C
    mask               = current_model.mask
    K_tilde_prev       = current_model.final_kernel_values.K_tilde    # Complete Ktilde used in the prev iteration
    kernfun            = current_model.kernfun

    # 100 computations of K_tilde this way take ~0.03s.
    assert xtilde_new[-1].mean() == new_img.mean(), 'The new image is not the last in the xtilde_new array'
    K_tilde_column  = kernfun(theta, xtilde_new[:,mask], new_img[:,mask], C=C, dC=None, diag=False) 
    K_tilde_new     = torch.cat((K_tilde_prev, K_tilde_column[:-1]), axis=1)
    K_tilde_new     = torch.cat((K_tilde_new, K_tilde_column.T), axis=0)  

    if current_model.ntilde==current_model.in_use_idx.shape[0]: K_new = K_tilde_new
    else: raise NotImplementedError('Fast calculation of K not implemented for ntilde != ntrain')

    Kvec_new        = kernfun(theta, X_in_use_new[:,mask],x2=None, C=C, dC=None, diag=True) 

    return K_tilde_new, K_new, Kvec_new

def get_new_model_kernels( current_model, X_in_use_new, xtilde_new, new_img ):
    '''
    Adds one image the kernel matrices K_tilde, K and Kvec with a new image saving them to model.working_kernel_values

    - Calculates the kernel matrices K_tilde, K and Kvec
    - Projects them into the new subspace of the largest eigenvectors

    '''

    # Calculate the kernel matrices for the new image efficently
    K_tilde_new, K_new, Kvec_new = add_one_img_to_kernel( 
        current_model, X_in_use_new, xtilde_new, new_img )
    
    # Now project into the subspace of biggest eigenvectors:

    B_new, K_tilde_b, K_b, K_tilde_inv_b, KKtilde_inv_b = project_kernel_matrices(
        K_tilde_new, K_new )
    
    # if nMstep_init != 0: 
        # raise NotImplementedError('nMstep_init must be 0 for now')
    
    return K_tilde_new, K_new, Kvec_new, B_new, K_tilde_b, K_b, K_tilde_inv_b, KKtilde_inv_b

def set_new_model_f_params_and_theta( current_model, new_model ):
    '''
    Sets the new model f_params and hyperparameters theta based on the current model f_params
    '''
    new_model.f_params = current_model.f_params.copy()

    return

def generate_new_active_model(current_model, x_idx_chosen, img_train, new_spike_counts, light_model=False):
    '''
    Generate new active model with a new image added to the training set

    Args:
        model (GPModel): The current active model
        x_idx_best (int): The index of the best image in the remaining images ( to be added to the training set)
        img_train (torch.Tensor): The full image dataset
    
    Returns: new_active_model (GPModel):
        The new untrained active model with the new image added to the kernel
    
    '''
    new_model = GPModel( 
        old_model=current_model, # This only copies the parameters unrelated to the kernel and other learned params
        C = current_model.C,
        mask = current_model.mask, 
        spike_counts = new_spike_counts,
            )

    # Update the index variables for the new model with the new image idx
    X_in_use_new, xtilde_new, new_img = set_new_model_idxs( current_model, new_model, x_idx_chosen, img_train )

    if not light_model:
        # Writes Var parameters m and V in the FULL space ( no projection)                
        set_new_model_variational_params( current_model, new_model )

        # Calculate the kernel matrices for the new model with the new image
        kernel_matrices_tuple = get_new_model_kernels(
            current_model, X_in_use_new, xtilde_new, new_img )
        
        K_tilde, K, Kvec, B, K_tilde_b, K_b, K_tilde_inv_b, KKtilde_inv_b = kernel_matrices_tuple

        # Set initial values for the kernel values of new model
        new_model.init_kernel_values = new_model.KernelValues(
            C                 = new_model.C,
            mask              = new_model.mask, # only true if Mstep=0 and hyperparams dont change
            K_tilde           = K_tilde,
            K                 = K,
            Kvec              = Kvec,
            B                 = B,
            K_tilde_b         = K_tilde_b,
            K_b               = K_b,
            K_tilde_inv_b     = K_tilde_inv_b,
            KKtilde_inv_b     = KKtilde_inv_b,
        )
        
        set_new_model_f_params_and_theta( current_model, new_model )

        new_model.hyperparams_obj = current_model.hyperparams_obj.copy()
        new_model.update_hyperparams_tuple()

    return new_model

def update_models_with_new_responses( active_model, random_model, 
                                     prev_active_model_spike_counts, prev_random_model_spike_counts, 
                                     x_idx_best, x_idx_rand, 
                                     img_train, spike_counts ):

    '''
    Legacy Function used to update both active and random models with new responses collected in response to a pair of images
    sent as vec file. In the current version we send one imagea t the time so this is useless
    Update the models being trained in parallel with the new responses that have jsut been collected.

    In the closed loop experiment, we train two models in parallel, one with
    active images and one with the random images.

    This function updates the models with the new responses that have just been collected.
    
    '''

    # Update the training set with the new responses
    new_spike_counts_active = torch.cat((prev_active_model_spike_counts, spike_counts[0][None]), axis=0)
    new_spike_counts_random = torch.cat((prev_random_model_spike_counts, spike_counts[1][None]), axis=0)

    # Initialize a new model with the new image arrays and response
    new_active_model = generate_new_active_model( 
        current_model=active_model, x_idx_chosen=x_idx_best, img_train=img_train, 
        new_spike_counts=new_spike_counts_active  
    )

    new_random_model = generate_new_active_model(
        current_model=random_model, x_idx_chosen=x_idx_rand, img_train=img_train,
        new_spike_counts=new_spike_counts_random, light_model=True
    )

    return new_active_model, new_random_model
##################  Preprocessing  ##################

def get_idx_for_training_testing_validation(X, R, ntrain, ntilde, ntest_lk):
    
    # NB: Test set in this case is simply a subset of the original X and R.
    #     to allow the its use for performance comparison using the loglikelihood estimation.
    #     we call it test_lk set

    '''
    Generate training and testing indices to be uset to generate the datasets 
    given to the model.

    if X and R are not None:
    Can also return directly the actual datasets with the indices already applied.

    Especially useful for the active training setup

    Args:
    X : torch.tensor shape (nimages, npx, npx)
        Stimuli
    R : torch.tensor shape (nimages, ncells)
        Responses
    ntrain : int
        Number of training points
    ntilde : int
        Number of inducing points
    ntest_lk : int
        Number of test points for the test loglikelihood estimation
    
    '''
    
    all_idx       = torch.arange(0, X.shape[0], device=DEVICE)                 # Indices of the whole dataset  
    all_idx_perm  = torch.randperm(all_idx.shape[0], device=DEVICE)            # Random permutation of the indices

    test_lk_idx   = all_idx_perm[:ntest_lk]                                    # These will be the indices of the test_lk set
    all_idx_perm  = all_idx_perm[~torch.isin( all_idx_perm, test_lk_idx )]     # Remove the test set indices from the permutation
    rndm_idx      = all_idx_perm[:]                                            # These will be the indices of the training. 

    # Choose the indices of the training set. This is overkill here, but in the active learning these indices are constantly changing
    in_use_idx    = rndm_idx[:ntrain]
    xtilde_idx    = in_use_idx[:ntilde] 
    remaining_idx = all_idx_perm[~torch.isin( all_idx_perm, in_use_idx )]

    # Set the starting set
    xtilde        = X[xtilde_idx,:]       # In the simplest case the starting points are all inducing points
    X_in_use      = X[in_use_idx,:]
    X_remaining   = X[remaining_idx,:]
    X_test_lk     = X[test_lk_idx,:]

    R_remaining   = R[remaining_idx]
    R_in_use      = R[in_use_idx]
    R_test_lk     = R[test_lk_idx]

    X_tuple   = (xtilde, X_in_use, X_remaining, X_test_lk)
    R_tuple   = (R_remaining, R_in_use, R_test_lk)
    idx_tuple = (xtilde_idx, in_use_idx, remaining_idx, test_lk_idx)

    return X_tuple, R_tuple, idx_tuple

def set_hyperparameters( X_in_use, R_in_use, n_px_side, theta=None, freeze_list=[]):
    # Set the hyperparameters of the model
    # If theta is None, the hyperparameters are set based on the STAs
    # If theta is not None, the hyperparameters are set based on the values in theta

    # In this code the learnt hyperparameters are the one in the dictionary 'theta'
    # One can set them direcly or let generate_theta() set them based on the training set STAs
    # To override the choice of generate_theta() just give theta as input 

    # If one wants to compare the hyperparemeters set in Matthews's / Samuels's code one has to set
    # logbetasam : and transform it to logbetaexpr with the function fromlogbetasam_to_logbetaexpr
    # logrhosam  : and transform it to logrhoexpr with the function fromlogrhosam_to_logrhoexpr
    # logsigma_0 : and transform it to sigma_0 exponetiating it

    # logbetaexpr = utils.fromlogbetasam_to_logbetaexpr( logbetasam=torch.tensor(5.5) )# Logbetaexpr in this code is equal to logbeta in Samuele's code. Samuele's code set logbeta to 5.5
    # logrhoexpr  = utils.fromlogrhosam_to_logrhoexpr( logrhosam=torch.tensor(5)) 

    # Set the gradient of the hyperparemters to be updateable 
    for key, value in theta.items():
    # to exclude a single hyperparemeters from the optimization ( to exclude them all just set nMstep=0)
        if key in freeze_list:
            continue
        theta[key] = value.requires_grad_()

    if theta is None:
        hyperparams_tuple = generate_theta( x=X_in_use, r=R_in_use, n_px_side=n_px_side, display=True)
    else:
        hyperparams_tuple = generate_theta( x=X_in_use, r=R_in_use, n_px_side=n_px_side, display=True, **theta)
    return hyperparams_tuple, theta

def set_f_params( logA, lambda0):
    '''
    Generate the f_params dict with the link function parameters

    Set logA to be a learneaable parameter with requires_grad_()
    '''
    # We are not learning the lambda0, since given an A there is a closed form for it that minimised the loss
    f_params = {'logA': logA, 'lambda0':lambda0}
    f_params['logA'] = f_params['logA'].requires_grad_()

    return f_params

def load_stimuli_responses( dataset_path ):
    with open( dataset_path, 'rb') as file:
        loaded_data = pickle.load(file)
    # loaded_data is a Dataset object from module Data with attributes "images_train, _val, _test" as well as responses

    X_train = torch.tensor(loaded_data.images_train).to(device, dtype=TORCH_DTYPE) # shape (2910,108,108,1) where 108 is the number of pixels. 2910 is the amount of training points
    X_val   = torch.tensor(loaded_data.images_val).to(device, dtype=TORCH_DTYPE)
    X_test  = torch.tensor(loaded_data.images_test).to(device, dtype=TORCH_DTYPE)  # shape (30,108,108,1) # nimages, npx, npx

    R_train = torch.tensor(loaded_data.responses_train).to(device, dtype=TORCH_DTYPE) # shape (2910,41) 2910 is the amount of training data, 41 is the number of cells
    R_val   = torch.tensor(loaded_data.responses_val).to(device, dtype=TORCH_DTYPE)
    R_test  = torch.tensor(loaded_data.responses_test).to(device, dtype=TORCH_DTYPE)  # shape (30,30,42) 30 repetitions, 30 images, 42 cells

    return X_train, X_val, X_test, R_train, R_val, R_test

def load_multi_unit_responses(narutal_image_stimuli_pietro, natural_image_single_unit_train_dataset, natural_image_single_unit_test_dataset):

    with open( narutal_image_stimuli_pietro + "_train.npy", 'rb') as file:
        loaded_data = np.load(file)
        X = torch.tensor(loaded_data, dtype=TORCH_DTYPE)
        X_train = X[:2910]
        X_val   = X[2910:3160]
    
    with open( narutal_image_stimuli_pietro + "_test.npy", 'rb') as file:
        loaded_data = np.load(file)
        X_test  = torch.tensor(loaded_data, dtype=TORCH_DTYPE)

    with open( natural_image_single_unit_train_dataset, 'rb') as file:
        loaded_data_single_unit = np.load(file)
    # We upload the data and we create dummy training and validation sets
    R_train = torch.tensor(loaded_data_single_unit).to(device, dtype=TORCH_DTYPE)[:2910]    
    R_val   = torch.tensor(loaded_data_single_unit).to(device, dtype=TORCH_DTYPE)[2910:3160]

    with open( natural_image_single_unit_test_dataset, 'rb') as file:
        loaded_data_single_unit_test = np.load(file)

    R_test = torch.tensor(loaded_data_single_unit_test).to(device, dtype=TORCH_DTYPE)

    return X_train, X_val, X_test, R_train, R_val, R_test

def preprocess_dataset(X_train, X_val, R_train, R_val, R_test, select_cell=True, cellid=None):
    # Stacks the training and validation sets
    # Flatten images
    # Choose the cellid

    X = torch.cat( (X_train, X_val), axis=0,) 
    R = torch.cat( (R_train, R_val), axis=0,)

    n_px_side = X.shape[1]  

    # Reshape images to 1D vector and choose a cell
    X = torch.reshape(X, ( X.shape[0], X.shape[1]*X.shape[2])) 

    if select_cell:
        R = R[...,cellid] 
        R_test = R_test[...,cellid] 

    return X, R, R_test, n_px_side

def estimate_memory_usage(X, R):
    # Calculate memory usage for each tensor
    X_memory = X.element_size() * X.nelement()
    r_memory = R.element_size() * R.nelement()
    # Total memory usage in bytes
    total_memory_bytes = X_memory + r_memory
    # Convert bytes to megabytes (MB)
    total_memory_MB = total_memory_bytes / (1024 ** 2)
    print(f'Total dataset memory on GPU: {total_memory_MB:.2f} MB')
    return total_memory_MB

def get_cell_STA(X, R, zscore=True, show=True,  return_tensor=False):

    if isinstance(X, torch.Tensor): 
        if X.device != 'cpu': 
            X = X.cpu().clone().detach()
        X = X.numpy()
    if isinstance(R, torch.Tensor): 
        if R.device != 'cpu': 
            R = R.cpu().clone().detach()
        R = R.numpy()

    n_px_side = int(np.sqrt(X.shape[1]))

    if zscore:
        X_zsorted = scipy.stats.zscore(  X, axis=0)    # Z-score the images
    else:
        X_zsorted = X
    STA = np.multiply( R[:,None], X_zsorted ).sum(axis=0) / R.sum()
    STA = STA.reshape(n_px_side,n_px_side)

    if not show:
        return STA if not return_tensor else torch.from_numpy(STA).to(DEVICE, dtype=TORCH_DTYPE)
    else:
        plt.imshow(STA, origin='lower', cmap='bwr',  vmax=STA.max(), vmin=STA.min())
        plt.show()

    return STA if not return_tensor else torch.from_numpy(STA).to(DEVICE, dtype=TORCH_DTYPE)

# this works better but i dont know how

def whiten_STA_new(STA, images, ridge_frac=0.1):
    """
    Simple whitening: project onto data subspace and apply (C + λI)^(-1/2).
    STA: (H,W), images: (n,H,W)
    """
    imgs = images.to(DEVICE, dtype=TORCH_DTYPE)
    n = imgs.shape[0]
    X = (imgs - imgs.mean(0, keepdim=True)).reshape(n, -1)            # (n, p)
    STA_vec = STA.reshape(-1).to(DEVICE, dtype=TORCH_DTYPE)           # (p)

    # Economy SVD of centered data (cov = V diag(S^2/(n-1)) V^T)
    U, S, Vt = torch.linalg.svd(X / torch.sqrt(torch.tensor(max(n-1,1),
                               device=DEVICE, dtype=TORCH_DTYPE)),
                                full_matrices=False)
    # Eigenvalues of covariance: eig = S^2
    eig = S**2
    ridge = ridge_frac * eig.mean()
    denom = torch.sqrt(eig + ridge)                                   # inverse sqrt

    # Project STA into subspace spanned by rows of Vt (same as columns of V)
    proj = Vt @ STA_vec                                               # (n)
    proj_w = proj / denom                                             # whiten scaling
    STA_w = (proj_w @ Vt).reshape_as(STA)                             # back to pixel space

    return STA_w

def whiten_STA(STA_squared, images ):
    
    flat_STA = STA_squared.flatten().to(DEVICE)

    mean_images = torch.mean(images, dim=0).to(DEVICE)

    images_centered = images - mean_images

    # traspose cause cov expects the variable to be the row 
    # ( variable is a pixel, observationsare the number of images, different vals of that pixel)
    cov_matrix_images = torch.cov(images_centered.view(images_centered.shape[0], -1).T)

    # Regularize ill conditioned covariance matrix
    alpha = 0.1 * torch.mean(torch.diag(cov_matrix_images)) 

    regularized_covariance = cov_matrix_images + alpha * torch.eye(cov_matrix_images.shape[0], device=DEVICE, dtype=TORCH_DTYPE)

    inv_covariance = torch.linalg.inv(regularized_covariance)

    STA_corrected_vector = inv_covariance @ flat_STA

    STA_corrected_squared = STA_corrected_vector.reshape(STA_squared.shape)

    return STA_corrected_squared

################## Visualization and Saving ##################

def save_model(model, directory, name, additional_description=None, force_overwrite=False, print_lock=NullLock()):
    """
    Save the model parameters and metadata to a specified directory with robust error handling.

    Args:
        model: Dictionary containing all the models results and parameters
        directory (Path): The directory path to save the model
        additional_description (str, optional): Additional text to add to the description
        overwrite (bool, optional): Whether to overwrite an existing directory
    """
    model_pathname = directory / name

    print(model_pathname)

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        with print_lock:
                print(f"Directory {directory} did not exist, creating it" )
        os.makedirs(directory)

    if os.path.exists(model_pathname):
        if not force_overwrite:
            with print_lock:
                answer = input(f"Model {model_pathname} already exists in directory. Overwrite? [y/N]: ")

            if not answer.strip().lower().startswith('y'):
                with print_lock:
                    print(f'Model not saved')
                return
            else:
                print(f"Overwriting model")


    # Helper function to safely format values with proper checks
    def safe_format(value, format_spec=">8.4f", default_value="N/A"):
        """Format a value safely, handling None, missing keys, and formatting errors."""
        if value is None:
            return default_value
            
        try:
            # Handle torch tensors
            if hasattr(value, 'item'):
                try:
                    return f"{value.item():{format_spec}}"
                except (ValueError, TypeError):
                    return str(value.item())
            # Handle regular values
            return f"{value:{format_spec}}"
        except (ValueError, TypeError):
            # If formatting fails, return as string
            return str(value)

    # Helper to safely get nested values from dictionary
    def safe_get(d, key_path, default=None):
        """Safely get a value from nested dictionaries."""
        if not isinstance(d, dict):
            return default
            
        keys = key_path.split('.')
        value = d
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value

    # Check if the required keys exist
    fit_params = safe_get(model, 'fit_parameters', {})
    values_track = safe_get(model, 'values_track', {})
    theta_track = safe_get(values_track, 'theta_track', {})
    f_par_track = safe_get(values_track, 'f_par_track', {})

    # Build description with robust checks
    description = f"""\
        Model Description:
        Cell ID:       {safe_format(safe_get(fit_params, 'cellid'), ">8", "None")}
        ntilde:        {safe_format(safe_get(fit_params, 'ntilde'), ">8", "None")}
        maxiter:       {safe_format(safe_get(fit_params, 'maxiter'), ">8", "None")}
        nMstep:        {safe_format(safe_get(fit_params, 'nMstep'), ">8", "None")}
        nEstep:        {safe_format(safe_get(fit_params, 'nEstep'), ">8", "None")}
        MIN_TOLERANCE: {safe_format(safe_get(fit_params, 'min_tolerance'), ">8.12f", "N/A")}
        EIGVAL_TOL:    {safe_format(safe_get(fit_params, 'eigval_tol'), ">8.4f", "N/A")}
        """

    # Add hyperparameters section if they exist
    if theta_track and all(len(v) > 0 for v in theta_track.values()):
        description += f"""
        Hyperparameters results:
        Start                 ->   End
        """
        
        for key in ['sigma_0', 'eps_0x', 'eps_0y', 'Amp', '-2log2beta', '-log2rho2']:
            if key in theta_track and len(theta_track[key]) > 0:
                start_val = theta_track[key][0] if len(theta_track[key]) > 0 else None
                end_val = theta_track[key][-1] if len(theta_track[key]) > 0 else None
                description += f"{key}:     {safe_format(start_val)} -> {safe_format(end_val)}\n        "

        # Add derived parameters if possible
        if '-2log2beta' in theta_track and len(theta_track['-2log2beta']) > 0:
            start_beta = logbetaexpr_to_beta(theta_track['-2log2beta'][0]) if callable(globals().get('logbetaexpr_to_beta')) else None
            end_beta = logbetaexpr_to_beta(theta_track['-2log2beta'][-1]) if callable(globals().get('logbetaexpr_to_beta')) else None
            description += f"beta:        {safe_format(start_beta)} -> {safe_format(end_beta)}\n        "

        if '-log2rho2' in theta_track and len(theta_track['-log2rho2']) > 0:
            start_rho = logrhoexpr_to_rho(theta_track['-log2rho2'][0]) if callable(globals().get('logrhoexpr_to_rho')) else None
            end_rho = logrhoexpr_to_rho(theta_track['-log2rho2'][-1]) if callable(globals().get('logrhoexpr_to_rho')) else None
            description += f"rho:         {safe_format(start_rho)} -> {safe_format(end_rho)}\n        "

    # Add f_params if they exist
    if f_par_track:
        description += """
        Link function results [f_params]:
        """
        for key in ['logA', 'lambda0', 'loglambda0']:
            if key in f_par_track and len(f_par_track[key]) > 0:
                start_val = f_par_track[key][0] if len(f_par_track[key]) > 0 else None
                end_val = f_par_track[key][-1] if len(f_par_track[key]) > 0 else None
                description += f"\n        {key}:        {safe_format(start_val)} -> {safe_format(end_val)}"

        # Add derived A value if logA exists
        if 'logA' in f_par_track and len(f_par_track['logA']) > 0:
            try:
                start_A = torch.exp(f_par_track['logA'][0]) if torch.is_tensor(f_par_track['logA'][0]) else None
                end_A = torch.exp(f_par_track['logA'][-1]) if torch.is_tensor(f_par_track['logA'][-1]) else None
                description += f"\n\n        A:           {safe_format(start_A)} -> {safe_format(end_A)}"
            except Exception:
                pass  # Skip if calculation fails

    # Add additional description if provided
    if additional_description is not None:
        description += f"\n\n{additional_description}"
    
    model['description'] = description

    # Save the file
    with open(model_pathname, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata_path = os.path.join(directory, f'{name}_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(description)

    with print_lock:
        print(f"Model saved successfully to {model_pathname}")
    return

def upload_model( directory, model_name):
    ''''''
    # Load the model
    model_pathname = directory / model_name
    with open(model_pathname, 'rb') as f:
        model = pickle.load(f)

    return model



def plot_loss_and_theta_notebook(model, linestyle='-', marker='o', figsize=(10, 10), xlim=None, ylim_logmarg=None, ylim_lambda0=None, ylim_eigvals=None):

    #region Extract the data
    values_track = model['values_track']
    nMstep  = model['fit_parameters']['nMstep']
    nEstep  = model['fit_parameters']['nEstep']
    maxiter = model['fit_parameters']['maxiter']
    cellid  = model['fit_parameters']['cellid']
    ntilde  = model['fit_parameters']['ntilde']

    logmarginal   = values_track['loss_track']['logmarginal'].cpu().detach().numpy()
    loglikelihood = values_track['loss_track']['loglikelihood'].cpu().detach().numpy()
    KL            = values_track['loss_track']['KL'].cpu().detach().numpy()

    # Extract the variational parameters
    m_b_tuple = values_track['variation_par_track']['m_b']
    V_b_tuple = values_track['variation_par_track']['V_b']

    n_eigvals = [ m_b.shape[0] for  m_b in m_b_tuple]
    m_b_mean  = [ m_b.mean().item() for m_b in m_b_tuple]
    V_b_mean  = [ torch.diag(V_b).mean().item() for V_b in V_b_tuple]


    # Extract the data of the f params
    A       =  torch.exp(values_track['f_par_track']['logA']).cpu().detach().numpy()
    if 'lambda0' in values_track['f_par_track'].keys():
        lambda0 = values_track['f_par_track']['lambda0'].cpu().detach().numpy()
    if 'loglambda0' in values_track['f_par_track'].keys():
        lambda0 = torch.exp(values_track['f_par_track']['loglambda0']).cpu().detach().numpy()
    # lambda0 = torch.exp(values_track['f_par_track']['loglambda0']).cpu().detach().numpy()
    # If we want lambda0 to be negative
    # lambda0 = -lambda0
    # lambda0 = torch.atanh(values_track['f_par_track']['tanhlambda0']).cpu().detach().numpy()

    # Extract the data for the second plot (Hypoerparameters)
    theta_sigma_0 = values_track['theta_track']['sigma_0'].cpu().detach().numpy()
    theta_eps_0x  = values_track['theta_track']['eps_0x'].cpu().detach().numpy()
    theta_eps_0y  = values_track['theta_track']['eps_0y'].cpu().detach().numpy()
    # theta_log2beta = values_track['theta_track']['-2log2beta'].cpu().detach().numpy()
    theta_beta = logbetaexpr_to_beta(values_track['theta_track']['-2log2beta']).cpu().detach().numpy()
    # theta_log2rho2 = values_track['theta_track']['-log2rho2'].cpu().detach().numpy()
    theta_rho  = logrhoexpr_to_rho(values_track['theta_track']['-log2rho2']).cpu().detach().numpy()
    theta_Amp = values_track['theta_track']['Amp'].cpu().detach().numpy()
    #endregion

    # Create a plot
    fig, ((ax1, ax4), (ax10, ax22) ) = plt.subplots(2, 2, figsize=figsize, )#sharex=True)

    # To format the y-axis
    def format_func(value, tick_number):
        if abs(value) < 1:
            return f'{value:.2f}'
        else:
            return f'{value:.4g}'
    formatter = FuncFormatter(format_func)

    iterations = np.arange(0, len(logmarginal))
    if len(iterations) != model['fit_parameters']['maxiter']:
        print(f'Iterations: {len(iterations)} != maxiter: {model["fit_parameters"]["maxiter"]}')

    #region
    # Plot logmarginal on the first y-axis
    ax1.plot(iterations, -logmarginal, label='-logmarginal', color='blue', linestyle=linestyle, marker=marker)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('-logmarginal', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.yaxis.set_major_formatter(formatter)
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=40))  # Set x-axis to display only integer values 
    ax1.set_ylim(ylim_logmarg)

    # Create a second y-axis for loglikelihood
    ax2 = ax1.twinx()
    ax2.plot(iterations, loglikelihood, label='loglikelihood', color='green',  linestyle=linestyle, marker=marker)
    ax2.set_ylabel('loglikelihood', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.yaxis.set_major_formatter(formatter)

    # Create a third y-axis for KL
    ax3 = ax1.twinx()
    ax3.plot(iterations, KL, label='KL', color='red', linestyle=linestyle, marker=marker)
    ax3.set_ylabel('KL', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    ax4.yaxis.set_major_formatter(formatter)
    # Adjust the position of the third y-axis
    ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward

    # Add a title
    # ax1.set_title(f'Loss = -logmarginal = KL - loglikelihood nMstep = {nMstep}, nEstep = {nEstep}, maxiter = {maxiter}. Cell :{cellid}')
    # ax1.grid()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #endregion

    #region
    # Plot the f parameter values:
    ax10.plot(iterations, A, label='A', color='purple', linestyle=linestyle, marker=marker)
    ax10.set_ylabel('A', color='purple')
    ax10.tick_params(axis='y', labelcolor='purple')
    ax10.yaxis.set_major_formatter(formatter)
    # ax10.spines['right'].set_position(('outward', 120))  # Move the third y-axis outward

    ax11 = ax10.twinx()
    ax11.plot(iterations, lambda0, label='lambda0', color='orange', linestyle=linestyle, marker=marker, )
    ax11.set_ylabel('lambda0', color='orange')
    ax11.tick_params(axis='y', labelcolor='orange')
    ax11.yaxis.set_major_formatter(formatter)
    # ax11.spines['right'].set_position(('outward', 180))  # Move the third y-axis outward
    ax11.set_ylim(ylim_lambda0)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #endregion
    
    #region # Plot the variational parameters
    ax22.plot(iterations, n_eigvals, label='n_eigvals', color='blue', linestyle='', marker='o')
    ax22.set_xlabel('Iteration')
    ax22.set_ylabel('n_eigvals', color='blue')
    ax22.tick_params(axis='y', labelcolor='blue')
    ax22.yaxis.set_major_formatter(formatter)
    if ylim_eigvals is not None: ax22.set_ylim( ylim_eigvals )
    else:                        ax22.set_ylim(0, ntilde)


    ax23 = ax22.twinx()
    ax23.plot(iterations, m_b_mean, label='m_b mean', color='green', linestyle=linestyle, marker=marker)
    ax23.set_ylabel('m_b mean', color='green')
    ax23.tick_params(axis='y', labelcolor='green')
    ax23.yaxis.set_major_formatter(formatter)

    ax24 = ax22.twinx()
    ax24.plot(iterations, V_b_mean, label='V_b diag mean', color='orange', linestyle=linestyle, marker=marker)
    ax24.set_ylabel('V_b diag mean', color='orange')
    ax24.tick_params(axis='y', labelcolor='orange')
    ax24.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #endregion
    
    # Second subplot: Theta values
    #region
    iterations_theta = iterations-0.5 # The theta saved as the first iterations are the ones that generated the eigenspace used in the Estep first iteration
    ax4.plot(iterations_theta, theta_sigma_0, label='sigma_0', color='blue', linestyle=linestyle, marker=marker)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('sigma_0', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.yaxis.set_major_formatter(formatter)

    # Create additional y-axes for each theta parameter
    ax5 = ax4.twinx()
    ax5.plot(iterations_theta, theta_eps_0x, label='eps_0x', color='green', linestyle=linestyle, marker=marker)
    ax5.set_ylabel('eps_0x', color='green')
    ax5.tick_params(axis='y', labelcolor='green')
    ax5.yaxis.set_major_formatter(formatter)

    ax6 = ax4.twinx()
    ax6.plot(iterations_theta, theta_eps_0y, label='eps_0y', color='red', linestyle=linestyle, marker=marker)
    ax6.set_ylabel('eps_0y', color='red')
    ax6.tick_params(axis='y', labelcolor='red')
    ax6.spines['right'].set_position(('outward', 60))
    ax6.yaxis.set_major_formatter(formatter)

    ax7 = ax4.twinx()
    ax7.plot(iterations_theta, theta_beta, label='beta', color='purple', linestyle=linestyle, marker=marker)
    ax7.set_ylabel('beta', color='purple')
    ax7.tick_params(axis='y', labelcolor='purple')
    ax7.spines['right'].set_position(('outward', 120))
    ax7.yaxis.set_major_formatter(formatter)

    ax8 = ax4.twinx()
    ax8.plot(iterations_theta, theta_rho, label='rho', color='orange', linestyle=linestyle, marker=marker)
    ax8.set_ylabel('rho', color='orange')
    ax8.tick_params(axis='y', labelcolor='orange')
    ax8.spines['right'].set_position(('outward', 180))
    ax8.yaxis.set_major_formatter(formatter)

    ax9 = ax4.twinx()
    ax9.plot(iterations_theta, theta_Amp, label='Amp', color='brown', linestyle=linestyle, marker=marker)
    ax9.set_ylabel('Amp', color='brown')
    ax9.tick_params(axis='y', labelcolor='brown')
    ax9.spines['right'].set_position(('outward', 240))
    ax9.yaxis.set_major_formatter(formatter)
    #endregion

    # Add a title and legend
    # ax4.set_title(f'Theta Parameters Over Iterations. nMstep = {nMstep}, nEstep = {nEstep}, maxiter = {maxiter}')

    # set the xlim if specified
    axes = [ax1, ax2, ax3, ax10, ax11, ax4, ax5, ax6, ax7, ax8, ax9, ax22, ax23, ax24]
    for ax in axes:
        if xlim is not None:
            ax.set_xlim(xlim)
    ax1.grid()
    ax4.grid()
    ax10.grid()
    ax22.grid()

    # Show the plot
    # fig.tight_layout()  # Adjust layout to prevent overlap
    fig.subplots_adjust(left=0.00, wspace=0.4)

    fig.suptitle(f'Loss = -logmarginal = KL - loglikelihood nMstep = {nMstep}, nEstep = {nEstep}, maxiter = {maxiter}. Cell :{cellid}')

    plt.show()

def load_model(directory):
    """
    Load the model parameters and metadata from a specified directory.

    Args:
        directory (str): The directory to load the model and parameters from.

    Returns:
        dict: The loaded model parameters and metadata.
    """
    with open(f'{directory}/model', 'rb') as f:
        model_b = pickle.load(f)
    return model_b

def print_hyp( theta ):
        key_width = 12
        number_width = 8
        for key in theta.keys():
            if key == '-2log2beta':
                print(f' {key:<{key_width}}: {theta[ key ]:>{number_width}.4f} --> beta: {logbetaexpr_to_beta(theta[key]):>{number_width}.4f}')  
                continue
            if key == '-log2rho2':
                print(f' {key:<{key_width}}: {theta[ key ]:>{number_width}.4f} --> rho : {logrhoexpr_to_rho(theta[key]):>{number_width}.4f}')  
                continue

            print(f' {key:<{key_width}}: {theta[ key ]:>{number_width}.4f}')     

def plot_hyperparams_on_STA(fit_model, STA=None, ax=None, **kwargs):
    """
    Backward-compatible version:
      - Returns ax (as legacy) OR (fig, ax) if return_fig=True.
      - Keeps default savefig=True.
      - Accepts same kwargs (label, center_color, width_color, show_values, name, savefig, return_fig).
      - Draws image first (better visibility) unless keep_legacy_order=True.
    """
    if isinstance(fit_model, GPModel):
        fit_model = fit_model.to_dict()

    label            = kwargs.get('label', None)
    center_color     = kwargs.get('center_color', 'k')
    width_color      = kwargs.get('width_color', 'k')
    show_values      = kwargs.get('show_values', True)
    name             = kwargs.get('name', 'noname')
    savefig          = kwargs.get('savefig', True)          # legacy default
    return_fig       = kwargs.get('return_fig', False)      # new opt
    keep_legacy_order= kwargs.get('keep_legacy_order', False)


    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        created_fig = True
    else:
        fig = ax.figure

    n_px_side = fit_model['fit_parameters']['n_px_side']
    center_idxs = torch.tensor([(n_px_side-1)/2, (n_px_side-1)/2])
    hp = fit_model['hyperparams_tuple'][0]
    eps_0x_fit = hp['eps_0x']; eps_0y_fit = hp['eps_0y']
    logbetaexpr_fit = hp['-2log2beta']; logrhoexpr_fit = hp['-log2rho2']
    sigma_0_fit = hp['sigma_0']; amp_fit = hp['Amp']

    eps_idxs_fit = torch.tensor([
        center_idxs[0]*(1+eps_0x_fit),
        center_idxs[1]*(1+eps_0y_fit)
    ])

    beta_value = logbetaexpr_to_beta(logbetaexpr_fit).item()
    rho_value  = logrhoexpr_to_rho(logrhoexpr_fit).item()
    beta_value_px = beta_value * n_px_side/2
    rho_value_px  = rho_value  * n_px_side/2

    # Grid
    ycord, xcord = torch.meshgrid(
        torch.linspace(-1, 1, n_px_side),
        torch.linspace(-1, 1, n_px_side),
        indexing='ij')
    xflat = xcord.flatten(); yflat = ycord.flatten()
    logalpha_fit = -torch.exp(logbetaexpr_fit)*((xflat - eps_0x_fit)**2 + (yflat - eps_0y_fit)**2)
    alpha_local_fit = torch.exp(logalpha_fit)
    alpha_img = alpha_local_fit.reshape(n_px_side, n_px_side)

    # Draw order
    if keep_legacy_order:
        # Contours first
        levels = torch.tensor([np.exp(-4.5), np.exp(-2), np.exp(-0.5)])
        ax.contour(alpha_img.cpu(), levels=levels.cpu(), colors=width_color, alpha=0.5)
        if STA is not None:
            vmax = float(STA.max())
            vmin = -vmax 
            ax.imshow(STA, vmax=vmax, vmin=vmin, cmap='bwr')
    else:
        if STA is not None:
            vmax = float(STA.max())
            vmin = -vmax 
            ax.imshow(STA, vmax=vmax, vmin=vmin, cmap='bwr')
        levels = torch.tensor([np.exp(-4.5), np.exp(-2), np.exp(-0.5)])
        ax.contour(alpha_img.cpu(), levels=levels.cpu(), colors=width_color, alpha=0.6, linewidths=1.0)

    ax.scatter(eps_idxs_fit[0].cpu(), eps_idxs_fit[1].cpu(),
               color=center_color, s=30, marker="o", label=label)
    ax.set_title('Hyperparameters on STA')
    # ax.set_xticks([]); ax.set_yticks([])

    if show_values:
        param_text = (f"Center: ({eps_0x_fit.item():.2f},{eps_0y_fit.item():.2f}) "
                      f"| px({eps_idxs_fit[0].item():.1f},{eps_idxs_fit[1].item():.1f})\n"
                      f"Beta: {beta_value:.2f} ({beta_value_px:.0f}px)\n"
                      f"Rho:  {rho_value:.2f} ({rho_value_px:.0f}px)\n"
                      f"Sigma0: {sigma_0_fit.item():.2f}  Amp: {amp_fit.item():.2f}")
        ax.text(0.03, 0.97, param_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, pad=0.3))

    if savefig and created_fig:
        os.makedirs( figs_dir, exist_ok=True)
        fig.savefig( figs_dir / f'{name}.png', dpi=200, bbox_inches='tight')

    # Close figure to free memory if we created it and are not returning it
    if created_fig and not return_fig:
        plt.close(fig)

    if return_fig:
        return fig, ax
    return ax

def plot_hyperparams_on_STA_old(fit_model, STA=None, ax=None,  **kwargs):
    '''
    Plot the hyperparameters on top of the STA image. Converts model to dictionary if it is a GPModel object

    Args:
        fit_model (dict): The fitted model dictionary containing the hyperparameters and STA
        STA (np.ndarray, optional): The STA image to plot


    '''

    if isinstance(fit_model, GPModel):
        fit_model = fit_model.to_dict()

    label = kwargs.get('label', None)
    center_color = kwargs.get('center_color', 'k')
    width_color  = kwargs.get('width_color', 'k')
    show_values = kwargs.get('show_values', True)  # Option to show parameter values
    name = kwargs.get('name', 'noname')
    savefig = kwargs.get('savefig', True)
    created_fig = (ax is None)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5)) 

    n_px_side = fit_model['fit_parameters']['n_px_side']
    
    # Eps_0 : Center of the receptive field
    center_idxs = torch.tensor([(n_px_side-1)/2, (n_px_side-1)/2])
    eps_0x_fit = fit_model['hyperparams_tuple'][0]['eps_0x']
    eps_0y_fit = fit_model['hyperparams_tuple'][0]['eps_0y']
    logbetaexpr_fit = fit_model['hyperparams_tuple'][0]['-2log2beta']
    logrhoexpr_fit = fit_model['hyperparams_tuple'][0]['-log2rho2']
    sigma_0_fit = fit_model['hyperparams_tuple'][0]['sigma_0']
    amp_fit = fit_model['hyperparams_tuple'][0]['Amp']

    eps_idxs_fit = torch.tensor([
        center_idxs[0]*(1+eps_0x_fit), 
        center_idxs[1]*(1+eps_0y_fit)
    ])

    # Convert log parameters to their actual physical interpretable values
    beta_value = logbetaexpr_to_beta(logbetaexpr_fit).item()
    rho_value = logrhoexpr_to_rho(logrhoexpr_fit).item()

    # We also want to print them in pixel coordinates
    beta_value_px = beta_value * n_px_side/2
    rho_value_px  = rho_value * n_px_side/2

    # Beta : Width of the receptive field - Implemented by the "alpha_local" part of the C covariance matrix
    ycord, xcord = torch.meshgrid(torch.linspace(-1, 1, n_px_side), torch.linspace(-1, 1, n_px_side), indexing='ij') # a grid of 108x108 points between -1 and 1
    xcord = xcord.flatten()
    ycord = ycord.flatten()
    logalpha_fit = -torch.exp(logbetaexpr_fit)*((xcord - eps_0x_fit)**2+(ycord - eps_0y_fit)**2)
    alpha_local_fit = torch.exp(logalpha_fit)    # aplha_local in the paper

    # Levels of the contour plot for distances [1sigma, 2sigma, 3sigma]
    # (x**2 + y**2) = n*sigma -> alpha_local = exp( - (n*sigma)^2 / (2*sigma^2) )
    levels = torch.tensor([np.exp(-4.5), np.exp(-2), np.exp(-1/2)])
    contours = ax.contour(alpha_local_fit.reshape(n_px_side,n_px_side).cpu(), levels=levels.cpu(), colors=width_color, alpha=0.5)
    center_point = ax.scatter(eps_idxs_fit[0].cpu(), eps_idxs_fit[1].cpu(), color=center_color, s=30, marker="o", label=label)
    
    # Set title and add the hyperparameters to the legend
    ax.set_title('Hyperparameters on STA')

    # Create a string with parameter values if requested
    if show_values:
        # Format parameter values for display, including both normalized and pixel coordinates
        param_text = (f"Center: ({eps_0x_fit.item():.2f}, {eps_0y_fit.item():.2f})  |  ({eps_idxs_fit[0].item():.1f}, {eps_idxs_fit[1].item():.1f})px\n"
                     f"Beta: {beta_value:.2f}  | {beta_value_px:.0f}px\n"
                     f"Rho: {rho_value:.2f}  | {rho_value_px:.0f}px\n"
                     f"Sigma₀: {sigma_0_fit.item():.2f}\n"
                     f"Amp: {amp_fit.item():.2f}")
        
        # Add text box with parameter values
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    # Show the STA if provided
    if STA is not None:
        ax.imshow(STA, vmax=STA.max(), vmin=STA.min(), cmap='bwr')

    # Show the updated image
    # plt.show(block=False)
    
    # save the figure
    if savefig:
        fig.savefig( session_data_path / f'STA_w_hyp_{label}_{name}.png')

    # Close figure to free memory if we created it
    if created_fig:
        plt.close(fig)

    return ax

def plot_final_and_intermediate_fit(fit_model, init_model, X_in_use, R_in_use, X_test_avg, R_test_avg_cell, cells_reliability, intermediate_model_iteration=2):

    '''
    Does inference on the X_test_avg stimuli and plots the predicted responses vs the true responses.

    Also calculates the correlation between the predicted and the true responses.
    '''
    
    cellid = fit_model['fit_parameters']['cellid']

    # region _______ Plot STA and HPs ______
    fig, ax = plt.subplots(1, 3, figsize=(22,5)) 
    STA     = get_cell_STA(X_in_use, R_in_use, zscore=True)

    intermediate_model = get_model_at_iteration(fit_model, intermediate_model_iteration)

    plot_hyperparams_on_STA( intermediate_model, STA, ax[0], center_color='black', label='Inter HP', width_color='k',)
    plot_hyperparams_on_STA( fit_model, STA, ax[0], center_color='blue', label='Final HP', width_color='m',)
    plot_hyperparams_on_STA( init_model, STA, ax[0], center_color='white', label='Initial HP', width_color='white',)
    ax[0].legend(loc='upper right')
    fig.suptitle(f'Sta of cell: {cellid} with reliability: {cells_reliability[cellid].cpu():.3f}')
    # endregion

    # _______ Inference ______
    f_mean, r, r2 = inference_and_correlation_cell(fit_model, X_test_avg, R_test_avg_cell)

    # _______ Plot final fit ______
    ax[1].plot( R_test_avg_cell.cpu() , label='True response + min of train set'  , marker='o')  
    ax[1].plot( f_mean.cpu(), label='Predicted response',  marker='o')  

    ax[2].scatter( R_test_avg_cell.cpu(), f_mean.cpu(),  marker='o')
    ax[2].set_title('Predicted vs True responses')
    ax[2].set_xlabel('True responses')
    ax[2].set_ylabel('Predicted responses')
    ax[1].legend(loc='upper right')

    ax[1].set_title(f'Correlation: {r:.3f}, R^2: {r2:.3f}')

    return f_mean, r, r2

def visualize_tensor(tensor, title=None, cmap='viridis', origin='lower', return_fig=False):
    """
    Visualize a tensor as a square image for debugging purposes.
    
    Parameters:
    -----------
    tensor : torch.Tensor or numpy.ndarray
        The tensor to visualize. If 1D, it will be reshaped to a square.
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap to use (default: 'viridis')
    origin : str, optional
        Origin position ('lower' or 'upper')
    return_fig : bool, optional
        If True, returns the figure object instead of displaying it
    
    Returns:
    --------
    fig : matplotlib.figure.Figure, optional
        Figure object if return_fig is True
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # Convert to numpy if it's a PyTorch tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Flatten if it has more than 2 dimensions
    if tensor.ndim > 2:
        tensor = tensor.reshape(-1)
    
    # If 1D, reshape to a square (or as square as possible)
    if tensor.ndim == 1:
        # Try to find factors close to square
        length = tensor.shape[0]
        side = int(np.sqrt(length))
        
        # If not a perfect square, find the closest dimensions
        if side*side == length:
            # Perfect square
            tensor = tensor.reshape(side, side)
        else:
            # Find the best rectangular shape
            for i in range(side, 0, -1):
                if length % i == 0:
                    tensor = tensor.reshape(i, length // i)
                    break
            else:
                # If no exact divisor found, use the original side and pad
                tensor = np.pad(tensor, (0, side*side - length), 'constant')
                tensor = tensor.reshape(side, side)
    
    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(tensor, cmap=cmap, origin=origin)
    plt.colorbar(im, ax=ax)
    
    if title:
        ax.set_title(title)
    
    # Add dimension info
    ax.set_xlabel(f"Shape: {tensor.shape}")
    
    # Show the plot or return the figure
    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()
        plt.close()

####### Get info from model #########
def get_K_and_Kinv_B(K_tilde):
    
    eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')  
    EIGVAL_TOL    = 1.e-4                                 
    ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)  
    B = eigvecs[:, ikeep]                                           
    K_tilde_b     = torch.diag(eigvals[ikeep])
    K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep]) 

    return K_tilde_b, K_tilde_inv_b, B

def get_final_K_vals(model, mask_override=None):

    if isinstance(model, GPModel):
        model = model.to_dict()
    final_kernel_vals = model['final_kernel']

    C    = final_kernel_vals['C']
    mask = final_kernel_vals['mask']
    K_tilde = final_kernel_vals['K_tilde']  # This is correct
    K_tilde_b, K_tilde_inv_b, B = get_K_and_Kinv_B(K_tilde)

    return C, mask, K_tilde_b, K_tilde_inv_b, B

def pred_firing_rate(model_dict, x):

    """
    Predict the firing of image
    x shape     : (n, nx) n images

    """
    kernfun     = model_dict['fit_parameters'].get('kernfun')
    xtilde      = model_dict.get('xtilde').to(DEVICE, dtype=TORCH_DTYPE)

    C, mask, K_tilde_b, K_tilde_inv_b, B = get_final_K_vals(model_dict)

    theta       = model_dict.get('hyperparams_tuple')[0]
    f_params      = model_dict.get('f_params')

    m_b           = model_dict.get('m_b')
    V_b           = model_dict.get('V_b')

    Kvec    = kernfun(theta, x[:,mask], x2=None, C=C, dC=None, diag=True)

    K   = kernfun(theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)
    K_b       = K @ B
    KKtilde_inv_b = K_b @ K_tilde_inv_b

    lambda_m, lambda_var = lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta, kernfun=kernfun)
    f_mean               = mean_f_given_lambda_moments(f_params, lambda_m, lambda_var)

    return f_mean, lambda_m, lambda_var


#### gradient related functions ####

def compare_gradient_patterns(predicted_grad, reference_grad):
        """Compare gradient patterns regardless of scale/offset differences"""
        
        # Option 1: Correlation coefficient instead of R²
        pred_flat = predicted_grad.flatten()
        ref_flat = reference_grad.flatten()
        correlation = torch.corrcoef(torch.stack([pred_flat, ref_flat]))[0,1]
        
        # Option 2: Z-score normalize before computing R²
        pred_norm = (predicted_grad - torch.mean(predicted_grad)) / (torch.std(predicted_grad) + 1e-8)
        ref_norm = (reference_grad - torch.mean(reference_grad)) / (torch.std(reference_grad) + 1e-8)
        
        ss_tot = torch.sum((ref_norm)**2) 
        ss_res = torch.sum((pred_norm - ref_norm)**2)
        r2_normalized = 1 - ss_res / (ss_tot + 1e-8)
        
        return correlation, r2_normalized

def reconstruct_gradient_image(dlambda, mask, n_px_side=108):
    """
    Reconstructs a full image from the gradient values computed on the masked pixels.
    
    Parameters:
    ----------
    dlambda : torch.Tensor
        The gradient vector (contains only values for masked pixels)
    mask : torch.Tensor
        Boolean mask indicating which pixels were used in the computation
    n_px_side : int
        The side length of the original square image
        
    Returns:
    -------
    torch.Tensor
        The reconstructed square image with gradient values
    """
    # Create a zero tensor with the total number of pixels in the original image
    full_size = n_px_side * n_px_side
    gradient_image = torch.zeros(full_size, device=dlambda.device, dtype=dlambda.dtype)
    
    assert dlambda.shape[0] == mask.sum().item(), "dlambda size must match number of True values in mask"

    # Place the gradient values back into their original positions
    gradient_image[mask] = dlambda
    
    # Reshape to square image
    gradient_image = gradient_image.reshape(n_px_side, n_px_side)
    
    return gradient_image

def overlay_gradient_on_image(image, gradient, mask, STA, alpha=0.7, n_px_side=108):
    """
    Overlay the gradient on the original image and show the masked area boundary.
    
    Parameters:
    ----------
    image : torch.Tensor
        Original input image
    gradient : torch.Tensor
        2D gradient image (already reconstructed to full size)
    mask : torch.Tensor
        Boolean mask indicating pixels used by the model
    alpha : float
        Transparency for gradient overlay
    n_px_side : int
        Side length of square image
    """
    plt.figure(figsize=(11, 3))
    
    # Create 2D version of mask for visualization
    mask_2d = torch.zeros((n_px_side * n_px_side), device=mask.device)
    mask_2d[mask] = 1.0
    mask_2d = mask_2d.reshape(n_px_side, n_px_side).cpu().numpy()

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image.reshape(n_px_side, n_px_side).cpu().detach(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Gradient with mask boundary
    plt.subplot(1, 4, 2)
    plt.imshow(gradient.cpu().detach(), cmap='RdBu_r', 
               vmin=-torch.max(torch.abs(gradient)), 
               vmax=torch.max(torch.abs(gradient)))
    
    # Add contour around the mask
    plt.contour(mask_2d, levels=[0.5], colors='yellow', linewidths=1.5)
    plt.title('Gradient with Mask Boundary')
    plt.axis('off')
    
    # Overlay with mask boundary
    plt.subplot(1, 4, 3)
    plt.imshow(image.reshape(n_px_side, n_px_side).cpu().detach(), cmap='gray')
    plt.imshow(gradient.cpu().detach(), cmap='RdBu_r', alpha=alpha,
               vmin=-torch.max(torch.abs(gradient)), 
               vmax=torch.max(torch.abs(gradient)))
    
    # Add contour around the mask
    plt.contour(mask_2d, levels=[0.5], colors='yellow', linewidths=1.5)
    plt.title('Overlay with Mask Boundary')
    plt.axis('off')

    # STA
    plt.subplot(1, 4, 4)
    plt.imshow(STA.reshape(n_px_side, n_px_side).cpu().detach(), cmap='bwr', 
               vmin=-torch.max(torch.abs(STA)), 
                vmax=torch.max(torch.abs(STA)))
    plt.title('STA')
    plt.axis('off')
    
    rows, cols = np.where(mask_2d == 1)
    if rows.size > 0 and cols.size > 0:
        ymin, ymax = np.min(rows), np.max(rows)
        xmin, xmax = np.min(cols), np.max(cols)
        
        # Add some padding
        padding = 10
        xlim = (max(0, xmin - padding), min(n_px_side, xmax + padding))
        # For imshow with default origin='upper', the y-axis is inverted
        ylim = (min(n_px_side, ymax + padding), max(0, ymin - padding))

        # Apply zoom to all subplots
        for ax in axs:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    plt.tight_layout()

def gradients_wrt_dx(model, x_i, mask_override=None):

    ''' Compute the gradient of the mean firing rate, of lambda moments and K(xtilde,x_i) w.r.t. x_i
        for one or mode image x_i UNMASKED
        
        x_i must be of shape (nx) . ONLY ONE IMAGE SUPPORTED FOR NOW
        '''

    theta = model.get('hyperparams_tuple')[0]
    m_b   = model['m_b']
    V_b   = model['V_b']
    f_params = model['f_params']

    C, mask, K_tilde_b, K_tilde_inv_b, B = get_final_K_vals(model, mask_override=mask_override)

    xtilde = model['xtilde']

    xtilde_m = xtilde[:,mask]
    x_i_m    = x_i[mask]

    # Get the gradient of K(xtilde, x_i) w.r.t. x_i
    # K, dK_x    = acosker(theta, xtilde_m, x_i_m[None,:], C, get_dK_x=True) # (1, ntilde) and (ntilde, 1, nx)

    # Note we are not using batch dimensions correctly, n of images is middle dimension

    # dlambda, dvar_lambda, dK_vec_x, dsecond_term = lambda_moments_gradient_wrt_x(theta, xtilde_m, x_i_m[None,:], C, m_b, V_b, K_tilde_b, K_tilde_inv_b, B ) #[:,-1]

    lambda_m, lambda_var, dlambda, dvar_lambda = lambda_moments_and_gradient_wrt_x(
        theta, xtilde_m, x_i_m[None,:], C, m_b, V_b, K_tilde_b, K_tilde_inv_b, B ) 

    f_mean, df_mean = mean_f_gradients_wrt_x_given_lambda_moments(
        f_params, lambda_m, lambda_var, dlambda, dvar_lambda)


    # return dlambda, dvar_lambda, dK_vec_x, dsecond_term, dK_x, K
    return f_mean, df_mean, dlambda, dvar_lambda#, dK_x, K


def test_dK_gradients_wrt_dx(model, x_i):

    ''' Test if out gradient of K(xtilde,x_i) w.r.t. x_i is correct numerically
    
        eps not too low to avoid catastrophic cancellation
        
        giving model as argument of gradients_wrt_dx '''


    theta = model.get('hyperparams_tuple')[0]
    m_b   = model['m_b']
    V_b   = model['V_b']

    C, mask, K_tilde_b, K_tilde_inv_b, B = get_final_K_vals(model)

    xtilde = model['xtilde']

    xtilde_m = xtilde[:,mask]
    x_i_m    = x_i[mask]

    print(f' mean masked xi: {x_i_m.mean().cpu().item():.4f}')

    maxdiffs_K = []

    for pxl in range(  x_i[mask].shape[0] ):
        
        # temporary kernel computation of shape ( ntilde, 1)

        eps = 1.e-2
    
        x_perturbed_1 = x_i_m.clone()
        x_perturbed_1[pxl] = x_perturbed_1[pxl] + eps

        x_perturbed_2 = x_i_m.clone()
        x_perturbed_2[pxl] = x_perturbed_2[pxl] - eps

        # _,_, _,_  = gradients_wrt_dx(model, x_i) # -> note x_i is unmasked. evarything is calculated in gradients_wrt_x
        # dK_x = dK_x.squeeze()

        K, dK_x  = acosker(theta, xtilde_m, x_i_m[None,:], C, get_dK_x=True) # (1, ntilde) and (ntilde, 1, nx)
        dK_x     = dK_x.squeeze()

        K_1, _   = acosker(theta, xtilde_m, x_perturbed_1[None,:], C, get_dK_x=True) # (1, ntilde)
        K_2, _   = acosker(theta, xtilde_m, x_perturbed_2[None,:], C, get_dK_x=True) # (1, ntilde)

        dK_x_num = (K_1-K_2).flatten() / (2*eps)

        maxdiff = torch.max(torch.abs(dK_x[:,pxl] - dK_x_num))

        maxdiffs_K.append(maxdiff.cpu().item())
        # print(f" Pixel {pxl}: max diff dK_x: {maxdiff.cpu().item():.4e}")
        if maxdiff > 1.e-1:
            raise ValueError("Gradient check failed")

        # dlambda, Sigma_dlambda = lambda_moments_gradient_wrt_x(theta, xtilde_m, x_i_m, C, m_b, V_b, K_tilde_inv_b, B ) #[:,-1]

    maxdiffs_K = np.array(maxdiffs_K)
    print(f" Gradient check (analytic vs numeric) max abs diff over {maxdiffs_K.shape[0]} pixels:")
    print(f" max difference: {np.max(maxdiffs_K):.4e}, mean: {np.mean(maxdiffs_K):.4e}, std: {np.std(maxdiffs_K):.4e}, eps used: {eps:.4e}")    

def lambda_moments_and_gradient_wrt_x(theta, xtilde, x, C, m_b, V_b, K_tilde_b, K_tilde_inv_b, B, lambda_m=None, lambda_var=None):

    '''Compute gradients of mean and covariance of lambda wrt z
    
    x can be one or more images (n, nx) (batch dimension first)
    '''

    # K_tilde_inv_b: (ntilde-B, ntilde_b) projection onto eigenspace of K_tilde_inv
    # xtilde : (ntilde, nx)
    # x : (nx) -> note its a single image
    # C : (nx_masked, nx_masked)

    # In MATLAB we got the pseudoinverse. No need here, saved in the model
    # K_tilde_inv = torch.linalg.pinv(K) ( shape (ntilde, ntilde) )
    

    # Derivative ok K(xtilde,x) wrt x ( note that we are computing K calling it with xtilde first )
    # K (ntilde, 1), dK_x (ntilde, 1, nx)
    K, dK_x = acosker(theta, xtilde, x, C, get_dK_x=True) # (ntilde, 1) and (n1,n2,nx)=(ntilde,1,nx)

    # New version to tet
    K_vec, dK_vec_x = acosker(theta, x, x2=None, C=C, get_dK_x=True, diag=True) # (1,1) and (n1,nx)=(1,nx)

    # Torch expects batch dimension as first 
    dK_x = dK_x.permute(1,2,0) #( ntilde, 1, nx ) -> (1,nx,ntilde)

    # Project dK_x onto eigenspace
    dK_x_b = dK_x @ B #(1, nx, ntilde_b)
    del dK_x

    # for dsigma we also need K. Usually we dont transpose it cause we call acosker as K(x,xtilde), but im used to a=KKtilde_inv
    K_b = K.T @ B #(1, ntilde_b)
    del K

    # KKtilde_inv_b 
    a  = K_b @ K_tilde_inv_b #(1, ntilde_b)
    da = dK_x_b @ K_tilde_inv_b # (1, nx, ntilde_b) 
    del dK_x_b
    del K_tilde_inv_b


    # Compute mean gradient (da' * m in MATLAB)
    dlambda = da @ m_b #(1, nx)
    
    dsecond_term = 2*(a@(V_b-K_tilde_b)@da.transpose(-1,-2)) #(1,1,nx)
    # squeeze teh ntilde dimension ( first is batch, last is pixels)
    dsecond_term = dsecond_term.squeeze(1) #(1,nx)

    dsigma_lambda = dK_vec_x + dsecond_term #(1,nx)

    # we return also the actual mean and variance for the given x

    if lambda_m is None or lambda_var is None:
        lambda_m, lambda_var = lambda_moments( x, K_tilde_b, a, K_vec, K_b, C, m_b, V_b, theta, kernfun=acosker)

    # return dlambda, dsigma_lambda, dK_vec_x, dsecond_term 
    return lambda_m, lambda_var, dlambda, dsigma_lambda


##################   Utility functions  ##################

# def nd_mean_noise_entropy(p_response, log_r2d_fact, sigma2, mu ):
#     # Computes the conditional noise entropy < H( r|f,x ) >_p(f|D) [eq 33 Paper PNAS]
#     # INPUTS:
#     # sigma2: 
#     #     - [nstar]
#     # mu:
#     #     - [nstar]
#     # p_response: set of probabilities p(r|x,D) for r that goes from 0 to a low number, set in utility(). Should go to infninty but the mean responses are low
#     #     - [r, nstar]
#     # log_r2d_fact: log(r!) argument of the sum, remember gamma(r+1) = r!
#     #     - [r, nstar]   
#     # In the nstar=1 case it was:  p_times_logr_sum = p_response@torch.lgamma(r+1) shape (1)
    
#     p_times_logr_sum = torch.sum( p_response*log_r2d_fact, dim=0 ) # shape (nstar)

#     # TODO: check this formula. In the paper
#     H_mean = -torch.exp(mu + 0.5*sigma2)*(mu + sigma2 - 1) + p_times_logr_sum

#     return H_mean

# def nd_lambda_r_mean(r, sigma2, mu):
#     # Computes the  argmax of the first row of the laplace approxmated logp(r|x,D) [eq 32 Paper PNAS] . Eq 33,34
#     # Its called lambda but its really representing log(f)
#     # this is NOT the lambda that we learn with the GP.
    
#     # r is a tensor of values from 0 to r_cutoff, its the max numver for the sum in eq 29 Paper PNAS
#     # sigma2: shape (nstar) 

#     rsigma2 = torch.outer(r,sigma2)                          # shape (r, nstar): every column is r*sigma2[i] (columns zero is rsigma2[:,0])
#     z       = torch.exp( rsigma2 + mu) * sigma2.unsqueeze(0) # shape (r, nstar): we add a first (1) dimension and multiply each sigma[i] by the corresponding column    

#     # Avoid overflowing in the exponential
#     sum_mask = z != torch.inf                                   # shape (r, nstar)
#     z        = torch.where(sum_mask, z, torch.tensor(0.))       # shape (r, nstar)
#     rsigma2  = torch.where(sum_mask, rsigma2, torch.tensor(0.)) # shape (r, nstar)



#     # TODO: I think its pretty important to avoid this copying to cpu and back to gpu. LambertW on the GPU would be great
#     '''
#     A little test gave
#     Elapsed time for the CPU copy: 0.000021
#     Elapsed time for the lambertw: 0.000082
#     Elapsed time for the GPU copy: 0.000041 
#     (results of this order)
#     So its not a bottleneck but it still doubles the time of the function
#     '''

#     z_cpu       = z.cpu()
#     lambertWcpu = scipy.special.lambertw( z=z_cpu, k=0, tol=1.e-8)
#     lambW       = rsigma2 + mu - torch.real(lambertWcpu.to(DEVICE)) # Take only the real part

#     # print(f' Kept {z.shape[0]} values for the summation in the Utility function')
#     return lambW, sum_mask


# ========================= CUSTOM LAMBERT W WITH STABLE GRADIENTS =========================

# class LambertWFunction(torch.autograd.Function):
#     """
#     Custom autograd function for Lambert W with numerically stable gradient.

#     Forward pass: Uses torchlambertw.special.lambertw (iterative solver)
#     Backward pass: Uses analytical formula dW/dz = W / (z * (1 + W))

#     The analytical gradient is more stable than differentiating through the
#     iterative solver, especially for edge cases where z→0 or W→-1.
#     """

#     @staticmethod
#     def forward(ctx, z, k=0):
#         """
#         Forward pass: Compute Lambert W function.

#         Args:
#             z: Input tensor
#             k: Branch (0 for principal, -1 for non-principal)

#         Returns:
#             w: Lambert W function of z
#         """
#         w = torch_lambertw(z, k=k)
#         ctx.save_for_backward(z, w)
#         ctx.k = k
#         return w

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass: Compute gradient using analytical formula.

#         The derivative of Lambert W is: dW/dz = W / (z * (1 + W))

#         This can be numerically unstable when:
#         - z ≈ 0 (division by zero)
#         - W ≈ -1 (division by zero, occurs at z = -1/e)

#         We handle these cases by clamping the denominator and filtering non-finite values.
#         """
#         z, w = ctx.saved_tensors

#         # Analytical gradient: dW/dz = W / (z * (1 + W))
#         denominator = z * (1.0 + w)

#         # Avoid division by zero - clamp denominator away from zero
#         eps = 1e-10
#         safe_denom = torch.where(
#             torch.abs(denominator) > eps,
#             denominator,
#             torch.sign(denominator) * eps + (denominator == 0).float() * eps
#         )

#         grad_z = grad_output * w / safe_denom

#         # Set gradient to zero for invalid values (inf, nan)
#         grad_z = torch.where(
#             torch.isfinite(grad_z),
#             grad_z,
#             torch.zeros_like(grad_z)
#         )

#         return grad_z, None  # None for k (not differentiable)


# def lambertw_stable(z, k=0):
#     """
#     Lambert W function with stable gradients for backpropagation.

#     Uses custom autograd implementation that computes gradients via the
#     analytical formula dW/dz = W/(z(1+W)) rather than differentiating
#     through the iterative solver.

#     Args:
#         z: Input tensor (real-valued)
#         k: Branch selection (0 for principal, -1 for non-principal)

#     Returns:
#         W(z): Lambert W function of z
#     """
#     return LambertWFunction.apply(z, k)


# def nd_lambda_r_mean_torch(r, sigma2, mu):
#     """
#     PyTorch-native version of nd_lambda_r_mean using torchlambertw with stable gradients.

#     Computes the argmax of the first row of the laplace approximated logp(r|x,D) [eq 32 Paper PNAS].
#     This version uses lambertw_stable() which provides numerically stable gradients via
#     analytical formula dW/dz = W/(z(1+W)) rather than differentiating through the iterative solver.

#     Args:
#         r: tensor of values from 0 to r_cutoff (max number for the sum in eq 29 Paper PNAS)
#         sigma2: tensor of shape (nstar)
#         mu: tensor of shape (nstar)

#     Returns:
#         lambW: tensor of shape (r, nstar) - the computed lambda values
#         sum_mask: tensor of shape (r, nstar) - mask indicating valid (non-infinite) values
#     """
#     # Ensure all inputs are on the same device
#     device = r.device
#     eps = 1e-10  # Small epsilon for numerical stability

#     rsigma2 = torch.outer(r, sigma2)                          # shape (r, nstar)

#     # Prevent overflow in exp() by clamping the exponent
#     # exp(88) ≈ 1.65e38 (close to float32 max)
#     # exp(709) ≈ 8.22e307 (close to float64 max)
#     # We use 85 to be safe and allow some margin
#     max_exp = 85.0
#     exponent = rsigma2 + mu
#     exponent_clamped = torch.clamp(exponent, max=max_exp)

#     # Compute z with clamped exponent to prevent inf
#     z = torch.exp(exponent_clamped) * sigma2.unsqueeze(0)     # shape (r, nstar)

#     # Track which values would have overflowed (for masking in caller)
#     # Note: We still compute with clamped values to maintain gradient flow
#     would_overflow = exponent > max_exp
#     sum_mask = ~would_overflow                                 # shape (r, nstar)

#     # For values that would overflow, set them to a small value
#     # This prevents numerical issues in Lambert W
#     z = torch.where(sum_mask, z, torch.full_like(z, eps))

#     # Also mask rsigma2 for consistency
#     rsigma2 = torch.where(sum_mask, rsigma2, torch.tensor(0., device=device))  # shape (r, nstar)

#     # Use stable Lambert W with custom gradient implementation
#     lambertW_result = lambertw_stable(z, k=0)
#     lambW = rsigma2 + mu - lambertW_result.real

#     return lambW, sum_mask

def nd_p_r_given_xD(r, sigma2, mu):
    # Computes p( r|x,D ) [eq 31 Paper PNAS]. It's the first term of I(r;f|x,D) [eq:27] 

    # Calculating lambda mean for different values of r
    # Note that the sum over r was already reduced by a r_cutoff set in the utility function
    # If despite this cutoff, the exponential still goes to infinity for certain r values, they are removed from the sum
    # lambda_mean, sum_mask = nd_lambda_r_mean(r, sigma2, mu)          # shape (r, nstar) , (r, nstar)
    
    lambda_mean, sum_mask = nd_lambda_r_mean_torch(r, sigma2, mu)    # shape (r, nstar) , (r, nstar)

    # Mask lambda_mean to prevent overflow values from affecting probability calculations
    lambda_mean = torch.where(sum_mask, lambda_mean, torch.tensor(0., device=lambda_mean.device))

    ex_lambda_mean = torch.exp(lambda_mean)                          # shape (r, nstar)


    # To calculate log(r!) we use the fact that the Gamma function of an integer is the factorial of that integer -1
    # G(r+1) = r! and torch. provides exactly the logarithm of this
    # torch.lgamma(r+1) = log( G(r+1) ) = log( r! )
    log_r_fact = torch.lgamma(r+1)

    # r needs to be a 2D tensor, and so dows log_r_fact
    r = r.unsqueeze(1)                                                # shape (r) -> shape (r, 1)
    r = r.repeat(1, sigma2.shape[0])                                  # shape (r, 1) -> shape (r, nstar)
    log_r_fact = log_r_fact.unsqueeze(1)                              # shape (r) -> shape (r, 1) ( equivalent to [:,None])
    log_r_fact = log_r_fact.repeat(1, sigma2.shape[0])                # shape (r, 1) -> shape (r, nstar)

    log_r_fact = torch.where(sum_mask, log_r_fact, torch.tensor(0., device=log_r_fact.device))  # shape (r, nstar)
    r          = torch.where(sum_mask, r, torch.tensor(0., device=r.device))           # shape (r, nstar)

    # We woulnd't need to unsqueeze / add an empty first dimension to sigma2. It's just to show that we are dividing each column of lambda_mean by the corresponding sigma2
    log_p = lambda_mean*r - ex_lambda_mean - ((lambda_mean-mu)**2)/(2*sigma2.unsqueeze(0)) - 0.5*safe_log( ex_lambda_mean*sigma2 + 1) - log_r_fact # TODO: is this factorial too slow?

    return torch.exp(log_p), log_p, r, log_r_fact


def nd_utility( mu, sigma2, r_masked):
    # Computes the utility function [eq 27 Paper PNAS]

    # mu, sigma2 are mean and variance of log(f) with f the firing rate. 
    # They are not the mean and variance of the lambda that we learn with the GP.
    # And they are not even the log of the mean and variance of firing rate. ( look at (37) notes)
    # sigma2: shape (nstar)
    # mu:     shape (nstar)

    # Generates the tensor of the probability of the responses p(r|x,D)
    # r_cutoff: for r that goes from 0 to a low number set by r_cutoff, 
    # r_tensor_masked: this will as well be reduced if the exponential in the lambda_r_mean function goes to infinity

    if sigma2.ndim == 0:
        sigma2 = sigma2[None]
        mu     = mu[None]
    # Returns the the Laplace approximation of p(r|x,D) and the masked r tensor to use in the sum in mean_noise entropy. 
    p_response, log_p_response, r_masked_2d, log_r2d_fact_masked = nd_p_r_given_xD( r=r_masked, sigma2=sigma2, mu=mu, )  # shape (r, nstar), shape (r, nstar), shape (r, nstar)
    
    # Response entropy H(r|x,D) [eq 28 Paper PNAS]
    H_r_xD = -torch.sum( p_response*log_p_response, dim=0 )  # shape (1, nstar)

    E_H_r_f = nd_mean_noise_entropy( p_response, log_r2d_fact_masked, sigma2, mu,  )
    U = H_r_xD - E_H_r_f

    # print(f'Sigma2 = {sigma2.item():.4f}, H(r|x,D) = {a.item():.4f}, <H(r|f,x)> = {b.item():.4f}, U = {U.item():.4f}')
    return U

def mean_noise_entropy(p_response, r, sigma2, mu ):
    # Computes the conditional noise entropy < H( r|f,x ) >_p(f|D) [eq 33 Paper PNAS]
    # INPUTS:
    # p_response: set of probabilities p(r|x,D) for r that goes from 0 to a low number, set in utility(). Should go to infninty but the mean responses are low
 
    # argument of the sum, remember gamma(r+1) = r!
    p_times_logr_sum = p_response@torch.lgamma(r+1)

    # TODO: check this formula. In the paper
    H_mean = -torch.exp(mu + 0.5*sigma2)*(mu + sigma2 - 1) + p_times_logr_sum

    return H_mean

def lambda_r_mean(r, sigma2, mu):
    # Computes the  argmax of the first row of the laplace approxmated logp(r|x,D) [eq 32 Paper PNAS] . Eq 33,34
    # Its called lambda but its really representing log(f)
    # this is NOT the lambda that we learn with the GP.
    
    # r is a tensor of values from 0 to r_cutoff, its the max numver for the sum in eq 29 Paper PNAS

    rsigma2 = r*sigma2                           # shape (r) -> shape (r, nstar)
    z       = sigma2 * torch.exp( rsigma2 + mu)  # shape (r) -> shape (r, nstar)

    # Avoid overflowing in the exponential
    sum_mask = z != torch.inf                    # shape (r) -> shape (r, nstar)
    z        = z[sum_mask]                       # shape (r_reduces) -> shape (r_reduced, nstar)
    rsigma2  = rsigma2[sum_mask]                 # shape (r_reduced) -> shape (r_reduced, nstar)

    # TODO: I think its pretty important to avoid this copying to cpu and back to gpu. LambertW on the GPU would be great
    '''
    A little test gave
    Elapsed time for the CPU copy: 0.000021
    Elapsed time for the lambertw: 0.000082
    Elapsed time for the GPU copy: 0.000041 
    (results of this order)
    So its not a bottleneck but it still doubles the time of the function
    '''

    z_cpu = z.cpu()
    lambertWcpu = scipy.special.lambertw( z=z_cpu, k=0, tol=1.e-8)
    lamb    = rsigma2 + mu - torch.real(lambertWcpu.to(DEVICE)) # Take only the real part

    # print(f' Kept {z.shape[0]} values for the summation in the Utility function')
    return lamb, sum_mask 

def p_r_given_xD(r, sigma2, mu):
    # Computes p( r|x,D ) [eq 31 Paper PNAS]. It's the first term of I(r;f|x,D) [eq:27] 

    # Calculating lambda mean for different values of r
    # Note that the sum over r was already reduced by a r_cutoff set in the utility function
    # If despite this cutoff, the exponential still goes to infinity for certain r values, they are removed from the sum
    lambda_mean, sum_mask = lambda_r_mean(r, sigma2, mu)
    ex_lambda_mean = torch.exp(lambda_mean)

    r = r[sum_mask]  

    # To calculate log(r!) we use the fact that the Gamma function of an integer is the factorial of that integer -1
    # G(r+1) = r! and torch. provides exactly the logarithm of this
    # torch.lgamma(r+1) = log( G(r+1) ) = log( r! )
    log_p = lambda_mean*r - ex_lambda_mean - ((lambda_mean-mu)**2)/(2*sigma2) - 0.5*safe_log( sigma2*ex_lambda_mean + 1) - torch.lgamma(r+1) # TODO: is this factorial too slow?

    # print(f'(- 0.5*safe_log( sigma2*ex_lambda_mean + 1))[:4] {(- 0.5*safe_log( sigma2*ex_lambda_mean + 1))[:4]}')


    # print(f'log_p[0:4] = {log_p[0:4].cpu().detach().numpy()}')

    return torch.exp(log_p), log_p, r

@torch.no_grad()
def utility( sigma2, mu, r_masked):
    # Computes the utility function [eq 27 Paper PNAS]

    # mu, sigma2 are mean and variance of log(f) with f the firing rate. They are not the mean and variance of the lambda that we learn with the GP.

    # Generates the tensor of the probability of the responses p(r|x,D)
    # r_cutoff: for r that goes from 0 to a low number set by r_cutoff, 
    # r_tensor_masked: this will as well be reduced if the exponential in the lambda_r_mean function goes to infinity

    # Returns the the Laplace approximation of p(r|x,D) and the masked r tensor to use in the sum in mean_noise entropy. 
    p_response, log_p_response, r_masked2 = p_r_given_xD( r=r_masked, sigma2=sigma2, mu=mu, ) 

    # print(f'p_response0:3 = {p_response[0:3].cpu().detach().numpy()}')
    # print(f'log_p_response0:3 = {log_p_response[0:3].cpu().detach().numpy()}')
    # print(f'r_masked2.shape = {r_masked2.shape}')

    H_r_xD = -torch.sum( p_response*log_p_response ) # Response entropy H(r|x,D) [eq 28 Paper PNAS]
    E_H_r_f = mean_noise_entropy( p_response, r_masked2, sigma2, mu,  )
    U = H_r_xD - E_H_r_f

    # print(f'Sigma2 = {sigma2.item():.4f}, H(r|x,D) = {a.item():.4f}, <H(r|f,x)> = {b.item():.4f}, U = {U.item():.4f}')
    return U

def get_utility(xstar, xtilde, C, mask, theta, m, V, K_tilde, K_tilde_inv, kernfun):
            
    if K_tilde_inv is None:
        eye = torch.eye(K_tilde.shape[0], device=DEVICE, dtype=TORCH_DTYPE)
        K_tilde_inv = torch.linalg.solve(K_tilde, eye)

    xstar = xstar.unsqueeze(0)

    # Inference on new input(s)
    mu_star, sigma2_star = lambda_moments_star(xstar[:,mask], xtilde[:,mask], C, theta, K_tilde_inv , m, V, kernfun=kernfun)

    return utility( sigma2=sigma2_star, mu=mu_star )

def batch_utility_w_grad(
    model_active,
    imgs_train,
    remaining_active_idx,
    max_r_cap=100,
    test_rms_constraint=False,
    test_L2norm_constraint=False,
    n_steps=5,
):
    """
    Compute Utility of all images, best image and gradient of utility w.r.t. the best image

    Parameters:
    -----------
    model_active : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    remaining_active_idx : torch.Tensor
        Indices of candidate images
    max_r_cap : int
        Maximum spike count for utility computation
    test_gradient_ascent : bool
        If True, perform 5 gradient ascent steps and return trajectory data

    Returns:
    --------
    u2d : torch.Tensor
        Utility values for all candidate images
    x_idx_best : torch.Tensor
        Index of best image in original dataset
    dU_best : torch.Tensor
        Gradient of utility w.r.t. best image
    trajectory_data : dict or None
        If test_gradient_ascent=True, returns dict with:
            - 'initial_img': initial image tensor
            - 'traj': list of image tensors at each step
            - 'util_hist': list of utility values
            - 'grad_norms': list of gradient norms
        Otherwise returns None
    """

    r_masked = torch.arange(0, max_r_cap, dtype=TORCH_DTYPE, device=DEVICE)

    X_remaining = imgs_train[remaining_active_idx]

    kernfun     = model_active.kernfun
    xtilde      = model_active.xtilde

    theta = model_active.theta
    m_b   = model_active.m_b
    V_b   = model_active.V_b

    A = torch.exp(model_active.f_params['logA'])
    lambda0 = model_active.f_params['lambda0']

    C, mask, K_tilde_b, K_tilde_inv_b, B = get_final_K_vals(model_active)

    Kvec = kernfun(theta, X_remaining[:,mask], x2=None, C=C, dC=None, diag=True)

    K   = kernfun(theta, X_remaining[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)
    K_b = K @ B
    KKtilde_inv_b = K_b @ K_tilde_inv_b

    # === Utility of all remaining images ===
    lambda_m_batch, lambda_var_batch = lambda_moments( 
        X_remaining[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta)  

    # NOTE: These are mean and variance of log_f = A*lambda + lambda0
    #   - Not the log of mean and variance of f
    #   - Not the mean and variance of lambda
    logf_mean_batch = A*lambda_m_batch + lambda0
    logf_var_batch  = A**2 * lambda_var_batch

    # === Utility of all remaining images ===
    # Keep gradient tracking enabled (minimal overhead) for potential gradient computation later
    with torch.enable_grad():
        logf_mean_batch_grad = logf_mean_batch.clone().requires_grad_(True)
        logf_var_batch_grad  = logf_var_batch.clone().requires_grad_(True)

        u2d = nd_utility(logf_mean_batch_grad, logf_var_batch_grad, r_masked)
        i_best = u2d.argmax()  # Index of the best image in the utility vector
        x_idx_best = remaining_active_idx[i_best]  # Index of the best image in the original dataset

    # Initialize return variables (will be populated if test_rms_constraint=True)
    dU_dx_init = None
    trajectory_data = None

    if test_rms_constraint:
        # L-BFGS optimization with RMS constraint (mean + standard deviation)
        # Key idea: Optimize unconstrained θ, transform to x_masked = μ_target + σ_target * (θ - mean(θ)) / std(θ)
        # Constraints automatically satisfied by construction (no post-step projection needed!)
        # This is more restrictive than L2 norm (2 constraints vs 1)
        # Matches Walker et al. 2019 Nature Neuroscience "Inception loops" paper approach
        print("\n=== Start L-BFGS Optimization (RMS reparameterization) ===")

        # === 1. Define reparameterization variables (single definition) ===
        assert imgs_train[x_idx_best].mean() == X_remaining[i_best].mean(), "Image means do not match!"

        initial_img = imgs_train[x_idx_best].clone()
        initial_unmasked = initial_img[~mask].clone()
        initial_masked = initial_img[mask].clone()

        # Target statistics from this specific image (not global across images)
        μ_target = initial_masked.mean().item()
        σ_target = initial_masked.std().item()
        # print(f"  Target mean (luminance): {μ_target:.6f}")
        # print(f"  Target std (RMS contrast): {σ_target:.6f}")

        # REPARAMETERIZATION: Optimization variable is θ (unconstrained)
        # We transform: x_masked = μ_target + σ_target * (θ - mean(θ)) / std(θ)
        # This ensures mean(x_masked) = μ_target and std(x_masked) = σ_target automatically!
        θ = initial_masked.clone().detach().requires_grad_(True)

        # Numerical safety: prevent std(θ) from collapsing
        θ_std_min = 0.01
        # print(f"  θ std will be clamped to ≥ {θ_std_min}")

        # === 2. Compute hybrid gradient (∂U/∂x via custom GP + torch autograd) ===
        with torch.enable_grad():
            # Compute the gradient of lambda moments wrt chosen x
            lambda_m, lambda_var, dlambda, dvar_lambda = lambda_moments_and_gradient_wrt_x(
                theta, xtilde[:,mask], X_remaining[i_best,mask][None,:], C, m_b, V_b, K_tilde_b, K_tilde_inv_b, B,
                lambda_m=lambda_m_batch[i_best], lambda_var=lambda_var_batch[i_best])

            # Get ∂U/∂logf_mean and ∂U/∂logf_var using torch autograd
            grad_U_mean_full, grad_U_var_full = torch.autograd.grad(
                u2d[i_best],
                [logf_mean_batch_grad, logf_var_batch_grad],
                retain_graph=False
            )
            grad_U_mean = grad_U_mean_full[i_best]
            grad_U_var = grad_U_var_full[i_best]

            dlogf_mean = A * dlambda
            dlogf_var = A**2 * dvar_lambda

            # Chain rule: ∂U/∂x = (∂U/∂logf_mean)(∂logf_mean/∂x) + (∂U/∂logf_var)(∂logf_var/∂x)
            dU_dx_init = grad_U_mean * dlogf_mean + grad_U_var * dlogf_var  # Shape (1, nx)

        print(f'\nU: {u2d[i_best].item():<8.8f}, ||∂U/∂x|| = {torch.linalg.vector_norm(dU_dx_init).item():<6.4e}, dU_dx_init mean: {torch.mean(dU_dx_init).item():<8.7f}')

        # === 3. Compute torch gradients for tracking/comparison (∂U/∂θ and ∂U/∂x via torch) ===
        with torch.enable_grad():
            # Transform θ → x_masked for initial utility
            θ_mean_init = θ.mean()
            θ_std_init = θ.std()
            x_masked_init = μ_target + σ_target * (θ - θ_mean_init) / θ_std_init

            x_full_init = initial_img.clone()
            x_full_init[mask] = x_masked_init  # Keep gradient flow through θ

            U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)

            dU_dtheta_init, dU_dx_init_torch = torch.autograd.grad(
                U_init,
                [θ, x_masked_init])

        # DIAGNOSTIC: Print initial Utility and gradient norms
        print(f"U: {U_init.item():<8.8f}, ||∂U/∂θ|| = {torch.linalg.vector_norm(dU_dtheta_init).item():<6.4e}, dU_dtheta_init mean: {torch.mean(dU_dtheta_init).item():<8.7f}")
        print(f"||∂U/∂x|| (hybrid) = {torch.linalg.vector_norm(dU_dx_init).item():<6.4e}, ||∂U/∂x|| (torch) = {torch.linalg.vector_norm(dU_dx_init_torch).item():<6.4e}")

        # === 4. Initialize tracking lists ===
        x_traj = [initial_img.clone()]
        util_hist = [u2d[i_best].item()]
        grad_norms = [torch.linalg.vector_norm(dU_dtheta_init).item()]  # ||∂U/∂θ||
        grad_hist = [dU_dtheta_init.detach().clone()]  # Store ∂U/∂θ

        # === 5. Setup L-BFGS optimizer ===
        history_size = 10

        # Track NaN gradient occurrences
        nan_grad_count = [0]
        max_nan_grad_allowed = 50

        # Create optimizer - Optimizes θ (not x_masked directly)
        optimizer = torch.optim.LBFGS(
            [θ],
            lr=1.e4,
            max_iter=500,
            history_size=history_size,
            line_search_fn='strong_wolfe',  # Changed from 'strong_wolfe' to 'armijo'
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            verbose=True,  # Enable detailed line search debugging
            # Armijo line search parameters - optimized for image optimization
            armijo_c1=1e-3,              # Less strict than default 1e-4 (accepts 0.1% decrease)
            armijo_rho=0.7,              # Gentler backtracking than default 0.5 (30% reduction per iter)
            armijo_min_step_size=1e-5,   # Prevent microscopic steps in pixel space
        )

        # === 6. Run L-BFGS optimization loop ===
        # DIAGNOSTIC: Track closure calls per step
        closure_call_count = [0]

        for step in range(n_steps):
            # print(f"\n--- optimizer.step {step} ---")
            def closure():
                """
                Closure for L-BFGS optimizer with RMS constraint via reparameterization.

                KEY: Optimize θ, transform to x_masked = μ_target + σ_target * (θ - mean(θ)) / std(θ)
                Constraints automatically satisfied by construction! Gradients flow: U → x_masked → θ
                More restrictive than L2 norm: fixes both brightness and contrast.
                """
                # DIAGNOSTIC: Count closure calls
                closure_call_count[0] += 1

                # NUMERICAL SAFETY: Clamp std(θ) to prevent division by zero
                with torch.no_grad():
                    θ_std_current = θ.std()
                    if θ_std_current < θ_std_min:
                        print(f"    [Warning] C-call {closure_call_count[0]}] Clamping θ std from {θ_std_current.item():.6f} to {θ_std_min}")
                        θ.data = (θ - θ.mean()) * (θ_std_min / θ_std_current) + θ.mean()

                optimizer.zero_grad()

                # REPARAMETERIZATION: Transform θ → x_masked (constraints satisfied by construction!)
                # Gradients will flow through this transformation: ∂U/∂θ = ∂U/∂x_masked * ∂x_masked/∂θ
                with torch.enable_grad():
                    θ_mean = θ.mean()
                    θ_std = θ.std()
                    x_masked_constrained = μ_target + σ_target * (θ - θ_mean) / θ_std  # mean=μ_target, std=σ_target automatically!

                    # Reconstruct full image
                    x_full = torch.empty_like(initial_img)
                    x_full[~mask] = initial_unmasked
                    x_full[mask] = x_masked_constrained  # Gradients flow: x_full → x_masked → θ

                    # Compute utility
                    U = compute_utility_single_image(model_active, x_full, max_r_cap)

                # Reject invalid utility values
                if torch.isnan(U) or torch.isinf(U):
                    print(f"    [Closure] U is {'NaN' if torch.isnan(U) else 'Inf'}, rejecting step")
                    return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

                # L-BFGS minimizes, so return negative utility
                loss = -U


                loss.backward()  # Computes ∂U/∂θ via chain rule through normalization

                # Check if backward pass produced invalid gradients in θ
                if θ.grad is not None:
                    # DIAGNOSTIC: Print loss value and utility gradient for first few closure calls
                    # if closure_call_count[0] <= 50:
                        # print(f"      [Closure call {closure_call_count[0]}] U={U.item():.8f}, dU/dθ norm={torch.linalg.vector_norm(θ.grad).item():.8f}")
                    if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                        nan_grad_count[0] += 1

                        if nan_grad_count[0] <= max_nan_grad_allowed:
                            print(f"    [Closure] Warning: Gradient is NaN/Inf (occurrence {nan_grad_count[0]}/{max_nan_grad_allowed}), clearing and rejecting step")
                            optimizer.zero_grad()
                            return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                        else:
                            raise RuntimeError(
                                f"Gradient contains NaN or Inf after backward pass (occurred {nan_grad_count[0]} times)! "
                                f"U={U.item():.6f}, loss={loss.item():.6f}. "
                                f"This indicates systematic numerical instability."
                            )
                else:
                    raise Exception(f"    [Closure] Warning: Gradient is None after backward pass!")
                return loss

            # DIAGNOSTIC: Store θ before step for comparison
            θ_before = θ.detach().clone()

            # Perform one L-BFGS step (includes line search)
            print(f"\n=== L-BFGS Step {step} ===")
            optimizer.step(closure)
            # DIAGNOSTIC: Print accepted step size t from line search
            state = optimizer.state[optimizer._params[0]]
            # print(f"    Accepted line-search step t = {state['t']:.3e}")
            # NO POST-STEP PROJECTION NEEDED!
            # Constraints are satisfied by construction via reparameterization

            # DIAGNOSTIC: Compute θ changes after step
            with torch.no_grad():
                θ_diff = θ - θ_before
                θ_max_change = torch.max(torch.abs(θ_diff)).item()
                θ_mean_change = torch.mean(torch.abs(θ_diff)).item()
                θ_std_after = θ.std().item()

            # Reconstruct x_masked from θ for tracking
            with torch.no_grad():
                θ_mean_current = θ.mean()
                θ_std_current = θ.std()
                x_masked_current = μ_target + σ_target * (θ - θ_mean_current) / θ_std_current

                x_full_current = torch.empty_like(initial_img)
                x_full_current[~mask] = initial_unmasked
                x_full_current[mask] = x_masked_current

            # Compute utility and gradient for tracking (∂U/∂θ, not ∂U/∂x)
            with torch.enable_grad():
                # Recompute utility with gradient tracking for θ
                θ_mean_t = θ.mean()
                θ_std_t = θ.std()
                x_masked_t = μ_target + σ_target * (θ - θ_mean_t) / θ_std_t

                x_full_t = torch.empty_like(initial_img)
                x_full_t[~mask] = initial_unmasked
                x_full_t[mask] = x_masked_t

                U_t = compute_utility_single_image(model_active, x_full_t, max_r_cap)
                dU_dtheta_t = torch.autograd.grad(U_t, θ, retain_graph=False)[0]

            # Verify constraints are satisfied (should be exact by reparameterization)
            with torch.no_grad():
                μ_actual = x_masked_current.mean().item()
                σ_actual = x_masked_current.std().item()
                μ_error = abs(μ_actual - μ_target) / abs(μ_target) if abs(μ_target) > 1e-8 else 0.0
                σ_error = abs(σ_actual - σ_target) / σ_target if σ_target > 1e-8 else 0.0

            # print(f"--- L-BFGS End step {step}: U={U_t.item():.6f}, ||∂U/∂θ||={torch.linalg.vector_norm(dU_dtheta_t).item():.6f}---")

            # DIAGNOSTIC: Print line search and θ change statistics
            # print(f"    Closure called {closure_call_count[0]} times ")
            # print(f"    θ change: max={θ_max_change:.6e}, mean={θ_mean_change:.6e}")
            # print(f"    θ_std: {θ_std_after:.6f}")
            # DIAGNOSTIC: Print constraint verification
            if μ_error > 1e-4 or σ_error > 1e-4:
                print(f"   [Warning] Constraint errors: μ={μ_actual:.6f} (target: {μ_target:.6f}, error: {μ_error*100:.4f}%)")
                print(f"                                σ={σ_actual:.6f} (target: {σ_target:.6f}, error: {σ_error*100:.4f}%)")

            # Store trajectory data
            util_hist.append(U_t.item())
            grad_norms.append(torch.linalg.vector_norm(dU_dtheta_t).item())  # ||∂U/∂θ||
            x_traj.append(x_full_current.clone())
            grad_hist.append(dU_dtheta_t.detach().clone())  # Full ∂U/∂θ vector

            # DIAGNOSTIC: Warn if minimal progress detected
            # if θ_max_change < 1e-6:
                # print(f"    [Warning]: θ changed less than 1e-6 everywhere.")

            # DIAGNOSTIC: Reset closure counter for next step
            closure_call_count[0] = 0

        # === 7. Store trajectory data ===
        trajectory_data = {
            'initial_img': initial_img,
            'x_traj': x_traj,
            'util_hist': util_hist,
            'grad_norms': grad_norms,
            'grad_hist': grad_hist
        }

        # === 8. Print final diagnostics ===
        # Verify monotonic utility increase
        monotonic = all(util_hist[i] >= util_hist[i-1] for i in range(1, len(util_hist)))
        utility_increase = util_hist[-1] - util_hist[0]
        relative_increase = (utility_increase / util_hist[0] * 100) if util_hist[0] > 1e-8 else 0.0

        print(f"=== L-BFGS Complete (RMS Constraint via Reparameterization) ===")
        print(f"  Initial U: {util_hist[0]:.6f}")
        print(f"  Final U:   {util_hist[-1]:.6f}")
        print(f"  Change:    +{utility_increase:.6f} ({relative_increase:+.2f}%)")
        print(f"  Monotonic: {'✓ Yes' if monotonic else '✗ No (WARNING!)'}")
        # print(f"  Constraints: μ = {μ_target:.6f}, σ = {σ_target:.6f} (satisfied by construction)\n")

        print(f"  ||∂U/∂θ||_2   = {torch.linalg.vector_norm(dU_dtheta_init).item():.6e}")
        print(f"  ||∂U/∂θ||_∞   = {dU_dtheta_init.abs().max().item():.6e}")
        # print(f"  n_masked_pixels = {dU_dtheta_init.numel()}")

    elif test_L2norm_constraint:
        # L-BFGS optimization with L2 norm constraint
        # Key idea: Optimize unconstrained θ, transform to x_masked = (θ / ||θ||) * sqrt(n_pixels_in_mask)
        # Constraint automatically satisfied by construction (no post-step projection needed!)
        # This is less restrictive than RMS constraint (1 constraint vs 2)
        # All images normalized to same L2 norm (no mean/std preservation)
        print("\n=== Start L-BFGS Optimization (L2 norm constraint) ===")

        # === 1. Define reparameterization variables (single definition) ===
        assert imgs_train[x_idx_best].mean() == X_remaining[i_best].mean(), "Image means do not match!"

        initial_img = imgs_train[x_idx_best].clone()
        initial_unmasked = initial_img[~mask].clone()
        initial_masked = initial_img[mask].clone()

        # Target L2 norm: FIXED for all images (global constraint)
        n_pixels_in_mask = initial_masked.numel()
        L2_target = torch.sqrt(torch.tensor(n_pixels_in_mask, dtype=TORCH_DTYPE, device=DEVICE)).item()
        print(f"  Target L2 norm: {L2_target:.6f} (= sqrt({n_pixels_in_mask}))")

        # REPARAMETERIZATION: Optimization variable is θ (unconstrained)
        # We transform: x = (θ / ||θ||) * L2_target
        # This ensures ||x|| = L2_target automatically!
        θ = initial_masked.clone().detach().requires_grad_(True)

        # Numerical safety: prevent ||θ|| from collapsing to zero
        θ_norm_min = 0.01

        # Compute x₀: the STARTING POINT for optimization
        # KEY DIFFERENCE from RMS: x₀ ≠ initial_masked for L2!
        # - initial_masked = original pixels from pool (arbitrary ||x||)
        # - x₀ = normalized version with ||x₀|| = sqrt(n)
        # This is what the L2 reparameterization produces at initialization
        with torch.no_grad():
            initial_norm = torch.linalg.norm(initial_masked)
            x0_masked = (initial_masked / initial_norm) * L2_target  # Starting point x₀

        print(f"  ||initial_masked|| (original): {initial_norm:.6f}")
        print(f"  ||x₀|| (starting point): {torch.linalg.norm(x0_masked).item():.6f} = {L2_target:.6f}")

        # === 2. Compute hybrid gradient AT x₀ (∂U/∂x via custom GP + torch autograd) ===
        # IMPORTANT: Both hybrid and torch gradients must be evaluated at x₀ (not original!)

        with torch.enable_grad():
            # Compute GP predictions at x₀ (starting point)
            lambda_m_at_x0, lambda_var_at_x0, dlambda_dx, dvar_lambda_dx = lambda_moments_and_gradient_wrt_x(
                theta, xtilde[:,mask], x0_masked[None,:], C, m_b, V_b, K_tilde_b, K_tilde_inv_b, B,
                lambda_m=None, lambda_var=None)  # Don't use cached values from original image

            # Transform to log-firing rate at x₀
            logf_mean_at_x0 = (A * lambda_m_at_x0 + lambda0).requires_grad_(True)
            logf_var_at_x0 = (A**2 * lambda_var_at_x0).requires_grad_(True)

            # Compute utility at x₀
            U_at_x0 = nd_utility(logf_mean_at_x0, logf_var_at_x0, r_masked)

            # Get ∂U/∂logf_mean and ∂U/∂logf_var at x₀
            grad_U_mean, grad_U_var = torch.autograd.grad(
                U_at_x0,
                [logf_mean_at_x0, logf_var_at_x0],
                retain_graph=False
            )

            dlogf_mean_dx = A * dlambda_dx
            dlogf_var_dx = A**2 * dvar_lambda_dx

            # Chain rule: ∂U/∂x = (∂U/∂logf_mean)(∂logf_mean/∂x) + (∂U/∂logf_var)(∂logf_var/∂x)
            dU_dx_init = grad_U_mean * dlogf_mean_dx + grad_U_var * dlogf_var_dx  # Shape (1, nx)

        print(f'\nU (at x₀): {U_at_x0.item():<8.8f}, ||∂U/∂x|| = {torch.linalg.vector_norm(dU_dx_init).item():<6.4e}, dU_dx_init mean: {torch.mean(dU_dx_init).item():<8.7f}')

        # === 3. Compute torch gradients AT x₀ for tracking/comparison (∂U/∂θ and ∂U/∂x via torch) ===
        with torch.enable_grad():
            # Apply L2 reparameterization: θ → x₀
            # At initialization, θ = initial_masked, so this produces x₀
            θ_norm_init = torch.linalg.norm(θ)
            x0_from_theta = (θ / θ_norm_init) * L2_target  # This equals x₀!

            assert torch.allclose(x0_from_theta, x0_masked, atol=1e-6), "x₀ from θ does not match precomputed x₀!"


            # Reconstruct full image with x₀ in masked region
            x_full_at_x0 = initial_img.clone()
            x_full_at_x0[mask] = x0_from_theta  # Keep gradient flow through θ

            # Compute utility at x₀ via torch
            U_torch_at_x0 = compute_utility_single_image(model_active, x_full_at_x0, max_r_cap)

            # Get gradients: ∂U/∂θ and ∂U/∂x both at x₀
            dU_dtheta_init, dU_dx_torch_at_x0 = torch.autograd.grad(
                U_torch_at_x0,
                [θ, x0_from_theta])

        # DIAGNOSTIC: Print initial utility and gradient norms
        print(f"U (torch at x₀): {U_torch_at_x0.item():<8.8f}, ||∂U/∂θ|| = {torch.linalg.vector_norm(dU_dtheta_init).item():<6.4e}, dU_dtheta_init mean: {torch.mean(dU_dtheta_init).item():<8.7f}")
        print(f"||∂U/∂x|| (hybrid at x₀) = {torch.linalg.vector_norm(dU_dx_init).item():<6.4e}, ||∂U/∂x|| (torch at x₀) = {torch.linalg.vector_norm(dU_dx_torch_at_x0).item():<6.4e}")
        print(f"Utilities match: {abs(U_at_x0.item() - U_torch_at_x0.item()) < 1e-6}")

        # === 4. Initialize tracking lists ===
        # NOTE: We track the full image (11664 pixels) but optimization only affects masked region
        x_traj = [initial_img.clone()]  # This still contains ORIGINAL pixels in masked region

        # IMPORTANT: Track utility at x₀ (starting point), not at original image
        util_hist = [U_at_x0.item()]  # Utility at x₀ (normalized version)
        grad_norms = [torch.linalg.vector_norm(dU_dtheta_init).item()]  # ||∂U/∂θ|| at x₀
        grad_hist = [dU_dtheta_init.detach().clone()]  # ∂U/∂θ at x₀

        # === 5. Setup L-BFGS optimizer ===
        history_size = 10

        # Track NaN gradient occurrences
        nan_grad_count = [0]
        max_nan_grad_allowed = 50

        # Create optimizer - Optimizes θ (not x_masked directly)
        optimizer = torch.optim.LBFGS(
            [θ],
            lr=1.0,
            max_iter=20,
            history_size=history_size,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            verbose=False,
        )

        # === 6. Run L-BFGS optimization loop ===
        # DIAGNOSTIC: Track closure calls per step
        closure_call_count = [0]

        for step in range(n_steps):
            # print(f"\n--- optimizer.step {step} ---")
            def closure():
                """
                Closure for L-BFGS optimizer with L2 norm constraint via reparameterization.

                KEY: Optimize θ, transform to x_masked = (θ / ||θ||) * L2_target
                Constraint automatically satisfied by construction! Gradients flow: U → x_masked → θ
                Less restrictive than RMS constraint: fixes only L2 norm (not mean/std).
                """
                # DIAGNOSTIC: Count closure calls
                closure_call_count[0] += 1

                # NUMERICAL SAFETY: Clamp ||θ|| to prevent division by zero
                with torch.no_grad():
                    θ_norm_current = torch.linalg.norm(θ)
                    if θ_norm_current < θ_norm_min:
                        print(f"    [Warning] C-call {closure_call_count[0]}] Clamping ||θ|| from {θ_norm_current.item():.6f} to {θ_norm_min}")
                        θ.data = (θ / θ_norm_current) * θ_norm_min

                optimizer.zero_grad()

                # REPARAMETERIZATION: Transform θ → x_masked (constraint satisfied by construction!)
                # Gradients will flow through this transformation: ∂U/∂θ = ∂U/∂x_masked * ∂x_masked/∂θ
                with torch.enable_grad():
                    θ_norm = torch.linalg.norm(θ)
                    x_masked_constrained = (θ / θ_norm) * L2_target  # ||x|| = L2_target automatically!

                    # Reconstruct full image
                    x_full = torch.empty_like(initial_img)
                    x_full[~mask] = initial_unmasked
                    x_full[mask] = x_masked_constrained  # Gradients flow: x_full → x_masked → θ

                    # Compute utility
                    U = compute_utility_single_image(model_active, x_full, max_r_cap)

                # Reject invalid utility values
                if torch.isnan(U) or torch.isinf(U):
                    print(f"    [Closure] U is {'NaN' if torch.isnan(U) else 'Inf'}, rejecting step")
                    return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

                # L-BFGS minimizes, so return negative utility
                loss = -U

                loss.backward()  # Computes ∂U/∂θ via chain rule through normalization

                # Check if backward pass produced invalid gradients in θ
                if θ.grad is not None:
                    if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                        nan_grad_count[0] += 1

                        if nan_grad_count[0] <= max_nan_grad_allowed:
                            print(f"    [Closure] Warning: Gradient is NaN/Inf (occurrence {nan_grad_count[0]}/{max_nan_grad_allowed}), clearing and rejecting step")
                            optimizer.zero_grad()
                            return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                        else:
                            raise RuntimeError(
                                f"Gradient contains NaN or Inf after backward pass (occurred {nan_grad_count[0]} times)! "
                                f"U={U.item():.6f}, loss={loss.item():.6f}. "
                                f"This indicates systematic numerical instability."
                            )
                else:
                    raise Exception(f"    [Closure] Warning: Gradient is None after backward pass!")
                return loss

            # DIAGNOSTIC: Store θ before step for comparison
            θ_before = θ.detach().clone()

            # Perform one L-BFGS step (includes line search)
            optimizer.step(closure)
            # NO POST-STEP PROJECTION NEEDED!
            # Constraint is satisfied by construction via reparameterization

            # DIAGNOSTIC: Compute θ changes after step
            with torch.no_grad():
                θ_diff = θ - θ_before
                θ_max_change = torch.max(torch.abs(θ_diff)).item()
                θ_mean_change = torch.mean(torch.abs(θ_diff)).item()
                θ_norm_after = torch.linalg.norm(θ).item()

            # Reconstruct x_masked from θ for tracking
            with torch.no_grad():
                θ_norm_current = torch.linalg.norm(θ)
                x_masked_current = (θ / θ_norm_current) * L2_target

                x_full_current = torch.empty_like(initial_img)
                x_full_current[~mask] = initial_unmasked
                x_full_current[mask] = x_masked_current

            # Compute utility and gradient for tracking (∂U/∂θ, not ∂U/∂x)
            with torch.enable_grad():
                # Recompute utility with gradient tracking for θ
                θ_norm_t = torch.linalg.norm(θ)
                x_masked_t = (θ / θ_norm_t) * L2_target

                x_full_t = torch.empty_like(initial_img)
                x_full_t[~mask] = initial_unmasked
                x_full_t[mask] = x_masked_t

                U_t = compute_utility_single_image(model_active, x_full_t, max_r_cap)
                dU_dtheta_t = torch.autograd.grad(U_t, θ, retain_graph=False)[0]

            # Verify constraint is satisfied (should be exact by reparameterization)
            with torch.no_grad():
                L2_actual = torch.linalg.norm(x_masked_current).item()
                L2_error = abs(L2_actual - L2_target) / L2_target if L2_target > 1e-8 else 0.0

            # DIAGNOSTIC: Print constraint verification
            if L2_error > 1e-4:
                print(f"   [Warning] L2 norm error: ||x||={L2_actual:.6f} (target: {L2_target:.6f}, error: {L2_error*100:.4f}%)")

            # Store trajectory data
            util_hist.append(U_t.item())
            grad_norms.append(torch.linalg.vector_norm(dU_dtheta_t).item())  # ||∂U/∂θ||
            x_traj.append(x_full_current.clone())
            grad_hist.append(dU_dtheta_t.detach().clone())  # Full ∂U/∂θ vector

            # DIAGNOSTIC: Reset closure counter for next step
            closure_call_count[0] = 0

        # === 7. Store trajectory data ===
        trajectory_data = {
            'initial_img': initial_img,
            'x_traj': x_traj,
            'util_hist': util_hist,
            'grad_norms': grad_norms,
            'grad_hist': grad_hist
        }

        # === 8. Print final diagnostics ===
        # Verify monotonic utility increase
        monotonic = all(util_hist[i] >= util_hist[i-1] for i in range(1, len(util_hist)))
        utility_increase = util_hist[-1] - util_hist[0]
        relative_increase = (utility_increase / util_hist[0] * 100) if util_hist[0] > 1e-8 else 0.0

        print(f"=== L-BFGS Complete (L2 Norm Constraint) ===")
        print(f"  Target L2 norm: {L2_target:.6f} (= sqrt({n_pixels_in_mask}))")
        print(f"  Initial U: {util_hist[0]:.6f}")
        print(f"  Final U:   {util_hist[-1]:.6f}")
        print(f"  Change:    +{utility_increase:.6f} ({relative_increase:+.2f}%)")
        print(f"  Monotonic: {'✓ Yes' if monotonic else '✗ No (WARNING!)'}")
        print(f"  L2 norm constraint: ||x_masked||₂ = {L2_target:.6f} (satisfied by construction)\n")

        print(f"  ||∂U/∂θ||_2   = {torch.linalg.vector_norm(dU_dtheta_init).item():.6e}")
        print(f"  ||∂U/∂θ||_∞   = {dU_dtheta_init.abs().max().item():.6e}")

    # Determine the optimized image to return
    # If optimization was performed (trajectory_data exists), use the final optimized image
    # Otherwise, return the initial (unoptimized) best image
    if trajectory_data is not None:
        optimized_img = trajectory_data['x_traj'][-1].clone().detach()
    else:
        optimized_img = imgs_train[x_idx_best].clone().detach()

    # NOTE: x_idx_best is the index of the INITIAL best image in the original dataset img_train
    # The optimized_img is a modified version of imgs_train[x_idx_best] after gradient ascent
    return {
        'optimized_img': optimized_img,                              # Final optimized image (shape: [11664] or [108,108])
        'img_idx': x_idx_best[None],                                 # Original image index (shape: [1])
        'utility_batch': u2d,                                        # Utility values for all candidate images
        'gradient': dU_dx_init.squeeze(0) if dU_dx_init is not None else None,  # Gradient (only if test_rms_constraint=True)
        'trajectory': trajectory_data                                # Full optimization trajectory (dict or None)
    }

def most_useful_remaining_idx( active_model, img_train, remaining_idx, verbose=True ):

    '''
    Works with active model instance of the GPModel class

    Returns a [1] shaped index of the most useful image in the remaining dataset

    So that it can be concatenated to the in_use_idx
    '''
    max_r_cap = 100

    X_remaining = img_train[remaining_idx]
    xtilde      = img_train[active_model.xtilde_idx]

    # Extract model parameters needed for utility calculation
    theta         = active_model.theta
    kernfun       = active_model.kernfun

    C             = active_model.C
    mask          = active_model.mask
    B             = active_model.B
    K_tilde_b     = active_model.K_tilde_b
    K_tilde_inv_b = active_model.K_tilde_inv_b

    m_b           = active_model.m_b
    V_b           = active_model.V_b
    A             = torch.exp(active_model.f_params['logA'])
    lambda0       = active_model.f_params['lambda0']

    # Calculate the matrices to compute the lambda moments. They are referred to the unseen images X_remaining
    Kvec_star = kernfun(theta, X_remaining[:,mask], x2=None, C=C, dC=None, diag=True)
    K_star    = kernfun(theta, X_remaining[:,mask], x2=xtilde[:,mask], C=C, dC=None, diag=False)
    K_star_b  = K_star @ B 

    lambda_m, lambda_var = lambda_moments( 
        X_remaining[:,mask], K_tilde_b, K_star_b@K_tilde_inv_b, Kvec_star, K_star_b, C, m_b, V_b, theta)  

    # We write the Utility function in terms of the expectation values of g(lambda) = A*lambda + lambda_0 
    # This is indeed equal to log(f) with f the firing rate but note that  we do not need the distribution of
    # log(f)
    logf_mean = A*lambda_m + lambda0
    logf_var  = A**2 * lambda_var

    # Estimate the utility and cap the maximum r ( used in a summation to infinity )
    r_masked = torch.arange(0, max_r_cap, dtype=TORCH_DTYPE, device=DEVICE)
    u2d      = nd_utility(logf_mean, logf_var, r_masked )

    if torch.any( torch.isnan(u2d) ):
        print('W - NaN U')
    if torch.any( torch.isinf(u2d) ):
        print('W - Inf U')

    i_best     = u2d.argmax()             # Index of the best image in the utility vector
    x_idx_best = remaining_idx[i_best]    # Index of the best image in the image dataset indices
    if verbose:
        print(f'Utility: {u2d[i_best].item():<8.6f} |  Best image ID: {i_best}  | Best image index: {x_idx_best}')

    assert x_idx_best not in active_model.in_use_idx, 'The best image idx is already in use. This shoudld not be possible for now'

    return x_idx_best[None]


    
def plot_optimization_trajectory(initial_img, x_traj, util_hist, grad_norms, grad_hist, model_active,
                                 img_idx, n_images, save_path, constraint_name, n_frames=5):
    """
    Create comprehensive visualization of image optimization trajectory.

    Creates a single figure with:
    - Utility curve over iterations (with best marked)
    - Gradient norm curve over iterations
    - Selected image frames including best image (highlighted)
    - Gradient of utility (∂U/∂X) for each frame

    All images are zoomed to show only the masked region with a small margin.

    Parameters:
    -----------
    initial_img : torch.Tensor
        Initial image (108*108,)
    traj : list
        List of images during optimization
    util_hist : list
        Utility values at each iteration
    grad_norms : list
        Gradient norms at each iteration
    mask : torch.Tensor
        Boolean mask for receptive field
    img_idx : int
        Image index
    n_images : int
        Current training iteration
    save_path : Path
        Directory to save figures
    n_frames : int
        Maximum number of image frames to plot (will include best + first + final)

    Note:
    -----
    The best image (highest utility) is always included in the frame selection
    and is highlighted with a red border and ★ marker.
    
    Gradients are computed from trajectory differences (masked pixels only).
    """
    import matplotlib.gridspec as gridspec

    n_px_side = 108
    n_iters = len(x_traj)

    # Find best iteration (maximum utility)
    best_iter = np.argmax(util_hist)
    best_utility = util_hist[best_iter]
    final_utility = util_hist[-1]

    # Smart frame selection: always include first, best, final
    mandatory_frames = {0, best_iter, n_iters - 1}

    if n_iters <= n_frames:
        frame_indices = list(range(n_iters))
    else:
        # Add evenly spaced middle frames
        n_middle = max(0, n_frames - 3)  # Reserve space for mandatory frames
        if n_middle > 0:
            middle_frames = set(np.linspace(1, n_iters-2, n_middle, dtype=int))
            all_frames = mandatory_frames | middle_frames
        else:
            all_frames = mandatory_frames

        # Sort and limit to n_frames
        frame_indices = sorted(list(all_frames))[:n_frames]

    # Prepare mask for plotting
    C, mask, K_tilde_b, K_tilde_inv_b, B = get_final_K_vals(model_active)

    mask_2d = torch.zeros((n_px_side * n_px_side), device=mask.device)
    mask_2d[mask] = 1.0
    mask_2d = mask_2d.reshape(n_px_side, n_px_side).cpu().numpy()

    # === COMPUTE ZOOM REGION FROM MASK ===
    # Find bounding box of masked region
    rows, cols = np.where(mask_2d == 1)
    if rows.size > 0 and cols.size > 0:
        ymin, ymax = np.min(rows), np.max(rows)
        xmin, xmax = np.min(cols), np.max(cols)
        
        # Add margin (10 pixels on each side, clamped to image bounds)
        margin = 10
        ymin_zoom = max(0, ymin - margin)
        ymax_zoom = min(n_px_side - 1, ymax + margin)
        xmin_zoom = max(0, xmin - margin)
        xmax_zoom = min(n_px_side - 1, xmax + margin)
    else:
        # Fallback: show entire image if mask is empty
        ymin_zoom, ymax_zoom = 0, n_px_side - 1
        xmin_zoom, xmax_zoom = 0, n_px_side - 1

    # Image normalization will be done independently per frame (see plotting loop below)
    # This ensures each frame uses its full dynamic range

    # === NEW: Compute gradients for visualization ===
    # Compute gradient direction from trajectory differences (MASKED PIXELS ONLY)

    # Find consistent gradient vmin/vmax for all frames
    all_grads_masked = torch.stack([grad_hist[i] for i in frame_indices])
    grad_vmax = torch.max(torch.abs(all_grads_masked)).item()

    # Create single figure with GridSpec for precise layout
    # Now 4 rows: utility curve, gradient norm, images, gradient images
    fig = plt.figure(figsize=(max(16, 3*len(frame_indices)), 16))
    gs = gridspec.GridSpec(4, len(frame_indices),
                          height_ratios=[2, 2, 3, 3],
                          hspace=0.3, wspace=0.05,
                          figure=fig)

    # Row 1: Utility curve (spans all columns)
    ax_util = fig.add_subplot(gs[0, :])

    # Row 2: Gradient curve (spans all columns)
    ax_grad = fig.add_subplot(gs[1, :], sharex=ax_util)

    # Row 3: Image frames (one per column)
    ax_imgs = [fig.add_subplot(gs[2, i]) for i in range(len(frame_indices))]
    
    # Row 4: Gradient images (one per column)
    ax_grads = [fig.add_subplot(gs[3, i]) for i in range(len(frame_indices))]

    # ===== Plot utility curve with best marker =====
    iterations = np.arange(len(util_hist))
    ax_util.plot(iterations, util_hist, 'b-', linewidth=2, label='Utility', zorder=2)

    # Mark best iteration
    ax_util.axvline(best_iter, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Best (iter {best_iter})', zorder=3)
    ax_util.scatter([best_iter], [best_utility], c='red', s=300,
                   marker='*', zorder=5, edgecolors='darkred', linewidths=2,
                   label=f'★ Max U={best_utility:.4f}')

    # Mark plotted frames
    ax_util.scatter(frame_indices, [util_hist[i] for i in frame_indices],
                   c='orange', s=80, zorder=4, alpha=0.7, edgecolors='darkorange',
                   linewidths=1.5, label='Plotted frames')

    ax_util.set_ylabel('Utility U', fontsize=14)
    ax_util.set_title(f'Optimization Trajectory: Image {img_idx} (n_images={n_images})',
                     fontsize=16, fontweight='bold')
    ax_util.grid(True, alpha=0.3)
    ax_util.legend(loc='best', fontsize=11)

    ax_util.xaxis.set_major_locator(MaxNLocator(integer=True)) 

    # ===== Plot gradient norm with best marker =====
    ax_grad.plot(iterations, grad_norms, 'g-', linewidth=2,
                    label='||∇U||', zorder=2)

    # Mark best iteration
    ax_grad.axvline(best_iter, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, zorder=3)

    # Mark plotted frames
    ax_grad.scatter([i for i in frame_indices if i > 0],
                   [grad_norms[i] for i in frame_indices if i > 0],
                   c='orange', s=80, zorder=4, alpha=0.7, edgecolors='darkorange',
                   linewidths=1.5)

    ax_grad.set_ylabel('||∇U|| ', fontsize=14)
    # ax_grad.set_xlabel('Iteration', fontsize=14)
    ax_grad.grid(True, alpha=0.3)

    ax_grad.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ===== Plot image frames and gradient frames =====
    for plot_idx, iter_idx in enumerate(frame_indices):
        # === Image row ===
        ax_img = ax_imgs[plot_idx]
        img = x_traj[iter_idx].reshape(n_px_side, n_px_side).cpu().numpy()

        assert ~np.isnan(img).any(), f"NaN detected in image at iter {iter_idx}"

        # Compute normalization for THIS frame independently
        vmin_frame = img.min()
        vmax_frame = img.max()

        # Show image with per-frame normalization
        im = ax_img.imshow(img, cmap='gray', vmin=vmin_frame, vmax=vmax_frame)
        ax_img.contour(mask_2d, levels=[0.5], colors='yellow',
                      linewidths=1.5, alpha=0.7)

        # === APPLY ZOOM ===
        ax_img.set_xlim(xmin_zoom, xmax_zoom)
        ax_img.set_ylim(ymax_zoom, ymin_zoom)  # Inverted for imshow convention

        # Build title with markers
        title_parts = []

        if iter_idx == best_iter:
            # BEST image - special highlighting

            # Add red border
            for spine in ax_img.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)

            # Background color
            ax_img.patch.set_facecolor('#ffe6e6')
            ax_img.patch.set_alpha(0.3)

        title_parts.append(f"Iter {iter_idx}")

        title = "\n".join(title_parts)

        title += f"\nU={util_hist[iter_idx]:.4f}"

        # if iter_idx > 0:
        # title += f"\n|Mask|={np.linalg.norm(img).item():.2e}"

        ax_img.set_title(title, fontsize=10,
                        fontweight='bold' if iter_idx == best_iter else 'normal',
                        color='darkred' if iter_idx == best_iter else 'black')
        ax_img.axis('off')

        # === Gradient row ===
        ax_grad_img = ax_grads[plot_idx]
        
        # Reconstruct gradient image: create full image with zeros, fill masked region
        grad_full = torch.zeros((n_px_side * n_px_side), device=all_grads_masked[plot_idx].device)
        grad_full[mask] = all_grads_masked[plot_idx]
        grad_2d = grad_full.reshape(n_px_side, n_px_side).cpu().numpy()

        # Show gradient with diverging colormap
        im_grad = ax_grad_img.imshow(grad_2d, cmap='RdBu_r', 
                                     vmin=-grad_vmax, vmax=grad_vmax)
        ax_grad_img.contour(mask_2d, levels=[0.5], colors='yellow',
                           linewidths=1.5, alpha=0.7)
        
        # === APPLY ZOOM ===
        ax_grad_img.set_xlim(xmin_zoom, xmax_zoom)
        ax_grad_img.set_ylim(ymax_zoom, ymin_zoom)  # Inverted for imshow convention
        
        # Add colorbar
        # plt.colorbar(im_grad, ax=ax_grad_img, fraction=0.046)
        
        grad_title = f"∂U"
        grad_title += f"\n||∂U||={np.linalg.matrix_norm(grad_2d):.2e}"

        
        ax_grad_img.set_title(grad_title, fontsize=10)
        ax_grad_img.axis('off')

        # Highlight best iteration in gradient row too
        if iter_idx == best_iter:
            for spine in ax_grad_img.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)
            ax_grad_img.patch.set_facecolor('#ffe6e6')
            ax_grad_img.patch.set_alpha(0.3)

    # ===== Add statistics text box =====
    stats_text = (
        f"Initial U: {util_hist[0]:.4f}\n"
        f"Best U: {best_utility:.4f} at iter {best_iter}\n"
        f"Final U: {final_utility:.4f}\n"
        f"Improvement: {best_utility - util_hist[0]:.4f} "
        f"({100*(best_utility-util_hist[0])/abs(util_hist[0]):.1f}%)\n"
        f"Zoom: [{ymin_zoom}:{ymax_zoom}, {xmin_zoom}:{xmax_zoom}]"
    )

    if best_iter != len(util_hist) - 1:
        diff = best_utility - final_utility
        stats_text += f"\n[!] Best != Final (diff: {diff:.4f}, {100*diff/final_utility:.1f}%)"

    # Add text box to figure
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            family='monospace')

    # Suppress tight_layout warning (text box placed manually with fig.text)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for text box

    # Save single comprehensive figure
    save_name = f'{constraint_name}_optimization_traj_n{n_images:03d}_img{img_idx:04d}.png'
    fig.savefig(save_path / save_name, dpi=150, bbox_inches='tight')
    plt.close(fig)

#################   Numerical estimation / problems functions  ####################

def is_posdef(tensor, name='M'):
                
    # L_upper, info_upper = torch.linalg.cholesky_ex(tensor, upper=True)
    # L_lower, info_lower = torch.linalg.cholesky_ex(tensor, upper=True)

    # if torch.any(info_upper != 0):
        # warnings.warn('Not positive definite using UPPER triangular')
    # if torch.any(info_lower != 0):
        # warnings.warn('Not positive definite using LOWER triangular')

    if is_simmetric(tensor, name=name):
        smallest_eig = torch.linalg.eigh(tensor)[0].min()
        if smallest_eig <= 0.:
            warnings.warn(f'Matrix {name} is simmetric but has an eigenvalue smaller than 0 ')
            return False
        if smallest_eig <= MIN_TOLERANCE:
            warnings.warn(f'Matrix {name} is simmetric but has an eigenvalue smaller than MIN_TOLERANCE: {MIN_TOLERANCE}')
            return False
        else:
            return True  
    else:
        warnings.warn('The matrix is not symmetric, cannot check if it is positive definite')
        return False

def is_simmetric(tensor, name='M'):
    difference = torch.abs(tensor - tensor.T)
    if torch.any(difference > MIN_TOLERANCE):
        print(f'Matrix {name} is not symmetric, maximum difference is {difference.max()}')
        return False
    
    else: return True 

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

def log_det(M, name='M', ignore_warning=False):

    try:
        # Try a Cholesky decomposition, L is the matrix that multiplied by its (cunjugate) transpose gives M
        L=torch.linalg.cholesky(M, upper=True)
        # The determinant of M is then the product (the * 2 ) of the product of the diagonal elements of L and L.T (the same if real) 
        # Its log is 2*torch.log(torch.product(torch.diag(L))) which corresponds to
        return 2*torch.sum(safe_log(torch.diag(L))) 
    except:
        if is_simmetric(M, name=name):

            # If the matrix is simmetric we can use eigh
            eigenvalues, eigenvectors = torch.linalg.eigh(M) # .eigh() is only for simmetrix matrices, otherwise .eig() 

            # We can also check what kind of small or neative eigenvalues there are
            # this is basically a copy of the function 'is_posdef'
            ikeep = eigenvalues > max(eigenvalues.max() * EIGVAL_TOL, EIGVAL_TOL)
            large_eigenvals      = eigenvalues[ikeep]
            large_real_eigenvals = torch.real(large_eigenvals)
        
            if not ignore_warning:
                smallest_eig = eigenvalues.min()
                warnings.warn(f"Matrix {name} in logdet is simmetric but not posdef, using eigendecomposition to calculate the log determinant")

                if smallest_eig <= 0.:
                    warnings.warn(f'Matrix {name} in logdet is simmetric but has an eigenvalue smaller than 0 ')

                elif smallest_eig <= 1.e-10:
                    warnings.warn(f'Matrix {name} in logdet is simmetric but has an eigenvalue smaller than 1e-10 ')
       
            return torch.sum(safe_log( large_real_eigenvals )) 
        else:
            warnings.warn(f"Matrix {name} in logdet is not simmetric in log_det used in KL_divergence")
            return 0

def estimate_memory_usage(X):
    # Calculate memory usage for each tensor
    X_memory = X.element_size() * X.nelement()
    # Total memory usage in bytes
    total_memory_bytes = X_memory 
    # Convert bytes to megabytes (MB)
    total_memory_MB = total_memory_bytes / (1024 ** 2)
    print(f'Memory on GPU: {total_memory_MB:.2f} MB')
    return total_memory_MB

def block_matrix_inverse(orig_inv, new_column):
    '''
    Use Sherman-Woodbury matrix update for the inverse of the N+1 X N+1 matrix
    Compute the inverse of the N+1 X N+1 matrix given the inverse of the N X N matrix
    the N+1 X N+1 matrix is assumed of the form [[K, b], [b.T, d]] where new_column = [b, d]
    '''
    b = new_column[:-1]
    d = new_column[-1]

    e = orig_inv @ b
    g = 1/( d - b.T @ e )

    updated_inv = torch.cat( ( orig_inv + g * e @ e.T, -g * e),                         axis=1)
    updated_inv = torch.cat( ( updated_inv ,          torch.cat((-g * e, g), axis=0).T),axis=0)

    return updated_inv

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

def generate_xtilde(ntilde, x, xtilde_idxs=None ):
    '''
    Returns the inducing points dataset from the original dataset x

    Can extract randomly or get xtildes from the indices
    
    '''
    if xtilde_idxs is not None:
        xtilde = x[xtilde_idxs, :]
        return xtilde

    else:
        xtilde_idxs = torch.randperm(ntilde)
        
    return xtilde

def logbetaexpr_to_beta(logbetaexpr):

    # Go from the logarithmic expression of beta used in the learning algorithm to beta of the PNAS Paper 
    beta_paper = torch.exp(-0.5*logbetaexpr) * torch.tensor(0.5)
    return beta_paper

def logrhoexpr_to_rho(logrhoexpr):
    # Go from the logarithmic expression of rho used in the learning algorithm to rho of the PNAS Paper
    # rho_paper = torch.exp(-0.5 * theta['-log2rho2']) /  torch.sqrt( torch.tensor(2)) 

    rho_paper = torch.exp(-0.5*logrhoexpr) /  torch.sqrt( torch.tensor(2)) 
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

def generate_theta(x, r, n_px_side, display_hyper=False, **kwargs):
        
        # Initializes the values of the Kernel and C hyperparameters and their limits
        # Some of the hyperaparameters are learnt as their log with some factors.
        # These factors where different in Matthew's code and Samuele's code. See hyperparameters_conversion.txt for details. 

        # sigma_0 : Acoskern specific hyperparameter.
        # Amp :     Amplitude of the localker 
       
        # eps_0 = (eps_0x, eps0y) : Center of the receptive field as real numbers from -1 to 1. 
            # They are obtained from the STA that gives them initially as integers (pixel position)
       
        # logbetaexpr : a logarithmic expression of beta, which is is the scale of local filter of C in the paper.    )
        # logrhoexpr  : a logarithmic expression of rho, which is the scale of the smoothness filter part of C.

        # The hyperparameters will be used in: 
        # C_smooth : (nx, nx) gaussian shaped exponential of distance between x. Can be seen as the covariance of the gaussian for the weights (in weights view)


        # _____ Sigma_0 and A _____
        # logsigma_0 = 0 # Samuele's code set the log of sigma
        # sigma_0 = torch.tensor(1.0,requires_grad=True) # This makes sigma_0 = 1
        
        # Amp     = torch.tensor(1., requires_grad=True) # Amplitude of the receptive field Amp = 1, NOT PRESENT IN SAMUELE'S CODE

        # # Center and size of the receptive field RF as sta and its variance (rf_width_pxl2). eps are indeces from 0 to 107
        # sta, rf_width_pxl2, (eps_0x, eps_0y) = get_sta(x, r, n_px_side)  # rf_width_pxl is manually chosen for now TODO

        # # eps_0 go from 0 to 107, n_px_side = 108. I then bring it to [-1,1] multiplying by 2 and shifting
        # eps_0x_rescaled = ( eps_0x / (n_px_side - 1))*2 - 1
        # eps_0y_rescaled = ( eps_0y / (n_px_side - 1))*2 - 1       

        # # _____ Temporary _____
        # rf_width_pxl2   = torch.tensor(10, dtype=TORCH_DTYPE)  
        # # eps_0x = torch.tensor(2, dtype=TORCH_DTYPE) # Center of the RF in pixels
        # # eps_0y = torch.tensor(3, dtype=TORCH_DTYPE) # Center of the RF in pixels
        # eps_0x_rescaled = torch.tensor(0.0, requires_grad=True)
        # eps_0y_rescaled = torch.tensor(0.0, requires_grad=True)

        # # Make them learnable
        # eps_0x_rescaled.requires_grad = True
        # eps_0y_rescaled.requires_grad = True

        # if not (low_lim <= eps_0x_rescaled <= up_lim) or not (low_lim <= eps_0y_rescaled <= up_lim):
        #     raise ValueError(f"eps_0x_rescaled and eps_0y_rescaled must be within the range of {low_lim} and {up_lim}.")
        
        # # _____ Beta and Rho _____
        # # Here the caracteristic lenght is Dict of the sqrt the  variance of the sta = receptive field pixel size squared.  TODO check
        # # It is chosen in pixels but Hransfered to [0,2]. It is also stored as theta[i]=-2log(2*beta) in the dict
        # # so that e^(theta[i]) = 1/(4beta2) can be multiplied in the exponent of a_local of the kernel
        # rf_width_pxl = torch.sqrt(rf_width_pxl2)
        
        # beta         = (rf_width_pxl / n_px_side) * (up_lim-low_lim) #sqrt of variance of sta brought to [0,2]

        # logbetaexpr      = -2*safe_log(2*beta) # we call it logbetaexpr cause of the factors in the expression (see hyperparameters_conversion.txt)
        # logbetaexpr.requires_grad = True
        
        # # Smoothness of the localkernel. Dict of half of caracteristic lenght of the RF
        # # It is chosen in Hut then transfered to [0,2]. It is also stored as theta[i]=-log2rho2 in the dict
        # rho    = beta/2
        # logrhoexpr = -safe_log(torch.tensor(2.0)*(rho*rho)) # we call it logrhoexpr cause of some factors in the expression (see hyperparameters_conversion.txt)
        # logrhoexpr.requires_grad = True

        # theta = {'sigma_0':sigma_0, 'eps_0x':eps_0x_rescaled, 'eps_0y':eps_0y_rescaled, '-2log2beta': logbetaexpr, '-log2rho2': logrhoexpr, 'Amp': Amp }


        # # Print the learnable hyperparameters
        # if display_hyper:
        #     print(' Before overloading')
        #     print(f' Hyperparameters have been SET as  : beta = {beta:.4f}, rho = {rho:.4f}')
        #     print(f' Samuele hyperparameters           : logbetasam = {-torch.log(2*beta*beta):.4f}, logrhosam = {-2*safe_log(rho):.4f}')
            
        #     kwargs.get
        #     print('\n After overloading')
        #     print(f' Dict of learnable hyperparameters : {", ".join(f"{key} = {value.item():.4f}" for key, value in theta.items())}')
        #     print(f' Hyperparameters from the logexpr  : beta = {logbetaexpr_to_beta(logbetaexpr):.4f}, rho = {logrhoexpr_to_rho(logrhoexpr):.4f}')
        #     beta = logbetaexpr_to_beta(logbetaexpr)
        #     rho  = logrhoexpr_to_rho(logrhoexpr)
        #     print(f' Samuele hyperparameters           : logbetasam = {-torch.log(2*beta*beta):.4f}, logrhosam = {-2*safe_log(rho):.4f}')

        # Lower bounds for these hyperparameters, considering that:
        # rho > 0 -> log(rho) > -inf
        # sigma_b > 0 -> log(sigma_b) > -inf
        # beta > e^4 (??) -> log(beta) > 4
        upp_lim =  torch.tensor(1.)
        low_lim = -torch.tensor(1.)
        # If theta is passed as a keyword argument, update the values of the learnable hyperparameters
        theta = {}
        for key, value in kwargs.items():
            # if key in theta:
                theta[key] = value
                if display_hyper: print(f'updated {key} to {value.cpu().item():.4f}')
        
        theta_lower_lims  = {'sigma_0': 0           , 'eps_0x':low_lim, 'eps_0y':low_lim, '-2log2beta': -float('inf'), '-log2rho2':-float('inf'), 'Amp': 0. }
        theta_higher_lims = {'sigma_0': float('inf'), 'eps_0x':upp_lim,  'eps_0y':upp_lim,  '-2log2beta':  float('inf'), '-log2rho2': float('inf'), 'Amp': float('inf') }
        
        return ( theta, theta_lower_lims, theta_higher_lims )

def gen_hyp_tuple(theta, freeze_list, display_hyper=True):
    '''
    (Better) Alternative to generate_theta.

    Generates the hyperparameters tuple (theta, theta_lower_lims, theta_higher_lims) 

    Sets the requires_grad attribute of hyp to True except for the ones in freeze_list

    Args:
        theta: dictionary of hyperparameters
    Returns:
        tuple of hyperparameters
    
    '''
    upp_lim =  torch.tensor(1.)
    low_lim = -torch.tensor(1.)
    theta_lower_lims  = {'sigma_0': 0           , 'eps_0x':low_lim, 'eps_0y':low_lim, '-2log2beta': -float('inf'), '-log2rho2':-float('inf'), 'Amp': 0. }
    theta_higher_lims = {'sigma_0': float('inf'), 'eps_0x':upp_lim,  'eps_0y':upp_lim,  '-2log2beta':  float('inf'), '-log2rho2': float('inf'), 'Amp': float('inf') }

    # Set the gradient of the hyperparemters to be updateable 
    for key, value in theta.items():
    # to exclude a single hyperparemeters from the optimization ( to exclude them all just set nMstep=0)
        if key in freeze_list:
            continue
        theta[key] = value.requires_grad_()
        if display_hyper: 
            print(f'{key} is {value.cpu().item():.4f}')

    return ( theta, theta_lower_lims, theta_higher_lims )

##################   Kernel related functions   ####################

def localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False):
    # Compute C, the part of the kernel responsible for implementing the receptive field and smoothness

    # Check that theta is inside the limits
    
    for key, value in theta.items():
        if not (theta_lower_lims[key] <= value <= theta_higher_lims[key]):
            raise ValueError(f"{key} = {value:.4f} is not within the limits of {theta_lower_lims[key]} and {theta_higher_lims[key]}")

    eps_0 = torch.stack([theta['eps_0x'], theta['eps_0y']]) # Do not create new tensor, just stack the two elements to preserve the gradient graph
    # xcord = torch.linspace(theta_lower_lims['eps_0y'], theta_higher_lims['eps_0x'], n_px_side)

    # ______ Samuele's code ______

    # spatial localised prior
    # Note: Matlab uses default indexing 'xy' while torch uses 'ij'. They make no difference because the arrays get flattened
    ycord, xcord = torch.meshgrid( torch.linspace(-1, 1, n_px_side), torch.linspace(-1, 1, n_px_side), indexing='ij')# a grid of 108x108 points between -1 and 1
    # ycord, xcord = torch.meshgrid( torch.linspace(-1, 1, n_px_side), torch.linspace(-1, 1, n_px_side), indexing='xy') #a grid of 108x108 points between -1 and 1
    xcord = xcord.flatten().to(DEVICE, dtype=TORCH_DTYPE)
    ycord = ycord.flatten().to(DEVICE, dtype=TORCH_DTYPE)
    logalpha    = -torch.exp(theta['-2log2beta'])*((xcord - eps_0[0])**2+(ycord - eps_0[1])**2)
    alpha_local =  torch.exp(logalpha)    # aplha_local in the paper

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


# NOTE: C_gradients_hyp, LocalkerCleanFunction, and localker_clean have been moved to
# gaussian_processes.Spatial_GP_repo.kernels.kernels
# They are imported at the top of this file for backward compatibility.


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
    # note : n1=n2 DOES NOT mean  xtilde = x, we might be doing inference
    if xtilde_case:
        return  (K + K.T)/2 + 1.e-9*torch.eye(K.shape[0]) # make sure it's simmetric  BUG put it on the right device
    if not xtilde_case:
        return  K 


def acosker(theta, x1, x2=None, C=None, dC=None, diag=False, get_dK_x=False):
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

    # Inputs are transposed because this function came from a translation of Samuele's code 

    x1 = x1.T # shape (nx, n1)
    if x2 is not None: x2 = x2.T # shape (nx, n2)
    n1 = x1.shape[-1]  # Take the shape given by the mask
    sigma_0 = theta['sigma_0']

    if C is None: C = torch.eye(n1, device=DEVICE, dtype=TORCH_DTYPE)

    if not diag:
        # n2 = x2.shape[1]

        X1 = torch.sqrt(torch.sum(x1*(C @ x1), dim=0) + sigma_0 ** 2) # torch.sum(x1*(C@x1), dim=0) is the same as Diag(x1.T @ C @ x1) #shape(n1)
        X2 = torch.sqrt(torch.sum(x2*(C @ x2), dim=0) + sigma_0 ** 2) # shape(n2,)

        X1X2 = torch.outer(X1,X2)               #shape(n1,n2)
        x1x2 = x1.T @ C @ x2 + sigma_0 ** 2   
        
        cosdelta = torch.clip(x1x2 / (X1X2 + 1e-7), -1, 1) #  TODO remove this clip (making it more numerically stable)

        delta = torch.arccos(cosdelta)  # angle theta in the paper and Samuele's code.
        # Angle dependend ezpression in eq (8) of the paper
        J = (torch.sqrt(1 - cosdelta ** 2) + torch.pi * cosdelta - delta * cosdelta) / torch.pi   #shape(n1,n2)
        #TODO check if this is not missing a *0.5 as in the paper (there it is /2pi)
        K = X1X2 * J       #shape(n1, n2) ( in case of x1=x and x2=xtilde -> shape (nt,ntilde) )

        if dC is not None:
            if get_dK_x:
                raise NotImplementedError("dK_x not implemented if dC is passed")
            # Define the gradient of K with respect to the hyperparameters ( including sigma_0 )
            dK = {}

            dX1X2 = sigma_0 ** 2 * (X2 / X1[:, None] + X1[:, None] / X2) #shape same as X1X2

            dcosdelta = (2 * sigma_0 ** 2 - cosdelta * dX1X2) / X1X2    #shape same as cosdelta

            dJ = -(delta - torch.pi) * dcosdelta / torch.pi           #shapesame as J
            # Add gradient for sigma_0. In Samuele's code what is learned is logsigma_0 so the expression is different

            # Test: the following derivative mst be wrt sigma_0 nos log_sigma_0 like in samueles code
            dK['sigma_0'] =  (X1X2 * dJ + dX1X2 * J) / sigma_0      #shape same as K
            # dK['sigma_0'] =  (X1X2 * dJ + dX1X2 * J)      #shape same as K

            # for j in range(1, dC.shape[2] + 1):
            for key, value in dC.items():
                if key == 'sigma_0':
                    continue

                dX1 = 0.5*torch.sum(x1*torch.matmul(dC[key], x1), dim=0)/X1  #shape(n1,)
                dX2 = 0.5*torch.sum(x2*torch.matmul(dC[key], x2), dim=0)/X2  #shape(n2,)
                
                dX1X2 = dX1[:, None]*X2 + X1[:, None]*dX2

                dcosdelta = (torch.matmul(x1.T, torch.matmul(dC[key], x2)) - cosdelta*dX1X2)/X1X2

                dJ = -(delta-torch.pi)*dcosdelta/torch.pi

                dK[key] = X1X2*dJ + dX1X2*J


        if get_dK_x and dC is None:
            
            if x1.shape[0] < x2.shape[0]:
                raise NotImplementedError("dK_x needs to be called with xtilde as first argument")
            if x2.ndim == 1:
                # in case we want gradient of more than one img at the time
                x2 = x2[:, None] # unsqueeze to make it a matrix of shape (nx,1)
  
            
            # Derivative of X2 with respect to x2
            # MATLAB: dX2 = x2'*C./X2 ( x2 transpose, matrix multiplied by C, element wise divided by X2)
            dX2 = torch.matmul(x2.T, C) / X2[:, None]  # shape (n2, nx)<=(n2, nx)/(n2,1) ( each row of x2.T*C divided by the corresponding X2 element)
            # Derivative of X1X2 with respect to x2
            # MATLAB: dX1X2 = X1'*dX2 ( no matrix multiplication despite * , its broadcasting as: )
            dX1X2 = X1[:, None, None] * dX2[None, :, :]  # shape (n1, n2, nx)
            
            # Derivative of cosdelta with respect to x2
            # MATLAB: darg = (x1'*C - arg.*dX1X2)./X1X2
            darg = (torch.matmul(x1.T, C)[:, None, :] - cosdelta[:, :, None] * dX1X2) / X1X2[:, :, None]  # shape (n1, n2, nx)
            
            # Derivative of J with respect to x2
            # MATLAB: dJ = -(theta-pi).*darg/pi
            dJ = -(delta - torch.pi)[:, :, None] * darg / torch.pi  # shape (n1, n2, nx)
            
            # Derivative of kernel K with respect to x2
            # MATLAB: dK_x = X1X2.*dJ + dX1X2.*J
            dK_x = X1X2[:, :, None] * dJ + dX1X2 * J[:, :, None]  # shape (n1, n2, nx)
            
            # NOTE we are returning the derivative of K with respect to x2, 
            # so you should call this function as K ( ntilde, n -> the x to be derived for)
            # IN THE OTHER TWO CASES WE USUALLY CALL K (n, ntilde)
            return K, dK_x # the K being returned is the same (n1,n2), dK_x is (n1,n2,nx)
        
    else: # In the diagonal case only the complete dataset passed as x1 is considered

        # return just diagonal of kernel
        K = torch.sum(x1*torch.matmul(C, x1), dim=0)[:, None]+sigma_0**2
        K = K.squeeze() # To return a vector of shape (n1,)
        # Gradient
        if dC is not None:
            dK = {}

            # Test: the following derivative mst be wrt sigma_0 nos log_sigma_0 like in samueles code
            ones = torch.ones((n1, 1), device=DEVICE, dtype=TORCH_DTYPE)
            dK['sigma_0'] = (2*sigma_0**2*ones).squeeze() / sigma_0
            # dK['sigma_0'] = (2*sigma_0**2*torch.ones((n1, 1))).squeeze()
            
            for key in dC.keys():
                if key == 'sigma_0':
                    continue
                dK[key] = torch.sum(x1 * torch.matmul(dC[key], x1), dim=0)

            #K += 1e-7 * torch.eye(n1, 1)

        if get_dK_x and dC is None:
            # Derivative of x1^t@C@x1 with respect to x1
            dK_x = 2 * torch.matmul(C, x1)  # shape (nx, n1)
            return K, dK_x.T # K shape (n1,), dK_x shape (n1,nx)

    # Returns
    if dC is not None:
        return K, dK    #shape(n1,n2), shape(n1,n2,6)
    else:
        return K    #shape (n1,n2)


# NOTE: acosker_clean, AcoskerCleanFunction, and acosker_with_grad have been moved to
# gaussian_processes.Spatial_GP_repo.kernels.kernels
# They are imported at the top of this file for backward compatibility.


def get_model_at_iteration(fit_model, iteration):
    """
    Constructs a fit_model dictionary using the tracked values at a specific iteration.
    
    Args:
        fit_model (dict): The completed model dictionary returned by varGP
        iteration (int): The iteration at which to construct the model state
        
    Returns:
        dict: A new fit_model dictionary with the state at the specified iteration
    """
    if iteration >= fit_model['fit_parameters']['maxiter']:
        raise ValueError(f"Iteration {iteration} is >= than total iterations {fit_model['fit_parameters']['maxiter']}")
    
    # Make a deep copy of fit_parameters and modify maxiter
    fit_parameters = copy.deepcopy(fit_model['fit_parameters'])
    fit_parameters['maxiter'] = iteration + 1
    
    # Get tracked values at the specified iteration
    values_track = {
        'loss_track': {
            key: value[:iteration + 1] 
            for key, value in fit_model['values_track']['loss_track'].items()
        },
        'theta_track': {
            key: value[:iteration + 1]
            for key, value in fit_model['values_track']['theta_track'].items()
        },
        'f_par_track': {
            key: value[:iteration + 1]
            for key, value in fit_model['values_track']['f_par_track'].items()
        },
        'variation_par_track': {
            'V_b': fit_model['values_track']['variation_par_track']['V_b'][:iteration + 1],
            'm_b': fit_model['values_track']['variation_par_track']['m_b'][:iteration + 1]
        }
    }
    
    # Get parameter values at the specified iteration
    theta = {
        key: value[iteration] 
        for key, value in fit_model['values_track']['theta_track'].items()
    }
    
    f_params = {}
    if 'lambda0' in fit_model['values_track']['f_par_track']:
        f_params = {
            'logA': fit_model['values_track']['f_par_track']['logA'][iteration],
            'lambda0': fit_model['values_track']['f_par_track']['lambda0'][iteration]
        }
    elif 'loglambda0' in fit_model['values_track']['f_par_track']:
        f_params = {
            'logA': fit_model['values_track']['f_par_track']['logA'][iteration],
            'loglambda0': fit_model['values_track']['f_par_track']['loglambda0'][iteration]
        }
    
    # Construct model at iteration
    model_at_iteration = {
        'fit_parameters': fit_parameters,
        'final_kernel': fit_model['final_kernel'],  # Using final kernel structure
        'err_dict': {'is_error': False, 'error_message': None},
        'xtilde': fit_model['xtilde'],
        'hyperparams_tuple': (
            theta,
            fit_model['hyperparams_tuple'][1],  # Lower bounds
            fit_model['hyperparams_tuple'][2]   # Upper bounds
        ),
        'f_params': f_params,
        'm_b': fit_model['values_track']['variation_par_track']['m_b'][iteration],
        'V_b': fit_model['values_track']['variation_par_track']['V_b'][iteration],
        'C': fit_model['C'],
        'mask': fit_model['mask'],
        'K_tilde_b': fit_model['K_tilde_b'],
        'K_tilde_inv_b': fit_model['K_tilde_inv_b'],
        'K_b': fit_model['K_b'],
        'Kvec': fit_model['Kvec'],
        'B': fit_model['B'],
        'values_track': values_track
    }
    
    return model_at_iteration

##################   X-Steps and Quantities   ####################
# NOTE: @torch.no_grad() decorator removed to allow PyTorch autograd for image optimization
# This function is called from compute_utility_single_image() which needs gradient flow
def lambda_moments( x, K_tilde, KKtilde_inv, Kvec, K, C, m, V, theta, kernfun=None, dK=None , dK_tilde=None, dK_vec=None, K_tilde_inv=None):
            # Calculate the mean and variance (diagonal of covariance matrix ) of (vec)lambda(of the training points) over the distribution given by:
            # p_cond(lambda|lambda_tilde,X,theta)*(N/q)_posterior(lambda_tilde|m,V) as ini eq (56)(57) of Notes for Pietro
            # We make use of the mean and variace of the same distribution, but oly its inducing point approximation ( m_b and V_b )

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


            lambda_var = Kvec + torch.sum(-K.T*a.T + a.T*(V@a.T), 0)

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
                    # pinv(Sigma)*(dki(:, :, i) - dSigma(:, :, i)*a );
                    # Derivative of the mean of lambda with respect to the hyperparameters
                    dlambda_m[key] = da[key]@m
                    # Derivative of the variance of lambda with respect to the hyperparameters
                    # dlambda_var[key] = dK_vec[key] + torch.einsum( 'ij,ji->i', 2*da[key], torch.linalg.solve(V_inv, a.T)) - torch.einsum( 'ij,ij->i', dK[key],a ) - torch.einsum( 'ij,ij->i', K, da[key] )
                    dlambda_var[key] = dK_vec[key] + torch.einsum( 'ij,ji->i', 2*da[key], V@a.T) - torch.einsum( 'ij,ij->i', dK[key],a ) - torch.einsum( 'ij,ij->i', K, da[key] )
                return lambda_m, lambda_var, dlambda_m, dlambda_var

            else :
                return lambda_m, lambda_var


def mean_f_given_lambda_moments( f_params, lambda_m, lambda_var,):
        '''Compute the expectation value of the vector of firing rates for every training point: 
                        <f> = exp(A*<lambda> + 0.5*A^2*Var(lambda) + lambda0)
        as shown (between other things) in (34)-(37) of Notes for Pietro
        
        Note: we cap the maximum firing rate to 1000 to avoid
        '''
        A       = torch.exp(f_params['logA'])
        # lambda0 = f_params['lambda0']

        lambda0 = torch.exp(f_params['loglambda0']) if 'loglambda0' in f_params else f_params['lambda0']

        f_mean = torch.exp(A*lambda_m + 0.5*A*A*lambda_var + lambda0 )

        # return torch.min( f_mean, torch.tensor(1000.))
        return f_mean

def mean_f_gradients_wrt_x_given_lambda_moments( f_params, lambda_m, lambda_var, dlambda, dvar_lambda):
        '''Compute the gradients of the expectation value of the vector of firing rates for every given point x wrt x:
                        d<f>/x = <f> * ( A*d<lambda>/dx + 0.5*A^2*dVar(lambda)/dx )

        
        '''
        A       = torch.exp(f_params['logA'])

        lambda0 = torch.exp(f_params['loglambda0']) if 'loglambda0' in f_params else f_params['lambda0']

        f_mean = torch.exp(A*lambda_m + 0.5*A*A*lambda_var + lambda0 )

        df = f_mean * ( A*dlambda + 0.5*A*A*dvar_lambda )
        
        return f_mean, df
        
def mean_f( f_params, calculate_moments, lambda_m=None, lambda_var=None,  x=None, K_tilde=None, KKtilde_inv=None, 
           Kvec=None, K=None, C=None, m=None, V=None, V_inv=None, theta=None, kernfun=None, dK=None, dK_tilde=None, 
           dK_vec=None, K_tilde_inv=None, r=None):
        
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
                lambda_m, lambda_var, dlambda_m, dlambda_var = lambda_moments( x, K_tilde, KKtilde_inv, Kvec, K, C, m, V, theta, kernfun=kernfun, dK=dK, dK_tilde=dK_tilde, dK_vec=dK_vec, K_tilde_inv=K_tilde_inv)
                # Calculate the actual mean of the firing rate

                # feature 2 lambda0
                if r is not None:
                    f_params_temp = {'logA':f_params['logA'], 'lambda0': lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)}
                    f_mean = mean_f_given_lambda_moments( f_params_temp, lambda_m, lambda_var)
                else:
                    f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var )
                
                return f_mean, lambda_m, lambda_var, dlambda_m, dlambda_var
            
            else:
                lambda_m, lambda_var = lambda_moments( x, K_tilde, KKtilde_inv, Kvec, K, C, m, V, theta, kernfun=kernfun)
                # Calculate the actual mean of the firing rate

                # feature 2 lambda0
                if r is not None:
                    f_params_temp = {'logA':f_params['logA'], 'lambda0': lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)}
                    f_mean = mean_f_given_lambda_moments( f_params_temp, lambda_m, lambda_var)
                else:
                    f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var )
                return f_mean, lambda_m, lambda_var
        
        else:
            # Calculate the actual mean of the firing rate

            # feature 2 lambda0
            if r is not None:
                f_params_temp = {'logA':f_params['logA'], 'lambda0': lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)}
                f_mean = mean_f_given_lambda_moments( f_params_temp, lambda_m, lambda_var)
            else:   
                f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var )
            return f_mean

def lambda0_given_logA( logA, r, lambda_m, lambda_var):

    '''Given A, the optimal lambda0 value is in closed form, given by the solution of equation dL/dlambda0 = 0
        with L = loglikelihood.'''
    
    A = torch.exp(logA)

    sumr = r.sum()
    # This expression is basically the mean log firing rate f_mean withouth the exp(lambda0) factor
    expexpr = torch.exp( A*lambda_m + 0.5*A*A*lambda_var)
    sumexpr = expexpr.sum()

    lambda0 = torch.log( sumr ) - torch.log( sumexpr )

    return lambda0

def compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=False, dlambda_m=None, dlambda_var=None):
    # Returns the Sum of <loglikelihood> terms of the logmarginal loss as in (51) Notes for Pietro   
    # TODO handle the thwo cases better (with and without gradients). Its returning a tuple in one case and and not int he other

    # NB: A here is the firing rate parameter, not the receptive field Amplitude one
    A         = torch.exp(f_params['logA'])
    # lambda0   = f_params['lambda0']
    lambda0   = torch.exp(f_params['loglambda0']) if 'loglambda0' in f_params else f_params['lambda0']

    rlambda_m = r@lambda_m  
    sum_r     = torch.sum(r)

    loglikelihood     = A*rlambda_m + lambda0*sum_r - torch.sum(f_mean)

    # It looks like its not lambda_m that is causing a diverging gradent
    # print(f'computing likelihood with lambda_m.mean = {lambda_m.mean()} and lambda_var.mean = {lambda_var.mean()}')

    if compute_grad_for_f_params:
        # This is used when -loglikelihood is used loss for optimizer of f_params
        # Derivative of the loglikelihood with respect to the parameters of the firing rate
        dloglikelihood = {}
        # dloglikelihood['A'] = rlambda_m - torch.dot(lambda_m + A*lambda_var, f_mean) # Derivative of the loglikelihood with respect to A
        dloglikelihood['logA'] = A*(rlambda_m - torch.dot(lambda_m + A*lambda_var, f_mean))
        if 'loglambda0' in f_params:  dloglikelihood['loglambda0'] = (sum_r - sum(f_mean))*lambda0
        elif 'lambda0'  in f_params:  dloglikelihood['lambda0']    = sum_r - sum(f_mean)

        # dloglikelihood['tanhlambda0'] = (sum_r - sum(f_mean))*(torch.cosh(lambda0)*torch.cosh(lambda0))  # Derivative of the loglikelihood with respect to tanhlambda0 is dlogLK/dlambda0 * dlambda0/dtanhlambda0 = dlogLK/dlambda0 * cosh^2(lambda0)
        
        return loglikelihood , dloglikelihood

    if dlambda_m is not None and dlambda_var is not None:
        dloglikelihood = {}
        for key in dlambda_m.keys():
            # dloglikelihood[key] = r@dlambda_m[key] - A*f_mean@dlambda_m[key] - 0.5*A*A*f_mean@dlambda_var[key] 
            # In Matthews code:
            dloglikelihood[key] = A*r@dlambda_m[key] - A*f_mean@dlambda_m[key] - 0.5*A*A*f_mean@dlambda_var[key] 
        return loglikelihood , dloglikelihood    
    else:
        return loglikelihood, rlambda_m, sum_r # Only here i return these two values cause i need them in updateA

def compute_KL_div( m, V, K_tilde, K_tilde_inv, dK_tilde=None, ignore_warning=False):
    
    # Computes the Kullback Leiber divergence term of the complete loss function (marginal likelihood)
    # D_KL( q(lambda_tilde) || p(lambda_tilde) 

    # INPUTS
    # m : mean     of the variational distribution q(lambda_tilde) = N(m, V), shape (ntilde, 1)
    # V : variance of the variational distribution q(lambda_tilde) = N(m, V), shape (ntilde, ntilde)
    # K_tilde : matrix of kernel values K_tilde(x_i, x_j) for every inducing point x_i in x, shape (ntilde, ntilde)
    # ignore_warning : if True, it will ignore warnings about the matrix not being positive definite or simmetric.
    #                   it's used to ignore the ones that would arise for the reprojected V_b after the M step

    c = V @ K_tilde_inv
    # c_inv = V_inv @ K_tilde
    b = K_tilde_inv @ m
    
    # c = torch.linalg.solve(K_tilde, V)       # Shape (ntilde, ntilde)# This is " C " in samuele code written as V @ K_tilde_inv
    # b = torch.linalg.solve(K_tilde, m)       # Shape (ntilde, 1)
    # derivative with respect to theta
    
    KL = -0.5*log_det(V, name='V', ignore_warning=ignore_warning) + 0.5*log_det(K_tilde, name='K_tilde') + 0.5*torch.matmul(m.T, b) + 0.5*torch.trace(c) #0.5*torch.sum(1./torch.linalg.eig(c_inv)[0])

    if dK_tilde is not None:
        dKL = {}
        for key in dK_tilde.keys():
            B = dK_tilde[key]@K_tilde_inv # Shape (ntilde, ntilde

            dKL[key] = 0.5*torch.trace(B) - 0.5*torch.trace(c@B) - 0.5*b.T@(B@m)

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
            # loglikelihood[-1]: loglikelihood reached by the last iteration

            count = 0 
            flag = True

            #L = np.zeros((int(nit), 1))
            # L = torch.tensor.zeros((nit, 1))
            loglikelihood = torch.zeros((nit, 1))

            while flag:
                f_mean    =  mean_f( f_params=f_params, calculate_moments=False, lambda_m=lambda_m, lambda_var=lambda_var )

                # f_mean    = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)
                loglikelihood[count], rlambda_m, sum_r  = compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params)
                
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
                    print(f'All eigenval of hessian are nonpositive? {they_negative}, the loglikelihood is not concave in this point. Iteration: {count}')

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
                
            return f_params, loglikelihood[count-1], mean_f_given_lambda_moments( f_params, lambda_m, lambda_var )

def Estep( r, KKtilde_inv, m, f_params, f_mean, K_tilde=None, K_tilde_inv=None, V=None, update_V_inv=False, alpha=1):

    '''
    Updates the value of the mean vector and covariance matrix of the variational distribution q(lambda_tilde) = N(m, V).
    m and V should be of shape (ntilde, 1) and (ntilde, ntilde) respectively but for stability reasons quantities are projected onto a subspace
    of the eigenspace of K_tilde. This means that the m and V that are passed are really m_b and V_b.

    Updates can be made on 
    V:     (update_V_inv=False) This allows also smaller step sizes for the Newton update, regulated by alpha.
        Note that for updating V , a projection on the eigenspace of K_tilde has been assumed (Kb_diag must be the diagonal of a diagonal matrix )
        Also, when updating V with alpha=1 we are not using the current value of V so there is no need for it to be positive definite for the result to be.
        Since this is the only implemented one, a warning 
    V_inv: (update_V_inv=True)  This latter case does not need V or V_inv to be passed as an input, K_tilde and m are enough.
    '''
    # TODO: To make the update on V work also for a generic K_tilde (not projected, non diagonal), the following update should be implemented:

    # Note that the operator .* in matlab has the same effect of * here

    A = torch.exp(f_params['logA'])
    g = A * KKtilde_inv.T @ (r - f_mean)
    G = A*A * KKtilde_inv.T@(KKtilde_inv*f_mean[:,None]) # f_mean is a vector
    # G = f_params['A']*f_params['A'] * torch.einsum('ij,jk->ikj', KKtilde_inv.T, KKtilde_inv) @ f_mean # Shape (ntilde, ntilde) #same as this, above is like Matthew did. TODO: Check what is faster
   
    # Updates on V can be performed on V or V_inv. Results are the same but the former is more stable cause it does not ivert directly V, but rather solves a linear system.
    # The update on V also allows for a smaller step size regulated by alpha.
    # Results are still the best with alpha=1 (static images) but if the E step was to give problems try changing alpha.
    if update_V_inv == False and K_tilde is not None:
        if alpha==1:
            eye = torch.eye(K_tilde.shape[0], dtype=TORCH_DTYPE).to(DEVICE)
            V_new = torch.linalg.solve( eye +  K_tilde@G, K_tilde)
            m_new = V_new @ (G @ m + g)  # shape(250,1)
        else:
            warnings.warn(' You are using a step size different from 1 in Estep, in case the eigenspace of K_tilde has increased in dimension, you could have a non invertible V_b here. It might mean non positive definite V_new.')
            # We haven't proved that V_new is positive even when V is not. To avoid instabilities and crashes, I'd avoid alpha!=1 for now.
            V_new = V @ torch.linalg.solve( (1-alpha)*K_tilde + alpha*V + alpha*(K_tilde@G)@V ,  K_tilde)
            eye = torch.eye(m.shape[0], dtype=TORCH_DTYPE).to(DEVICE)
            m_new = m - alpha*(  torch.linalg.solve( (eye + K_tilde@G ) , ( m-K_tilde@g ) ) )

        V_new = (V_new + V_new.T) / 2 
        return m_new, V_new    

    elif update_V_inv == True and K_tilde_inv is not None and alpha==1:
        warnings.warn(' You are updating V_inv in Estep, not V. Some artifacts to its diagonal are being added.')
        V_inv_new = ( K_tilde_inv + G ) # shape (ntilde, ntilde)
        
        # This ugly control on the positive definiteness of V is also what makes updating V preferable
        eye = torch.eye(V_inv_new.shape[0], dtype=TORCH_DTYPE, device=DEVICE)
        V_inv_new = (V_inv_new + V_inv_new.T) / 2 + torch.finfo(TORCH_DTYPE).eps*1.e-7*eye # making sure it is symmetric
        try:
            V_new     = torch.linalg.inv(V_inv_new) # shape (ntilde, ntilde)
        except:
            warnings.warn('V_inv_new is not invertible in Estep')

        # m_new = torch.linalg.solve(V_inv_new , (G @ m + g))  #shape(ntilde)
        m_new = V_new @ (G @ m + g)  #shape(ntilde)
        eye = torch.eye(V_new.shape[0], dtype=TORCH_DTYPE, device=DEVICE)
        V_new = (V_new + V_new.T) / 2 + torch.finfo(TORCH_DTYPE).eps*1.e-7*eye # making sure it is symmetric
        return m_new, V_new
    else:
        warnings.warn('The update of V is not implemented for the inverse of V with alpha != 0 now in Estep')
        raise NotImplementedError
    
##################   Inference and Testing  ########################

def inference_and_correlation_cell(fit_model, X_test_avg, R_test_avg_cell):

    '''
    Does inference on the X_test_avg stimuli and calculates the correlation between the predicted and the true responses.

    Its just plot_final_and_intermediate_fit without the plots.
    '''

    # region _______ Inference ______
    # Calculate the matrices to compute the lambda moments. They are referred to the unseen images xstar
    kernfun    = fit_model['fit_parameters']['kernfun']
    xtilde     = fit_model['xtilde']

    C    = fit_model['C']
    mask = fit_model['mask']

    B             = fit_model['B']
    m_b           = fit_model['m_b']
    V_b           = fit_model['V_b']
    K_tilde_b     = fit_model['K_tilde_b']
    K_tilde_inv_b = fit_model['K_tilde_inv_b']

    theta_fit     = fit_model['hyperparams_tuple'][0]
    A_fit         = fit_model['f_params']['logA'].exp()
    lambda0_fit   = fit_model['f_params']['lambda0']

    Kvec = acosker(theta_fit, X_test_avg[:,mask], x2=None, C=C, dC=None, diag=True)
    K    = acosker(theta_fit, X_test_avg[:,mask], x2=xtilde[:,mask], C=C, dC=None, diag=False)
    K_b  = K @ B 

    lambda_m_t, lambda_var_t = lambda_moments( X_test_avg[:,mask], K_tilde_b, K_b@K_tilde_inv_b, Kvec, K_b, C, m_b, V_b, theta_fit, kernfun)  

    f_mean    = torch.exp(A_fit*lambda_m_t + 0.5*A_fit*A_fit*lambda_var_t  + lambda0_fit)

    r, r2 = calculate_correlation(R_test_avg_cell, f_mean, return_r2=True)

    return f_mean, r, r2

def lambda_moments_star( xstar, xtilde, C, theta, K_tilde, K_tilde_inv, m, V, B, kernfun):
    # Computes lambda_mean and lambda_var for a single test point xstar

    # B : is the matrix of eigenvectors of K_tilde corresponding to big eigenvelues
    #     all the kernels, m and V here are projected onto this subspace. The only one missing is the newly created Kvec_star (below)

    if kernfun == 'acosker': kernfun = acosker
    else: raise Exception('Kernel function not recognized')

    # Kvec_star is the covariance of the prior of the testing points
    Kvec_star = kernfun(theta, xstar, xtilde, C=C, dC=None, diag=False) # shape (nt, ntilde) in this case nt=1 (xstar is a single point)
    Kvec_star = Kvec_star @ B # All of the quantities in mu sig

    KKtilde_inv = Kvec_star @ K_tilde_inv

    mu_star =  KKtilde_inv @ m #

    # Scalar covariance between input xstar and itself. In VarGP it's a vector because it's calculated for all the training points. Here its only one point so its a scalar
    K_star = kernfun(theta, xstar, x2=None, C=C, dC=None, diag=True)              # shape (nt)]    


    # lambda_var = Kvec + torch.sum(-K.T*KKtilde_inv.T + KKtilde_inv.T*(V@KKtilde_inv.T), 0)
    sigma_star2 = K_star + torch.diag(KKtilde_inv@(V-K_tilde)@KKtilde_inv.T)

    return mu_star, torch.reshape(sigma_star2, (xstar.shape[0],))

def explained_variance(rtst, f_pred, sigma=True):

    # Compute the observed r2 for the sequence of images
    # rtst   = ( repetitions, nimages )
    # f_pred = ( nimages )

    # Even and odd repetitions of the same image, mean response. First index is repetitions
    reven = torch.mean(rtst[0::2,:], axis=0)
    rodd  = torch.mean(rtst[1::2,:], axis=0)

    # stacked_R = torch.stack( (r, f ) )
   
    reliability = torch.abs(torch.corrcoef( torch.stack((reven,rodd))))[0,1]
    accuracy_o  = torch.corrcoef(torch.stack((f_pred, rodd)))[0,1]
    accuracy_e  = torch.corrcoef(torch.stack((f_pred, reven)))[0,1]
    r2          = 0.5 * (accuracy_o + accuracy_e) / reliability

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

    R_predicted = R_predicted.cpu().numpy()
    rtst = rtst.cpu().numpy()

    # fig = plt.figure(figsize=(12, 8))

    # plot wothout using plt.
    fig = plt.figure(figsize=(6, 9),)  
    gs = fig.add_gridspec(5, 5,
                left=0.1, right=0.9, bottom=0.1, top=0.9,
                wspace=0.3, hspace=0.7)
    dt = 0.05
    time_values = dt * np.arange( len(R_predicted) )
    ax = fig.add_subplot(gs[3:, :])

    ax.plot(time_values, np.mean(rtst, axis=0) , 'k', label='Neural Activity', linewidth=1)
    ax.plot(time_values, R_predicted , color='red', label='GP Prediction')
    
    ax.legend(['data', 'GP'], loc='upper right', fontsize=14)

    # ax.errorbar(time_values, R_predicted / 0.05, yerr=np.sqrt(sigma_r2[:,0].cpu()) / 0.05, color='red')

    # ax.errorbar(time_values, R_predicted  , yerr=np.sqrt(sigma_r2.cpu()), color='red')

    ax.errorbar(time_values, R_predicted  , yerr=np.sqrt(R_predicted), color='red')

    # ax.legend(['data', 'GP'], loc='upper right', fontsize=14)
    txt = f'Pietro adjusted r^2 = {r2:.2f} ± {sigma_r2:.2f} Cell: {cellid}'
    ax.set_title(f'{txt}')
    ax.grid()
    ax.set_ylabel('Spike count')
    ax.set_xlabel('Test images')
    # plt.show()
    # plt.close()
    return fig

@torch.no_grad()
def test(X_test, R_test_cell, xtilde, X_train=None, at_iteration=None, print_expl_var=True, **kwargs):

    # X_test # shape (30,108,108,1) # nimages, npx, npx

    maxiter     = kwargs['fit_parameters'].get('maxiter', 0)
    nEstep      = kwargs['fit_parameters'].get('nEstep', 0)
    nMstep      = kwargs['fit_parameters'].get('nMstep', 0)
    kernfun     = kwargs['fit_parameters'].get('kernfun')
    cellid      = kwargs['fit_parameters'].get('cellid')
    n_px_side   = kwargs['fit_parameters'].get('n_px_side')
    # kernfun   = acosker if kernfun == 'acosker' else print('Kernel function not recognized')
    mask        = kwargs.get('mask')
    theta       = kwargs.get('hyperparams_tuple')[0]
    C           = kwargs.get('C')
    m           = kwargs.get('m_b')
    V           = kwargs.get('V_b')
    B           = kwargs.get('B')
    K_tilde     = kwargs.get('K_tilde_b')
    K_tilde_inv = kwargs.get('K_tilde_inv_b')
    f_params    = kwargs.get('f_params')
    theta_lower_lims  = kwargs.get('hyperparams_tuple')[1]
    theta_higher_lims = kwargs.get('hyperparams_tuple')[2]

    R_predicted = torch.zeros(X_test.shape[0])

    A        = torch.exp(f_params['logA'])
    if 'lambda0' in f_params.keys():    lambda0  = f_params['lambda0']
    if 'loglambda0' in f_params.keys(): lambda0  = torch.exp(f_params['loglambda0'])
    # loglambda0 = f_params['loglambda0']
    # lambda0    = torch.exp(loglambda0)

    if at_iteration is not None and X_train is not None:

        theta = {}
        for key, val in kwargs['values_track']['theta_track'].items():
            theta[key] = val[at_iteration]

        m         = kwargs['values_track']['variation_par_track']['m_b'][at_iteration]
        V         = kwargs['values_track']['variation_par_track']['V_b'][at_iteration]
        f_params    = kwargs['values_track']['f_par_track']
        logA        = f_params['logA'][at_iteration]
        A           = torch.exp(logA)
        if 'lambda0' in f_params.keys():    lambda0 = f_params['lambda0'][at_iteration]
        if 'loglambda0' in f_params.keys(): lambda0 = torch.exp(f_params['loglambda0'][at_iteration])

        if kernfun == 'acosker':
            kernfun = acosker

        # If execution was interrupted, the values of the Kernel have yet to be updated
        C, mask    = localker(theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=False)
        K_tilde    = kernfun(theta, xtilde[:,mask],  xtilde[:,mask], C=C, diag=False)             # shape (ntilde, ntilde)

        eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')                                   # calculates the eigenvals for an assumed symmetric matrix, eigenvalues  are returned in ascending order. Uplo=L uses the lower triangular part of the matrix. Eigenvectors are columns
        ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)                             # Keep only the largest eigenvectors

        B = eigvecs[:, ikeep]                                     # shape (ntilde, n_eigen)            

        # Project onto eigenspace but keep the names
        K_tilde = torch.diag(eigvals[ikeep])                    # shape (n_eigen, n_eigen)
        K_tilde_inv = torch.diag_embed(1/eigvals[ikeep])        # shape (n_eigen, n_eigen)

    for i in range(X_test.shape[0]):
        xstar = X_test[i,:,:,:]
        xstar = torch.reshape(xstar, (1, xstar.shape[0]*xstar.shape[1]))

        kernfun = 'acosker'
        mu_star, sigma_star2 = lambda_moments_star(xstar[:,mask], xtilde[:,mask], C, theta, K_tilde, K_tilde_inv, m, V, B, kernfun)

        rate_star = torch.exp( A*mu_star + 0.5*A*A*sigma_star2 + lambda0 ) 

        R_predicted[i] = rate_star #ends up being of shape 30
        # print(f'rate_star: {rate_star.item():.4f}')


    expl_var, sigma_expl_var = explained_variance( R_test_cell, R_predicted, sigma=True)

    # Print the results
    R_pred_cell = R_predicted

    if print_expl_var:
        print(f"\n Explained variance: R2 = {expl_var:.2f} ± {sigma_expl_var:.2f} Cell: {cellid} maxiter = {maxiter}, nEstep = {nEstep}, nMstep = {nMstep} ")


    return R_test_cell, R_pred_cell, expl_var, sigma_expl_var

def calculate_correlation(observed, predicted, return_r2=False):
    """Calculate Pearson correlation between observed and predicted values"""
    
    # Ensure same device
    if observed.device != predicted.device:
        predicted = predicted.to(observed.device)
    # print(f'Calculating correlation on device: {observed.device}')

    # Calculate means
    obs_mean = observed.mean()
    pred_mean = predicted.mean()
    
    # Calculate correlation coefficient
    numerator = ((observed - obs_mean) * (predicted - pred_mean)).sum()
    denominator = torch.sqrt(((observed - obs_mean)**2).sum() * ((predicted - pred_mean)**2).sum())
    
    r = numerator / denominator
    
    if return_r2:
        return r, r**2
    return r
##########################################################

@torch.no_grad()
def varGP_original(x, r, **kwargs):
    #region _______ infos __________
    # Learn the hyperparameters of the GP model

    # INPUTS:

    # x = [nt, nx], stimulus
    # r = [1, n],  spike counts
    
    # OPTIONAL INPUTS:
    
    # ntilde: number of inducing data points if they are not provided as argument(increses approximation accuracy )
    # xtilde: inducing points [nx, ntilde]
    
    # nEstep:  number of  of iterations in E-step (updating m, and V)
    # nMstep:  number of steps in M-step (updating theta )
    # Display_hyper: Bool to display the initial hyperparameters
    
    # kernfun: kernel function ( acosker is default)
    # theta: dict of initial hyperparameters of the kernel
    # lb, ub: lower and upper bound for theta
    # m, V: initial mean and variance of variational distribution, q(lambda) = N(m, V). Requires_gradietn is set to False cause in the M step i keep them fixed
     
    # RETURNS
    
    # theta,
    # f_par, 2 Parameters of the firing rate (logA and lambda_0) in the paper, we update logA cause udpading A is not stable, since its in the exponent of the firing rate f_mean
    # m, V (tensors tracking the values of all of the above)
    # xtilde: set of inducing datapoints
    # L, loss function during learning

    # values_track:
    #     - ['loss_track']
    #         - ['logmarginal']
    #         - ['loglikelihood']
    #         - ['KL']
    #     - ['theta_track']
    #        - ['sigma_0']
    #        - ['eps_0x']
    #        - ['eps_0y']
    #        - ['-2log2beta']
    #        - ['-log2rho2']
    #        - ['Amp']
    #     - ['f_par_track'] 
    #        - ['logA']
        #    - ['loglambda0']   
    #     - ['variation_par_track']
    #        - ['V_b']
    #        - ['m_b']




    #endregion
    
    #region ________ Initialization __________
    start_time_before_init = time.time()
    err_dict = {'is_error': False, 'error_message': None}

    # number of pixels, number of training points 
    nt, nx = x.shape 

    # Update the parameters of the fit with the used global variables
    fit_parameters = copy.deepcopy(kwargs['fit_parameters'])
    fit_parameters['min_tolerance'] = MIN_TOLERANCE
    fit_parameters['eigval_tol']    = EIGVAL_TOL

    ntilde        = fit_parameters.get('ntilde',  100 if nt>100 else nt) # if no ntilde is provided try with 100, otherwise inducing points=x   
    maxiter       = fit_parameters.get('maxiter', 50)
    nEstep        = fit_parameters.get('nEstep',  50) 
    nMstep        = fit_parameters.get('nMstep',  20)
    nFparamstep   = fit_parameters.get('nFparamstep', 10)
    lr_Mstep      = fit_parameters.get('lr_Mstep', 0.1)
    lr_Fparamstep = fit_parameters.get('lr_Fparamstep', 0.1)
    display_hyper = fit_parameters.get('display_hyper', True)
    n_px_side     = fit_parameters.get('n_px_side', math.sqrt(nx))
    kernfun       = fit_parameters.get('kernfun', 'acosker')
    if kernfun == 'acosker': kernfun = acosker
    else: raise Exception('Kernel function not recognized')


    # Initialize hyperparameters of Kernel and parameters of the firing rate
    # Mutable objects are copied otherwise their values would be updated in the original args dictionary sent as argument

    xtilde            = kwargs['xtilde'] if 'xtilde' in kwargs else generate_xtilde(ntilde, x)
    if ntilde        != xtilde.shape[0]: raise Exception('Number of inducing points does not match ntilde')
    hyperparams_tuple = copy.deepcopy(kwargs['hyperparams_tuple']) if 'hyperparams_tuple' in kwargs.keys() else generate_theta(x, r, n_px_side, display_hyper)
    theta             = copy.deepcopy(kwargs.get( 'theta',             hyperparams_tuple[0]) )
    theta_lower_lims  = copy.deepcopy(kwargs.get( 'theta_lower_lims',  hyperparams_tuple[1] ))
    theta_higher_lims = copy.deepcopy(kwargs.get( 'theta_higher_lims', hyperparams_tuple[2] ))

    if 'f_params' not in kwargs.keys():
        raise Exception('f_params not provided')
    f_params          = copy.deepcopy(kwargs['f_params']) 
    # f_params          = copy.deepcopy(kwargs['f_params']) if 'f_params' in kwargs.keys() else {'logA': torch.log(torch.tensor(0.0001)), 'lambda0':torch.tensor(1)}
    for key in f_params.keys(): f_params[key] = f_params[key].requires_grad_(True)
    
    # f_params          = copy.deepcopy(kwargs.get( 'f_params', {'logA': torch.log(torch.tensor(0.0001)), 'lambda0':torch.tensor(-1)} )) # Parameters of the firing rate (A and lambda_0) in the paper
    # f_params          = copy.deepcopy(kwargs.get( 'f_params', {'logA': torch.log(torch.tensor(0.0001,)), 'loglambda0':torch.log(torch.tensor(-1))} )) # Parameters of the firing rate (A and lambda_0) in the paper

    # Calculate the part of the kernel responsible for implementing smoothness and the receptive field
    # TODO Calculate it only close to the RF (for now it's every pixel)

    # The following lines initialize the kernel values. 
    # They take care of setting the kernel of the whole dataset equal to the kernel on the inducing points (K_tilde) the same if the inducing points are the whole dataset
    # They also dont calculate the kernel if its starting values are passed as an argument
    # They also take care of projecting the kernel into the eigenspace of the largest eigenvectors of K_tilde
    C, mask = localker(theta=theta, theta_lower_lims=theta_lower_lims, theta_higher_lims=theta_higher_lims, n_px_side=n_px_side, grad=False) if 'init_kernel' not in kwargs else (kwargs['init_kernel']['C'], kwargs['init_kernel']['mask'])
    K_tilde = kernfun(theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)                                                       if 'init_kernel' not in kwargs else kwargs['init_kernel']['K_tilde']
    
    if ntilde != nt:  K  = kernfun(theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False) if 'init_kernel' not in kwargs else kwargs['init_kernel']['K']       # shape (nt, ntilde) set of row vectors K_i for every input 
    else:             K  = K_tilde
    
    Kvec = kernfun(theta, x[:,mask], x2=None, C=C, dC=None, diag=True)                         if 'init_kernel' not in kwargs else kwargs['init_kernel']['Kvec']    # shape (nt)
    if 'init_kernel' not in kwargs:
        eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')                                # calculates the eigenvals for an assumed symmetric matrix, eigenvalues  are returned in ascending order. Uplo=L uses the lower triangular part of the matrix. Eigenvectors are columns
        ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)                          # Keep only the largest eigenvectors

    B = eigvecs[:, ikeep]                                 if 'init_kernel' not in kwargs else kwargs['init_kernel']['B']             # shape (ntilde, n_eigen)            
    # make K_tilde_b and K_b a projection of K_tilde and K into the eigenspace of the largest eigenvectors
    K_tilde_b = torch.diag(eigvals[ikeep])                if 'init_kernel' not in kwargs else kwargs['init_kernel']['K_tilde_b']     # shape (n_eigen, n_eigen)
    K_b       = K @ B                                     if 'init_kernel' not in kwargs else kwargs['init_kernel']['K_b']           # shape (3190, n_eigen)


    K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep])    if 'init_kernel' not in kwargs else kwargs['init_kernel']['K_tilde_inv_b'] # shape (n_eigen, n_eigen)
    # K_tilde_inv_p = torch.diag_embed(1/eigvals)           if 'init_kernel' not in kwargs else kwargs['init_kernel']['K_tilde_inv_p'] # If we want to invert the whole K_tilde, not only the projected one, maybe outside VarGP for active loop
    if ntilde != nt:  KKtilde_inv_b = K_b @ K_tilde_inv_b if 'init_kernel' not in kwargs else kwargs['init_kernel']['KKtilde_inv_b'] # shape (nt, n_eigen) # this is 'a' in matthews code
    else:             KKtilde_inv_b = B                                                                                              # the resulting matrix of Ktildeb @ B @ B.T @ Ktildeb_inv @ B = B  

    # We always pass the non projected variational parameters because the dimensionality of the problem is determined by the lines above ( B ).
    m = copy.deepcopy(kwargs.get('m', torch.zeros( (ntilde) )).detach())
    V = copy.deepcopy(kwargs.get('V', K_tilde ).detach())

    # m_b = copy.deepcopy(kwargs.get('m_b', torch.zeros( (ntilde) )).detach())
    # V_b = copy.deepcopy(kwargs.get('V_b', K_tilde_b ).detach())

    V_b = B.T @ V @ B if 'V' in kwargs else K_tilde_b     # shape (n_eigen, n_eigen)
    m_b = B.T @ m 

    lambda_m, lambda_var = lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta, kernfun=kernfun)
    f_mean               = mean_f_given_lambda_moments(f_params, lambda_m, lambda_var)

    loglikelihood, _, __ = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=False)
    KL_div               = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None, ignore_warning=True )
    logmarginal          = loglikelihood - KL_div   

    # Tracking dictionary
    loss_track          = {'logmarginal'  : torch.zeros((maxiter)),                          # Loss to  maximise: Log Likelihood - KL
                            'loglikelihood': torch.zeros((maxiter)),
                            'KL'           : torch.zeros((maxiter)),
                            } 
    theta_track         = {key : torch.zeros((maxiter)) for key in theta.keys()}
    f_par_track         = {'logA': torch.zeros((maxiter)), 'lambda0': torch.zeros((maxiter))} if 'lambda0' in f_params else {'logA': torch.zeros((maxiter)), 'loglambda0': torch.zeros((maxiter))}
    # f_par_track         = {'logA': torch.zeros((maxiter)), 'loglambda0': torch.zeros((maxiter))} # track hyperparamers

    variation_par_track = {'V_b': (), 'm_b': ()}               # track the variation parameters
    # subspace_track      = {'eigvals': torch.zeros((maxiter, K_tilde.shape[0])), 
                            # 'eigvecs': torch.zeros((maxiter, *tuple(K_tilde.shape)))  }        # track the eigenvectors of the kernel
    values_track        = {'loss_track':      loss_track,   'theta_track': theta_track, 
                            'f_par_track':    f_par_track,  'variation_par_track': variation_par_track}
                            # 'subspace_track': subspace_track }
    
    # print(f'Initialization took: {(time.time()-start_time_before_init):.4f} seconds\n')

    #region _________ Memory usage___________
    # memory = 0
    # for dict in values_track.values():
    #     for key in dict.keys():
    #         if isinstance(dict[key], tuple):
    #             for i in range(len(dict[key])):
    #                 memory += dict[key][i].element_size() * dict[key][i].nelement()
    #             # print(f'{key} memory: {memory / (1024 ** 2):.2f} MB')
    #         else:
    #             memory += dict[key].element_size() * dict[key].nelement()
    #             # print(f'{key} memory: {dict[key].element_size() * dict[key].nelement() / (1024 ** 2):.2f} MB')

    # # Convert bytes to megabytes (MB)
    # total_memory_MB = memory / (1024 ** 2)
    # print(f'Total values_track memory on GPU: {total_memory_MB:.2f} MB')
    # # Allocated memory
    # allocated_bytes = torch.cuda.memory_allocated()
    # allocated_MB = allocated_bytes / (1024 ** 2)
    # print(f"\nAfter initialization Allocated memory: {allocated_MB:.2f} MB")

    # # Reserved (cached) memory
    # reserved_bytes = torch.cuda.memory_reserved()
    # reserved_MB = reserved_bytes / (1024 ** 2)
    # print(f"\nAfter initialization Reserved (cached) memory: {reserved_MB:.2f} MB")
    #endregion _________ Memory usage___________

    #endregion ______________________________
    try: 
        # Loop variables
        start_time_loop        = time.time()
        time_estep_total       = 0
        time_f_params_total    = 0
        time_mstep_total       = 0
        time_computing_kernels = 0
        time_computing_loss    = 0
        time_lambda0_estimation= 0

        #region ________________ Initialize tracking dict ______________________

        # Update the tracking dictionaries. Remember that mutable objects are passed by reference so any modification to them would reflect in the dictionary if we dont copy
        values_track['loss_track']['loglikelihood'][0].copy_(loglikelihood)
        values_track['loss_track']['KL'][0].copy_(KL_div)
        values_track['loss_track']['logmarginal'][0].copy_(loglikelihood-KL_div)

        print(f'Initial Loss: {-(loglikelihood-KL_div):.4f}')

        # Theta before the Mstep of 0 "i" is the one used to build the kernel of the E-step of 0 "i+1". 
        # The theta we are saving here is the one we just used.
        for key in theta.keys():
            values_track['theta_track'][key][0].copy_(theta[key])

        values_track['f_par_track']['logA'][0].copy_(f_params['logA'])
        if 'lambda0' in f_params:
            values_track['f_par_track']['lambda0'][0].copy_(f_params['lambda0'])
        elif 'loglambda0' in f_params:
            values_track['f_par_track']['loglambda0'][0].copy_(f_params['loglambda0'])

        values_track['variation_par_track']['V_b'] += (V_b.clone(),)
        values_track['variation_par_track']['m_b'] += (m_b.clone(),)
        #endregion
        
        #_______________________ Main Loop ___________
        for iteration in range(1,maxiter):

            # print(f'*Iteration*: {iteration}', end='')

            #region ________________ Computing Kernel and Stabilization____________________
            # Compute starting Kernel, if no M-step -> only compute it once cause it's not changing
            start_time_computing_kernels = time.time()
            if nMstep > 0 and iteration > 1:
                #________________ Compute the KERNELS after M-Step and the inverse of V _____________
                C, mask    = localker(theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=False)                
                K_tilde    = kernfun( theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)                                   # shape (ntilde, ntilde)
                K          = kernfun( theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)      if ntilde != nt else K_tilde      # shape (nt, ntilde) set of row vectors K_i for every input 
                Kvec       = kernfun( theta, x[:,mask], x2=None, C=C, dC=None, diag=True)                                                # shape (nt)
                
                eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')                                # calculates the eigenvals for an assumed symmetric matrix, eigenvalues  are returned in ascending order. Uplo=L uses the lower triangular part of the matrix. Eigenvectors are columns
                ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)                          # Keep only the largest eigenvectors
            
                B_old = B 
                B = eigvecs[:, ikeep]                                                  # shape (ntilde, n_eigen)            
                # make K_tilde_b and K_b a projection of K_tilde and K into the eigenspace of the largest eigenvectors
                K_tilde_b     = torch.diag(eigvals[ikeep])                                 # shape (n_eigen, n_eigen)
                # K_tilde_inv_p = torch.diag_embed(1/eigvals)                              # We keep the latest inverse of the complete K_tilde just to return it and be fast in computing the inverse of the incremented K_tilde if needed ( active learning )
                K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep])                     # shape (n_eigen, n_eigen)
                K_b           = K @ B                                                      # shape (3190, n_eigen)
                KKtilde_inv_b = K_b @ K_tilde_inv_b if ntilde != nt else B             # shape (nt, n_eigen) # this is 'a' in matthews code                         
        
                # In the following iterations we already have V_b (maybe updated in an E-step) and if Mstep > 0 we changed the eigenspace,
                # Get V_b_new referring to this new eigenspace as:
                #       V_b_new = (B_new.T@V)@B_new 
                # Where V       = B_old@V_b_old@B_old.T, so

                # V_b_new       = B_new.T@(B_old@V_b_old@B_old.T)@B_new and
                # m_b_new       = B_new.T @ B_old @ m_b_old 

                # Note that we might have augmented the dimension of the eigenspace, this might leave very small eigenvalues in V_b_new
                # This will not be necessaraly invertible (or posdef ). This might be problem in the Estep when using alpha != 1.
                # This matrix is not numerically simmetric for precision higher than 1.e-13 even if it should be arount 1.e-15, hence the choice of MIN_TOLERANCE 1.e-13
                # V_b is guaranteed to be simmetric (and posdef) only when coming out of E step
                # It will be used only in each first estep iteration. To calculate the lambda moments. It never gave numerical problems but might be a source loss of precision
                V_b_new = B.T@(B_old@V_b@B_old.T)@B                   
                V_b     = V_b_new                                     

                # smallest_eig = torch.linalg.eigh(V_b)[0].min()
                # if smallest_eig <= 0.:
                    # warnings.warn(f'Matrix V_b is simmetric but has an eigenvalue smaller than 0 ')

                m_b_new = B.T @ B_old @ m_b
                m_b     = m_b_new

            time_computing_kernels += time.time() - start_time_computing_kernels
            #endregion 

            #region  _______________ Control over possible Nans ______
            # for tensor in [C, K_tilde_b, K_b, KKtilde_inv_b, V_b, m_b, f_params['logA'], f_params['lambda0']]:
            # for tensor in [C, K_tilde_b, K_b, KKtilde_inv_b, V_b, m_b, f_params['logA'], f_params['loglambda0']]:                
            # # for tensor in [C, K_tilde_b, K_b, KKtilde_inv_b, V_b, m_b, f_params['logA'], f_params['tanhlambda0']]:                                
            #     if torch.any(torch.isnan(tensor)):
            #         variable_name = [k for k, v in locals().items() if v is tensor][0]
            #         raise ValueError(f'NaN in {variable_name}')
            #     if torch.any(torch.isinf(tensor)):
            #         variable_name = [k for k, v in locals().items() if v is tensor][0]
            #         raise ValueError(f'Inf in {variable_name}')
            #endregion
            
            #region ________________ E-Step : Update on m & V and f(lambda) parameters ________
            start_time_estep = time.time()
            if nEstep > 0:
                # print(f'Estep in iteration {iteration}')

                for i_estep in range(nEstep):
                    # print(f'   Estep n {i_estep}')

                    # Update lambda moments only if the kernel has changed or if it's the first iteration
                    # They are update again after the Estep

                    if i_estep == 0 and nMstep > 0:
                        lambda_m, lambda_var = lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta, kernfun=kernfun)  

                        # feature 2: lambda0
                        # f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)

                    # We should also count this in the time for the f_params update, the f_mean computation would not be here if there was no update
                    f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var) # Since f_params influece f_mean, we need to update it at each estep

                    #region ____________ Update m, V ______________
                    m_b, V_b = Estep( r=r, KKtilde_inv=KKtilde_inv_b, m=m_b, f_params=f_params, f_mean=f_mean, 
                                        K_tilde=K_tilde_b, K_tilde_inv=K_tilde_inv_b, update_V_inv=False, alpha=1  ) # Do not change udpate_V_inv or alpha, read Estep docs

                    # And the things that depend on them ( moments of lambda )
                    f_mean, lambda_m, lambda_var  =  mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], 
                                                            K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, 
                                                            K=K_b, C=C, m=m_b, V=V_b, theta=theta, kernfun=kernfun, 
                                                            lambda_m=None, lambda_var=None  )
                    #endregion

                    #region ____________ Update f_params ______________ 
                    start_time_f_params = time.time()
                    f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)

                    # lr_f_params = 0.01 # learning rate
                   #lr_Fparamstep = 0.1  # what we use usually
                    # lr_f_params = 1
                    optimizer_f_params = torch.optim.LBFGS([f_params['logA']], lr=lr_Fparamstep, max_iter=nFparamstep, 
                                                            tolerance_change=1.e-9, tolerance_grad=1.e-7,
                                                            history_size=nFparamstep, line_search_fn='strong_wolfe')
                    CLOSURE2_COUNTER = [0]
                    @torch.no_grad()
                    def closure_f_params( ):
                        CLOSURE2_COUNTER[0] += 1
                        optimizer_f_params.zero_grad()
                        nonlocal f_mean          # Update f_mean of the outer scope each time the closure is called
                        # Lambda0 feature 3

                        # Each time the closure is called the optimizer expects the value of the loss. It might be using it to explore how big of a step to take (line search) or actually updating the parameters ( logA)
                        # We need the optimizer to evaluate the loss with the optimal lambda0 parameter given logA, so we update it here, before computing all the other things that depend on it.

                        f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)
                        f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)   

                        loglikelihood, dloglikelihood = compute_loglikelihood(  r,  f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=True )                       
                        # print(f' -logmarginal = {(-loglikelihood.item() + KL.item()):.4f} -loglikelihood = {-loglikelihood.item():.4f}  KL = {KL.item():.4f}')

                        # Update gradients of the loss with respect to the firing rate parameters
                        # The minus here is because we are minimizing the negative loglikelihood
                        f_params['logA'].grad    = -dloglikelihood['logA']    #if f_params['logA'].requires_grad else None


                        if 'lambda0' in f_params:
                            f_params['lambda0'].grad = -dloglikelihood['lambda0']        if f_params['lambda0'].requires_grad else None
                        elif 'loglambda0' in f_params:
                            f_params['loglambda0'].grad = -dloglikelihood['loglambda0']  if f_params['loglambda0'].requires_grad else None
 
                        # if torch.any(torch.isnan(f_mean)):
                            # raise ValueError(f'Nan in f_mean during f param update in Estep, closure has been called {CLOSURE2_COUNTER[0]} times in estep {i_estep} iteration. Try substituting them with inf.')
                        # if  torch.any( f_mean > 1.e4):
                            # raise ValueError(f'f_mean is too large in Estep, closure has been called {CLOSURE2_COUNTER[0]} times in estep {i_estep} iteration')

                        if f_mean.mean() > 100 or torch.any(torch.isnan(f_mean)):
                            print(f'f_mean mean is {f_mean.mean()} at i_step {i_estep} iteration {iteration} at closure call {CLOSURE2_COUNTER[0]}, returning infinite loss')
                            return torch.tensor(float('inf'))
                        
                        return -loglikelihood

                    optimizer_f_params.step(closure_f_params)        
                    
                    f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var) # the optimal logA value found by the optimizer might not be the one used in the last closure call. We need to make sure lambda0 is updated.

                    time_f_params_total += time.time()-start_time_f_params
                    #endregion
            else: print('No E-step')

            time_estep = time.time()-start_time_estep
            time_estep_total += time_estep 
            # print(f'\r*Iteration*: {iteration:>3} E-step took: {time_estep:.4f}s', end='')
            #endregion                       
        
            #region ________________ Update the tracking dictionaries _______________  

            # Update the value every x iterations. 
            # We are doing it here to avoid having to project V_b and m_b to the updated eigenspace. 
            # This might pose numerical problem if alpha!=1 as explained in the docs, 
            # But even in the alpha=1 case the value of the almost singular reprojected V_b would be used for the loss, leading to expliding values sometimes
            #  ( its only a tracking problem ) 
            
            if iteration % 1 == 0 or iteration == maxiter-1:

                start_time_computing_loss = time.time()
                # lambda_m, lambda_var = lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta, kernfun=kernfun)
                
                f_mean               = mean_f_given_lambda_moments(f_params, lambda_m, lambda_var)

                loglikelihood, _, __ = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=False)

                KL_div               = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None, ignore_warning=True )
                logmarginal          = loglikelihood - KL_div   

                time_computing_loss += time.time()-start_time_computing_loss
                # print(f" TOT time for loss computation at iteration {iteration:>2}: {tot_elapsed_time:>2.6f} s")
                # print(f"     time for loss computation at iteration {iteration:>2}: {elapsed_time:>2.6f} s")

            # Update the tracking dictionaries. Remember that mutable objects are passed by reference so any modification to them would reflect in the dictionary if we dont copy
            values_track['loss_track']['loglikelihood'][iteration].copy_(loglikelihood)
            values_track['loss_track']['KL'][iteration].copy_(KL_div)
            values_track['loss_track']['logmarginal'][iteration].copy_(loglikelihood-KL_div)

            # Theta before the Mstep of iteration "i" is the one used to build the kernel of the E-step of iteration "i+1". 
            # The theta we are saving here is the one we just used.
            for key in theta.keys():
                values_track['theta_track'][key][iteration].copy_(theta[key])

            values_track['f_par_track']['logA'][iteration].copy_(f_params['logA'])
            if 'lambda0' in f_params:
                values_track['f_par_track']['lambda0'][iteration].copy_(f_params['lambda0'])
            elif 'loglambda0' in f_params:
                values_track['f_par_track']['loglambda0'][iteration].copy_(f_params['loglambda0'])                


            values_track['variation_par_track']['V_b'] += (V_b.clone(),)
            values_track['variation_par_track']['m_b'] += (m_b.clone(),)


            print(f'Loss iter {iteration}: {-(loglikelihood-KL_div):.4f}')

            # region _________ Check loss stabilization __________
            # If loss hasn't changed in the last 5 iterations, break the loop
            if iteration >= 5:
                # Get the loss values for the last 5 iterations
                recent_losses = [values_track['loss_track']['logmarginal'][i] for i in range(iteration-4, iteration+1)]
                recent_losses_tensor = torch.tensor(recent_losses)
                loss_range = torch.abs(recent_losses_tensor.max() - recent_losses_tensor.min())
                if loss_range < LOSS_STOP_TOL:
                    print(f'Loss stabilization detected (loss range {loss_range.item():.2e} < tolerance {LOSS_STOP_TOL:.2e}). Stopping training.')
                    raise LossStagnationError(f'Loss stabilization detected (loss range {loss_range.item():.2e} < tolerance {LOSS_STOP_TOL:.2e}). Stopping training.')
            # endregion

            #endregion

            #region ________________ M-Step : Update on hyperparameters theta  ________________

            start_time_mstep = time.time()
            if nMstep > 0 and iteration < maxiter-1: 
                # Skip the M-step in the last iteration to avoid generating a new eigenspace that will not be used by V and m

                print(f' Mstep of iteration {iteration}')
                if iteration > 1:
                    del optimizer_hyperparams
                optimizer_hyperparams = torch.optim.LBFGS(theta.values(), lr=lr_Mstep, max_iter=nMstep, line_search_fn='strong_wolfe', 
                                                          tolerance_change=1.e-9, tolerance_grad=1.e-7, history_size=100)
    
                CLOSURE2_COUNTER = [0]
                @torch.no_grad()
                def closure_hyperparams( ):
                    CLOSURE2_COUNTER[0] += 1
                    optimizer_hyperparams.zero_grad()
                    # if any hyperparameter is out of bounds, return infinite loss to signal the optimizer to revaluate the step size
                    return_infinite_loss = False
                    for key, value in theta.items():
                        if not (theta_lower_lims[key] <= value <= theta_higher_lims[key]):
                            return_infinite_loss = True
                            print(f"{key} = {value:.4f} is not within the limits of {theta_lower_lims[key]} and {theta_higher_lims[key]}, returning infinite loss in closure call {CLOSURE2_COUNTER[0]}")
                            if theta[key].requires_grad:
                                theta[key].grad = torch.tensor(float('inf'))
                    if return_infinite_loss: return torch.tensor(float('inf'))

                    C, mask, dC       = localker(theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=True)
                    K_tilde, dK_tilde = kernfun( theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=dC, diag=False)
                    K, dK             = kernfun( theta, x[:,mask], xtilde[:,mask], C=C, dC=dC, diag=False) if ntilde != nt else (K_tilde, dK_tilde) 
                    Kvec, dKvec       = kernfun( theta, x[:,mask], x2=None, C=C, dC=dC, diag=True) 

                    #region ____________Stabilization____________________
                    # Note on Stabilization
                    # The eigenvector matrix is not recalculated during the M-step. 
                    # This is not entirely precise because a change in hyperparameters could change the eigenvalues 
                    # over the threshold (and therefore change the dimension of the subspace I'm projecting onto)
                    # But this most likely has a minimal effect. And it saves nMstep eigenvalue decompositions per iteration.
                    # NOTE that even if I am saving resources by not recalculating the eigenspace of K_tilde, I still have to recalculate the inverse of K_tilde in the M-step... still On^3

                    # Projecting the Kernel into the same eigenspace used in the E-step (its not changing with the changing hyperparameters/Kernel)
                    K_tilde_b = B.T@K_tilde@B                 # Projection of K_tilde into eigenspace (n_eigen,n_eigen) 
                    K_tilde_b = (K_tilde_b + K_tilde_b.T)*0.5 # make sure it is symmetric
                    K_b  = K @ B                              # Project K into eigenspace, shape (3190, n_eigen)

                    # If eigenspace B has been recalculated, one has to reproject m and V into the new eigenspace
                    # V_b_new = B.T@(B_old@V_b@B_old.T)@B
                    # V_b     = V_b_new
                    # m_b_new = B.T @ B_old @ m_b
                    # m_b = m_b_new

                    # Projection of the gradients of the Kernel into the eigenspace
                    dK_tilde_b, dK_b = {}, {}
                    for key in dK_tilde.keys():
                        dK_tilde_b[key] = B.T@dK_tilde[key]@B
                        dK_b[key]       = dK[key] @ B                     
                    #endregion

                    # K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep]) # shape (n_eigen, n_eigen) To use if I have recalculated the eigenspace of K_tilde
                    # NOTE that even if I am saving resources by not recalculating the eigenspace of K_tilde, I still have to recalculate the inverse of K_tilde in the M-step... still On^3
                    eye = torch.eye(K_tilde_b.shape[0], device=DEVICE, dtype=TORCH_DTYPE)
                    K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye)
                    KKtilde_inv_b = K_b @ K_tilde_inv_b if ntilde != nt else B

                    f_mean, lambda_m, lambda_var, dlambda_m, dlambda_var  =  mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, K=K_b,  
                                                                                C=C, m=m_b, V=V_b, theta=theta, kernfun=kernfun, lambda_m=None, lambda_var=None, dK=dK_b, dK_tilde=dK_tilde_b, dK_vec=dKvec, K_tilde_inv=K_tilde_inv_b) # Shape (nt
                    
                    # feature 2: lambda0
                    # lambda0_estimation_start_time = time.time()
                    # temp_f_params = {'logA':f_params['logA'], 'lambda0':lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)}
                    # f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)
                    # time_lambda0_estimation += time.time()-lambda0_estimation_start_time

                    # loglikelihood, dloglikelihood = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, temp_f_params, dlambda_m=dlambda_m, dlambda_var=dlambda_var )
                    loglikelihood, dloglikelihood = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, f_params, dlambda_m=dlambda_m, dlambda_var=dlambda_var )
                    KL, dKL                       = compute_KL_div(m_b, V_b, K_tilde_b, K_tilde_inv=K_tilde_inv_b, dK_tilde=dK_tilde_b)
                    logmarginal                   = loglikelihood - KL
                    
                    l = -logmarginal
                    # print(f' {l.item():.4f} = logmarginal in m-step - closure call {CLOSURE2_COUNTER[0]}')   


                    # Dictionary of gradients of the -loss with respect to the hyperparameters, to be assigned to the gradients of the parameters
                    # Update the gradients of the loss with respect to the hyperparameters ( its minus the gradeints of the logmarginal)
                    dlogmarginal = {}
                    for key in theta.keys():
                        dlogmarginal[key] = dloglikelihood[key] - dKL[key]
                        if theta[key].requires_grad:
                            theta[key].grad = -dlogmarginal[key]
                        # dlogmarginal[key] = -dKL[key]
                        # if theta[key].requires_grad:
                        #         theta[key].grad = -dlogmarginal[key]                            

                    # print_hyp(theta)
            
                    # In case you want to implement the simplest gradient descent, you can use this and call closure_hyperparams() directly   
                    # for key in theta.keys():
                    #     if theta[key].requires_grad:
                    #         theta[key] = theta[key] - 0.0001*theta[key].grad
                    # print(f'   m-step loss: {-logmarginal.item():.4f}')
                    # return -KL
                    return l
            
                optimizer_hyperparams.step(closure_hyperparams)

                # mstep_time = time.time()-start_time_mstep
                # print(f'\r*Iteration*: {iteration:>3} E-step took: {time_estep:.4f}s, M-step took: {mstep_time:.4f}s', end= '\n')

            else: 
                if iteration < maxiter-1: print(' No M-step')
            time_mstep        = time.time()-start_time_mstep
            time_mstep_total += time_mstep
            #endregion __________________________________________

    except KeyboardInterrupt as e:

        print(' ===================  Interrupted  ===================\n')
        print(f'During iteration: {iteration}, there should be {iteration} completed iterations')

        #region _________ Adjust to the last available values _________
        fit_parameters['maxiter'] = iteration
        if fit_parameters['maxiter'] <= 1: 
            print('Too few iterations iterations were done to save')
            err_dict['is_error'] = True
            err_dict['error'] = e    
            raise e

        last_theta = {}
        for theta_key in theta.keys():
            last_theta[theta_key] = values_track['theta_track'][theta_key][iteration-1] # We go back 2 steps cause that is the value of theta for which f_params were optimized 
        theta = last_theta                                                              # and eigenvectors were calculated ( therefore onto which the last used V-b was projected )

        f_params['logA']    = values_track['f_par_track']['logA'][iteration-1]
        if 'lambda0' in f_params:
            f_params['lambda0'] = values_track['f_par_track']['lambda0'][iteration-1]
        elif 'loglambda0' in f_params:
            f_params['loglambda0'] = values_track['f_par_track']['loglambda0'][iteration-1] 
        # f_params['tanhlambda0'] = values_track['f_par_track']['tanhlambda0'][iteration-1]

        V_b = values_track['variation_par_track']['V_b'][iteration-1]
        m_b = values_track['variation_par_track']['m_b'][iteration-1]

        # eigvals = values_track['subspace_track']['eigvals'][iteration-1]
        # eigvecs = values_track['subspace_track']['eigvecs'][iteration-1]


        err_dict['is_error'] = True
        err_dict['error'] = e 

    except Exception as e: # Handle any other exception in the same way as KeyboardInterrupt
        
        if isinstance( e, LossStagnationError):
            print(f' ===================  Loss stagnating at iteration: {iteration} =================== \n')
            print(f'During iteration: {iteration}, there should be {iteration} completed iterations')
        else:            
            print(f' ===================  Error During iteration: {iteration} =================== \n')
            print(f'During iteration: {iteration}, there should be {iteration} completed iterations')

        #region _________ Adjust to the last available values _________
        fit_parameters['maxiter'] = iteration
        if fit_parameters['maxiter'] <= 1: 
            print('Too few iterations iterations were done to save')
            err_dict['is_error'] = True
            err_dict['error'] = e    
            raise e

        last_theta = {}
        for theta_key in theta.keys():
            last_theta[theta_key] = values_track['theta_track'][theta_key][iteration-1] # We go bag 2 steps cause that is the value of theta for whihc f_params were optimized and eigenvectors 
            # were calculated ( therefore onto which V-b was projected )
        theta = last_theta

        f_params['logA']    = values_track['f_par_track']['logA'][iteration-1]
        if 'lambda0' in f_params:
            f_params['lambda0'] = values_track['f_par_track']['lambda0'][iteration-1]
        elif 'loglambda0' in f_params:
            f_params['loglambda0'] = values_track['f_par_track']['loglambda0'][iteration-1]            

        V_b = values_track['variation_par_track']['V_b'][iteration-1]
        m_b = values_track['variation_par_track']['m_b'][iteration-1]

        err_dict['is_error'] = True
        err_dict['error'] = e 

    finally: 

            final_start_time = time.time()
            if err_dict['is_error']:
                # If execution was interrupted, the values of the Kernel have yet to be updated
                C, mask    = localker(theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=False)
                K_tilde    = kernfun(theta, xtilde[:,mask], xtilde[:,mask], C=C, diag=False)        # shape (ntilde, ntilde)
                K          = kernfun(theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)    if ntilde != nt else K_tilde
                Kvec       = kernfun(theta, x[:,mask], x2=None, C=C, dC=None, diag=True)            # shape (nt)]

                eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')                                   
                ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)  
                B = eigvecs[:, ikeep]                                           
                K_tilde_b     = torch.diag(eigvals[ikeep])
                K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep])                  
                # K_tilde_inv_p = torch.diag_embed(1/eigvals)                                         # Complete inverse of K_tilde, projected onto the eigenspace. This would be used outside the function to invert the rank+1 kernel after choosing new point             
                K_b           = K @ B 
                KKtilde_inv_b = K_b @ K_tilde_inv_b if ntilde != nt else B


                '''# if not err_dict['is_error']:
            #     B_old = B
            #     eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')         # calculates the eigenvalues for an assumed symmetric matrix, eigenvalues are returned in ascending order. Uplo=L uses the lower triangular part of the matrix

            #     ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)   # Keep only the largest eigenvectors
            #     B = eigvecs[:, ikeep]                                           # shape (ntilde, n_eigen)            
    
            #     V_b = B.T@(B_old@V_b@B_old.T)@B                                 # NOTE: there is a high chance of this V not being posdef since is being projected on a new eigenspace
            #     m_b = B.T @ B_old @ m_b'''
        
                #region ___________ Final loss ___________
                f_mean, lambda_m, lambda_var  =  mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, K=K_b, C=C, m=m_b, V=V_b, 
                                                theta=theta, kernfun=kernfun, lambda_m=None, lambda_var=None  )

                loglikelihood       = compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params)[0]                
                KL                  = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None )
                logmarginal         = loglikelihood - KL 

                values_track['loss_track']['loglikelihood'][fit_parameters['maxiter']-1] = loglikelihood
                values_track['loss_track']['KL'][fit_parameters['maxiter']-1]            = KL
                values_track['loss_track']['logmarginal'][fit_parameters['maxiter']-1]   = logmarginal

            # If they have not just been updated, these values come from the beginning of the last iteration
            final_kernel = {}
            final_kernel['C']             = C
            final_kernel['mask']          = mask
            final_kernel['K_tilde']       = K_tilde
            # final_kernel['K_tilde_inv_p'] = K_tilde_inv_p
            final_kernel['K']             = K
            final_kernel['Kvec']          = Kvec
            # final_kernel['eigvecs']       = eigvecs

            if not is_simmetric(V_b, 'V_b'): 
                print('Final V_b is not simmetric, maximum difference: ', torch.max(torch.abs(V_b - V_b.T)))
                V_b = (V_b + V_b.T)/2
            if not is_posdef(V_b, 'V_b'):
                print('Final V_b is not posdef, this should not be possible if you are skipping the last M-step')   
                V_b += torch.eye(V_b.shape[0], device=DEVICE, dtype=TORCH_DTYPE)*EIGVAL_TOL

            # print(f'Final Loss: {-logmarginal.item():.4f}' ) 

            print(f'\nTime spent for E-steps:       {time_estep_total:.3f}s,') 
            print(f'Time spent for f params:      {time_f_params_total:.3f}s')
            # print(f'Time spent computing Lambda0: {time_lambda0_estimation:.3f}s')
            print(f'Time spent for m / V update:  {time_estep_total-time_f_params_total:.3f}s')
            print(f'Time spent for M-steps:       {time_mstep_total:.3f}s')
            print(f'Time spent for All-steps:     {time_estep_total+time_mstep_total:.3f}s')
            print(f'Time spent computing Kernels: {time_computing_kernels:.3f}s')
            print(f'Time spent computing Loss:    {time_computing_loss:.3f}s')
            print(f'\nTime total after init:        {time.time()-start_time_loop:.3f}s')
            print(f"Time total before init:       {time.time()-start_time_before_init:.3f}s")

            print(f'Final Loss: {-logmarginal.item():.4f}' )
            # Reduce values_track dictionary to the first 'iteration' elements`
            for key in values_track.keys():
                for subkey in values_track[key].keys():
                    values_track[key][subkey] = values_track[key][subkey][:fit_parameters['maxiter']] # Last index not be included 

            hyperparams_tuple = (theta, theta_lower_lims, theta_higher_lims)

            fit_model = {
                'fit_parameters':    fit_parameters,
                'final_kernel':      final_kernel,
                'err_dict':          err_dict,
                'xtilde':            xtilde,
                'hyperparams_tuple': hyperparams_tuple,
                'f_params':          f_params,
                'm_b':               m_b,
                'V_b':               V_b,
                'C':                 C,
                'mask':              mask,
                'K_tilde_b':         K_tilde_b,
                'K_tilde_inv_b':     K_tilde_inv_b,
                'K_b':               K_b,
                'Kvec':              Kvec,
                'B':                 B,
                'values_track':      values_track
            }


                #region _________ Memory usage___________
            # memory = 0
            # for dict in values_track.values():
            #     for key in dict.keys():
            #         if isinstance(dict[key], tuple):
            #             for i in range(len(dict[key])):
            #                 memory += dict[key][i].element_size() * dict[key][i].nelement()
            #             # print(f'{key} memory: {memory / (1024 ** 2):.2f} MB')
            #         else:
            #             memory += dict[key].element_size() * dict[key].nelement()
            #             # print(f'{key} memory: {dict[key].element_size() * dict[key].nelement() / (1024 ** 2):.2f} MB')

            # # Convert bytes to megabytes (MB)
            # total_memory_MB = memory / (1024 ** 2)
            # print(f'\nFinal Total values_track memory on GPU: {total_memory_MB:.2f} MB')
            # # Allocated memory
            # allocated_bytes = torch.cuda.memory_allocated()
            # allocated_MB = allocated_bytes / (1024 ** 2)
            # print(f"Final Allocated memory: {allocated_MB:.2f} MB")

            # # Reserved (cached) memory
            # reserved_bytes = torch.cuda.memory_reserved()
            # reserved_MB = reserved_bytes / (1024 ** 2)
            # print(f"Final Reserved (cached) memory: {reserved_MB:.2f} MB")
            #endregion _________ Memory usage___________
            return fit_model, err_dict
    
        # else:
            # raise Exception('Error')


@torch.no_grad()
def varGP(x, r, **kwargs):
    #region _______ infos __________
    # Learn the hyperparameters of the GP model

    # INPUTS:

    # x = [nt, nx], stimulus
    # r = [1, n],  spike counts
    
    # OPTIONAL INPUTS:
    
    # ntilde: number of inducing data points if they are not provided as argument(increses approximation accuracy )
    # xtilde: inducing points [nx, ntilde]
    
    # nEstep:  number of  of iterations in E-step (updating m, and V)
    # nMstep:  number of steps in M-step (updating theta )
    # Display_hyper: Bool to display the initial hyperparameters
    
    # kernfun: kernel function ( acosker is default)
    # theta: dict of initial hyperparameters of the kernel
    # lb, ub: lower and upper bound for theta
    # m, V: initial mean and variance of variational distribution, q(lambda) = N(m, V). Requires_gradietn is set to False cause in the M step i keep them fixed
     
    # RETURNS
    
    # theta,
    # f_par, 2 Parameters of the firing rate (logA and lambda_0) in the paper, we update logA cause udpading A is not stable, since its in the exponent of the firing rate f_mean
    # m, V (tensors tracking the values of all of the above)
    # xtilde: set of inducing datapoints
    # L, loss function during learning

    # values_track:
    #     - ['loss_track']
    #         - ['logmarginal']
    #         - ['loglikelihood']
    #         - ['KL']
    #     - ['theta_track']
    #        - ['sigma_0']
    #        - ['eps_0x']
    #        - ['eps_0y']
    #        - ['-2log2beta']
    #        - ['-log2rho2']
    #        - ['Amp']
    #     - ['f_par_track'] 
    #        - ['logA']
        #    - ['loglambda0']   
    #     - ['variation_par_track']
    #        - ['V_b']
    #        - ['m_b']




    #endregion
    
    try:
        #region ________ Initialization __________
        start_time_before_init = time.time()
        err_dict = {'is_error': False, 'error_message': None, 'during_init': False}

        # number of pixels, number of training points 
        nt, nx = x.shape 

        # Update the parameters of the fit with the used global variables
        fit_parameters = copy.deepcopy(kwargs['fit_parameters'])
        fit_parameters['min_tolerance'] = MIN_TOLERANCE
        fit_parameters['eigval_tol']    = EIGVAL_TOL

        ntilde        = fit_parameters.get('ntilde',  100 if nt>100 else nt) # if no ntilde is provided try with 100, otherwise inducing points=x   
        maxiter       = fit_parameters.get('maxiter', 50)
        nEstep        = fit_parameters.get('nEstep',  50) 
        nMstep        = fit_parameters.get('nMstep',  20)
        nFparamstep   = fit_parameters.get('nFparamstep', 10)
        lr_Mstep      = fit_parameters.get('lr_Mstep', 0.1)
        lr_Fparamstep = fit_parameters.get('lr_Fparamstep', 0.1)
        display_hyper = fit_parameters.get('display_hyper', True)
        n_px_side     = fit_parameters.get('n_px_side', None)

        verbose       = kwargs.get('verbose', True)
        silent        = kwargs.get('silent', False)

        kernfun_arg       = fit_parameters.get('kernfun', None)

        if hasattr(kernfun_arg, '__name__') and \
            kernfun_arg.__name__ == 'acosker':
                kernfun = acosker
        else:
            raise Exception('Kernel function in varGP should be acosker for now')

        # assert fit_parameters['kernfun'] == acosker

        # if   kernfun ==  acosker:   pass
        # elif kernfun == 'acosker': kernfun = acosker
        # else: 
        #     print('Provided kernel function: ', kernfun)
        #     raise Exception('Kernel function not recognized')
        

        # Initialize hyperparameters of Kernel and parameters of the firing rate
        # Mutable objects are copied otherwise their values would be updated in the original args dictionary sent as argument
        if 'xtilde' in kwargs:
            xtilde            = kwargs['xtilde']  
        else:
            raise Exception('Inducing points not provided')
        if ntilde != xtilde.shape[0]: 
            raise Exception('Number of inducing points does not match ntilde')
        if 'hyperparams_tuple' in kwargs.keys():
            hyperparams_tuple = copy.deepcopy(kwargs['hyperparams_tuple'])  
            theta             = copy.deepcopy(hyperparams_tuple[0])
            theta_lower_lims  = copy.deepcopy(hyperparams_tuple[1])
            theta_higher_lims = copy.deepcopy(hyperparams_tuple[2])
        else:
            raise Exception('Hyperparameters not provided')
        


        if 'f_params' in kwargs.keys():
            f_params = copy.deepcopy(kwargs['f_params']) 
        else:
            raise Exception('f_params not provided')
        # f_params          = copy.deepcopy(kwargs['f_params']) if 'f_params' in kwargs.keys() else {'logA': torch.log(torch.tensor(0.0001)), 'lambda0':torch.tensor(1)}
        for key in f_params.keys(): 
            f_params[key] = f_params[key].requires_grad_(True)
        
        # f_params          = copy.deepcopy(kwargs.get( 'f_params', {'logA': torch.log(torch.tensor(0.0001)), 'lambda0':torch.tensor(-1)} )) # Parameters of the firing rate (A and lambda_0) in the paper
        # f_params          = copy.deepcopy(kwargs.get( 'f_params', {'logA': torch.log(torch.tensor(0.0001,)), 'loglambda0':torch.log(torch.tensor(-1))} )) # Parameters of the firing rate (A and lambda_0) in the paper

        # Calculate the part of the kernel responsible for implementing smoothness and the receptive field
        # TODO Calculate it only close to the RF (for now it's every pixel)

        # The following lines initialize the kernel values. 
        # They take care of setting the kernel of the whole dataset equal to the kernel on the inducing points (K_tilde) the same if the inducing points are the whole dataset
        # They also dont calculate the kernel if its starting values are passed as an argument
        # They also take care of projecting the kernel into the eigenspace of the largest eigenvectors of K_tilde
        if 'init_kernel' not in kwargs:
            C, mask = localker(theta=theta, theta_lower_lims=theta_lower_lims, theta_higher_lims=theta_higher_lims, n_px_side=n_px_side, grad=False)  
            K_tilde = kernfun(theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)
            Kvec    = kernfun(theta, x[:,mask], x2=None, C=C, dC=None, diag=True)
            if ntilde != nt:
                K   = kernfun(theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)
            else:
                K = K_tilde
            eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')                                # calculates the eigenvals for an assumed symmetric matrix, eigenvalues  are returned in ascending order. Uplo=L uses the lower triangular part of the matrix. Eigenvectors are columns
            ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)                          # Keep only the largest eigenvectors
            B = eigvecs[:, ikeep]
            K_tilde_b = torch.diag(eigvals[ikeep])
            K_b       = K @ B
            K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep])
            if ntilde != nt:  
                KKtilde_inv_b = K_b @ K_tilde_inv_b
            else:
                KKtilde_inv_b = B

        else: 
            C, mask = (kwargs['init_kernel']['C'], kwargs['init_kernel']['mask'])
            K_tilde = kwargs['init_kernel']['K_tilde']
            Kvec    = kwargs['init_kernel']['Kvec']
            if ntilde != nt:  
                K   = kwargs['init_kernel']['K']
            else:
                K = K_tilde
            B = kwargs['init_kernel']['B']
            # make K_tilde_b and K_b a projection of K_tilde and K into the eigenspace of the largest eigenvectors
            K_tilde_b = kwargs['init_kernel']['K_tilde_b']  # shape (n_eigen, n_eigen)
            K_b       = kwargs['init_kernel']['K_b']
            K_tilde_inv_b = kwargs['init_kernel']['K_tilde_inv_b'] # shape (n_eigen, n_eigen)
            if ntilde != nt:
                kwargs['init_kernel']['KKtilde_inv_b']
            else:             
                KKtilde_inv_b = B # the resulting matrix of Ktildeb @ B @ B.T @ Ktildeb_inv @ B = B  

        # K_tilde_inv_p = torch.diag_embed(1/eigvals) if 'init_kernel' not in kwargs else kwargs['init_kernel']['K_tilde_inv_p'] # If we want to invert the whole K_tilde, not only the projected one, maybe outside VarGP for active loop

        # We always pass the non projected variational parameters because the dimensionality of the problem is determined by the lines above ( B ).
        m = copy.deepcopy(kwargs.get('m', torch.zeros( (ntilde) )).detach()).to(DEVICE, dtype=TORCH_DTYPE)
        V = copy.deepcopy(kwargs.get('V', K_tilde ).detach()).to(DEVICE, dtype=TORCH_DTYPE)

        # m_b = copy.deepcopy(kwargs.get('m_b', torch.zeros( (ntilde) )).detach())
        # V_b = copy.deepcopy(kwargs.get('V_b', K_tilde_b ).detach())

        V_b = B.T @ V @ B if 'V' in kwargs else K_tilde_b     # shape (n_eigen, n_eigen)
        m_b = B.T @ m 

        lambda_m, lambda_var = lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta, kernfun=kernfun)
        f_mean               = mean_f_given_lambda_moments(f_params, lambda_m, lambda_var)

        loglikelihood, _, __ = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=False)
        KL_div               = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None, ignore_warning=True )
        logmarginal          = loglikelihood - KL_div   

        # Tracking dictionary
        loss_track          = {'logmarginal'  : torch.zeros((maxiter)).to(DEVICE, dtype=TORCH_DTYPE),                          # Loss to  maximise: Log Likelihood - KL
                                'loglikelihood': torch.zeros((maxiter)),
                                'KL'           : torch.zeros((maxiter)),
                                } 
        theta_track         = {key : torch.zeros((maxiter)).to(DEVICE, dtype=TORCH_DTYPE) for key in theta.keys()}
        f_par_track         = {'logA': torch.zeros((maxiter)).to(DEVICE, dtype=TORCH_DTYPE), 
                            'lambda0': torch.zeros((maxiter)).to(DEVICE, dtype=TORCH_DTYPE)} if 'lambda0' in f_params else \
                                {'logA': torch.zeros((maxiter)).to(DEVICE, dtype=TORCH_DTYPE), 
                                'loglambda0': torch.zeros((maxiter)).to(DEVICE, dtype=TORCH_DTYPE)}
        # f_par_track         = {'logA': torch.zeros((maxiter)), 'loglambda0': torch.zeros((maxiter))} # track hyperparamers

        variation_par_track = {'V_b': (), 'm_b': ()}               # track the variation parameters
        # subspace_track      = {'eigvals': torch.zeros((maxiter, K_tilde.shape[0])), 
                                # 'eigvecs': torch.zeros((maxiter, *tuple(K_tilde.shape)))  }        # track the eigenvectors of the kernel
        values_track        = {'loss_track':      loss_track,   'theta_track': theta_track, 
                                'f_par_track':    f_par_track,  'variation_par_track': variation_par_track}
                                # 'subspace_track': subspace_track }
        
        # print(f'Initialization took: {(time.time()-start_time_before_init):.4f} seconds\n')

        #region _________ Memory usage___________
        # memory = 0
        # for dict in values_track.values():
        #     for key in dict.keys():
        #         if isinstance(dict[key], tuple):
        #             for i in range(len(dict[key])):
        #                 memory += dict[key][i].element_size() * dict[key][i].nelement()
        #             # print(f'{key} memory: {memory / (1024 ** 2):.2f} MB')
        #         else:
        #             memory += dict[key].element_size() * dict[key].nelement()
        #             # print(f'{key} memory: {dict[key].element_size() * dict[key].nelement() / (1024 ** 2):.2f} MB')

        # # Convert bytes to megabytes (MB)
        # total_memory_MB = memory / (1024 ** 2)
        # print(f'Total values_track memory on GPU: {total_memory_MB:.2f} MB')
        # # Allocated memory
        # allocated_bytes = torch.cuda.memory_allocated()
        # allocated_MB = allocated_bytes / (1024 ** 2)
        # print(f"\nAfter initialization Allocated memory: {allocated_MB:.2f} MB")

        # # Reserved (cached) memory
        # reserved_bytes = torch.cuda.memory_reserved()
        # reserved_MB = reserved_bytes / (1024 ** 2)
        # print(f"\nAfter initialization Reserved (cached) memory: {reserved_MB:.2f} MB")
        #endregion _________ Memory usage___________

        #endregion ______________________________
        # Loop variables
        start_time_loop        = time.time()
        time_estep_total       = 0
        time_f_params_total    = 0
        time_mstep_total       = 0
        time_computing_kernels = 0
        time_computing_loss    = 0
        time_lambda0_estimation= 0

        #region ________________ Initialize tracking dict ______________________

        # Update the tracking dictionaries. Remember that mutable objects are passed by reference so any modification to them would reflect in the dictionary if we dont copy
        values_track['loss_track']['loglikelihood'][0].copy_(loglikelihood)
        values_track['loss_track']['KL'][0].copy_(KL_div)
        values_track['loss_track']['logmarginal'][0].copy_(loglikelihood-KL_div)
        
        initial_loss = -(loglikelihood-KL_div)
        if not silent:
            print(f'Initial Loss: {initial_loss:.4f}')
        if initial_loss == torch.inf or initial_loss == -torch.inf or initial_loss == torch.nan:
            raise LossInfError('Initial loss is infinite or NaN')


        # Theta before the Mstep of 0 "i" is the one used to build the kernel of the E-step of 0 "i+1". 
        # The theta we are saving here is the one we just used.
        for key in theta.keys():
            values_track['theta_track'][key][0].copy_(theta[key])

        values_track['f_par_track']['logA'][0].copy_(f_params['logA'])
        if 'lambda0' in f_params:
            values_track['f_par_track']['lambda0'][0].copy_(f_params['lambda0'])
        elif 'loglambda0' in f_params:
            values_track['f_par_track']['loglambda0'][0].copy_(f_params['loglambda0'])

        values_track['variation_par_track']['V_b'] += (V_b.clone(),)
        values_track['variation_par_track']['m_b'] += (m_b.clone(),)
        #endregion
    
    except Exception as e:
        print(f'...GP Thread: Error during initialization : {e}')
        err_dict['is_error'] = True
        err_dict['error'] = e
        err_dict['during_init'] = True

    else:

        if not err_dict['during_init']:
            try:    
                #_______________________ Main Loop ___________
                for iteration in range(1,maxiter):

                    # print(f'*Iteration*: {iteration}', end='')

                    #region ________________ Computing Kernel and Stabilization____________________
                    # Compute starting Kernel, if no M-step -> only compute it once cause it's not changing
                    start_time_computing_kernels = time.time()
                    if nMstep > 0 and iteration > 1:
                        #________________ Compute the KERNELS after M-Step and the inverse of V _____________
                        C, mask    = localker(theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=False)                
                        K_tilde    = kernfun( theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)                                   # shape (ntilde, ntilde)
                        K          = kernfun( theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)      if ntilde != nt else K_tilde      # shape (nt, ntilde) set of row vectors K_i for every input 
                        Kvec       = kernfun( theta, x[:,mask], x2=None, C=C, dC=None, diag=True)                                                # shape (nt)
                        
                        eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')                                # calculates the eigenvals for an assumed symmetric matrix, eigenvalues  are returned in ascending order. Uplo=L uses the lower triangular part of the matrix. Eigenvectors are columns
                        ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)                          # Keep only the largest eigenvectors
                    
                        B_old = B 
                        B = eigvecs[:, ikeep]                                                  # shape (ntilde, n_eigen)            
                        # make K_tilde_b and K_b a projection of K_tilde and K into the eigenspace of the largest eigenvectors
                        K_tilde_b     = torch.diag(eigvals[ikeep])                                 # shape (n_eigen, n_eigen)
                        # K_tilde_inv_p = torch.diag_embed(1/eigvals)                              # We keep the latest inverse of the complete K_tilde just to return it and be fast in computing the inverse of the incremented K_tilde if needed ( active learning )
                        K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep])                     # shape (n_eigen, n_eigen)
                        K_b           = K @ B                                                      # shape (3190, n_eigen)
                        KKtilde_inv_b = K_b @ K_tilde_inv_b if ntilde != nt else B             # shape (nt, n_eigen) # this is 'a' in matthews code                         
                
                        # In the following iterations we already have V_b (maybe updated in an E-step) and if Mstep > 0 we changed the eigenspace,
                        # Get V_b_new referring to this new eigenspace as:
                        #       V_b_new = (B_new.T@V)@B_new 
                        # Where V       = B_old@V_b_old@B_old.T, so

                        # V_b_new       = B_new.T@(B_old@V_b_old@B_old.T)@B_new and
                        # m_b_new       = B_new.T @ B_old @ m_b_old 

                        # Note that we might have augmented the dimension of the eigenspace, this might leave very small eigenvalues in V_b_new
                        # This will not be necessaraly invertible (or posdef ). This might be problem in the Estep when using alpha != 1.
                        # This matrix is not numerically simmetric for precision higher than 1.e-13 even if it should be arount 1.e-15, hence the choice of MIN_TOLERANCE 1.e-13
                        # V_b is guaranteed to be simmetric (and posdef) only when coming out of E step
                        # It will be used only in each first estep iteration. To calculate the lambda moments. It never gave numerical problems but might be a source loss of precision
                        V_b_new = B.T@(B_old@V_b@B_old.T)@B                   
                        V_b     = V_b_new                                     

                        # smallest_eig = torch.linalg.eigh(V_b)[0].min()
                        # if smallest_eig <= 0.:
                            # warnings.warn(f'Matrix V_b is simmetric but has an eigenvalue smaller than 0 ')

                        m_b_new = B.T @ B_old @ m_b
                        m_b     = m_b_new

                    time_computing_kernels += time.time() - start_time_computing_kernels
                    #endregion 

                    #region  _______________ Control over possible Nans ______
                    # for tensor in [C, K_tilde_b, K_b, KKtilde_inv_b, V_b, m_b, f_params['logA'], f_params['lambda0']]:
                    # for tensor in [C, K_tilde_b, K_b, KKtilde_inv_b, V_b, m_b, f_params['logA'], f_params['loglambda0']]:                
                    # # for tensor in [C, K_tilde_b, K_b, KKtilde_inv_b, V_b, m_b, f_params['logA'], f_params['tanhlambda0']]:                                
                    #     if torch.any(torch.isnan(tensor)):
                    #         variable_name = [k for k, v in locals().items() if v is tensor][0]
                    #         raise ValueError(f'NaN in {variable_name}')
                    #     if torch.any(torch.isinf(tensor)):
                    #         variable_name = [k for k, v in locals().items() if v is tensor][0]
                    #         raise ValueError(f'Inf in {variable_name}')
                    #endregion
                    
                    #region ________________ E-Step : Update on m & V and f(lambda) parameters ________
                    start_time_estep = time.time()
                    if nEstep > 0:
                        # print(f'Estep in iteration {iteration}')

                        for i_estep in range(1):
                            # print(f'   Estep n {i_estep}')

                            # Update lambda moments only if the kernel has changed or if it's the first iteration
                            # They are update again after the Estep
                            if i_estep == 0 and nMstep > 0:
                                lambda_m, lambda_var = lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta, kernfun=kernfun)  

                                # feature 2: lambda0
                                # f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)

                            # Tracking the time for the f_params update, the f_mean computation would not be here if there was no update
                            start_time_f_params = time.time()
                            f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var) # Since f_params influece f_mean, we need to update it at each estep
                            time_f_params_total += time.time()-start_time_f_params

                            #region ____________ Update m, V ______________
                            for _ in range(nEstep):
                                # print(f'estep {_} of {nEstep} in iteration {iteration}')

                                m_b_prev = m_b.clone()
                                V_b_prev = V_b.clone()
                                f_mean_prev = f_mean.clone()

                                m_b, V_b = Estep( r=r, KKtilde_inv=KKtilde_inv_b, m=m_b, f_params=f_params, f_mean=f_mean, 
                                                    K_tilde=K_tilde_b, K_tilde_inv=K_tilde_inv_b, update_V_inv=False, alpha=1  ) # Do not change udpate_V_inv or alpha, read Estep docs

                                # And the things that depend on them ( moments of lambda )
                                f_mean, lambda_m, lambda_var  =  mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], 
                                                                        K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, 
                                                                        K=K_b, C=C, m=m_b, V=V_b, theta=theta, kernfun=kernfun, 
                                                                        lambda_m=None, lambda_var=None  )
                            
                                # avoid numerical instability, revert if needed
                                if f_mean.mean() > 1000:
                                    if verbose:
                                        print(f'f_mean mean = {f_mean.mean():.1f} after Estep iteration {iteration}, back to prev m_b and V_b')
                                    m_b = m_b_prev
                                    V_b = V_b_prev
                                    f_mean = f_mean_prev

                                    while f_mean.mean() > 1000:
                                        print(f'Warning: f_mean still very large taking previous values of m and V, lowering A and lambda0')

                                        f_params['logA'] -= 0.1
                                        f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)

                                        f_mean = mean_f( f_params=f_params, calculate_moments=True, lambda_m=lambda_m, 
                                                        lambda_var=lambda_var, x=x[:,mask], K_tilde=K_tilde_b, 
                                                        KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, K=K_b, C=C, 
                                                        m=m_b, V=V_b, theta=theta, kernfun=kernfun, )

                                    break

                                # Check convergence ( early stopping )
                                if _ > 0: # skip first iteration
                                    # We implement norm based earlystopping on f_mean
                                    # - f_mean integrates all the parameters we are optimizing
                                    # - if it was element wise, small firing rate increase of 100% would count 
                                    #   the same 
                                    rel_change = torch.norm(f_mean - f_mean_prev) / (torch.norm(f_mean_prev) + 1e-6)           

                                    if rel_change < 1.e-5:
                                        # if verbose:
                                            # print(f'Estep converged after {_+1} iterations, (relative change: {rel_change:.8f})')
                                        break    
                                #endregion

                            #region ____________ Update f_params ______________ 
                            # if i_estep > 0:
                            f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)


                            #lr_Fparamstep = 0.1  # what we use usually
                            optimizer_f_params = torch.optim.LBFGS([f_params['logA']], lr=lr_Fparamstep, max_iter=nFparamstep, 
                                                                    tolerance_change=1.e-9, tolerance_grad=1.e-7,
                                                                    history_size=nFparamstep, line_search_fn='strong_wolfe')
                            start_time_f_params = time.time()
                            CLOSURE2_COUNTER = [0]
                            @torch.no_grad()
                            def closure_f_params( ):
                                CLOSURE2_COUNTER[0] += 1
                                optimizer_f_params.zero_grad()
                                nonlocal f_mean          # Update f_mean of the outer scope each time the closure is called
                                # Lambda0 feature 3

                                # Each time the closure is called the optimizer expects the value of the loss. 
                                # It might be using it to explore how big of a step to take (line search) or actually updating the parameters ( logA)
                                # We need the optimizer to evaluate the loss with the optimal lambda0 parameter given logA, 
                                # so we update it here, before computing all the other things that depend on it.

                                f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)
                                f_mean = mean_f_given_lambda_moments( f_params, lambda_m, lambda_var)   

                                loglikelihood, dloglikelihood = compute_loglikelihood(  r,  f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=True )                       
                                # print(f' -logmarginal = {(-loglikelihood.item() + KL.item()):.4f} -loglikelihood = {-loglikelihood.item():.4f}  KL = {KL.item():.4f}')

                                # Update gradients of the loss with respect to the firing rate parameters
                                # The minus here is because we are minimizing the negative loglikelihood
                                f_params['logA'].grad    = -dloglikelihood['logA']    #if f_params['logA'].requires_grad else None


                                if 'lambda0' in f_params:
                                    f_params['lambda0'].grad = -dloglikelihood['lambda0']        if f_params['lambda0'].requires_grad else None
                                elif 'loglambda0' in f_params:
                                    f_params['loglambda0'].grad = -dloglikelihood['loglambda0']  if f_params['loglambda0'].requires_grad else None
        
                                # if torch.any(torch.isnan(f_mean)):
                                    # raise ValueError(f'Nan in f_mean during f param update in Estep, closure has been called {CLOSURE2_COUNTER[0]} times in estep {i_estep} iteration. Try substituting them with inf.')
                                # if  torch.any( f_mean > 1.e4):
                                    # raise ValueError(f'f_mean is too large in Estep, closure has been called {CLOSURE2_COUNTER[0]} times in estep {i_estep} iteration')


                                # If the optimizer is exploring values of A that make the firing rate too large, 
                                # signal the loss is too big so that it goes back on its step
                                if f_mean.mean() > 100 or torch.any(torch.isnan(f_mean)):
                                    if verbose:
                                        print(f'f_mean mean is {f_mean.mean()} at i_step {i_estep} iteration {iteration} at closure call {CLOSURE2_COUNTER[0]}, returning infinite loss')
                                    return torch.tensor(float('inf'))
                                
                                return -loglikelihood

                            optimizer_f_params.step(closure_f_params)        
                            
                            f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var) # the optimal logA value found by the optimizer might not be the one used in the last closure call. We need to make sure lambda0 is updated.

                            if f_mean.mean() > 100:
                                if verbose:
                                    print(f'f_mean mean is {f_mean.mean()} at i_step {i_estep} iteration {iteration} after closure .step')


                            time_f_params_total += time.time()-start_time_f_params
                            #endregion
                    else: 
                        if verbose:
                            print('No E-step')

                    time_estep = time.time()-start_time_estep
                    time_estep_total += time_estep 
                    # print(f'\r*Iteration*: {iteration:>3} E-step took: {time_estep:.4f}s', end='')
                    #endregion                       
                
                    #region ________________ Update the tracking dictionaries _______________  

                    # Update the value every x iterations. 
                    # We are doing it here to avoid having to project V_b and m_b to the updated eigenspace. 
                    # This might pose numerical problem if alpha!=1 as explained in the docs, 
                    # But even in the alpha=1 case the value of the almost singular reprojected V_b would be used for the loss, leading to expliding values sometimes
                    #  ( its only a tracking problem ) 
                    
                    if iteration % 1 == 0 or iteration == maxiter-1:

                        start_time_computing_loss = time.time()
                        # lambda_m, lambda_var = lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta, kernfun=kernfun)
                        
                        f_mean               = mean_f_given_lambda_moments(f_params, lambda_m, lambda_var)

                        loglikelihood, _, __ = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=False)

                        KL_div               = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None, ignore_warning=True )
                        logmarginal          = loglikelihood - KL_div   

                        time_computing_loss += time.time()-start_time_computing_loss
                        # print(f" TOT time for loss computation at iteration {iteration:>2}: {tot_elapsed_time:>2.6f} s")
                        # print(f"     time for loss computation at iteration {iteration:>2}: {elapsed_time:>2.6f} s")

                    # Update the tracking dictionaries. Remember that mutable objects are passed by reference so any modification to them would reflect in the dictionary if we dont copy
                    values_track['loss_track']['loglikelihood'][iteration].copy_(loglikelihood)
                    values_track['loss_track']['KL'][iteration].copy_(KL_div)
                    values_track['loss_track']['logmarginal'][iteration].copy_(loglikelihood-KL_div)

                    # Theta before the Mstep of iteration "i" is the one used to build the kernel of the E-step of iteration "i+1". 
                    # The theta we are saving here is the one we just used.
                    for key in theta.keys():
                        values_track['theta_track'][key][iteration].copy_(theta[key])

                    values_track['f_par_track']['logA'][iteration].copy_(f_params['logA'])
                    if 'lambda0' in f_params:
                        values_track['f_par_track']['lambda0'][iteration].copy_(f_params['lambda0'])
                    elif 'loglambda0' in f_params:
                        values_track['f_par_track']['loglambda0'][iteration].copy_(f_params['loglambda0'])                


                    values_track['variation_par_track']['V_b'] += (V_b.clone(),)
                    values_track['variation_par_track']['m_b'] += (m_b.clone(),)

                    if verbose:
                        print(f'Loss iter {iteration}: {-(loglikelihood-KL_div):.4f}')
                    if loglikelihood == torch.inf or loglikelihood == -torch.inf or loglikelihood == torch.nan:
                        raise LossInfError(f'loglikelihood is infinite or NaN at iteration {iteration}')
                    if KL_div == torch.inf or KL_div == -torch.inf or KL_div == torch.nan:
                        raise LossInfError(f'KL_div is infinite or NaN at iteration {iteration}')
                            

                    # region _________ Check loss stabilization __________
                    # If loss hasn't changed in the last 5 iterations, break the loop
                    if iteration >= 5:
                        # Get the loss values for the last 5 iterations
                        recent_losses = [values_track['loss_track']['logmarginal'][i] for i in range(iteration-4, iteration+1)]
                        recent_losses_tensor = torch.tensor(recent_losses)
                        loss_range = torch.abs(recent_losses_tensor.max() - recent_losses_tensor.min())
                        if loss_range < LOSS_STOP_TOL:
                            if verbose:
                                print(f'Loss stabilization detected (loss range {loss_range.item():.2e} < tolerance {LOSS_STOP_TOL:.2e}). Stopping training.')
                            raise LossStagnationError(f'Loss stabilization detected (loss range {loss_range.item():.2e} < tolerance {LOSS_STOP_TOL:.2e}). Stopping training.')
                    # endregion

                    #endregion

                    #region ________________ M-Step : Update on hyperparameters theta  ________________

                    start_time_mstep = time.time()
                    if nMstep > 0 and iteration < maxiter-1: 
                        # Skip the M-step in the last iteration to avoid generating a new eigenspace that will not be used by V and m
                        if verbose:
                            print(f' Mstep of iteration {iteration}')
                        if iteration > 1:
                            del optimizer_hyperparams
                        optimizer_hyperparams = torch.optim.LBFGS(theta.values(), lr=lr_Mstep, max_iter=nMstep, line_search_fn='strong_wolfe', 
                                                                tolerance_change=1.e-9, tolerance_grad=1.e-7, history_size=100)
            
                        CLOSURE2_COUNTER = [0]
                        @torch.no_grad()
                        def closure_hyperparams( ):
                            CLOSURE2_COUNTER[0] += 1
                            optimizer_hyperparams.zero_grad()
                            # if any hyperparameter is out of bounds, return infinite loss to signal the optimizer to revaluate the step size
                            return_infinite_loss = False
                            for key, value in theta.items():
                                if not (theta_lower_lims[key] <= value <= theta_higher_lims[key]):
                                    return_infinite_loss = True
                                    if verbose:
                                        print(f"{key} = {value:.4f} is not within the limits of {theta_lower_lims[key]} and {theta_higher_lims[key]}, returning infinite loss in closure call {CLOSURE2_COUNTER[0]}")
                                    if theta[key].requires_grad:
                                        theta[key].grad = torch.tensor(float('inf'), device=DEVICE)
                            if return_infinite_loss: return torch.tensor(float('inf'), device=DEVICE)

                            C, mask, dC       = localker(theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=True)
                            K_tilde, dK_tilde = kernfun( theta, xtilde[:,mask], xtilde[:,mask], C=C, dC=dC, diag=False)
                            K, dK             = kernfun( theta, x[:,mask], xtilde[:,mask], C=C, dC=dC, diag=False) if ntilde != nt else (K_tilde, dK_tilde) 
                            Kvec, dKvec       = kernfun( theta, x[:,mask], x2=None, C=C, dC=dC, diag=True) 

                            #region ____________Stabilization____________________
                            # Note on Stabilization
                            # The eigenvector matrix is not recalculated during the M-step. 
                            # This is not entirely precise because a change in hyperparameters could change the eigenvalues 
                            # over the threshold (and therefore change the dimension of the subspace I'm projecting onto)
                            # But this most likely has a minimal effect. And it saves nMstep eigenvalue decompositions per iteration.
                            # NOTE that even if I am saving resources by not recalculating the eigenspace of K_tilde, I still have to recalculate the inverse of K_tilde in the M-step... still On^3

                            # Projecting the Kernel into the same eigenspace used in the E-step (its not changing with the changing hyperparameters/Kernel)
                            K_tilde_b = B.T@K_tilde@B                 # Projection of K_tilde into eigenspace (n_eigen,n_eigen) 
                            K_tilde_b = (K_tilde_b + K_tilde_b.T)*0.5 # make sure it is symmetric
                            K_b  = K @ B                              # Project K into eigenspace, shape (3190, n_eigen)

                            # If eigenspace B has been recalculated, one has to reproject m and V into the new eigenspace
                            # V_b_new = B.T@(B_old@V_b@B_old.T)@B
                            # V_b     = V_b_new
                            # m_b_new = B.T @ B_old @ m_b
                            # m_b = m_b_new

                            # Projection of the gradients of the Kernel into the eigenspace
                            dK_tilde_b, dK_b = {}, {}
                            for key in dK_tilde.keys():
                                dK_tilde_b[key] = B.T@dK_tilde[key]@B
                                dK_b[key]       = dK[key] @ B                     
                            #endregion

                            # K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep]) # shape (n_eigen, n_eigen) To use if I have recalculated the eigenspace of K_tilde
                            # NOTE that even if I am saving resources by not recalculating the eigenspace of K_tilde, I still have to recalculate the inverse of K_tilde in the M-step... still On^3
                            eye = torch.eye(K_tilde_b.shape[0], device=DEVICE, dtype=TORCH_DTYPE)
                            K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye)
                            KKtilde_inv_b = K_b @ K_tilde_inv_b if ntilde != nt else B

                            f_mean, lambda_m, lambda_var, dlambda_m, dlambda_var  =  mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, K=K_b,  
                                                                                        C=C, m=m_b, V=V_b, theta=theta, kernfun=kernfun, lambda_m=None, lambda_var=None, dK=dK_b, dK_tilde=dK_tilde_b, dK_vec=dKvec, K_tilde_inv=K_tilde_inv_b) # Shape (nt
                            
                            # feature 2: lambda0
                            # lambda0_estimation_start_time = time.time()
                            # temp_f_params = {'logA':f_params['logA'], 'lambda0':lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)}
                            # f_params['lambda0'] = lambda0_given_logA( f_params['logA'], r, lambda_m, lambda_var)
                            # time_lambda0_estimation += time.time()-lambda0_estimation_start_time

                            # loglikelihood, dloglikelihood = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, temp_f_params, dlambda_m=dlambda_m, dlambda_var=dlambda_var )
                            loglikelihood, dloglikelihood = compute_loglikelihood(r, f_mean, lambda_m, lambda_var, f_params, dlambda_m=dlambda_m, dlambda_var=dlambda_var )
                            KL, dKL                       = compute_KL_div(m_b, V_b, K_tilde_b, K_tilde_inv=K_tilde_inv_b, dK_tilde=dK_tilde_b)
                            logmarginal                   = loglikelihood - KL
                            
                            l = -logmarginal
                            # print(f' {l.item():.4f} = logmarginal in m-step - closure call {CLOSURE2_COUNTER[0]}')   


                            # Dictionary of gradients of the -loss with respect to the hyperparameters, to be assigned to the gradients of the parameters
                            # Update the gradients of the loss with respect to the hyperparameters ( its minus the gradeints of the logmarginal)
                            dlogmarginal = {}
                            for key in theta.keys():
                                dlogmarginal[key] = dloglikelihood[key] - dKL[key]
                                if theta[key].requires_grad:
                                    theta[key].grad = -dlogmarginal[key]
                                # dlogmarginal[key] = -dKL[key]
                                # if theta[key].requires_grad:
                                #         theta[key].grad = -dlogmarginal[key]                            

                            # print_hyp(theta)
                    
                            # In case you want to implement the simplest gradient descent, you can use this and call closure_hyperparams() directly   
                            # for key in theta.keys():
                            #     if theta[key].requires_grad:
                            #         theta[key] = theta[key] - 0.0001*theta[key].grad
                            # print(f'   m-step loss: {-logmarginal.item():.4f}')
                            # return -KL
                            return l
                    
                        optimizer_hyperparams.step(closure_hyperparams)

                        # mstep_time = time.time()-start_time_mstep
                        # print(f'\r*Iteration*: {iteration:>3} E-step took: {time_estep:.4f}s, M-step took: {mstep_time:.4f}s', end= '\n')

                    else: 
                        if iteration < maxiter-1: 
                            if verbose:
                                print(' No M-step')
                    time_mstep        = time.time()-start_time_mstep
                    time_mstep_total += time_mstep
                    #endregion __________________________________________

            except KeyboardInterrupt as e:

                print(' ===================  Interrupted  ===================\n')
                print(f'During iteration: {iteration}, there should be {iteration} completed iterations')

                #region _________ Adjust to the last available values _________
                fit_parameters['maxiter'] = iteration
                if fit_parameters['maxiter'] <= 1: 
                    print('Too few iterations iterations were done to save')
                    err_dict['is_error'] = True
                    err_dict['error'] = e    
                    raise e

                last_theta = {}
                for theta_key in theta.keys():
                    last_theta[theta_key] = values_track['theta_track'][theta_key][iteration-1] # We go back 2 steps cause that is the value of theta for which f_params were optimized 
                theta = last_theta                                                              # and eigenvectors were calculated ( therefore onto which the last used V-b was projected )

                f_params['logA']    = values_track['f_par_track']['logA'][iteration-1]
                if 'lambda0' in f_params:
                    f_params['lambda0'] = values_track['f_par_track']['lambda0'][iteration-1]
                elif 'loglambda0' in f_params:
                    f_params['loglambda0'] = values_track['f_par_track']['loglambda0'][iteration-1] 
                # f_params['tanhlambda0'] = values_track['f_par_track']['tanhlambda0'][iteration-1]

                V_b = values_track['variation_par_track']['V_b'][iteration-1]
                m_b = values_track['variation_par_track']['m_b'][iteration-1]

                # eigvals = values_track['subspace_track']['eigvals'][iteration-1]
                # eigvecs = values_track['subspace_track']['eigvecs'][iteration-1]


                err_dict['is_error'] = True
                err_dict['error'] = e 

            except Exception as e: # Handle any other exception in the same way as KeyboardInterrupt
                
                if isinstance( e, LossStagnationError):
                    if not silent:
                        print(f' ===================  Loss stagnating at iteration: {iteration} =================== \n')
                        print(f'During iteration: {iteration}, there should be {iteration} completed iterations')
                else:            
                    print(f' ===================  Error During iteration: {iteration} =================== \n')
                    print(f'During iteration: {iteration}, there should be {iteration} completed iterations')

                #region _________ Adjust to the last available values _________
                fit_parameters['maxiter'] = iteration
                if fit_parameters['maxiter'] <= 1: 
                    print('Too few iterations iterations were done to save')
                    err_dict['is_error'] = True
                    err_dict['error'] = e    
                    raise e

                last_theta = {}
                for theta_key in theta.keys():
                    last_theta[theta_key] = values_track['theta_track'][theta_key][iteration-1] # We go bag 2 steps cause that is the value of theta for whihc f_params were optimized and eigenvectors 
                    # were calculated ( therefore onto which V-b was projected )
                theta = last_theta

                f_params['logA']    = values_track['f_par_track']['logA'][iteration-1]
                if 'lambda0' in f_params:
                    f_params['lambda0'] = values_track['f_par_track']['lambda0'][iteration-1]
                elif 'loglambda0' in f_params:
                    f_params['loglambda0'] = values_track['f_par_track']['loglambda0'][iteration-1]            

                V_b = values_track['variation_par_track']['V_b'][iteration-1]
                m_b = values_track['variation_par_track']['m_b'][iteration-1]

                err_dict['is_error'] = True
                err_dict['error'] = e 

    finally: 
        if verbose:
            print('Startig finally block')
        final_start_time = time.time()

        if not err_dict['during_init']:
            if err_dict['is_error']:
                # If execution was interrupted, the values of the Kernel have yet to be updated
                C, mask    = localker(theta=theta, theta_higher_lims=theta_higher_lims, theta_lower_lims=theta_lower_lims, n_px_side=n_px_side, grad=False)
                K_tilde    = kernfun(theta, xtilde[:,mask], xtilde[:,mask], C=C, diag=False)        # shape (ntilde, ntilde)
                K          = kernfun(theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)    if ntilde != nt else K_tilde
                Kvec       = kernfun(theta, x[:,mask], x2=None, C=C, dC=None, diag=True)            # shape (nt)]

                eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')                                   
                ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)  
                B = eigvecs[:, ikeep]                                           
                K_tilde_b     = torch.diag(eigvals[ikeep])
                K_tilde_inv_b = torch.diag_embed(1/eigvals[ikeep])                  
                # K_tilde_inv_p = torch.diag_embed(1/eigvals)                                         # Complete inverse of K_tilde, projected onto the eigenspace. This would be used outside the function to invert the rank+1 kernel after choosing new point             
                K_b           = K @ B 
                KKtilde_inv_b = K_b @ K_tilde_inv_b if ntilde != nt else B


                '''# if not err_dict['is_error']:
            #     B_old = B
            #     eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')         # calculates the eigenvalues for an assumed symmetric matrix, eigenvalues are returned in ascending order. Uplo=L uses the lower triangular part of the matrix

            #     ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)   # Keep only the largest eigenvectors
            #     B = eigvecs[:, ikeep]                                           # shape (ntilde, n_eigen)            

            #     V_b = B.T@(B_old@V_b@B_old.T)@B                                 # NOTE: there is a high chance of this V not being posdef since is being projected on a new eigenspace
            #     m_b = B.T @ B_old @ m_b'''
        
                #region ___________ Final loss ___________
                f_mean, lambda_m, lambda_var  =  mean_f( f_params=f_params, calculate_moments=True, x=x[:,mask], K_tilde=K_tilde_b, KKtilde_inv=KKtilde_inv_b, Kvec=Kvec, K=K_b, C=C, m=m_b, V=V_b, 
                                                theta=theta, kernfun=kernfun, lambda_m=None, lambda_var=None  )

                loglikelihood       = compute_loglikelihood( r,  f_mean, lambda_m, lambda_var, f_params)[0]                
                KL                  = compute_KL_div( m_b, V_b, K_tilde_b, K_tilde_inv_b, dK_tilde=None )
                logmarginal         = loglikelihood - KL 

                values_track['loss_track']['loglikelihood'][fit_parameters['maxiter']-1] = loglikelihood
                values_track['loss_track']['KL'][fit_parameters['maxiter']-1]            = KL
                values_track['loss_track']['logmarginal'][fit_parameters['maxiter']-1]   = logmarginal

            # If they have not just been updated, these values come from the beginning of the last iteration
            final_kernel = {}
            final_kernel['C']             = C
            final_kernel['mask']          = mask
            final_kernel['K_tilde']       = K_tilde
            # final_kernel['K_tilde_inv_p'] = K_tilde_inv_p
            final_kernel['K']             = K
            final_kernel['Kvec']          = Kvec
            # final_kernel['eigvecs']       = eigvecs

            if not is_simmetric(V_b, 'V_b'): 
                print('Final V_b is not simmetric, maximum difference: ', torch.max(torch.abs(V_b - V_b.T)))
                V_b = (V_b + V_b.T)/2
            if not is_posdef(V_b, 'V_b'):
                print('Final V_b is not posdef, this should not be possible if you are skipping the last M-step')   
                eye = torch.eye(V_b.shape[0], device=DEVICE, dtype=TORCH_DTYPE)
                V_b += eye*EIGVAL_TOL

            # print(f'Final Loss: {-logmarginal.item():.4f}' ) 

            if verbose:
                print(f'\nTime spent for E-steps:       {time_estep_total:.3f}s,') 
                print(f'Time spent for f params:      {time_f_params_total:.3f}s')
                # print(f'Time spent computing Lambda0: {time_lambda0_estimation:.3f}s')
                print(f'Time spent for m / V update:  {time_estep_total-time_f_params_total:.3f}s')
                print(f'Time spent for M-steps:       {time_mstep_total:.3f}s')
                print(f'Time spent for All-steps:     {time_estep_total+time_mstep_total:.3f}s')
                print(f'Time spent computing Kernels: {time_computing_kernels:.3f}s')
                print(f'Time spent computing Loss:    {time_computing_loss:.3f}s')
                print(f'\nTime total after init:        {time.time()-start_time_loop:.3f}s')
                print(f"Time total before init:       {time.time()-start_time_before_init:.3f}s")
            if not silent:
                print(f'Final Loss: {-logmarginal.item():.4f}' )
            # Reduce values_track dictionary to the first 'iteration' elements`
            for key in values_track.keys():
                for subkey in values_track[key].keys():
                    values_track[key][subkey] = values_track[key][subkey][:fit_parameters['maxiter']] # Last index not be included 

            hyperparams_tuple = (theta, theta_lower_lims, theta_higher_lims)

            fit_model = {
                'fit_parameters':    fit_parameters,
                'final_kernel':      final_kernel,
                'err_dict':          err_dict,
                'xtilde':            xtilde,
                'spike_counts':      r,
                'hyperparams_tuple': hyperparams_tuple,
                'f_params':          f_params,
                'm_b':               m_b,
                'V_b':               V_b,
                'C':                 C,
                'mask':              mask,
                'K_tilde_b':         K_tilde_b,
                'K_tilde_inv_b':     K_tilde_inv_b,
                'K_b':               K_b,
                'Kvec':              Kvec,
                'B':                 B,
                'values_track':      values_track
            }

            #region _________ Memory usage___________
            # memory = 0
            # for dict in values_track.values():
            #     for key in dict.keys():
            #         if isinstance(dict[key], tuple):
            #             for i in range(len(dict[key])):
            #                 memory += dict[key][i].element_size() * dict[key][i].nelement()
            #             # print(f'{key} memory: {memory / (1024 ** 2):.2f} MB')
            #         else:
            #             memory += dict[key].element_size() * dict[key].nelement()
            #             # print(f'{key} memory: {dict[key].element_size() * dict[key].nelement() / (1024 ** 2):.2f} MB')

            # # Convert bytes to megabytes (MB)
            # total_memory_MB = memory / (1024 ** 2)
            # print(f'\nFinal Total values_track memory on GPU: {total_memory_MB:.2f} MB')
            # # Allocated memory
            # allocated_bytes = torch.cuda.memory_allocated()
            # allocated_MB = allocated_bytes / (1024 ** 2)
            # print(f"Final Allocated memory: {allocated_MB:.2f} MB")

            # # Reserved (cached) memory
            # reserved_bytes = torch.cuda.memory_reserved()
            # reserved_MB = reserved_bytes / (1024 ** 2)
            # print(f"Final Reserved (cached) memory: {reserved_MB:.2f} MB")
            #endregion _________ Memory usage___________
        else:
            print("Returning None as fit model")
            fit_model = None

        return fit_model, err_dict
    
        # else:
            # raise Exception('Error')

