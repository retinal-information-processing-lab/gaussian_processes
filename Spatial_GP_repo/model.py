import os
import torch
import pickle
import shutil
import copy
from typing import Dict, Tuple, List, Any, Optional, Callable, Union
import numpy as np
import matplotlib.pyplot as plt

class GPModel:
    """
    A class representing a Gaussian Process model for neural response prediction.
    
    This class encapsulates the model parameters, hyperparameters, and training state
    that were previously stored in a dictionary structure.
    """
    class KernelValues:
        """
        A nested class to store kernel-related values and matrices.
        
        This replaces the dictionary-based storage of kernel components with a
        structured class with proper typing and documentation.
        """
        
        def __init__(self, 
                    C: Optional[torch.Tensor] = None,
                    mask: Optional[torch.Tensor] = None, 
                    K_tilde: Optional[torch.Tensor] = None,
                    K: Optional[torch.Tensor] = None,
                    Kvec: Optional[torch.Tensor] = None,
                    B: Optional[torch.Tensor] = None,
                    K_tilde_b: Optional[torch.Tensor] = None,
                    K_tilde_inv_b: Optional[torch.Tensor] = None,
                    K_b: Optional[torch.Tensor] = None,
                    KKtilde_inv_b: Optional[torch.Tensor] = None):
            """
            Initialize KernelValues with kernel matrices and related components.
            
            Args:
                C: Spatial covariance matrix
                mask: Mask for active pixels
                K_tilde: Kernel matrix for inducing points
                K: Kernel matrix between data and inducing points
                Kvec: Diagonal elements of the kernel matrix
                B: Eigenvectors matrix for projection
                K_tilde_b: Projected inducing point kernel matrix
                K_tilde_inv_b: Inverse of the projected inducing point kernel matrix
                K_b: Projected cross-covariance matrix
                KKtilde_inv_b: Product of K_b and K_tilde_inv_b
            """
            self.C = C
            self.mask = mask
            self.K_tilde = K_tilde
            self.K = K
            self.Kvec = Kvec
            self.B = B
            self.K_tilde_b = K_tilde_b
            self.K_tilde_inv_b = K_tilde_inv_b
            self.K_b = K_b
            self.KKtilde_inv_b = KKtilde_inv_b
        
        @classmethod
        def from_dict(cls, kernel_dict: Dict) -> 'GPModel.KernelValues':
            """Create a KernelValues instance from a dictionary."""
            return cls(
                C=kernel_dict.get('C'),
                mask=kernel_dict.get('mask'),
                K_tilde=kernel_dict.get('K_tilde'),
                K=kernel_dict.get('K'),
                Kvec=kernel_dict.get('Kvec'),
                B=kernel_dict.get('B'),
                K_tilde_b=kernel_dict.get('K_tilde_b'),
                K_tilde_inv_b=kernel_dict.get('K_tilde_inv_b'),
                K_b=kernel_dict.get('K_b'),
                KKtilde_inv_b=kernel_dict.get('KKtilde_inv_b')
            )
        
        def to_dict(self) -> Dict:
            """Convert to a dictionary for backward compatibility."""
            kernel_dict = {}
            for attr in ['C', 'mask', 'K_tilde', 'K', 'Kvec', 'B',
                        'K_tilde_b', 'K_tilde_inv_b', 'K_b', 'KKtilde_inv_b']:
                value = getattr(self, attr)
                if value is not None:
                    kernel_dict[attr] = value
            return kernel_dict
        
        def __repr__(self) -> str:
            """String representation of KernelValues."""
            attributes = []
            for attr in ['C', 'mask', 'K_tilde', 'K', 'Kvec', 'B',
                        'K_tilde_b', 'K_tilde_inv_b', 'K_b', 'KKtilde_inv_b']:
                value = getattr(self, attr)
                if value is not None:
                    if hasattr(value, 'shape'):
                        attributes.append(f"{attr}[shape={tuple(value.shape)}]")
                    else:
                        attributes.append(f"{attr}={value}")
            return f"KernelValues({', '.join(attributes)})"

    class Hyperparameters:
        """
        A nested class to store and manage model hyperparameters that were previously
        stored in hyperparams_tuple.
        """
        
        def __init__(self, theta=None, theta_lower_lims=None, theta_higher_lims=None):
            """
            Initialize hyperparameters with values and bounds.
            
            Args:
                theta: Dictionary of hyperparameters or None
                theta_lower_lims: Dictionary of lower bounds or None
                theta_higher_lims: Dictionary of upper bounds or None
            """
            # Store values directly as public attributes
            if theta is not None:
                self.theta = {k: v for k, v in theta.items()}
            else:
                self.theta = {}
                
            # Default lower bounds if not provided
            if theta_lower_lims is not None:
                self.theta_lower_lims = theta_lower_lims
            else:
                low_lim = -torch.tensor(1.)
                self.theta_lower_lims = {
                    'sigma_0': 0, 
                    'eps_0x': low_lim, 
                    'eps_0y': low_lim, 
                    '-2log2beta': -float('inf'), 
                    '-log2rho2': -float('inf'), 
                    'Amp': 0.
                }
            
            # Default upper bounds if not provided
            if theta_higher_lims is not None:
                self.theta_higher_lims = theta_higher_lims
            else:
                upp_lim = torch.tensor(1.)
                self.theta_higher_lims = {
                    'sigma_0': float('inf'), 
                    'eps_0x': upp_lim, 
                    'eps_0y': upp_lim, 
                    '-2log2beta': float('inf'), 
                    '-log2rho2': float('inf'), 
                    'Amp': float('inf')
                }
        
        # Dictionary-like access for direct parameter access
        def __getattr__(self, key):
            """Allow accessing parameters as attributes (e.g., hyper.eps_0x)"""
            if key in self.theta:
                return self.theta[key]
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")
                
        def __getitem__(self, key):
            """Dictionary-style access (e.g., hyper['eps_0x'])"""
            return self.theta.get(key)
            
        def __setitem__(self, key, value):
            """Dictionary-style setting (e.g., hyper['eps_0x'] = value)"""
            self.theta[key] = value
            
        def items(self):
            """Return items from theta dictionary for iteration"""
            return self.theta.items()
            
        def set_requires_grad(self, freeze_list=None):
            """
            Set requires_grad=True for parameters not in freeze_list.
            
            Args:
                freeze_list: List of parameter names to freeze (keep requires_grad=False)
            """
            if freeze_list is None:
                freeze_list = []
                
            for key, value in self.theta.items():
                if hasattr(value, 'requires_grad_') and key not in freeze_list:
                    self.theta[key] = value.requires_grad_()
        
        def to_tuple(self):
            """Convert to the legacy (theta, lower_bounds, upper_bounds) tuple."""
            return (
                self.theta,
                self.theta_lower_lims,
                self.theta_higher_lims
            )
            
        @classmethod
        def from_tuple(cls, hyperparams_tuple):
            """Create from the legacy tuple representation."""
            if not hyperparams_tuple or len(hyperparams_tuple) < 3:
                Warning("Creating empty Hyperparameters object")
                return cls()
                
            return cls(
                theta=hyperparams_tuple[0],
                theta_lower_lims=hyperparams_tuple[1],
                theta_higher_lims=hyperparams_tuple[2]
            )
            
        def copy(self):
            """Create a deep copy of the hyperparameters."""
            import copy
            return self.__class__(
                theta=copy.deepcopy(self.theta),
                theta_lower_lims=copy.deepcopy(self.theta_lower_lims),
                theta_higher_lims=copy.deepcopy(self.theta_higher_lims)
            )
            
        def __repr__(self):
            """String representation showing key parameter values."""
            parts = []
            for key, value in self.theta.items():
                if hasattr(value, 'item'):
                    parts.append(f"{key}={value.item():.4f}")
                else:
                    parts.append(f"{key}={value}")
            return f"Hyperparameters({', '.join(parts)})"

    def __init__(self, model_dict=None, **kwargs):
        """
        Initialize a GPModel either from an existing model dictionary or from scratch.
        
        Args:
            model_dict (dict, optional): An existing model dictionary to convert to a class instance.
            **kwargs: Parameters to initialize a model from scratch.
        """
        # Direct fit parameter attributes
        self.ntilde = None
        self.maxiter = None
        self.nMstep = None
        self.nEstep = None
        self.nFparamstep = None
        self.kernfun = None
        self.cellid = None
        self.n_px_side = None
        self.in_use_idx = None
        self.all_idx_perm = None  # The permutation of the indexes of the images
        self.xtilde_idx = None
        self.start_idx = None
        self.lr_Mstep = None
        self.lr_Fparamstep = None
        self.min_tolerance = None
        self.eigval_tol = None
        
        # Other model components - no more leading underscores
        self.xtilde = None
        self.hyperparams_tuple = None
        self.hyperparams_obj = None  # Public attribute for Hyperparameters object
        self.f_params = {}
        self.values_track = None
        self.description = None
        
        # Kernel specific components - no more leading underscores
        self.final_kernel_dict = None   # kept for backward compatibility
        self.init_kernel_dict = None    # kept for backward compatibility
        self.final_kernel_values = None # KernelValues instance for final kernel
        self.init_kernel_values = None # KernelValues instance for init_kernel

        self.err_dict = {'is_error': False, 'error_message': None, 'during_init': False}
        self.m_b = None
        self.V_b = None
        self.C = None
        self.mask = None
        self.K_tilde_b = None
        self.K_tilde_inv_b = None
        self.K_b = None
        self.Kvec = None
        self.B = None
        
        # If a model dictionary is provided, populate from it
        if model_dict is not None:
            self.from_dict(model_dict)
            if self.hyperparams_tuple:
                self.hyperparams_obj = self.Hyperparameters.from_tuple(
                    self.hyperparams_tuple)
        else:
            # Initialize from kwargs

            # Check for old model to copy general parameters that dont change with new images
            for key, value in kwargs.items():
                if key == 'old_model':
                    self.set_general_model_params(kwargs['old_model'])                
                setattr(self, key, value)
            # Check for kernel values in kwargs - special handling for dictionary conversion
            if 'init_kernel' in kwargs:
                self.init_kernel_values = self.KernelValues.from_dict(kwargs['init_kernel'])
    
    def from_dict(self, model_dict: Dict):
        """Convert a model dictionary to GPModel attributes."""
        # Extract fit parameters to direct attributes
        if 'fit_parameters' in model_dict:
            fit_params = model_dict['fit_parameters']
            
            # Set each parameter as a direct attribute
            self.ntilde = fit_params.get('ntilde')
            self.maxiter = fit_params.get('maxiter')
            self.nMstep = fit_params.get('nMstep')
            self.nEstep = fit_params.get('nEstep')
            self.nFparamstep = fit_params.get('nFparamstep')
            self.kernfun = fit_params.get('kernfun')
            self.cellid = fit_params.get('cellid')
            self.n_px_side = fit_params.get('n_px_side')
            self.in_use_idx = fit_params.get('in_use_idx')
            self.all_idx_perm = fit_params.get('all_idx_perm')
            self.xtilde_idx = fit_params.get('xtilde_idx')
            self.start_idx = fit_params.get('start_idx')
            self.lr_Mstep = fit_params.get('lr_Mstep')
            self.lr_Fparamstep = fit_params.get('lr_Fparamstep')
            self.min_tolerance = fit_params.get('min_tolerance')
            self.eigval_tol = fit_params.get('eigval_tol')
        
        # Set kernel and GP-specific components
        if 'final_kernel' in model_dict:
            self.final_kernel_dict = model_dict['final_kernel']
            self.final_kernel_values = self.KernelValues.from_dict(model_dict['final_kernel'])
            
        if 'init_kernel' in model_dict:
            self.init_kernel_dict = model_dict['init_kernel']
            self.init_kernel_values = self.KernelValues.from_dict(model_dict['init_kernel'])
          
        # Set other model components
        if 'xtilde' in model_dict:
            self.xtilde = model_dict['xtilde']
            
        if 'hyperparams_tuple' in model_dict:
            self.hyperparams_tuple = model_dict['hyperparams_tuple']
            
        if 'f_params' in model_dict:
            self.f_params = copy.deepcopy(model_dict['f_params'])
            
        if 'values_track' in model_dict:
            self.values_track = model_dict['values_track']
            
        if 'description' in model_dict:
            self.description = model_dict['description']
                        
        if 'err_dict' in model_dict:
            self.err_dict = model_dict['err_dict']
            
        if 'm_b' in model_dict:
            self.m_b = model_dict['m_b']
            
        if 'V_b' in model_dict:
            self.V_b = model_dict['V_b']
            
        if 'C' in model_dict:
            self.C = model_dict['C']
            
        if 'mask' in model_dict:
            self.mask = model_dict['mask']
            
        if 'K_tilde_b' in model_dict:
            self.K_tilde_b = model_dict['K_tilde_b']
            
        if 'K_tilde_inv_b' in model_dict:
            self.K_tilde_inv_b = model_dict['K_tilde_inv_b']
            
        if 'K_b' in model_dict:
            self.K_b = model_dict['K_b']
            
        if 'Kvec' in model_dict:
            self.Kvec = model_dict['Kvec']
            
        if 'B' in model_dict:
            self.B = model_dict['B']
        
        # Sync kernel components with KernelValues for consistency
        # self.sync_kernel_components()
    
    def sync_kernel_components(self):
        """Synchronize individual kernel components with kernel values objects."""
        # Use final kernel values if available, otherwise use working kernel values
        kernel_values = self.final_kernel_values or self.init_kernel_values
        
        if kernel_values:
            self.C = kernel_values.C
            self.mask = kernel_values.mask
            self.K_tilde_b = kernel_values.K_tilde_b
            self.K_tilde_inv_b = kernel_values.K_tilde_inv_b
            self.K_b = kernel_values.K_b
            self.Kvec = kernel_values.Kvec
            self.B = kernel_values.B
    
    def to_dict(self) -> Dict:
        """Convert the GPModel to a dictionary for backward compatibility."""
        # Create fit parameters dictionary
        fit_parameters = {}
        for attr in ['ntilde', 'maxiter', 'nMstep', 'nEstep', 'nFparamstep', 'kernfun', 
                    'cellid', 'n_px_side', 'in_use_idx', 'all_idx_perm', 'xtilde_idx', 'start_idx',
                    'lr_Mstep', 'lr_Fparamstep', 'min_tolerance', 'eigval_tol']:
            value = getattr(self, attr)
            if value is not None:
                fit_parameters[attr] = value        
        
        model_dict = {'fit_parameters': fit_parameters}

        # Add kernel values
        if self.final_kernel_values is not None:
            model_dict['final_kernel'] = self.final_kernel_values.to_dict()
        elif self.final_kernel_dict is not None:
            model_dict['final_kernel'] = self.final_kernel_dict
            
        if self.init_kernel_values is not None:
            model_dict['init_kernel'] = self.init_kernel_values.to_dict()
        elif self.init_kernel_dict is not None:
            model_dict['init_kernel'] = self.init_kernel_dict
            
        # Add other model components
        if self.xtilde is not None:
            model_dict['xtilde'] = self.xtilde
            
        if self.hyperparams_tuple is not None:
            model_dict['hyperparams_tuple'] = self.hyperparams_tuple
            
        if self.f_params:
            model_dict['f_params'] = self.f_params
            
        if self.values_track is not None:
            model_dict['values_track'] = self.values_track
            
        if self.description is not None:
            model_dict['description'] = self.description
            
        # Add the individual components for backward compatibility
        if self.err_dict is not None:
            model_dict['err_dict'] = self.err_dict
            
        if self.m_b is not None:
            model_dict['m_b'] = self.m_b
            
        if self.V_b is not None:
            model_dict['V_b'] = self.V_b
            
        if self.C is not None:
            model_dict['C'] = self.C
            
        if self.mask is not None:
            model_dict['mask'] = self.mask
            
        if self.K_tilde_b is not None:
            model_dict['K_tilde_b'] = self.K_tilde_b
            
        if self.K_tilde_inv_b is not None:
            model_dict['K_tilde_inv_b'] = self.K_tilde_inv_b
            
        if self.K_b is not None:
            model_dict['K_b'] = self.K_b
            
        if self.Kvec is not None:
            model_dict['Kvec'] = self.Kvec
            
        if self.B is not None:
            model_dict['B'] = self.B
            
        return model_dict
    
    def set_general_model_params(self, old_model):
        '''
        Sets the general fit parameters for the new model, everything that does not changes by adding one image

        NB: values for f_params are not initialized even if they dont change by adding one image. 
            this is done to keep their initialization explicit in the script.

        '''
        self.maxiter       = old_model.maxiter
        self.nMstep        = old_model.nMstep
        self.nEstep        = old_model.nEstep
        self.nFparamstep   = old_model.nFparamstep
        self.kernfun       = old_model.kernfun
        self.n_px_side     = old_model.n_px_side
        self.all_idx_perm  = old_model.all_idx_perm
        self.lr_Mstep      = old_model.lr_Mstep
        self.lr_Fparamstep = old_model.lr_Fparamstep
        self.min_tolerance = old_model.min_tolerance
        self.eigval_tol    = old_model.eigval_tol

    # Only necessary getter - for derived property
    @property
    def theta(self):
        """Get the theta Hyperparameters object or create it if needed."""
        if self.hyperparams_obj is None:
            if self.hyperparams_tuple is not None and len(self.hyperparams_tuple) > 0:
                self.hyperparams_obj = self.Hyperparameters.from_tuple(self.hyperparams_tuple)
            else:
                Warning("Creating empty Hyperparameters object")
                self.hyperparams_obj = self.Hyperparameters()
        return self.hyperparams_obj
    
    # Optional: Update the hyperparams_tuple when needed for backward compatibility
    def update_hyperparams_tuple(self):
        """Update the hyperparams_tuple from the Hyperparameters object."""
        if self.hyperparams_obj:
            self.hyperparams_tuple = self.hyperparams_obj.to_tuple()

    def __repr__(self):
        """Return a detailed string representation of the GPModel."""
        parts = ["GPModel("]
        
        # Core model parameters
        if self.cellid is not None:
            parts.append(f"cell_id={self.cellid}")
        
        if self.n_px_side is not None:
            parts.append(f"n_px_side={self.n_px_side}")
        
        if self.ntilde is not None:
            parts.append(f"ntilde={self.ntilde}")
        
        # Training configuration
        if self.maxiter is not None:
            parts.append(f"maxiter={self.maxiter}")
        
        if self.nMstep is not None and self.nEstep is not None:
            parts.append(f"steps=M{self.nMstep}E{self.nEstep}")
        
        # Training status
        if self.values_track is not None and 'loss_track' in self.values_track:
            if 'logmarginal' in self.values_track['loss_track']:
                loss_vals = self.values_track['loss_track']['logmarginal']
                if len(loss_vals) > 0:
                    iter_completed = len(loss_vals) - loss_vals.eq(0).sum()
                    parts.append(f"trained={iter_completed}/{self.maxiter}")
                    if iter_completed > 0:
                        parts.append(f"final_loss={loss_vals[iter_completed-1]:.4f}")
        
        # Hyperparameters
        if self.hyperparams_tuple is not None and len(self.hyperparams_tuple) > 0:
            theta = self.hyperparams_tuple[0]
            if 'eps_0x' in theta and 'eps_0y' in theta:
                eps_x = theta['eps_0x'].item() if hasattr(theta['eps_0x'], 'item') else theta['eps_0x']
                eps_y = theta['eps_0y'].item() if hasattr(theta['eps_0y'], 'item') else theta['eps_0y']
                parts.append(f"center=({eps_x:.2f},{eps_y:.2f})")
            
            if '-2log2beta' in theta and hasattr(theta['-2log2beta'], 'item'):
                try:
                    from gaussian_processes.Spatial_GP_repo.utils import logbetaexpr_to_beta
                    beta = logbetaexpr_to_beta(theta['-2log2beta']).item()
                    parts.append(f"beta={beta:.2f}")
                except (ImportError, AttributeError):
                    pass
        
        # Dataset information
        if self.in_use_idx is not None:
            parts.append(f"train_samples={len(self.in_use_idx)}")

        if self.all_idx_perm is not None:
            parts.append(f"total_samples={len(self.all_idx_perm)}")
        
        # Error status if any
        if self.err_dict and self.err_dict.get('is_error'):
            parts.append(f"has_error=True")
            
        parts.append(")")
        return " ".join(parts)