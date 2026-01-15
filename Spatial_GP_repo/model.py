# Backward compatibility shim
# model.py has been moved to GP_model/model.py
# This file re-exports everything for existing imports to continue working

from gaussian_processes.Spatial_GP_repo.GP_model.model import *
from gaussian_processes.Spatial_GP_repo.GP_model.model import GPModel
