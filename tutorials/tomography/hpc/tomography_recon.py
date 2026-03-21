from quantem.tomography.tomography import TomographyConventional, Tomography
from quantem.tomography.dataset_models import TomographyPixDataset, TomographyINRDataset, TomographyINRPretrainDataset, DatasetConstraintParams
from quantem.tomography.object_models import ObjectINR, ObjectPixelated, ObjConstraintParams
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.core.ml.inr import HSiren
from quantem.core.ml.optimizer_mixin import SchedulerParams, OptimizerParams
import numpy as np

from quantem.core.utils.tomography_utils import fourier_binning
from quantem.core.visualization import show_2d
import torch

import warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context") # Ignoring CuBLAS errors for now since it fixes it self.

"""
Example script to run the tomography reconstruction on HPC (NERSC).

- Major difference is not needing to specify the device, as it will be automatically set by the DDP framework.

*Cedric Lim, 2/2/26*
"""


# Load Phantom Dataset
tilt_series = np.load('../../../data/tilt_series_1_deg_tilt_axis.npy')
tilt_angles = np.load('../../../data/tilt_angles_1_deg_tilt_axis.npy')

tilt_series = np.array([fourier_binning(img, (100, 100)) for img in tilt_series]) # Cropped down to 100x100 for speed

dset = TomographyINRDataset.from_data(
    tilt_stack = tilt_series,
    tilt_angles = tilt_angles,
)

# Initialize INR Model
model = HSiren(alpha = 1, winner_initialization = True)

# Initialize INR Object
obj_inr = ObjectINR.from_model(
    shape = (100, 100, 100),
    model = model,
)

# Define a logger
logger = LoggerTomography(
    log_dir = "../../../outputs/tomography/tutorial_02_scripts/",
    run_prefix = "inr_tomography_warmup_cosineanneal_hpc",
    run_suffix = "",
    log_images_every = 2,
)

# Initialize INR-Based Tomography Object
tomo_inr = Tomography.from_models(
    dset = dset,
    obj_model = obj_inr,
    logger = logger,
    verbose = False,
)

# Define optimizer and scheduler parameters - only optimizing the object.

optimizer_params = {
    "object": OptimizerParams.Adam(
        lr = 1e-4,
    ),
    # "pose": OptimizerParams.Adam(
    #     lr = 1e-2,
    # )
}
"""
All available scheduler params are in `core/ml/optimizer_mixin.py`

Scheduler types: 'cyclic', 'plateau', 'exp', 'gamma', 'linear', 'cosine_annealing'
Keyword arguments follow PyTorch scheduler documentation.
"""

scheduler_params = {
    "object": SchedulerParams.Plateau(
        mode = "min",
        factor = 0.5,
        patience = 10,
        threshold = 1e-3,
        min_lr = 1e-7,
    ),
    # "pose": SchedulerParams.Plateau(
    #     mode = "min",
    #     factor = 0.5,
    #     patience = 10,
    #     threshold = 1e-3,
    #     min_lr = 1e-7,
    # )
}

"""
Defining the constraints that we want to apply to the object and dataset. In this case
adding a total-variational loss, enforcing positivity, and a shrinkage constraint.

For the dataset we can add a 1-D total-variational loss to the shifts and z-shifts.
However this may not be necessary depending on the dataset.
"""

obj_constraints = ObjConstraintParams.ObjINRConstraints(
    positivity = True,
    sparsity = 1e-6,
    tv_vol = 1e-4,
)

## Dataset constraints not necessarily needed.

dataset_constraints = DatasetConstraintParams.BaseTomographyDatasetConstraints(
    tv_shifts = 1e-6, # 1-D regularizer for the shift optimization
    tv_zs = 1e-6, # 1-D regularizer for the z-shift optimization.
)


# Warmup Schedule for 10 epochs

num_samples_per_ray = [
    (0, 20),
    (1, 20),
    (2, 40),
    (3, 40),
    (4, 60),
    (4, 60),
    (6, 80),
    (7, 80),
    (8, 100),
    (9, 100),
]

tomo_inr.reconstruct(
    num_iter = 10,
    batch_size = 256,
    optimizer_params = optimizer_params,
    scheduler_params = scheduler_params,
    obj_constraints = obj_constraints,
    dset_constraints = dataset_constraints,
    num_samples_per_ray = num_samples_per_ray,
    num_workers = 32,
)

# Initialzie pose optimizer 

optimizer_params = {
    "pose": {
        "type": "adam",
        "lr": 1e-2,
    }
}

# Define new schedulers for both optimizers using CosineAnnealing
scheduler_params = {
    "object": {
        "type": "cosine_annealing",
    },
    "pose": {
        "type": "cosine_annealing",
    }
}

# Reconstruct

tomo_inr.reconstruct(
    num_iter = 100,
    optimizer_params = optimizer_params,
    scheduler_params = scheduler_params,
    num_samples_per_ray = 100,
)

    
torch.distributed.destroy_process_group()