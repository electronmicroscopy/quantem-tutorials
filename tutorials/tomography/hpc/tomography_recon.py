from quantem.tomography.tomography import TomographyConventional, Tomography
from quantem.tomography.dataset_models import TomographyPixDataset, TomographyINRDataset, TomographyINRPretrainDataset
from quantem.tomography.object_models import ObjectINR, ObjectPixelated
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.core.ml.inr import HSiren
import numpy as np

from quantem.core.utils.tomography_utils import fourier_binning
from quantem.core.visualization import show_2d
import torch

"""
Example script to run the tomography reconstruction on HPC (NERSC).

- Major difference is not needing to specify the device, as it will be automatically set by the DDP framework.

*Cedric Lim, 2/2/26*
"""


# Load Phantom Dataset
tilt_series = np.load('../../../data/tilt_series_1_deg_tilt_axis.npy')
tilt_angles = np.load('../../../data/tilt_angles_1_deg_tilt_axis.npy')

tilt_series = np.array([fourier_binning(img, (100, 100)) for img in tilt_series]) # Cropped down to 100x100 for speed

dset = TomographyINRDataset(
    tilt_stack = tilt_series,
    tilt_angles = tilt_angles,
)

# Initialize INR Model
model = HSiren(alpha = 1, winner_initialization = True)

# Initialize INR Object
obj_inr = ObjectINR(
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
tomo_inr = Tomography(
    dset = dset,
    obj_model = obj_inr,
    logger = logger,
)

# Define optimizer and scheduler parameters - only optimizing the object.

optimizer_params = {
    "object": {
        "type": "adam",
        "lr": 5e-4,
    },
}

scheduler_params = {
    "object": {
        "type": "linear",
    },
}

# Define constraints

constraints = {
    "tv_vol": 5e-7,
    "positivity": True,
}

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
    optimizer_params = optimizer_params,
    scheduler_params = scheduler_params,
    constraints = constraints,
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

    
