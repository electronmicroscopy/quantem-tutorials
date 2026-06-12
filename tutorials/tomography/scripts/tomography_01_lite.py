from quantem.tomography.dataset_models import TomographyPixDataset, TomographyINRDataset
from quantem.tomography.tomography_lite import TomographyLiteConv, TomographyLiteINR
import numpy as np
from quantem.core.visualization import show_2d

from quantem.tomography.utils import fourier_cropping

"""
Example script to run the lite version of the INR-based tomography pipeline.

*Cedric Lim, 2/1/26*
"""

# Load Phantom Dataset
tilt_series = np.load('../../../data/tilt_series.npy')
tilt_angles = np.load('../../../data/tilt_angles.npy')

# Fourier crop
tilt_series = np.array([fourier_cropping(img, (100, 100)) for img in tilt_series]) # Cropped down to 100x100 for speed

# Initialize dataset
dset = TomographyINRDataset(
    tilt_stack = tilt_series,
    tilt_angles = tilt_angles,
)

# Initialize tomography object
tomography_inr = TomographyLiteINR.from_dataset(
    dset,
    device = 'cuda:0',
    log_dir = '../../../outputs/tomography/tutorial_01_scripts/tomo_inr_lite',
    log_images_every = 2,
)


constraints = {
    "positivity": True,
    "tv_vol": 1e-6,
}

tomography_inr.reconstruct(
    num_iter = 50,
    # reset = True,
    obj_lr = 1e-4,
    pose_lr = 1e-2,
    batch_size = 1024,
    num_workers = 32,
    learn_pose = True,
    scheduler_type = "cosine_annealing",
    constraints = constraints,
)

