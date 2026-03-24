# Tomography Reconstruction Notebooks

Included in this folder are three files:

- `tomography_01_lite.ipynb`: A lite version of the reconstruction notebook, useful for quick testing and debugging.
- `tomography_02_full.ipynb`: The full reconstruction notebook, containing all the necessary code for a complete reconstruction.
- `tomography_03_preprocessing.ipynb`: A notebook for preprocessing the data before reconstruction. # IN PROGRESS

# Multi-GPU Support

These notebooks are not intended for multi-node or multi-GPU workflows and are designed for single-GPU use only. There are currently no clean ways of running a notebook with multiple GPUs, please see the `hpc` directory for multi-GPU support through scripts using `torchrun`.