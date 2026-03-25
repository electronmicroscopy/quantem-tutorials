# HPC (NERSC) Tomography Reconstructions

Included in this folder are two files:

- `run.sh`: Initializes all necessary parameters for `torch` to see all gpus across nodes using `torchrun`.
- `run_job.sh`: A sample job script for submitting to the Slurm scheduler.
- `tomography_recon.py`: Contains the full reconstruction script; similar to `Scripts/tomography_02_full.py`
- `tensorboard_nersc.ipynb`: A Jupyter notebook for visualizing tensorboard logs on NERSC.

**Note: The first iteration of the reconstruction takes some time to initialize due to `multiprocessing_context="spawn"` which is a slightly slower way to initialize the multiprocessing context but is more stable across different systems.**

# Compatibility Table

| System | Python Ver. | PyTorch Ver. | Status |
|--------|-------------|--------------|--------|
| NERSC Perlmutter | >=3.10 | >=2.10.0 | ✅ |

# Installing Conda Environments 

Please refer to: https://docs.nersc.gov/development/languages/python/nersc-python/. It is advised to put your environments in `/global/common/software` for the fastest performance.

You will need to clone the quantem repository: `https://github.com/electronmicroscopy/quantem/` and install it in your conda environment using `pip install -e .`.

## Installation Steps

We recommend the installation order as follows:

1. Create a conda environment in `/global/common/software` using the `--prefix` flag:
   ```bash
   conda create --prefix /global/common/software/your_env_name python=3.xx.xx
   ```
2. Install torch with CUDA support by looking at [PyTorch's official installation page](https://pytorch.org/get-started/locally/). **Note: Currently CUDA 13.0 is the default version of Torch that will be installed. You may want to specify the torch version if you need a specific one, see [here](https://pytorch.org/get-started/locally/) for more details**:

   ```bash
   conda install pytorch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuxxx 
   # replace xxx with your CUDA version, or remove this if CUDA 13.0 is wanted
   ```
3. Clone the `quantem` repository and install it in your conda environment:
   ```bash
   git clone https://github.com/electronmicroscopy/quantem.git
   cd quantem
   pip install -e .
   ```
4. Edit `run.sh` (for interactive nodes) and `run_job.sh` (for batch jobs):
    - `run.sh`:
        - Replace `/global/common/software/mxxxx/user/conda/quantem` with the path to your conda environment
    - `run_job.sh`:
        - Replace `mxxxx` with your NERSC project ID
        - Replace `'INSERT USER HERE'` with your NERSC username
        - Replace `/global/common/software/mxxxx/user/conda/quantem` with the path to your conda environment
        - Adjust the number of nodes and GPUs per node as needed
        - Adjust the walltime as needed (default is 4 hours)
        - *If* using this script for longer jobs, change the queue from `debug` to `regular` and adjust the walltime accordingly.
5. (Optional) Here is an example of how to allocate an interactive job on Perlmutter which will instantiate all the required variables for `torchrun` and `DDP`:

   ```bash
   salloc -q interactive -C gpu -A mxxxx --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 --time=04:00:00
   ```

# Shifter Docker Environment

**3/22/2026 In Progress**

# FAQ

## Compatibility with HPC Systems

These tutorials have been heavily tested using NERSC's Perlmutter system. They should work on other HPC systems with similar configurations, but may require adjustments to the module loading and conda environment paths.

## How to run?

There are two ways to run the script across multiple nodes. On Perlmutter, you can launch an interactive job using `salloc`. The maximum amount of time for an interactive job is 4 hours, and the command for allocating one can be used here:

`salloc -q interactive -C gpu -A m5241 --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=32 --time=04:00:00`

Once the job is allocated, you can just run `sh run.sh` or `batch run.sh`. 

Alternatively, you can submit a job using `sbatch run_job.sh`. Usual parameters can be adjusted within the file.

These files should work on these tutorial files, but will potentially require adjustments when using your own data.

## Tensorboard Visualization

See `tensorboard_nersc.ipynb` for an example of how to visualize tensorboard logs on NERSC. For different HPC systems, you may need to use a different method to visualize tensorboard logs via port-forwarding.