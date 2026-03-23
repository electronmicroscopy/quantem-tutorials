# HPC (NERSC) Tomography Reconstructions

Included in this folder are two files:

- `run.sh`: Initializes all necessary parameters for `torch` to see all gpus across nodes using `torchrun`.
- `run_job.sh`: A sample job script for submitting to the Slurm scheduler.
- `tomography_recon.py`: Contains the full reconstruction script; similar to `Scripts/tomography_02_full.py`

# Compatibility Table

| System | Python Ver. | PyTorch Ver. | Status |
|--------|-------------|--------------|--------|
| NERSC Perlmutter | >=3.10 | >=2.10.0 | ✅ |

# Installing Conda Environments 

Please refer to: https://docs.nersc.gov/development/languages/python/nersc-python/. It is advised to put your environments in `/global/common/software` for the fastest performance.

You will need to clone the quantem repository: `https://github.com/electronmicroscopy/quantem/` and install it in your conda environment using `pip install -e .`.

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
