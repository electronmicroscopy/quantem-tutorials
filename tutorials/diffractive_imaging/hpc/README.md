# HPC (NERSC) Ptychography Reconstructions

This folder contains scripts for running multi-node, multi-GPU ptychography reconstructions on NERSC's Perlmutter supercomputer. Single-node multi-GPU can be done directly in a notebook — see `ptycho_iter_08_multi_gpu.ipynb`.

**Files:**

- `run.sh` — launches `ptycho_ducky.py` via `torchrun` on an interactive allocation
- `run_job.sh` — SLURM batch job script
- `ptycho_ducky.py` — reconstruction script; reconstructs the ducky dataset across all allocated GPUs

---

# Compatibility Table

| System | Python Ver. | PyTorch Ver. | Status |
|--------|-------------|--------------|--------|
| NERSC Perlmutter | ≥3.10 | ≥2.1.0 | ✅ |

---

# Installing a Conda Environment

Please refer to the [NERSC Python docs](https://docs.nersc.gov/development/languages/python/nersc-python/). It is advised to put your environments in `/global/common/software` for the fastest performance.

## Installation Steps

1. Create a conda environment in `/global/common/software` using the `--prefix` flag:
   ```bash
   conda create --prefix /global/common/software/your_env_name python=3.11
   ```

2. Install PyTorch with CUDA support ([PyTorch installation page](https://pytorch.org/get-started/locally/)):
   ```bash
   conda install pytorch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   # Replace cu124 with your target CUDA version
   ```

3. Clone the `quantem` repository and install it:
   ```bash
   git clone https://github.com/electronmicroscopy/quantem.git
   cd quantem
   pip install -e .
   ```

4. Edit `run.sh` and `run_job.sh`:
   - Replace `mxxxx` with your NERSC project ID
   - Replace `INSERT_USER` / path with the path to your conda environment
   - Adjust `WORKDIR`, node count, walltime as needed

---

# Step-by-step Workflow

### 1. Preprocess on a workstation or login node

Run `ptycho_iter_08_multi_gpu.ipynb` through the preprocessing cell (or optionally a short single-GPU trial), then save:

```python
ptycho.save("ducky_preprocessed.zip")
```

### 2. Transfer files to Perlmutter

```bash
scp ducky_preprocessed.zip perlmutter.nersc.gov:/path/to/workdir/
scp ptycho_ducky.py perlmutter.nersc.gov:/path/to/workdir/
```

### 3a. Interactive job

Allocate nodes, then run `run.sh`:

```bash
salloc -q interactive -C gpu -A YOUR_ACCOUNT \
       --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
       --cpus-per-task=128 --time=04:00:00

bash run.sh
```

### 3b. Batch job

Edit `run_job.sh`, then submit:

```bash
sbatch run_job.sh
```

### 4. Transfer result back and visualize

```bash
scp perlmutter.nersc.gov:/path/to/workdir/ducky_result.zip .
```

```python
from quantem.diffractive_imaging import Ptychography
ptycho = Ptychography.from_file("ducky_result.zip")
ptycho.visualize()
```

---

# FAQ

## How does multi-node work?

`torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` environment variables. `ptycho.reconstruct()` detects these and automatically:

1. Initialises an NCCL process group across all ranks
2. Shards the diffraction dataset — rank `r` holds scan positions `[r·ceil(N/W), (r+1)·ceil(N/W)]`
3. Replicates the object and probe on every GPU
4. All-reduces object and probe gradients after each backward pass so all ranks stay in sync

## How do I tune batch size and learning rate?

Each rank processes `PTYCHO_BATCH_SIZE` scan positions per step, so the effective global batch is `PTYCHO_BATCH_SIZE × world_size`. A common starting point is to keep `batch_size` fixed and scale `lr` by `sqrt(world_size)`. Edit `PTYCHO_LR_OBJ` / `PTYCHO_LR_PROBE` in `run.sh` or `run_job.sh`.

## Memory planning

Peak VRAM per GPU ≈ `(N / world_size) × Qr × Qc × 4 bytes` for diffraction data, plus object and probe (replicated). Doubling the node count halves diffraction VRAM per GPU.

## NCCL hangs or slow multi-node

- Ensure `NCCL_SOCKET_IFNAME=hsn` is set — this routes traffic over Perlmutter's Slingshot interconnect instead of the slower management network.
- `FI_CXI_ATS=0` disables address translation for better NCCL performance.
- If `init_process_group` hangs, verify `MASTER_ADDR` is reachable from all worker nodes.

## `RuntimeError: shard() must be called after preprocess()`

The input zip was saved before `preprocess()` was called. Re-run the preprocessing notebook cell and re-save.
