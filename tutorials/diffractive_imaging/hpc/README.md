# HPC (NERSC) Ptychography Reconstructions

This folder contains scripts for running multi-node, multi-GPU ptychography reconstructions on NERSC's Perlmutter supercomputer. Single-node multi-GPU can be done directly in a notebook — see `ptycho_iter_08_multi_gpu.ipynb`.

**Files:**

- `ptycho_ducky.py` — reconstruction script. Loads + preprocesses + reconstructs end-to-end.
- `run.sh` — launches `ptycho_ducky.py` via `torchrun` on an interactive allocation.
- `run_job.sh` — SLURM batch job script.

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

# Step-by-step Workflow (all-in-one default)

`ptycho_ducky.py` runs end-to-end: load the raw 4D-STEM dataset, preprocess, then reconstruct across all allocated GPUs. With your environment + paths set in `run.sh` / `run_job.sh`, the script is meant to "just work".

### 1. Transfer the raw dataset and script to Perlmutter

```bash
scp ducky_dataset.zip perlmutter.nersc.gov:/path/to/workdir/
scp ptycho_ducky.py perlmutter.nersc.gov:/path/to/workdir/
```

### 2a. Interactive job

```bash
salloc -q interactive -C gpu -A YOUR_ACCOUNT \
       --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
       --cpus-per-task=128 --time=04:00:00

bash run.sh
```

### 2b. Batch job

Edit `run_job.sh`, then submit:

```bash
sbatch run_job.sh
```

### 3. Transfer the result back and visualize

```bash
scp perlmutter.nersc.gov:/path/to/workdir/ducky_result.zip .
```

```python
from quantem.diffractive_imaging import Ptychography
ptycho = Ptychography.from_file("ducky_result.zip")
ptycho.visualize()
```

---

# Recommended for large datasets: preprocess once, load on HPC

For datasets that don't comfortably fit in one GPU's VRAM, or when you want to iterate on optimizer / constraint settings without paying the preprocessing cost each time, it is **typically better to preprocess once on a workstation** and have the HPC job load the preprocessed `.zip` directly. Dataset preprocessing is not currently multi-GPU aware, so the all-in-one path redoes the same single-device work on every rank.

1. On a workstation:
   ```python
   import quantem as em
   from quantem.diffractive_imaging import (
       DetectorPixelated, ObjectPixelated, ProbePixelated,
       Ptychography, PtychographyDatasetRaster,
   )

   dset = em.io.load("ducky_dataset.zip")
   pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
   pdset.preprocess(com_fit_function="constant")

   ptycho = Ptychography.from_models(
       dset=pdset,
       obj_model=ObjectPixelated.from_uniform(
           obj_type="pure_phase", num_slices=1
       ),
       probe_model=ProbePixelated.from_params(
           probe_params={"energy": 80e3, "defocus": 500, "semiangle_cutoff": 20}
       ),
       detector_model=DetectorPixelated(),
       rng=42,
   )
   ptycho.preprocess(obj_padding_px=(32, 32))
   ptycho.save("ducky_preprocessed.zip", save_raw_data=True)
   ```

2. Transfer `ducky_preprocessed.zip` to Perlmutter and point `PTYCHO_INPUT` at it.

3. In `ptycho_ducky.py`, replace the all-in-one block with the commented-out template (a single `Ptychography.from_file(input_path)` call).

4. Submit as usual.

---

# FAQ

## How does multi-node work?

`torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` environment variables. `ptycho.reconstruct()` detects these and automatically:

1. Binds each rank to its `cuda:LOCAL_RANK` before initialising the NCCL process group (so NCCL communicator buffers land on the right GPU rather than leaking onto `cuda:0`).
2. Shards the diffraction dataset — rank `r` holds scan positions `[r · ceil(N/W), (r+1) · ceil(N/W)]`.
3. Replicates the object and probe on every GPU.
4. All-reduces object and probe gradients after each backward pass so all ranks stay in sync.

## Effective batch size with multi-GPU

Each rank processes `PTYCHO_BATCH_SIZE` scan positions per step, so the **effective batch size per optimizer step is `PTYCHO_BATCH_SIZE × world_size`**. Reconstructions with `W > 1` are therefore not bit-identical to a single-GPU run at the same `batch_size`. To match single-GPU optimizer dynamics, set `PTYCHO_BATCH_SIZE` to roughly `base_batch // world_size`; you give up some of the speedup but keep the optimizer trajectory comparable. LR scaling (`lr × W`) is sometimes suggested as compensation, but its benefit on these workloads isn't well established — prefer adjusting `batch_size` when in doubt.

## Memory planning

Peak VRAM per GPU ≈ `(N / world_size) × Qr × Qc × 4 bytes` for diffraction data, plus object and probe (replicated). Doubling the node count halves diffraction VRAM per GPU.

## NCCL hangs or slow multi-node

- Ensure `NCCL_SOCKET_IFNAME=hsn` is set — this routes traffic over Perlmutter's Slingshot interconnect instead of the slower management network.
- `FI_CXI_ATS=0` disables address translation for better NCCL performance.
- If `init_process_group` hangs, verify `MASTER_ADDR` is reachable from all worker nodes.
