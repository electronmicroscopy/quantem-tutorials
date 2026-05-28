"""
Multi-node multi-GPU ptychography reconstruction of the ducky dataset.

Prereqs
-------
Preprocess the ducky dataset from a notebook or login node first
(see ptycho_iter_08_multi_gpu.ipynb for the full workflow):

    from pathlib import Path
    from quantem.core.io import read_4dstem
    from quantem.diffractive_imaging import PtychoLite

    dset = read_4dstem(Path("path/to/ducky_dataset.zip"))
    ptycho = PtychoLite.from_dataset(dset, device="cpu")
    ptycho.preprocess()
    ptycho.save("ducky_preprocessed.zip")          # transfer this file to HPC

This script is then launched via torchrun (see run.sh / run_job.sh).
torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE automatically; quantem's
reconstruct() detects these and uses the distributed code path.
"""

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from quantem.core.ml.optimizer_mixin import OptimizerParams, SchedulerParams
from quantem.diffractive_imaging import Ptychography


def main() -> None:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    input_path = Path(os.environ.get("PTYCHO_INPUT", "ducky_preprocessed.zip"))
    output_path = Path(os.environ.get("PTYCHO_OUTPUT", "ducky_result.zip"))
    num_iters = int(os.environ.get("PTYCHO_ITERS", 200))
    batch_size_env = os.environ.get("PTYCHO_BATCH_SIZE", "")
    batch_size = int(batch_size_env) if batch_size_env else None
    lr_obj = float(os.environ.get("PTYCHO_LR_OBJ", 5e-2))
    lr_probe = float(os.environ.get("PTYCHO_LR_PROBE", 5e-2))

    if rank == 0:
        print(f"World size : {world_size}")
        print(f"Input      : {input_path}")
        print(f"Output     : {output_path}")
        print(f"Iterations : {num_iters}  |  batch_size/rank: {batch_size or 'all'}")

    # Each rank loads the full preprocessed state.
    # reconstruct() shards the diffraction data across ranks automatically.
    ptycho = Ptychography.from_file(input_path)

    if rank == 0:
        n = ptycho.dset.num_gpts
        roi = ptycho.dset.roi_shape
        full_mb = n * roi[0] * roi[1] * 4 / 1e6
        print(f"  {n} scan positions, detector {roi[0]}×{roi[1]}")
        print(f"  Full amplitudes ≈ {full_mb:.0f} MB  |  per-GPU ≈ {full_mb/world_size:.0f} MB")

    optimizer_params = {
        "object": OptimizerParams.Adam(lr=lr_obj),
        "probe": OptimizerParams.Adam(lr=lr_probe),
    }
    scheduler_params = {
        "object": SchedulerParams.Plateau(factor=0.5),
        "probe": SchedulerParams.Plateau(factor=0.5),
    }

    t0 = time.perf_counter()

    # reconstruct() detects RANK/WORLD_SIZE from torchrun, initialises NCCL,
    # shards the dataset, and all-reduces gradients after each backward pass.
    ptycho.reconstruct(
        num_iters=num_iters,
        reset=True,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        batch_size=batch_size,
        autograd=True,
    )

    elapsed = time.perf_counter() - t0

    if rank == 0:
        print(f"Finished in {elapsed:.1f}s  ({elapsed / num_iters:.2f} s/iter)")
        ptycho.save(output_path, mode="o")
        print(f"Saved to {output_path}")

    # Clean up distributed process group — required for torchrun multi-node.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
