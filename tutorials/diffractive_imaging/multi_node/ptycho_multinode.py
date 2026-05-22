"""
Multi-node multi-GPU ptychography reconstruction script.

Launch with torchrun (single-node example):
    torchrun --nproc_per_node=4 ptycho_multinode.py --input ptycho_preprocessed.zip

Launch with torchrun (multi-node via SLURM, see SLURM script in README.md):
    torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        ptycho_multinode.py --input ptycho_preprocessed.zip

torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE automatically.
quantEM's reconstruct() detects these and uses the distributed code path.

Workflow:
  1. Run ptycho_iter_07_multi_gpu.ipynb on a workstation or login node to
     preprocess and optionally do a short trial reconstruction.
  2. Save: ptycho.save("ptycho_preprocessed.zip")
  3. Transfer the zip and this script to NERSC (or your HPC cluster).
  4. Submit the SLURM job using the template in README.md.
  5. Transfer the result back and reload in a notebook:
       from quantem.diffractive_imaging import Ptychography
       ptycho = Ptychography.from_file("result.zip", dset=dset)
"""

import argparse
import os
import time
from pathlib import Path

import torch

from quantem.core import config
from quantem.diffractive_imaging import Ptychography
from quantem.core.ml import OptimizerParams, SchedulerParams


def parse_args():
    p = argparse.ArgumentParser(description="Multi-node ptychography reconstruction")
    p.add_argument("--input", required=True, type=Path,
                   help="Path to preprocessed ptycho .zip saved from notebook")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path (default: input stem + _result.zip)")
    p.add_argument("--num_iters", type=int, default=200,
                   help="Number of reconstruction iterations")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size per rank (default: all positions on rank)")
    p.add_argument("--lr_obj", type=float, default=5e-2)
    p.add_argument("--lr_probe", type=float, default=5e-2)
    p.add_argument("--scheduler", choices=["none", "plateau", "exp"], default="plateau")
    return p.parse_args()


def main():
    args = parse_args()

    # torchrun sets these; fall back to single-process defaults
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank == 0:
        print(f"World size: {world_size} (ranks across all nodes)")
        print(f"Loading from: {args.input}")

    # Each rank loads the full preprocessed state from the saved zip.
    # reconstruct() will shard the diffraction data across ranks automatically.
    ptycho = Ptychography.from_file(args.input)

    if rank == 0:
        n = ptycho.dset.num_gpts
        roi = ptycho.dset.roi_shape
        full_mb = n * roi[0] * roi[1] * 4 / 1e6
        print(f"  {n} scan positions, {roi[0]}×{roi[1]} detector")
        print(f"  Full amplitudes: {full_mb:.0f} MB, per-GPU: {full_mb/world_size:.0f} MB")
        print(f"Running {args.num_iters} iterations, batch_size={args.batch_size} per rank")

    opt_params = {
        "object": OptimizerParams.Adam(lr=args.lr_obj),
        "probe": OptimizerParams.Adam(lr=args.lr_probe),
    }
    scheduler_params = {
        "object": {"name": args.scheduler, "factor": 0.5},
        "probe": {"name": args.scheduler, "factor": 0.5},
    }

    t0 = time.perf_counter()

    # reconstruct() detects RANK/WORLD_SIZE env vars set by torchrun and
    # automatically shards data, initialises NCCL, and all-reduces gradients.
    ptycho.reconstruct(
        num_iters=args.num_iters,
        reset=True,
        optimizer_params=opt_params,
        scheduler_params=scheduler_params,
        batch_size=args.batch_size,
        autograd=True,
    )

    elapsed = time.perf_counter() - t0
    if rank == 0:
        print(f"Reconstruction finished in {elapsed:.1f}s ({elapsed/args.num_iters:.2f}s/iter)")

        out_path = args.output or args.input.parent / (args.input.stem + "_result.zip")
        ptycho.save(out_path, mode="o")
        print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
