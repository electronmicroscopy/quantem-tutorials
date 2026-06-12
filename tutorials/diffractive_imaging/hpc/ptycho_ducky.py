"""
Multi-node multi-GPU ptychography reconstruction of the ducky dataset.

This script runs end-to-end: load the raw 4D-STEM dataset, preprocess it,
and reconstruct. It is designed to "just work" with minimal changes — set
your account ID + paths in run.sh / run_job.sh and submit.

For large datasets it is typically better to preprocess once on a workstation
and load the preprocessed zip on HPC, so the reconstruction job doesn't
re-pay the preprocessing cost on every rank (and dataset preprocessing is
not currently multi-GPU aware). See the commented-out block in main() for
the alternative load-only path.

Launch via torchrun (see run.sh / run_job.sh). torchrun sets RANK, LOCAL_RANK,
and WORLD_SIZE; quantEM's reconstruct() detects them and uses the distributed
code path automatically.
"""


import os
import time
from pathlib import Path

import torch.distributed as dist

import quantem as em
from quantem.core.ml import OptimizerParams, SchedulerParams
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    Ptychography,
    PtychographyDatasetRaster,
    # Uncomment to use constraint dataclasses in the reconstruct() call below.
    # PtychoObjConstraintParams,
    # PtychoProbeConstraintParams,
)


def _env(name: str, default, cast=str):
    """Read an env var; fall back to ``default`` when unset or empty."""
    raw = os.environ.get(name, "")
    return cast(raw) if raw else default


def main() -> None:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # --- Configuration (override via env vars in run.sh / run_job.sh) ---
    input_path = Path(_env("PTYCHO_INPUT", "ducky_dataset.zip"))
    output_path = Path(_env("PTYCHO_OUTPUT", "ducky_result.zip"))
    num_iters = _env("PTYCHO_ITERS", 200, int)
    batch_size = _env("PTYCHO_BATCH_SIZE", None, int)
    lr_obj = _env("PTYCHO_LR_OBJ", 5e-2, float)
    lr_probe = _env("PTYCHO_LR_PROBE", 5e-2, float)
    obj_padding_px = _env("PTYCHO_OBJ_PADDING", 32, int)

    # Probe initial conditions — edit for your dataset
    probe_energy = _env("PTYCHO_PROBE_ENERGY", 80e3, float)  # eV
    probe_defocus = _env("PTYCHO_PROBE_DEFOCUS", 500.0, float)  # Å
    probe_semiangle = _env("PTYCHO_PROBE_SEMIANGLE", 20.0, float)  # mrad

    if rank == 0:
        print(f"World size : {world_size}")
        print(f"Input      : {input_path}")
        print(f"Output     : {output_path}")
        print(f"Iterations : {num_iters}  |  batch_size/rank: {batch_size or 'all'}")

    # =================================================================
    # All-in-one path: every rank loads the raw dataset, preprocesses,
    # and constructs the same Ptychography state from scratch. Determinism
    # is provided by ``rng=42`` and the broadcast in reconstruct() makes
    # rank 0's parameters authoritative before the first optimizer step.
    # =================================================================
    dset = em.io.load(input_path)
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(
        com_fit_function="constant", plot_rotation=False, plot_com=False
    )

    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1),
        probe_model=ProbePixelated.from_params(
            probe_params={
                "energy": probe_energy,
                "defocus": probe_defocus,
                "semiangle_cutoff": probe_semiangle,
            }
        ),
        detector_model=DetectorPixelated(),
        verbose=(rank == 0),
        rng=42,
    )
    ptycho.preprocess(obj_padding_px=(obj_padding_px, obj_padding_px))

    # =================================================================
    # Preferred for large datasets: replace the all-in-one block above
    # with a single from_file() call. Point PTYCHO_INPUT at the
    # preprocessed .zip.
    #
    #     ptycho = Ptychography.from_file(input_path)
    #
    # Build the preprocessed zip once on a workstation:
    #     dset = em.io.load("ducky_dataset.zip")
    #     pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    #     pdset.preprocess(com_fit_function="constant")
    #     ptycho = Ptychography.from_models(
    #         dset=pdset,
    #         obj_model=ObjectPixelated.from_uniform(
    #             obj_type="pure_phase", num_slices=1
    #         ),
    #         probe_model=ProbePixelated.from_params(probe_params=PROBE_PARAMS),
    #         detector_model=DetectorPixelated(),
    #         rng=42,
    #     )
    #     ptycho.preprocess(obj_padding_px=(32, 32))
    #     ptycho.save("ducky_preprocessed.zip", save_raw_data=True)
    # =================================================================

    if rank == 0:
        n = ptycho.dset.num_gpts
        roi = ptycho.dset.roi_shape
        full_mb = n * roi[0] * roi[1] * 4 / 1e6
        print(f"  {n} scan positions, detector {roi[0]}×{roi[1]}")
        print(
            f"  Full amplitudes ≈ {full_mb:.0f} MB  |  "
            f"per-GPU ≈ {full_mb / world_size:.0f} MB"
        )

    optimizer_params = {
        "object": OptimizerParams.Adam(lr=lr_obj),
        "probe": OptimizerParams.Adam(lr=lr_probe),
    }
    scheduler_params = {
        "object": SchedulerParams.Plateau(factor=0.5),
        "probe": SchedulerParams.Plateau(factor=0.5),
    }

    # Apply constraint dataclasses by uncommenting and passing constraints=...
    # See PtychoObjConstraintParams.Raster / PtychoProbeConstraintParams.Raster
    # for the full list of available fields.
    # constraints = {
    #     "object": PtychoObjConstraintParams.Raster(tv_weight_xy=0.05),
    #     "probe": PtychoProbeConstraintParams.Raster(center_probe=True),
    # }

    t0 = time.perf_counter()
    # reconstruct() detects RANK/WORLD_SIZE from torchrun, binds the GPU to
    # LOCAL_RANK before NCCL init, shards the dataset, and all-reduces
    # gradients after each backward pass.
    ptycho.reconstruct(
        num_iters=num_iters,
        reset=True,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        # constraints=constraints,
        batch_size=batch_size,
        autograd=True,
    )
    elapsed = time.perf_counter() - t0

    if rank == 0:
        print(f"Finished in {elapsed:.1f}s  ({elapsed / num_iters:.2f} s/iter)")
        ptycho.save(output_path, mode="o")
        print(f"Saved to {output_path}")

    # Clean shutdown — required so torchrun returns without hanging.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
