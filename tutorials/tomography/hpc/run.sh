#!/bin/bash

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
ml nccl/2.24.3
ml conda
# ml nccl
conda activate /global/common/software/m5020/cedlim/conda/quantem


srun -l torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT tomography_recon.py