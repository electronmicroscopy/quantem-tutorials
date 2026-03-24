#!/bin/bash

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8

echo "=== Loading modules ==="
ml nccl/2.29.2-cu13
ml cudatoolkit/13.0
ml conda
echo "=== Loaded modules ==="
module list

echo "=== Activating conda env ==="
conda activate /global/common/software/mxxxx/user/conda/quantem  # CHANGE: Replace with the path to your conda environment
echo "=== Active conda env ==="
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"

echo "=== Python being used ==="
which python
python --version

echo "=== Key packages ==="
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "=== NCCL ==="
echo "NCCL_HOME: $NCCL_HOME"

echo "=== Starting srun ==="
srun -l torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT tomography_recon.py