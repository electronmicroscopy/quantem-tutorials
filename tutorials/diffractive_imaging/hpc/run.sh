#!/bin/bash
# Interactive multi-node ptychography reconstruction on Perlmutter.
#
# Prerequisites:
#   1. Allocate an interactive job:
#        salloc -q interactive -C gpu -A YOUR_ACCOUNT \
#               --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
#               --cpus-per-task=128 --time=04:00:00
#   2. Transfer ducky_preprocessed.zip to $WORKDIR (see README.md).
#   3. Edit the CHANGE lines below, then: bash run.sh
#
# Configuration — edit these:
WORKDIR=$SCRATCH/ptycho_multinode              # CHANGE: working directory on scratch
SCRIPTDIR=$HOME/quantem-tutorials/tutorials/diffractive_imaging/hpc
CONDA_ENV=/global/common/software/<mxxxx>/<user>/conda/quantem  # CHANGE: path to your conda env
PTYCHO_INPUT=$WORKDIR/ducky_251105_20mrad_500A-df_4A-step_5e+04-dose_clean.zip
PTYCHO_OUTPUT=$WORKDIR/ducky_hpc_result.zip
PTYCHO_ITERS=2000
PTYCHO_LR_OBJ=5e-2
PTYCHO_LR_PROBE=5e-2
PTYCHO_BATCH_SIZE=256

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8

# Perlmutter NCCL tuning (routes traffic over Slingshot high-speed network)
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_ATS=0

# Export script parameters as env vars (ptycho_ducky.py reads them)
export PTYCHO_INPUT PTYCHO_OUTPUT PTYCHO_ITERS PTYCHO_LR_OBJ PTYCHO_LR_PROBE
[ -n "${PTYCHO_BATCH_SIZE+x}" ] && export PTYCHO_BATCH_SIZE

echo "=== Loading modules ==="
ml nccl/2.29.2-cu13
ml cudatoolkit/13.0
ml conda
echo "=== Loaded modules ==="
module list

echo "=== Activating conda env ==="
conda activate "$CONDA_ENV"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"

echo "=== Python ==="
which python
python --version
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "=== Starting torchrun (nodes=$SLURM_JOB_NUM_NODES, gpus/node=$SLURM_GPUS_PER_NODE) ==="
srun -l torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node="$SLURM_GPUS_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$SCRIPTDIR/ptycho_ducky.py"
