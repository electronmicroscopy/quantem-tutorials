#!/bin/bash
#SBATCH -A mxxxx                              # CHANGE: replace with your NERSC project ID
#SBATCH -C gpu
#SBATCH -q debug                              # CHANGE: use 'regular' for runs > 30 min
#SBATCH -t 00:30:00                           # CHANGE: adjust walltime as needed
#SBATCH -N 2                                  # CHANGE: number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=your@email.com            # CHANGE: your email

# === Configuration — edit these ===
WORKDIR=$SCRATCH/ptycho_multinode             # CHANGE: working directory on scratch
CONDA_ENV=/global/common/software/mxxxx/INSERT_USER/conda/quantem  # CHANGE: conda env path
PTYCHO_INPUT=$WORKDIR/ducky_preprocessed.zip
PTYCHO_OUTPUT=$WORKDIR/ducky_result.zip
PTYCHO_ITERS=200
PTYCHO_LR_OBJ=5e-2
PTYCHO_LR_PROBE=5e-2
# PTYCHO_BATCH_SIZE=256                       # uncomment to set explicit batch size per rank

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8

# Perlmutter NCCL tuning
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_ATS=0

# Export script parameters
export PTYCHO_INPUT PTYCHO_OUTPUT PTYCHO_ITERS PTYCHO_LR_OBJ PTYCHO_LR_PROBE
[ -n "${PTYCHO_BATCH_SIZE+x}" ] && export PTYCHO_BATCH_SIZE

ml nccl/2.29.2-cu13
ml conda
conda activate "$CONDA_ENV"

srun -l torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node="$SLURM_GPUS_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$WORKDIR/ptycho_ducky.py"
