#!/bin/bash
#SBATCH -A mxxxx  # CHANGE: Replace with your NERSC project ID
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=xxx@xxx.xxx  # CHANGE: Replace with your email address

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
ml nccl/2.29.2-cu13
ml conda
ml nccl
conda activate /global/common/software/mxxxx/'INSERT USER HERE'/conda/quantem  # CHANGE: Replace 'INSERT USER HERE' with your NERSC username and update path to your conda environment
srun -l torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT tomography_recon.py
