#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=overfit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output_overfit_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi
cd "$HOME/dutch-daniel2"

# Overfit test: tiny dataset slice, high LR, many epochs, large EGNN.
# Goal: loss should decrease to near-zero if the model + loss are working correctly.
python train.py \
    --sample_frac 0.002 \
    --lr 5e-4 \
    --max_epochs 500 \
    --check_val_every_n_epoch 50 \
    --num_workers 4 \
    --n_real_molecules 64 \
    --n_gen_molecules 64 \
    --hidden_dim 256 \
    --num_layers 8
