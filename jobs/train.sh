#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=drifting_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi
cd "$HOME/dutch-daniel2"

python train.py --sample_frac 0.05 --lr 1e-4 --check_val_every_n_epoch 2 --max_epochs 10 --num_workers 8 --n_real_molecules 2048 --hidden_dim 256 --num_layers 8 --atom_type_loss_weight 1.0 --geom_loss_weight 1.0 --gradient_clip_val 0.1 --equiv_phi_mode vector