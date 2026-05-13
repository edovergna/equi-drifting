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

python train.py \
  --group_tag test_moments \
  --sample_frac 0.01 \
  --n_real_molecules 4 \
  --n_gen_molecules 32 \
  --max_num_atoms 18 \
  --num_workers 8 \
  --max_epochs 50 \
  --check_val_every_n_epoch 5 \
  --hidden_dim 256 \
  --num_layers 8 \
  --phi_mode moments \
  --equiv_phi_mode vector
