#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pos_cond
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=30:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi

cd "$HOME/drifting-experiments"

python train_conditional.py \
  --overfit \
  --overfit_atom_count 9 \
  --overfit_sample_frac 0.001 \
  --n_real_molecules 16 \
  --max_epochs 2000 \
  --hidden_dim 128 \
  --num_layers 4 \
  --lr 2e-4 \
  --weight_decay 0.0 \
  --position_sigma 1.0 \
  --position_eta 1.0 \
  --group_tag "conditional_overfit"
