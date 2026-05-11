#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=comp_prior
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=comp_prior_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi

cd "$HOME/drifting-experiments"

python train.py \
  --group_tag comp_prior_exact_onehot \
  --sample_frac 0.05 \
  --n_real_molecules 64 \
  --n_gen_molecules 64 \
  --max_num_atoms 18 \
  --num_workers 8 \
  --max_epochs 30 \
  --check_val_every_n_epoch 1 \
  --hidden_dim 64 \
  --num_layers 6 \
  --lr 2e-4 \
  --weight_decay 5e-5 \
  --temperatures 0.02 0.05 0.2 \
  --loss_variant norm_based \
  --prior_pos_clamp 3.0 \
  --log_every_n_steps 10

# To test the soft alternative, add:
#   --prior_atom_dirichlet_strength 50.0
