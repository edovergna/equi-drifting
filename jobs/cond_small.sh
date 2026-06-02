#!/bin/bash

#SBATCH --partition=gpu_a100
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
  --min_num_atoms 3 \
  --max_num_atoms 9 \
  --n_real_molecules 128 \
  --n_gen_molecules 128 \
  --num_workers 8 \
  --max_epochs 500 \
  --hidden_dim 128 \
  --num_layers 4 \
  --lr 2e-4 \
  --weight_decay 5e-5 \
  --position_sigma 5.0 \
  --position_eta 0.4 \
  --position_weight 0.5 \
  --check_val_every_n_epoch 5 \
  --group_tag "conditional_small"
