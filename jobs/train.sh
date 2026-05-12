#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=test_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi

cd "$HOME/drifting-experiments"

python train.py \
  --group_tag test_train \
  --sample_frac 0.01 \
  --n_real_molecules 4 \
  --n_gen_molecules 32 \
  --max_num_atoms 18 \
  --num_workers 4 \
  --max_epochs 10 \
  --check_val_every_n_epoch 1 \
  --hidden_dim 64 \
  --num_layers 6
