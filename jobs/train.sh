#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=aligned
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
  --n_real_molecules 128 \
  --n_gen_molecules 128 \
  --num_workers 8 \
  --max_epochs 750 \
  --min_num_atoms 3 \
  --max_num_atoms 11 \
  --check_val_every_n_epoch 1 \
  --hidden_dim 256 \
  --num_layers 7 \
  --max_iter 1 \
  --sample_frac 1.0 \
  --position_sigma 2.0 \
  --types_sigma 1.0 \
  --position_eta 0.4 \
  --types_eta 0.7 \
  --lr 2e-4 \
  --position_weight 0.5 \
  --chem_refinement
