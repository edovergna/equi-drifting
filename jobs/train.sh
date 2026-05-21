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
  --n_real_molecules 64 \
  --n_gen_molecules 128 \
  --num_workers 4 \
  --max_epochs 500 \
  --min_num_atoms 3 \
  --max_num_atoms 7 \
  --check_val_every_n_epoch 1 \
  --hidden_dim 128 \
  --num_layers 5 \
  --max_iter 3 \
  --sample_frac 1.0 \
  --position_sigma 2.5 \
  --position_eta 1.0 \
  --types_eta 1.0 \
  --chem_refinement
