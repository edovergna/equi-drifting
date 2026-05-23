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
  --num_workers 4 \
  --max_epochs 1000 \
  --min_num_atoms 3 \
  --max_num_atoms 16 \
  --check_val_every_n_epoch 1 \
  --hidden_dim 128 \
  --num_layers 8 \
  --max_iter 1 \
  --sample_frac 1.0 \
  --position_sigma 2.0 \
  --types_sigma 1.0 \
  --position_eta 1.0 \
  --types_eta 1.0 \
  --lr 1e-5 \
  --position_weight 0.5 \
  --lambda_valence_excess 1.0 \
  --lambda_hydrogen_valence 1.0 \
  --lambda_clash 1.0 \
  --clash_threshold 1.0 \
  --chem_refinement
