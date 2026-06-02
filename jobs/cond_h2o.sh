#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pos_cond
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi

cd "$HOME/drifting-experiments"

# Baseline on 3-atom molecules (HCN/H2O), no chem refinement.
# Goal: confirm drift loss produces high validity + stability
# before scaling up or enabling chem refinement.
python train_conditional.py \
  --min_num_atoms 3 \
  --max_num_atoms 3 \
  --n_real_molecules 128 \
  --n_gen_molecules 128 \
  --num_workers 4 \
  --max_epochs 200 \
  --hidden_dim 256 \
  --num_layers 9 \
  --lr 2e-4 \
  --weight_decay 5e-5 \
  --position_sigma 5.0 \
  --position_eta 0.4 \
  --position_weight 0.5 \
  --types_weight 1.0 \
  --chem_refinement \
  --start_frac_epoch 0.25 \
  --clash_threshold 0.5 \
  --lambda_clash 1.0 \
  --lambda_valence_excess 1.0 \
  --lambda_hydrogen_valence 1.0 \
  --bond_temperature 0.1 \
  --check_val_every_n_epoch 5 \
  --group_tag "conditional_h2o"
