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

# Scale run on 3-12 atoms WITH chemical refinement.
# clash_threshold=0.5 (not 1.0): in the conditional model atom types
# are fixed, so bond distances are well-defined. 1.0 > H-O bond length
# (~0.96A) and breaks training; 0.5 is safely below all real bonds.
# Run cond_scale.sh first to confirm baseline before running this.
python train_conditional.py \
  --min_num_atoms 3 \
  --max_num_atoms 12 \
  --n_real_molecules 128 \
  --n_gen_molecules 128 \
  --num_workers 8 \
  --max_epochs 500 \
  --hidden_dim 256 \
  --num_layers 9 \
  --lr 2e-4 \
  --weight_decay 5e-5 \
  --position_sigma 5.0 \
  --position_eta 0.4 \
  --position_weight 0.5 \
  --types_weight 1.0 \
  --chem_refinement \
  --clash_threshold 0.5 \
  --lambda_clash 1.0 \
  --lambda_valence_excess 1.0 \
  --lambda_hydrogen_valence 1.0 \
  --bond_temperature 0.1 \
  --start_frac_epoch 0.8 \
  --check_val_every_n_epoch 5 \
  --group_tag "conditional_scale_chem"
