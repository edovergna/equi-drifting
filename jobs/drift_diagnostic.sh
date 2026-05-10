#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=drift_diag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=slurm_drift_diag_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
conda activate equiv

cd "$HOME/drifting-experiments"

python train.py \
  --offline \
  --group_tag drift-diagnostic \
  --sample_frac 0.05 \
  --batch_size 32 \
  --num_workers 4 \
  --max_epochs 10 \
  --check_val_every_n_epoch 1 \
  --log_every_n_steps 10 \
  --hidden_dim 64 \
  --num_layers 4 \
  --sigma_r 1.0 \
  --sigma_a 0.5 \
  --eta_pos 1.0 \
  --eta_type 1.0 \
  --weight_pos 1.0 \
  --weight_type 0.05
