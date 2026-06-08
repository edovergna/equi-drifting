#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=types
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=2:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi

cd "$HOME/drifting-experiments"

python train_types.py \
  --n_real_molecules 128 \
  --n_gen_molecules 128 \
  --num_workers 4 \
  --max_epochs 200 \
  --min_num_atoms 3 \
  --max_num_atoms 10 \
  --lr 2e-5