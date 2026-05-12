#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=many_geom
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=many_geom_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi

cd "$HOME/drifting-experiments"

python experiments/single_molecule_geometry/train_many.py \
  --root data/QM9 \
  --output_dir outputs/many_molecule_geometry \
  --n_molecules 100 \
  --start_index 0 \
  --max_num_atoms 18 \
  --steps 10000 \
  --log_every 100 \
  --hidden_dim 64 \
  --num_layers 6 \
  --lr 1e-3
