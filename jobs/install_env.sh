#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=ins_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/equi-drifting"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create -f conda_environment.yaml
