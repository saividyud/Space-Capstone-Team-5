#!/usr/bin/env bash

#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=lambert_scanning_1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=senthilnathan.11@osu.edu

#SBATCH --output="./Trajectory/lambert_scanner_analysis.out"

# Commands to run
module load mamba
mamba activate .venv_space
python ./Trajectory/lambert_scanner_analysis.py