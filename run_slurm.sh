#!/bin/bash
#SBATCH --job-name=test_PINN # Job name
#SBATCH --output=output.txt # Standard output file
#SBATCH --error=error.txt # Standard error file
#SBATCH --tasks=1 # Number of tasks
#SBATCH --gpus-per-node=1 # Require GPUs
#SBATCH --time=0-01:00 # Maximum runtime (D-HH:MM)
#SBATCH --partition=Chacha

apptainer run --nv app_test.sif