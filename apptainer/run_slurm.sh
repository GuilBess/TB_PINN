#!/bin/bash
#SBATCH --job-name=test_PINN # Job name
#SBATCH --output=output.txt # Standard output file
#SBATCH --error=error.txt # Standard error file
#SBATCH --tasks=1 # Number of tasks
#SBATCH --gpus-per-node=1 # Require GPUs
#SBATCH --time=0-02:00 # Maximum runtime (D-HH:MM)
#SBATCH --nodelist=calypso3 # Specific node [Optional]

apptainer run --nv mycontainer.sif --bind /home/guillaume.bessard/dist:.
