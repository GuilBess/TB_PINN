#!/bin/bash
#SBATCH --job-name=test_PINN    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=shard:24
#SBATCH --output=output.txt # Standard output file
#SBATCH --error=error.txt # Standard error file
#module purge

apptainer run --nv mycontainer.sif
