#!/bin/bash
#SBATCH --job-name=test_PINN    # create a short name for your job
#SBATCH --nodelist=disco
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --output=output.txt # Standard output file
#SBATCH --error=error.txt # Standard error file
#module purge

apptainer run --nv mycontainer.sif --bind ~/results.
