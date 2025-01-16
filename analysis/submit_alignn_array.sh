#!/bin/bash

#SBATCH --job-name=alignn_array
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=rack1,rack2e,rack3,rack4e,rack5,rack6
#SBATCH --array=1-999  # We'll replace 10000 with the actual number of rows
#SBATCH --output=logs/alignn_array_%A_%a.out
#SBATCH --error=logs/alignn_array_%A_%a.err

# If your HPC requires a module load or sourcing .bashrc, do that here
# module load anaconda
# source ~/.bashrc

#conda activate chipsff-alignn-dev-12_24


# The array index is stored in SLURM_ARRAY_TASK_ID
# That tells us which row we're processing
echo "[INFO] Starting row $SLURM_ARRAY_TASK_ID"

python run_alignn_single.py --row_index $SLURM_ARRAY_TASK_ID
