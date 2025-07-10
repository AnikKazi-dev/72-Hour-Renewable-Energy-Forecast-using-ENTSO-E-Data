#!/bin/bash -l
#
#SBATCH --job-name=cycle_lstm_summer
#SBATCH --partition=work
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_cyclelstm_summer_%j.out
#SBATCH --error=slurm_cyclelstm_summer_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# --- Your code starts here ---

echo "Job started on $(hostname)"
module load python
conda activate tf-nvidia
echo "Environment activated. Starting Python script for SUMMER..."

# Execute the Summer Python script
srun python3 summer_forecast.py

echo "Script finished with exit code $?."