#!/bin/bash
#SBATCH --job-name=bast
#SBATCH --ntasks=1
#SBATCH --output=slurm.txt
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 256
#SBATCH --time 0:01:00

module load releases/2021b
module load Python
module load SciPy-bundle
source ./env/bin/activate

srun python scripts/sandbox.py
