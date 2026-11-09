#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J doubleMM
#
# Number of MPI Tasks, e.g. 8:
#SBATCH --ntasks=8
# Memory usage [MB] of the job is required, e.g. 3000 MB per task:
#SBATCH --mem=24000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=twutz@bgc-jena.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
#module purge
#module load intel/2026.0 impi/2021.18

# Run the program:
# started from main directory so projects is a subdiretory
srun julia projects/doubleMM_site.jl > projects/prog.out