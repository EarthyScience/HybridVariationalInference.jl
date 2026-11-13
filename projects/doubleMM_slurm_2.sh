#!/bin/bash -l
# sbatch projects/doubleMM_slurm.sh
# squeue -u twutz
# Standard output and error:
#SBATCH -o ./tmp/job.out.%j
#SBATCH -e ./tmp/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J doubleMM_2
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
# Memory usage [MB] of the job is required, e.g. 2GB per CPU*task
#SBATCH --mem=6000  # get from last finished similar job
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=twutz@bgc-jena.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
#module purge
#module load intel/2026.0 impi/2021.18

# Run the program:
# started from main directory so projects is a subdiretory
srun julia --threads ${SLURM_CPUS_PER_TASK} projects/doubleMM_site_2.jl > projects/prog_2.out
