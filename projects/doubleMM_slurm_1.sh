#!/bin/bash -l
# sbatch projects/doubleMM_slurm_1.sh
# squeue -u twutz
# Standard output and error:
#SBATCH -o ./tmp/job.out.%j
#SBATCH -e ./tmp/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J dMM_1
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

# Run the program:
# started from main directory so projects is a subdiretory
srun julia --threads ${SLURM_CPUS_PER_TASK} projects/doubleMM_site_1.jl > projects/prog_1.out
