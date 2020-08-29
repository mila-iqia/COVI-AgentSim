#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=10
#SBATCH --gres gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -o /scratch/pratgupt/job_logs/slurm-%j.out  # Write the log in $SCRATCH
#SBATCH --mail-user=pg2455@columbia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python/3.8
module load cuda

source /home/pratgupt/covid19/bin/activate
PYTHON=/home/pratgupt/covid19/bin/python

$PYTHON /home/pratgupt/simulator/src/covid19sim/run.py tune=True n_people=2000 init_fraction_sick=0.004 seed=$5 \
        simulation_days=50 outdir=/scratch/pratgupt/$6/$1 \
        INTERVENTION_DAY=$2 intervention=$3 \
        APP_UPTAKE=$4 track=light GLOBAL_MOBILITY_SCALING_FACTOR=$7 \
        ORACLE_MUL_NOISE=$8 ORACLE_ADD_NOISE=$9 risk_mappings=uniform # these are not used if its not oracle
