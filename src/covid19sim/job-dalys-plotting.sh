#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -o /scratch/williaar/feb_24_asymp_dalys_csv/slurm-%j.out  # Write the log in $SCRATCH
#SBATCH --mail-user=williams.andrew1305@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python/3.8
module load cuda

source /home/williaar/py37/bin/activate
PYTHON=/home/williaar/py37/bin/python

outdir=/scratch/williaar/feb_24_asymp_dalys_csv
UPTAKE=$1
USE_CACHE=$2


python plotting/main.py plot=dalys \
    path=$SCRATCH/dalys/cost-benefit/dalys_S_Main_10000_init_0.004_UPTAKE_${UPTAKE}/scatter_Ax_0.75_Lx_0.02_Sx_0.20_test_0.001/normalized_mobility \
    use_cache=$USE_CACHE


