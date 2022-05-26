#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=2:50:00
#SBATCH -o /scratch/pratgupt/job_logs/slurm-%j.out  # Write the log in $SCRATCH
#SBATCH --mail-user=pg2455@columbia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

dirname=$1


SIM_OUTPUT_BASEDIR=$SCRATCH/
# SIM_OUTPUT_BASEDIR=/home/nrahaman/python/covi-simulator/exp/sensitivity_v3
COVISIM_REPO=/home/$USER/simulator


module load python/3.8.2
module load scipy-stack
module load cuda

source /home/pratgupt/covid19/bin/activate
cd ${COVISIM_REPO}/src/covid19sim/plotting

python main.py plot=normalized_mobility \
        path=$dirname/normalized_mobility/ load_cache=False use_cache=False normalized_mobility_use_extracted_data=False
# python $HOME/simulator/src/covid19sim/plotting/main.py plot=normalized_mobility \
#   path=/scratch/pratgupt/$dirname/normalized_mobility/ load_cache=True use_cache=True normalized_mobility_use_extracted_data=False


# srun -N 1 -n 1 -c 5 --time=2:00:00 --mem=60G --account=rrg-bengioy-ad ~/covid19/bin/python main.py plot=normalized_mobility \
#         path=$SCRATCH/new_scatter_5000_init_0.002_UPTAKE_0.6415/normalized_mobility/ load_cache=False use_cache=False normalized_mobility_use_extracted_data=False
