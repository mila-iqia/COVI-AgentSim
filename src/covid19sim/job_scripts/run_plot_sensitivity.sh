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

n_people=5000
init=0.004

SCENARIO=$1
UPTAKE=$2
ASYMP=$3
ALL_LEVELS_DROPOUT=$4
P_DROPOUT_SYMPTOM=$5
TEST=$6
dirname=$8/sensitivity_S_${SCENARIO}_${n_people}_init_${init}_UPTAKE_${UPTAKE}/scatter_Ax_${ASYMP}_Lx_${ALL_LEVELS_DROPOUT}_Sx_${P_DROPOUT_SYMPTOM}_test_${TEST}

INTERVENTION=$7

if [[ ! -z "${10}" ]]; then
  dirname=${dirname}_AIR_${10}
fi


SIM_OUTPUT_BASEDIR=$SCRATCH/
# SIM_OUTPUT_BASEDIR=/home/nrahaman/python/covi-simulator/exp/sensitivity_v3
COVISIM_REPO=/home/$USER/simulator

module load python/3.8
module load cuda

source /home/pratgupt/covid19/bin/activate
cd ${COVISIM_REPO}/src/covid19sim/plotting

python main.py plot=normalized_mobility \
        path=${SIM_OUTPUT_BASEDIR}/$dirname/normalized_mobility/ load_cache=False use_cache=False normalized_mobility_use_extracted_data=False
# python $HOME/simulator/src/covid19sim/plotting/main.py plot=normalized_mobility \
#   path=/scratch/pratgupt/$dirname/normalized_mobility/ load_cache=True use_cache=True normalized_mobility_use_extracted_data=False


# srun -N 1 -n 1 -c 5 --time=2:00:00 --mem=60G --account=rrg-bengioy-ad ~/covid19/bin/python main.py plot=normalized_mobility \
#         path=$SCRATCH/new_scatter_5000_init_0.002_UPTAKE_0.6415/normalized_mobility/ load_cache=False use_cache=False normalized_mobility_use_extracted_data=False
