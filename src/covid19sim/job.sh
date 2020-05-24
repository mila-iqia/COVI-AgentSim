#! /bin/bash
#SBATCH --partition=long                      # Ask for unkillable job | may use covi but this won't have gpus
#SBATCH --cpus-per-task=6                     # Ask for 6 CPUs
#SBATCH --mem=16G
#SBATCH --gres=gpu:1                          # Ask for 10 GB of RAM
#SBATCH --time=4:00:00                        # The job will run for 4 hours
#SBATCH -o /network/tmp1/schmidtv/covi-slurm-%j.out  # Write the log on tmp1

module load anaconda/3

source $CONDA_ACTIVATE

conda deactivate
conda activate covid

cd $HOME/simulator/src/covid19sim/

hydra_args=$@

echo "received $hydra_args"

echo "------------------------"


HYDRA_FULL_ERROR=1 python run.py $hydra_args