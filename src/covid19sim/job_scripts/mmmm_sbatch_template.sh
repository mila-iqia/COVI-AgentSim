#! /bin/bash
# /!\ THIS FILE WILL BE PYTHON-FORMATTED: DO NOT USE CURLY-BRACKETS IN TEXT
{partition}        # Partition to use
{cpu}              # Nb. of cpus (max(unkillable)=4, max(main)=6)
{mem}              # Require memory (16GB default should be enough)
{time}             # The job will run for 4 hours
{slurm_log}        # Write the logs in /network/tmp1/<user>/covi-slurm-%j.out
{gres}             # May use GPU to get allocation

module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda deactivate
conda activate {env_name} # covid is default

export PYTHONUNBUFFERED=1

# where is simulator's code? default: $HOME/simulator/src/covid19sim/
cd {code_loc}

echo $(pwd)
echo $(which python)

use_server={use_server}

if [ "$use_server" = true ] ; then
    python server_bootstrap.py -e {weights} -w {workers} {frontend} {backend}&
fi

echo "------------------------"

# THIS FILE WILL BE APPENDED TO. DO NOT WRITE AFTER THIS LINE