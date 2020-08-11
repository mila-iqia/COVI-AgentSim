#! /bin/bash
# /!\ THIS FILE WILL BE PYTHON-FORMATTED: DO NOT USE CURLY-BRACKETS IN TEXT
#SBATCH  --account=rrg-bengioy-ad
{cpu}              # Nb. of cpus (max(unkillable)=4, max(main)=6)
{mem}              # Require memory (16GB default should be enough)
{time}             # The job will run for 4 hours
{slurm_log}        # Write the logs in /network/tmp1/<user>/covi-slurm-%j.out

module purge
module load python/3.8.2
source ~/{env_name}/bin/activate

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
