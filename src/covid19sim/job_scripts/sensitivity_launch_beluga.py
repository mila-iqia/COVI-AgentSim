import sys
import os
import time
import copy
from covid19sim.utils.utils import log

APP_UPTAKE = 0.8415
MAX_JOBS = 1000
BATCHSIZE = 210
SLEEP_SECONDS = 1800
logfile = None
INTERVENTIONS = ["post-lockdown-no-tracing", "bdt1"]
# INTERVENTIONS = ["plot"]

# NOTE: values are arranged from optimistic to pessimistic
PARAMETERS = {
    "BASELINE_P_ASYMPTOMATIC": {
        "values": [0.1475, 0.2525, 0.3575], # asymptomatic-ratio =  0.20 0.30 0.40
        "no-effect":[]
    },
    "ALL_LEVELS_DROPOUT": {
        "values": [0.02, 0.08, 0.16], # 0.02 0.08 0.16
        "no-effect":["post-lockdown-no-tracing"]
    },
    "P_DROPOUT_SYMPTOM": {
        "values": [0.20, 0.40, 0.60], # 0.20 0.40 0.60
        "no-effect":["post-lockdown-no-tracing", "bdt1"]
    },
    "PROPORTION_LAB_TEST_PER_DAY": {
        "values": [0.004, 0.002, 0.001], # 0.004 0.002 0.001
        "no-effect":[]
    }
}

SCENARIO_PARAMETERS_IDX={
    "Optimistic" : 0,
    "Moderate": 1,
    "Pessimistic": 2
}

def run_sensitivity(INTERVENTION, BASELINE_P_ASYMPTOMATIC, P_DROPOUT_SYMPTOM, ALL_LEVELS_DROPOUT, PROPORTION_LAB_TEST_PER_DAY, n_jobs=None):
    if INTERVENTION == "plot":
        log(f"plotting sensitivity for {INTERVENTION} with A:{BASELINE_P_ASYMPTOMATIC} B:{P_DROPOUT_SYMPTOM} C:{ALL_LEVELS_DROPOUT} D:{PROPORTION_LAB_TEST_PER_DAY}", logfile)
        command = f"sbatch run_plot_sensitivity.sh {SCENARIO} {APP_UPTAKE} {BASELINE_P_ASYMPTOMATIC} {ALL_LEVELS_DROPOUT} {P_DROPOUT_SYMPTOM} {PROPORTION_LAB_TEST_PER_DAY} {INTERVENTION}"
    else:
        log(f"running sensitivity for {INTERVENTION} with A:{BASELINE_P_ASYMPTOMATIC} B:{P_DROPOUT_SYMPTOM} C:{ALL_LEVELS_DROPOUT} D:{PROPORTION_LAB_TEST_PER_DAY}", logfile)
        command = f"./run_exps_sensitivity.sh {SCENARIO} {APP_UPTAKE} {BASELINE_P_ASYMPTOMATIC} {ALL_LEVELS_DROPOUT} {P_DROPOUT_SYMPTOM} {PROPORTION_LAB_TEST_PER_DAY} {INTERVENTION} dev=True"

    stream = os.popen(command)
    output = stream.read()
    cmds = [x for x in output.split("\n") if "python run.py" in x]
    return cmds


def GET_ARGS(SCENARIO):
    idx = SCENARIO_PARAMETERS_IDX[SCENARIO]
    SCENARIO_PARAMETERS = {key:value['values'][idx] for key,value in PARAMETERS.items()}
    for intervention in INTERVENTIONS:
        yield {"INTERVENTION": intervention, **SCENARIO_PARAMETERS}

    for key, value_dict in PARAMETERS.items():
        values = value_dict['values']
        no_run = value_dict['no-effect']
        for intervention in INTERVENTIONS:
            if intervention in no_run:
                continue

            for val in values:
                if val == SCENARIO_PARAMETERS[key]:
                    continue
                new_params = copy.deepcopy(SCENARIO_PARAMETERS)
                new_params[key] = val
                yield {"INTERVENTION": intervention, **new_params}


all_commands = []
for SCENARIO in SCENARIO_PARAMETERS_IDX.keys():
    args = GET_ARGS(SCENARIO)
    for new_args in args:
        all_commands += run_sensitivity(**new_args, n_jobs=None)

print("number of commands that will be launched - ", len(all_commands))

node_launch_template = """#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=2:50:00
#SBATCH --mem=0
#SBATCH --job-name COVID19_SENSITIVITY_ANALYSIS
#SBATCH -o /scratch/pratgupt/job_logs/slurm-%j.out  # Write the log in $SCRATCH
#SBATCH --mail-user=pg2455@columbia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

OFF=0

"""


node_launch_srun_template = "srun -r$(( OFF=(OFF+1)%$SLURM_JOB_NUM_NODES )) \ \n\t-N 1 -n 1 -c 1 \ \n\t--time=01:00:00 \ \n\t"

for command in all_commands:
    cmd = command.replace("USE_INFERENCE_SERVER=False", "")
    node_launch_template += f"\n{node_launch_srun_template}{cmd}\n"

with open("node_launch.sh", "w") as bash:
    bash.write(node_launch_template)
