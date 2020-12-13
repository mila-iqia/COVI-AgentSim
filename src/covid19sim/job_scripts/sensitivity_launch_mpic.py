import sys
import os
import time
import copy
from covid19sim.utils.utils import log

APP_UPTAKE = 0.8415
MAX_JOBS = 1000
BATCHSIZE = 210
SLEEP_SECONDS = 1800
logfile = "sensitivity_runs_log.txt"
INTERVENTIONS = ["post-lockdown-no-tracing", "bdt1", "heuristicv4"]
# NOTE: values are arranged from optimistic to pessimistic
PARAMETERS = {
    "BASELINE_P_ASYMPTOMATIC": {
        "values": 0.1475, 0.2525, 0.3575], # asymptomatic-ratio =  0.20 0.30 0.40
        "no-effect": [],
    },
    "ALL_LEVELS_DROPOUT": {
        "values": [0.02, 0.08, 0.16],  # 0.02 0.08 0.16
        "no-effect": ["post-lockdown-no-tracing"],
    },
    "P_DROPOUT_SYMPTOM": {
        "values": [0.20, 0.40, 0.60],  # 0.20 0.40 0.60
        "no-effect": ["post-lockdown-no-tracing", "bdt1"],
    },
    "PROPORTION_LAB_TEST_PER_DAY": {
        "values": [0.004, 0.002, 0.001],  # 0.004 0.002 0.001
        "no-effect": [],
    },
}
SCENARIO_PARAMETERS_IDX = {"Optimistic": 0, "Moderate": 1, "Pessimistic": 2}


# def total_jobs():
#     command = "squeue -u pratgupt | wc -l"
#     stream = os.popen(command)
#     output = stream.read()
#     n_jobs = eval(output.strip()) - 1
#     log(f"jobs running currently - {n_jobs}", logfile)
#     return n_jobs


def run_sensitivity(
    INTERVENTION,
    BASELINE_P_ASYMPTOMATIC,
    P_DROPOUT_SYMPTOM,
    ALL_LEVELS_DROPOUT,
    PROPORTION_LAB_TEST_PER_DAY,
    n_jobs=None,
    *,
    scenario,
):
    log(
        f"running sensitivity for {INTERVENTION} with A:{BASELINE_P_ASYMPTOMATIC} B:{P_DROPOUT_SYMPTOM} C:{ALL_LEVELS_DROPOUT} D:{PROPORTION_LAB_TEST_PER_DAY}",
        logfile,
    )
    command = f"bash /lustre/home/nrahaman/python/covi-simulator/src/covid19sim/job_scripts/run_exps_sensitivity_mpic.sh {scenario} {APP_UPTAKE} {BASELINE_P_ASYMPTOMATIC} {ALL_LEVELS_DROPOUT} {P_DROPOUT_SYMPTOM} {PROPORTION_LAB_TEST_PER_DAY} {INTERVENTION}"
    print("Running: ", command)
    stream = os.popen(command)
    output = stream.read()
    return output
    # new_n_jobs = total_jobs()
    # log(f"New jobs launched: {new_n_jobs - n_jobs}", logfile)
    # pass


def GET_ARGS(scenario):
    idx = SCENARIO_PARAMETERS_IDX[scenario]
    SCENARIO_PARAMETERS = {
        key: value["values"][idx] for key, value in PARAMETERS.items()
    }
    for intervention in INTERVENTIONS:
        yield {"INTERVENTION": intervention, **SCENARIO_PARAMETERS}
    for key, value_dict in PARAMETERS.items():
        values = value_dict["values"]
        no_run = value_dict["no-effect"]
        for intervention in INTERVENTIONS:
            if intervention in no_run:
                continue
            for val in values:
                if val == SCENARIO_PARAMETERS[key]:
                    continue
                new_params = copy.deepcopy(SCENARIO_PARAMETERS)
                new_params[key] = val
                yield {"INTERVENTION": intervention, **new_params}


# SCENARIO = sys.argv[1]
# SCENARIO = "Optimistic"
# assert (
#     SCENARIO in SCENARIO_PARAMETERS_IDX
# ), f"{SCENARIO} not found in {SCENARIO_PARAMETERS_IDX.keys()}"
# args = GET_ARGS()

job_list = [
    run_sensitivity(**all_args, scenario=scenario)
    for scenario in SCENARIO_PARAMETERS_IDX.keys()
    for all_args in GET_ARGS(scenario)
]

final_output_string = "\n".join(job_list)

print(f"Dumping {len(final_output_string)} jobs:")

print(final_output_string)

# while True:
#     n_jobs = total_jobs()
#     if MAX_JOBS - n_jobs > BATCHSIZE:
#         new_args = next(args)
#         run_sensitivity(**new_args, n_jobs=n_jobs)
#     else:
#         time.sleep(SLEEP_SECONDS)
