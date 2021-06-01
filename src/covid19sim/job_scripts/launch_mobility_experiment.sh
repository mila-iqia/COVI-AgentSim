#!/bin/bash

# args
SCENARIO=$1
UPTAKE=$2
ASYMP=$3
ALL_LEVELS_DROPOUT=$4
P_DROPOUT_SYMPTOM=$5
TEST=$6
INTERVENTION=$7
TYPE=$9

n_people=${10}
init=${11}
dirname=$8/dalys_S_${SCENARIO}_${n_people}_init_${init}_UPTAKE_${UPTAKE}/scatter_Ax_${ASYMP}_Lx_${ALL_LEVELS_DROPOUT}_Sx_${P_DROPOUT_SYMPTOM}_test_${TEST}

ASYMP_INFECTION_RATIO=0.29
if [[ ! -z "${12}" ]]; then
  ASYMP_INFECTION_RATIO=${12}
  dirname=${dirname}_AIR_${ASYMP_INFECTION_RATIO}
fi

# simulation days
SIM_DAYS=60

# essentials
ENV_RELATIVE_PATH=py37
COVISIM_REPO=/home/$USER/COVI-AgentSim
SLURM_LOG_DIR=$SCRATCH/job_logs
SIM_OUTPUT_BASEDIR=$SCRATCH/
# SIM_OUTPUT_BASEDIR=/home/nrahaman/python/covi-simulator/exp/sensitivity_v3 ## NASIM
EMAIL=andrew.williams@mila.quebec # to be notified of every run

# transformer related
TRANSFORMER_FOLDER_NAME=(WORLDLY-GALAXY-801)
TRANSFORMER_EXP_BASEPATH=$SCRATCH/pra
REC_LEVEL_THRESHOLDS="[0,0,1]"
NAME_SUFFIX=_001_60


source ~/${ENV_RELATIVE_PATH}/bin/activate

cd ${COVISIM_REPO}/src/covid19sim/job_scripts
# normalized mobility

TIME="'5:00:00'"

if [ "$INTERVENTION" == "transformer" ]; then
  TIME="'10:00:00'"
fi

if [ "${n_people}" -ge "5000" ] ; then
  TIME="'15:00:00'"

  # if [ "${UPTAKE}" == "0.5618" ] ; then
  #   TIME="'36:00:00'"
  # fi

  # if [ "${UPTAKE}" == "0.8415" ] ; then
  #   TIME="'48:00:00'"
  # fi

fi

glomo_range=uniform_for_sensitivity
n_search=250

if [ "$TYPE" != "main-scenario" ] ; then
  # no need to have a GP if filter is the number of contacts
  n_search=60
  glomo_range=uniform_for_sensitivity_narrow
fi

if [ -x "$(command -v queue_monitor)" ]; then
  len=1
  if [ "$INTERVENTION" == "transformer" ]; then
    len=${#TRANSFORMER_FOLDER_NAME[@]}
  fi
  queue_monitor $(echo $((995 - len * $n_search))) && notify "launching $n_search jobs now" # will not run this script until there are are 225 jobs can be queued
fi

if [ "$INTERVENTION" == "post-lockdown-no-tracing" ] || [ "$INTERVENTION" == "bdt1" ] || [ "$INTERVENTION" == "heuristicv4" ] ; then
  cd $COVISIM_REPO/src/covid19sim/job_scripts

  python experiment.py exp_file=normalized_mobility env_name=${ENV_RELATIVE_PATH} \
    email_id=$EMAIL slurm_log=${SLURM_LOG_DIR} track=light code_loc=${COVISIM_REPO}/src/covid19sim \
    base_dir=${SIM_OUTPUT_BASEDIR}/$dirname/normalized_mobility simulation_days=${SIM_DAYS}\
    intervention=$INTERVENTION  global_mobility_scaling_factor_range=${glomo_range} \
    n_people=${n_people} init_fraction_sick=$init time=$TIME APP_UPTAKE=$UPTAKE \
    BASELINE_P_ASYMPTOMATIC=$ASYMP PROPORTION_LAB_TEST_PER_DAY=$TEST ASYMPTOMATIC_INFECTION_RATIO=${ASYMP_INFECTION_RATIO} \
    P_DROPOUT_SYMPTOM=${P_DROPOUT_SYMPTOM} ALL_LEVELS_DROPOUT=${ALL_LEVELS_DROPOUT}  seeds=9_seeds n_search=${n_search} CONTAGION_KNOB=27.5 cpus=10 #\
    # dev=True pure_command_output_file=${TYPE}_${n_people}_${init}.txt ## NASIM
fi

if [ "$INTERVENTION" == "transformer" ]; then
    for TRANSFORMER_NAME in "${TRANSFORMER_FOLDER_NAME[@]}"
    do
      python experiment.py exp_file=normalized_mobility env_name=${ENV_RELATIVE_PATH} \
      email_id=$EMAIL slurm_log=${SLURM_LOG_DIR} track=light code_loc=${COVISIM_REPO}/src/covid19sim \
      outdir=${SIM_OUTPUT_BASEDIR}/$dirname/normalized_mobility/${TRANSFORMER_NAME}${NAME_SUFFIX} simulation_days=${SIM_DAYS} \
      intervention=$INTERVENTION global_mobility_scaling_factor_range=${glomo_range}  \
      n_people=$n_people init_fraction_sick=$init time=$TIME APP_UPTAKE=$UPTAKE \
      BASELINE_P_ASYMPTOMATIC=${ASYMP} PROPORTION_LAB_TEST_PER_DAY=${TEST} ASYMPTOMATIC_INFECTION_RATIO=${ASYMP_INFECTION_RATIO} \
      P_DROPOUT_SYMPTOM=${P_DROPOUT_SYMPTOM} ALL_LEVELS_DROPOUT=${ALL_LEVELS_DROPOUT}  seeds=9_seeds n_search=${n_search} CONTAGION_KNOB=27.5 \
      REC_LEVEL_THRESHOLDS=${REC_LEVEL_THRESHOLDS} TRANSFORMER_EXP_PATH=${TRANSFORMER_EXP_BASEPATH}/${TRANSFORMER_NAME} cpus=10 # \
      # dev=True pure_command_output_file=${TYPE}_${n_people}_${init}.txt ## NASIM
    done
fi

if [ "$INTERVENTION" == "oracle" ]; then
    ORACLE_MUL_NOISE=1.0
    ORACLE_ADD_NOISE=3.0

    python experiment.py exp_file=normalized_mobility env_name=${ENV_RELATIVE_PATH} \
    email_id=$EMAIL slurm_log=${SLURM_LOG_DIR} track=light code_loc=${COVISIM_REPO}/src/covid19sim \
    outdir=${SIM_OUTPUT_BASEDIR}/$dirname/normalized_mobility/ORACLE_MUL_${ORACLE_MUL_NOISE}_ADD_${ORACLE_ADD_NOISE} simulation_days=${SIM_DAYS} \
    intervention=$INTERVENTION global_mobility_scaling_factor_range=${glomo_range}  \
    n_people=$n_people init_fraction_sick=$init time=$TIME APP_UPTAKE=$UPTAKE \
    BASELINE_P_ASYMPTOMATIC=${ASYMP} PROPORTION_LAB_TEST_PER_DAY=${TEST} ASYMPTOMATIC_INFECTION_RATIO=${ASYMP_INFECTION_RATIO} \
    P_DROPOUT_SYMPTOM=${P_DROPOUT_SYMPTOM} ALL_LEVELS_DROPOUT=${ALL_LEVELS_DROPOUT}  seeds=9_seeds n_search=${n_search} CONTAGION_KNOB=27.5 \
    REC_LEVEL_THRESHOLDS=${REC_LEVEL_THRESHOLDS} cpus=6 \
    ORACLE_MUL_NOISE=${ORACLE_MUL_NOISE} ORACLE_ADD_NOISE=${ORACLE_ADD_NOISE} # \
    # dev=True pure_command_output_file=${TYPE}_${n_people}_${init}.txt ## NASIM
fi
