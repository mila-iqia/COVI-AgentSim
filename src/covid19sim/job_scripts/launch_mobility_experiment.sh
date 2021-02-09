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
dirname=$8/sensitivity_S_${SCENARIO}_${n_people}_init_${init}_UPTAKE_${UPTAKE}/scatter_Ax_${ASYMP}_Lx_${ALL_LEVELS_DROPOUT}_Sx_${P_DROPOUT_SYMPTOM}_test_${TEST}

# simulation days
SIM_DAYS=60

# essentials
ENV_RELATIVE_PATH=covid19
COVISIM_REPO=/home/$USER/simulator
SLURM_LOG_DIR=$SCRATCH/job_logs
SIM_OUTPUT_BASEDIR=$SCRATCH/
EMAIL=pg2455@columbia.edu # to be notified of every run

source ~/${ENV_RELATIVE_PATH}/bin/activate

cd /home/pratgupt/simulator/src/covid19sim/job_scripts
# normalized mobility

TIME="'2:50:00'"

if [ "${n_people}" -ge "5000" ] ; then
  if [ "$INTERVENTION" == "heuristicv4" ] || [ "$INTERVENTION" == "transformer" ] ; then
    TIME="'9:50:00'"
  fi
fi

glomo_range=uniform_for_sensitivity
n_search=225

if [ "$TYPE" != "main-scenario" ] ; then
  # no need to have a GP if filter is the number of contacts
  n_search=60
  glomo_range=uniform_for_sensitivity_narrow
  # if [ "$INTERVENTION" == "bdt1" ] ; then
  #   glomo_range=uniform_for_sensitivity_medium
  # elif [ "$INTERVENTION" == "heuristicv4" ] ; then
  #   glomo_range=uniform_for_sensitivity_medium_low
  # elif [ "$INTERVENTION" == "transformer" ] ; then
  #   glomo_range=uniform_for_sensitivity_low
  # else
  #   glomo_range=uniform_for_sensitivity_high
  # fi
fi

if [ -x "$(command -v queue_monitor)" ]; then
  queue_monitor $(echo $((995 - $n_search))) && notify "launching $n_search jobs now" # will not run this script until there are are 225 jobs can be queued
fi

if [ "$INTERVENTION" == "post-lockdown-no-tracing" ] || [ "$INTERVENTION" == "bdt1" ] || [ "$INTERVENTION" == "heuristicv4" ] ; then
  cd $COVISIM_REPO/src/covid19sim/job_scripts

  python experiment.py exp_file=normalized_mobility env_name=${ENV_RELATIVE_PATH} \
    email_id=$EMAIL slurm_log=${SLURM_LOG_DIR} track=light code_loc=${COVISIM_REPO}/src/covid19sim \
    base_dir=${SIM_OUTPUT_BASEDIR}/$dirname/normalized_mobility simulation_days=${SIM_DAYS}\
    intervention=$INTERVENTION  global_mobility_scaling_factor_range=${glomo_range} \
    n_people=${n_people} init_fraction_sick=$init time=$TIME APP_UPTAKE=$UPTAKE \
    BASELINE_P_ASYMPTOMATIC=$ASYMP PROPORTION_LAB_TEST_PER_DAY=$TEST \
    P_DROPOUT_SYMPTOM=${P_DROPOUT_SYMPTOM} ALL_LEVELS_DROPOUT=${ALL_LEVELS_DROPOUT}  seeds=9_seeds n_search=${n_search} CONTAGION_KNOB=27.5 dev=True
fi

if [ "$INTERVENTION" == "transformer" ]; then
  echo NOT IMPLEMENTED
fi

# transformer
# STELLAR-HAZE-736 MISTY-WOOD-727 RESILIENT-STAR-735 STELLAR-MONKEY-732
# GENIAL-WIND-744 FLOWING-STAR-753 PLEASANT-SUNSET-746 GLAD-LAKE-754
# RADIANT-FOG-751
# ROYAL-ENERGY-749
# STOIC-MICROWAVE-755
# FLOWING-DRAGON-758
# for TRANSFORMER_FOLDER_NAME in STOIC-MICROWAVE-755
# do
#   python experiment.py exp_file=normalized_mobility env_name=covid19 \
#    email_id=pg2455@columbia.edu slurm_log=/scratch/pratgupt/job_logs/ track=light \
#    outdir=/scratch/pratgupt/$dirname/normalized_mobility/$TRANSFORMER_FOLDER_NAME \
#    intervention=transformer global_mobility_scaling_factor_range=uniform_for_sensitivity \
#    n_people=$n_people init_fraction_sick=$init REC_LEVEL_THRESHOLDS="[0,1,2]" \
#    TRANSFORMER_EXP_PATH=/scratch/pratgupt/pra_models/$TRANSFORMER_FOLDER_NAME \
#    APP_UPTAKE=$UPTAKE time="5:00:00" seeds=1_seed $OTHER_ARGS n_search=225
# done
