#!/bin/bash
# 0.8415 - 60
# 0.xxxx - 50
# 0.6415 - 45
# 0.5618 - 40
# 0.4215 - 30
# 0.3580 - 25
# 0.2850 - 20
# 0.2140 - 15
SCENARIO=$1
UPTAKE=$2
n_people=3000
init=0.002
ASYMP=$3
ALL_LEVELS_DROPOUT=$4
P_DROPOUT_SYMPTOM=$5
TEST=$6
dirname=sensitivity_S_${SCENARIO}_${n_people}_init_${init}_UPTAKE_${UPTAKE}/scatter_Ax_${ASYMP}_Lx_${ALL_LEVELS_DROPOUT}_Sx_${P_DROPOUT_SYMPTOM}_test_${TEST}
INTERVENTION=$7

source ~/.bashrc
conda activate cov19
source ~/prepenv.sh
cd /home/nrahaman/python/covi-simulator/src/covid19sim

# normalized mobility
if [ "$INTERVENTION" == "post-lockdown-no-tracing" ]; then
  # # no-tracing
  python job_scripts/experiment.py exp_file=normalized_mobility \
    track=light \
    base_dir=/home/nrahaman/python/covi-simulator/exp/sensitivity/$dirname/normalized_mobility \
    intervention=post-lockdown-no-tracing  global_mobility_scaling_factor_range=cartesian_low \
    n_people=$n_people init_fraction_sick=$init APP_UPTAKE=$UPTAKE \
    BASELINE_P_ASYMPTOMATIC=$ASYMP PROPORTION_LAB_TEST_PER_DAY=$TEST \
    P_DROPOUT_SYMPTOM=$P_DROPOUT_SYMPTOM ALL_LEVELS_DROPOUT=$ALL_LEVELS_DROPOUT dev=True
fi
if [ "$INTERVENTION" == "bdt1" ]; then
  # # bdt
  python job_scripts/experiment.py exp_file=normalized_mobility \
     track=light \
     base_dir=/home/nrahaman/python/covi-simulator/exp/sensitivity/$dirname/normalized_mobility \
     intervention=bdt1  global_mobility_scaling_factor_range=cartesian_medium \
     n_people=$n_people init_fraction_sick=$init APP_UPTAKE=$UPTAKE \
     BASELINE_P_ASYMPTOMATIC=$ASYMP PROPORTION_LAB_TEST_PER_DAY=$TEST \
     P_DROPOUT_SYMPTOM=$P_DROPOUT_SYMPTOM ALL_LEVELS_DROPOUT=$ALL_LEVELS_DROPOUT dev=True
fi
if [ "$INTERVENTION" == "heuristicv4" ]; then
   python job_scripts/experiment.py exp_file=normalized_mobility \
    base_dir=/home/nrahaman/python/covi-simulator/exp/sensitivity/$dirname/normalized_mobility \
    track=light \
    intervention=heuristicv4  global_mobility_scaling_factor_range=cartesian_medium \
    n_people=$n_people init_fraction_sick=$init APP_UPTAKE=$UPTAKE \
    BASELINE_P_ASYMPTOMATIC=$ASYMP PROPORTION_LAB_TEST_PER_DAY=$TEST \
    P_DROPOUT_SYMPTOM=$P_DROPOUT_SYMPTOM ALL_LEVELS_DROPOUT=$ALL_LEVELS_DROPOUT dev=True
fi

#python job_scripts/experiment.py exp_file=normalized_mobility \
#    track=light \
#    base_dir=/home/nrahaman/python/covi-simulator/exp/sensitivity/$dirname/normalized_mobility \
#    intervention=post-lockdown-no-tracing  global_mobility_scaling_factor_range=cartesian_low \
#    n_people=$n_people init_fraction_sick=$init APP_UPTAKE=$UPTAKE \
#    BASELINE_P_ASYMPTOMATIC=$ASYMP PROPORTION_LAB_TEST_PER_DAY=$TEST \
#    P_DROPOUT_SYMPTOM=$P_DROPOUT_SYMPTOM ALL_LEVELS_DROPOUT=$ALL_LEVELS_DROPOUT dev=True

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
#    intervention=transformer global_mobility_scaling_factor_range=cartesian_medium \
#    n_people=$n_people init_fraction_sick=$init REC_LEVEL_THRESHOLDS="[0,1,2]" \
#    TRANSFORMER_EXP_PATH=/scratch/pratgupt/pra_models/$TRANSFORMER_FOLDER_NAME \
#    APP_UPTAKE=$UPTAKE time="5:00:00"
# done