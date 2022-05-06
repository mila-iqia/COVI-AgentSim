
#!/bin/bash
TYPE=$1 # "user-behavior-Lx", "user-behavior-Sx", "test-quantity", "asymptomatic-infection-ratio", "adoption-rate"
ACTION=$2 # "plot" or "launch-jobs"
NAME=$3 # Name of the folder will be sensitivity_Yx_NAME
DEFAULTS=$4

N_PEOPLE=${5:-5000}
INIT=${6:-0.004}

# only used for outputing plot path
BASEDIR=/home/nrahaman/python/covi-simulator/exp/sensitivity_v3

# values
ALL_LEVELS_DROPOUT=(0.02 0.08 0.16 0.32 0.64)
P_DROPOUT_SYMPTOM=(0.20 0.40 0.60 0.80)
PROPORTION_LAB_TEST_PER_DAY=(0.005 0.003 0.0015 0.001 0.0005)
BASELINE_P_ASYMPTOMATIC=(0.5 0.625 0.75 0.875 1.0)
APP_UPTAKE=(0.8415 0.7170 0.5618 0.4215 0.2850)
ASYMPTOMATIC_INFECTION_RATIO=(0.29 0.35 0.40 0.45)

# default index
ALL_LEVELS_DROPOUT_DEFAULT_INDEX=0
P_DROPOUT_SYMPTOM_DEFAULT_INDEX=0
PROPORTION_LAB_TEST_PER_DAY_DEFAULT_INDEX=3
BASELINE_P_ASYMPTOMATIC_DEFAULT_INDEX=2
ASYMPTOMATIC_INFECTION_RATIO_DEFAULT_INDEX=0

# change ALL_LEVELS_DROPOUT
if [ "$TYPE" == "user-behavior-Lx" ]; then
  Sx=${P_DROPOUT_SYMPTOM[$P_DROPOUT_SYMPTOM_DEFAULT_INDEX]}
  Ax=${BASELINE_P_ASYMPTOMATIC[$BASELINE_P_ASYMPTOMATIC_DEFAULT_INDEX]}
  Tx=${PROPORTION_LAB_TEST_PER_DAY[$PROPORTION_LAB_TEST_PER_DAY_DEFAULT_INDEX]}
  FOLDER_NAME=${NAME}/sensitivity_Lx
  INTERVENTIONS=(transformer) #(bdt1 heuristicv4) #(transformer) # (bdt1 heuristicv4 transformer)

  if [ "$ACTION" == "launch-jobs" ]; then

    #
    END=$((${#ALL_LEVELS_DROPOUT[@]}-1))
    if [ "$DEFAULTS" -eq "1" ]; then
      START=0
    else
      START=1
    fi

    # launch all for 30% AR
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in $(seq $START $END);
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[3]} $Ax ${ALL_LEVELS_DROPOUT[$x]} $Sx $Tx \
                              $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done

    # exclude the case used for the main scenario (60% AR)
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in $(seq $START $END);
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[0]} $Ax ${ALL_LEVELS_DROPOUT[$x]} $Sx $Tx \
                              $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done

  fi

  # plot
  if [ "$ACTION" == "plot" ]; then
    # for all methods
    for ARx in  ${APP_UPTAKE[0]} ${APP_UPTAKE[3]}
    do
      for Lx in ${ALL_LEVELS_DROPOUT[@]}
      do
        if [ "$DEFAULTS" -eq "1" ]; then
          sbatch run_plot_sensitivity.sh Main $ARx $Ax $Lx $Sx $Tx XX $FOLDER_NAME $NAME $N_PEOPLE $INIT
        else
          echo "python main.py plot=normalized_mobility path=$BASEDIR/$FOLDER_NAME/sensitivity_S_Main_${N_PEOPLE}_init_${INIT}_UPTAKE_${ARx}/scatter_Ax_${Ax}_Lx_${Lx}_Sx_${Sx}_test_${Tx}/normalized_mobility load_cache=False use_cache=False normalized_mobility_use_extracted_data=False"
        fi
      done
    done
  fi

fi


# change ALL_LEVELS_DROPOUT
if [ "$TYPE" == "user-behavior-Sx" ]; then
  Lx=${ALL_LEVELS_DROPOUT[$ALL_LEVELS_DROPOUT_DEFAULT_INDEX]}
  Ax=${BASELINE_P_ASYMPTOMATIC[$BASELINE_P_ASYMPTOMATIC_DEFAULT_INDEX]}
  Tx=${PROPORTION_LAB_TEST_PER_DAY[$PROPORTION_LAB_TEST_PER_DAY_DEFAULT_INDEX]}
  FOLDER_NAME=${NAME}/sensitivity_Sx
  INTERVENTIONS=(transformer) #(heuristicv4) #(transformer) #(heuristicv4 transformer)

  if [ "$ACTION" == "launch-jobs" ]; then
    #
    END=$((${#P_DROPOUT_SYMPTOM[@]}-1))
    if [ "$DEFAULTS" -eq "1" ]; then
      START=0
    else
      START=1
    fi

    # launch all for 30% AR
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in $(seq $START $END);
      do

        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[3]} $Ax $Lx ${P_DROPOUT_SYMPTOM[$x]} $Tx \
                              $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done

    # (60% AR)
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in $(seq $START $END);
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[0]} $Ax $Lx ${P_DROPOUT_SYMPTOM[$x]} $Tx \
                              $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done
  fi

  # plot
  if [ "$ACTION" == "plot" ]; then
    # for all methods
    for ARx in  ${APP_UPTAKE[0]} ${APP_UPTAKE[3]}
    do
      for Sx in ${P_DROPOUT_SYMPTOM[@]}
      do
        if [ "$DEFAULTS" -eq "1" ]; then
          sbatch run_plot_sensitivity.sh Main $ARx $Ax $Lx $Sx $Tx XX $FOLDER_NAME $NAME $N_PEOPLE $INIT
        else
          echo "python main.py plot=normalized_mobility path=$BASEDIR/$FOLDER_NAME/sensitivity_S_Main_${N_PEOPLE}_init_${INIT}_UPTAKE_${ARx}/scatter_Ax_${Ax}_Lx_${Lx}_Sx_${Sx}_test_${Tx}/normalized_mobility load_cache=False use_cache=False normalized_mobility_use_extracted_data=False"
        fi
      done
    done
  fi

fi

if [ "$TYPE" == "adoption-rate" ]; then
  Lx=${ALL_LEVELS_DROPOUT[$ALL_LEVELS_DROPOUT_DEFAULT_INDEX]}
  Sx=${P_DROPOUT_SYMPTOM[$P_DROPOUT_SYMPTOM_DEFAULT_INDEX]}
  Ax=${BASELINE_P_ASYMPTOMATIC[$BASELINE_P_ASYMPTOMATIC_DEFAULT_INDEX]}
  Tx=${PROPORTION_LAB_TEST_PER_DAY[$PROPORTION_LAB_TEST_PER_DAY_DEFAULT_INDEX]}
  FOLDER_NAME=${NAME}/sensitivity_ARx
  INTERVENTIONS=(bdt1 heuristicv4)

  if [ "$ACTION" == "launch-jobs" ]; then

    if [ "$DEFAULTS" -eq "1" ]; then
      ARRAY=(0 1 2 3 4)
    else
      ARRAY=(1 2 4)
    fi

    # post-lockdown-no-tracing
    ./launch_mobility_experiment.sh Main -1 $Ax $Lx $Sx $Tx \
              post-lockdown-no-tracing $FOLDER_NAME $TYPE $N_PEOPLE $INIT

    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in "${ARRAY[@]}"
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[$x]} $Ax $Lx $Sx $Tx \
                  $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done

  # ml based

  fi


  if [ "$ACTION" == "plot" ]; then
    for ARx in "${APP_UPTAKE[@]}"
    do
      if [ "$DEFAULTS" -eq "1" ]; then
        sbatch run_plot_sensitivity.sh Main $ARx $Ax $Lx $Sx $Tx XX $FOLDER_NAME $NAME $N_PEOPLE $INIT
      else
        echo "python main.py plot=normalized_mobility path=$BASEDIR/$FOLDER_NAME/sensitivity_S_Main_${N_PEOPLE}_init_${INIT}_UPTAKE_${ARx}/scatter_Ax_${Ax}_Lx_${Lx}_Sx_${Sx}_test_${Tx}/normalized_mobility load_cache=False use_cache=False normalized_mobility_use_extracted_data=False"
      fi
    done
  fi

fi

if [ "$TYPE" == "test-quantity" ]; then
  Lx=${ALL_LEVELS_DROPOUT[$ALL_LEVELS_DROPOUT_DEFAULT_INDEX]}
  Sx=${P_DROPOUT_SYMPTOM[$P_DROPOUT_SYMPTOM_DEFAULT_INDEX]}
  Ax=${BASELINE_P_ASYMPTOMATIC[$BASELINE_P_ASYMPTOMATIC_DEFAULT_INDEX]}
  FOLDER_NAME=${NAME}/sensitivity_Tx
  INTERVENTIONS=(transformer) #(bdt1 heuristicv4) #(transformer) # (bdt1 heuristicv4 transformer)

  if [ "$ACTION" == "launch-jobs" ]; then

    if [ "$DEFAULTS" -eq "1" ]; then
      ARRAY=(0 1 2 3 4)
    else
      ARRAY=(0 1 2 4)
    fi

    # for x in "${ARRAY[@]}"
    # do
    #   ./launch_mobility_experiment.sh Main -1 $Ax $Lx $Sx ${PROPORTION_LAB_TEST_PER_DAY[$x]} \
    #             post-lockdown-no-tracing $FOLDER_NAME $TYPE $N_PEOPLE $INIT
    # done

    # 30% AR
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in "${ARRAY[@]}"
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[3]} $Ax $Lx $Sx ${PROPORTION_LAB_TEST_PER_DAY[$x]} \
                  $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done

    # 60% AR
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in "${ARRAY[@]}"
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[0]} $Ax $Lx $Sx ${PROPORTION_LAB_TEST_PER_DAY[$x]} \
                  $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done


  fi

  # plot
  if [ "$ACTION" == "plot" ]; then
    for ARx in ${APP_UPTAKE[0]} ${APP_UPTAKE[3]} -1
    do
      for Tx in ${PROPORTION_LAB_TEST_PER_DAY[@]}
      do
        if [ "$DEFAULTS" -eq "1" ]; then
          sbatch run_plot_sensitivity.sh Main $ARx $Ax $Lx $Sx $Tx XX $FOLDER_NAME $NAME $N_PEOPLE $INIT
        else
          echo "python main.py plot=normalized_mobility path=$BASEDIR/$FOLDER_NAME/sensitivity_S_Main_${N_PEOPLE}_init_${INIT}_UPTAKE_${ARx}/scatter_Ax_${Ax}_Lx_${Lx}_Sx_${Sx}_test_${Tx}/normalized_mobility load_cache=False use_cache=False normalized_mobility_use_extracted_data=False"
        fi
      done
    done
  fi

fi

if [ "$TYPE" == "asymptomatic-infection-ratio" ]; then
  Lx=${ALL_LEVELS_DROPOUT[$ALL_LEVELS_DROPOUT_DEFAULT_INDEX]}
  Sx=${P_DROPOUT_SYMPTOM[$P_DROPOUT_SYMPTOM_DEFAULT_INDEX]}
  Tx=${PROPORTION_LAB_TEST_PER_DAY[$PROPORTION_LAB_TEST_PER_DAY_DEFAULT_INDEX]}
  Ax=${BASELINE_P_ASYMPTOMATIC[$BASELINE_P_ASYMPTOMATIC_DEFAULT_INDEX]}
  FOLDER_NAME=${NAME}/sensitivity_AIRx
  INTERVENTIONS=(transformer) #(bdt1 heuristicv4) #(transformer) # (bdt1 heuristicv4 transformer)

  if [ "$ACTION" == "launch-jobs" ]; then

    if [ "$DEFAULTS" -eq "1" ]; then
      ARRAY=(0 1 2 3)
    else
      ARRAY=(1 2 3)
    fi

    for x in "${ARRAY[@]}"
    do
      ./launch_mobility_experiment.sh Main -1 $Ax $Lx $Sx $Tx \
                post-lockdown-no-tracing $FOLDER_NAME $TYPE $N_PEOPLE $INIT ${ASYMPTOMATIC_INFECTION_RATIO[$x]}
    done

    # 30% AR
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in "${ARRAY[@]}"
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[3]} $Ax $Lx $Sx $Tx \
                  $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT ${ASYMPTOMATIC_INFECTION_RATIO[$x]}
      done
    done

    # 60% AR
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in "${ARRAY[@]}"
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[0]} $Ax $Lx $Sx $Tx \
                  $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT ${ASYMPTOMATIC_INFECTION_RATIO[$x]}
      done
    done

  fi

  # plot
  if [ "$ACTION" == "plot" ]; then
    for ARx in ${APP_UPTAKE[0]} ${APP_UPTAKE[3]} -1
    do
      for AIRx in ${ASYMPTOMATIC_INFECTION_RATIO[@]}
      do
        if [ "$DEFAULTS" -eq "1" ]; then
          sbatch run_plot_sensitivity.sh Main $ARx $Ax $Lx $Sx $Tx XX $FOLDER_NAME $NAME $N_PEOPLE $INIT $AIRx
        else
          echo "python main.py plot=normalized_mobility path=$BASEDIR/$FOLDER_NAME/sensitivity_S_Main_${N_PEOPLE}_init_${INIT}_UPTAKE_${ARx}/scatter_Ax_${Ax}_Lx_${Lx}_Sx_${Sx}_test_${Tx}_AIR_${AIRx}/normalized_mobility load_cache=False use_cache=False normalized_mobility_use_extracted_data=False"
        fi
      done
    done
  fi

fi

if [ "$TYPE" == "asymptomatic-proportion" ]; then
  Lx=${ALL_LEVELS_DROPOUT[$ALL_LEVELS_DROPOUT_DEFAULT_INDEX]}
  Sx=${P_DROPOUT_SYMPTOM[$P_DROPOUT_SYMPTOM_DEFAULT_INDEX]}
  Tx=${PROPORTION_LAB_TEST_PER_DAY[$PROPORTION_LAB_TEST_PER_DAY_DEFAULT_INDEX]}
  FOLDER_NAME=${NAME}/sensitivity_Ax
  INTERVENTIONS=(transformer) # (bdt1 heuristicv4 transformer)

  if [ "$ACTION" == "launch-jobs" ]; then

    if [ "$DEFAULTS" -eq "1" ]; then
      ARRAY=(0 1 2 3 4)
    else
      ARRAY=(0 1 3 4)
    fi

    # for x in "${ARRAY[@]}"
    # do
    #   ./launch_mobility_experiment.sh Main -1 ${BASELINE_P_ASYMPTOMATIC[$x]} $Lx $Sx $Tx \
    #             post-lockdown-no-tracing $FOLDER_NAME $TYPE $N_PEOPLE $INIT
    # done

    # 30% AR
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in "${ARRAY[@]}"
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[3]} ${BASELINE_P_ASYMPTOMATIC[$x]} $Lx $Sx $Tx \
                  $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done

    # 60% AR
    for intervention in "${INTERVENTIONS[@]}"
    do
      for x in "${ARRAY[@]}"
      do
        ./launch_mobility_experiment.sh Main ${APP_UPTAKE[0]} ${BASELINE_P_ASYMPTOMATIC[$x]} $Lx $Sx $Tx \
                  $intervention $FOLDER_NAME $TYPE $N_PEOPLE $INIT
      done
    done


    fi

    # plot
    if [ "$ACTION" == "plot" ]; then
      for ARx in ${APP_UPTAKE[0]} ${APP_UPTAKE[3]} -1
      do
        for Ax in ${BASELINE_P_ASYMPTOMATIC[@]}
        do
          if [ "$DEFAULTS" -eq "1" ]; then
            sbatch run_plot_sensitivity.sh Main $ARx $Ax $Lx $Sx $Tx XX $FOLDER_NAME $NAME $N_PEOPLE $INIT
          else
            echo "python main.py plot=normalized_mobility path=$BASEDIR/$FOLDER_NAME/sensitivity_S_Main_${N_PEOPLE}_init_${INIT}_UPTAKE_${ARx}/scatter_Ax_${Ax}_Lx_${Lx}_Sx_${Sx}_test_${Tx}/normalized_mobility load_cache=False use_cache=False normalized_mobility_use_extracted_data=False"
          fi
        done
      done
    fi

fi




notify "$TYPE - $ACTION for $NAME:  all jobs are launched"
