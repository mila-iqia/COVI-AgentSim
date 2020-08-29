dirname=$1
intervention_day=0

# no tracing
for mobility_scaling_factor in 0.50 0.475 0.45 0.425 0.40 0.375 0.35 0.325 0.3 0.25 0.20
do
  for seed in 4001 5001 6001 7001 3000 3001 3002 3003 1300 3100 1900 3
  do
    echo post-lockdown-no-tracing $seed $mobility_scaling_factor/post-lockdown-no-tracing
    sbatch job-mobility.sh $mobility_scaling_factor/post-lockdown-no-tracing $intervention_day post-lockdown-no-tracing -1 $seed $dirname $mobility_scaling_factor
  done
done

# bdt1
for mobility_scaling_factor in 0.625 0.60 0.575 0.55 0.50 0.475 0.45 0.425 0.40 0.375 0.35
do
  for seed in 4001 5001 6001 7001 3000 3001 3002 3003 1300 3100 1900 3
  do
    echo bdt1 $seed $mobility_scaling_factor/bdt1
    sbatch job-mobility.sh $mobility_scaling_factor/bdt1_60 $intervention_day bdt1 0.8415 $seed $dirname $mobility_scaling_factor
  done
done

# heuristicv1
for mobility_scaling_factor in 1.0 0.95 0.9 0.85 0.80 0.75 0.7 0.65 0.60 0.55 0.50
do
  for seed in 4001 5001 6001 7001 3000 3001 3002 3003 1300 3100 1900 3
  do
    echo heuristic $seed $mobility_scaling_factor/heuristic
    sbatch job-mobility.sh $mobility_scaling_factor/heuristicv1_60 $intervention_day heuristicv1 0.8415 $seed $dirname $mobility_scaling_factor
  done
done


# oracle (tune for MUL and ADD Noise) 
for mobility_scaling_factor in 1.0 0.9 0.80 0.7 0.60
do
  for seed in 4001 5001 6001 7001 3000 3001 3002 3003
  do
    for ORACLE_MUL_NOISE in 0.8 0.6 0.4 0.2 0.5
    do
      for ORACLE_ADD_NOISE in 0.05 0.1 0.08 0.13 0.15
      do
        echo oracle $seed $mobility_scaling_factor/$ORACLE_MUL_NOISE/$ORACLE_ADD_NOISE/oracle_60
        sbatch job-mobility.sh $mobility_scaling_factor/$ORACLE_MUL_NOISE/$ORACLE_ADD_NOISE/oracle_60 $intervention_day oracle 0.8415 $seed $dirname $mobility_scaling_factor $ORACLE_MUL_NOISE $ORACLE_ADD_NOISE
      done
    done
  done
done
