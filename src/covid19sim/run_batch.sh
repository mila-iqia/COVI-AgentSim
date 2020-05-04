#!/bin/bash
#path="/dev/shm/output/batch"
path="output/batch"
experiment_name="1k_app_1.zip"
experimental_config="configs/experimental_config.yml"
n_people=1000
init_percent_sick=0.01
simulation_days=20
n_jobs=10
num_seeds=2

for (( i=0; i<$num_seeds; i++ ))
  do
    echo "running seed ${i}"
    mkdir "${path}/${i}"

    # Run the simulations
    python run.py sim --n_people $n_people --init_percent_sick $init_percent_sick --seed $i --outdir "$path/$i" --simulation_days $simulation_days --n_jobs $n_jobs &
  done

for (( i=0; i<$num_seeds; i++ ))
  do
    echo "running seed ${i}"
    mkdir "${path}/${i}"

    # Merge the outputs
    path2="$(ls $path/$i | grep sim)"
    python models/merge_outputs.py --data_path "$path/$i/$path2/daily_outputs" --output_path "$path/$i/$path2/$path2-output.zip"

    # Copy the experimental config into the output zip
    cp $experimental_config $path/$i/$path2/$experimental_config

    # Zip the merged outputs and config
    zip $path/$i/$path2/1k_app_1-public-test.zip $path/$i/$path2/*

    # Copy the merged outputs to AWS
    aws s3 cp  $path/$i/$path2/$path2-output.zip s3://covid-p2p-simulation
  done

# Cleanup
