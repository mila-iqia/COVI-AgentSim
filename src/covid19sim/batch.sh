#!/bin/bash
#path="/dev/shm/output/batch"
root_path="output"
batch_path="batch"
outzip_name="1k_app_1.zip"
config_dir="configs"
config_file="naive_config.yml"
n_people=1000
init_percent_sick=0.01
simulation_days=2
n_jobs=10
num_seeds=1

for (( i=0; i<$num_seeds; i++ ))
  do
    echo "running seed ${i}"
    mkdir "${root_path}/${batch_path}/${i}"

    # Run the simulations
    python run.py sim --n_people $n_people --init_percent_sick $init_percent_sick --seed $i --outdir "$root_path/$batch_path/$i" --simulation_days $simulation_days --n_jobs $n_jobs &
  done

wait

for (( i=0; i<$num_seeds; i++ ))
  do
    # Merge the outputs
    path2="$(ls $root_path/$batch_path/$i | grep sim)"
    echo "merge"
    echo "$root_path/$batch_path/$i/$path2/$path2-output.zip"
    python models/merge_outputs.py --data_path "$root_path/$batch_path/$i/$path2/daily_outputs" --output_path "$root_path/$path2-output.zip"
  done

# Copy the experimental config into the output zip
cp $config_dir/$config_file $root_path/$config_file

# Cleanup the output
rm -fr $root_path/$batch_path

# Zip the merged outputs and config
zip $root_path/$outzip_name $root_path/*

# Copy the merged outputs to AWS
aws s3 cp  $root_path/$outzip_name s3://covid-p2p-simulation

