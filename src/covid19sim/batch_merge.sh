#!/bin/bash
path="/dev/shm/output/batch"
for i in {0..0}
  do
    echo "running seed ${i}"
    mkdir "${path}/${i}"
    python run.py sim --n_people 1000 --init_percent_sick 0.01 --seed $i --outdir "$path/$i" --simulation_days 60

    path2="$(ls $path/$i | grep sim)"
    python models/merge_outputs.py --data_path "$path/$i/$path2/daily_outputs" --output_path "$path/$i/$path2/$path2-output.zip"
    aws s3 cp  $path/$i/$path2/$path2-output.zip s3://covid-p2p-simulation 
  done
