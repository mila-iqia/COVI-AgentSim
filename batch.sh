#!/bin/bash
path="output/batch"
for i in {0..9}
  do
    echo "running seed ${i}"
    mkdir "${path}/${i}"
    python run.py sim --n_people 30 --init_percent_sick 0.01 --seed $i --outdir "$path/$i" --simulation_days 30
  done
for output_f in $(ls $path)
  do
    echo "running seed ${i}"
    python run.py model --save_training_data --data_path "$path/$output_f/data.zip$ --n_jobs 1
  done
