#!/bin/bash
path="/dev/shm/output/batch"
for i in {0..9}
  do
    echo "running seed ${i}"
    mkdir "${path}/${i}"
    python run.py sim --n_people 1000 --init_percent_sick 0.01 --seed $i --outdir "$path/$i" --simulation_days 60
  done
for output_f in $(ls $path)
  do
    echo "running seed $output_f"
    output_f2="$(ls $path/$output_f | grep sim)"
    python run.py model --save_training_data --data_path "$path/$output_f/$output_f2/data.zip" --plot_path "$path/$output_f/$output_f2/plots/" --output_file "$path/$output_f/$output_f2/output/output.pkl" --cluster_path "$path/$output_f/$output_f2/clusters.json" --n_jobs 64
  done
