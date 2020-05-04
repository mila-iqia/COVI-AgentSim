#!/bin/bash
#path="/dev/shm/output/batch"
path="output/batch/"
mkdir $path
for i in {0..1}
  do
    echo "running seed ${i}"
    mkdir "${path}/${i}"
    python run.py sim --n_people 100 --init_percent_sick 0.05 --seed $i --outdir "$path/$i" --simulation_days 20 --n_jobs 10 &
  done
