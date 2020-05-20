#!/bin/bash

PYTHON=python

# run the server in background
if [ "$2" == "transformer" ]; then
  $PYTHON ~/covid_p2p_simulation/server_bootstrap.py -e ~/covid_p2p_risk_prediction/models/CTT-SHIPMENT-0/ --mp-threads 10 &
fi

# launch simulations
for seed in 1234 1235 1236 1237
do
  $PYTHON run.py tune --n_people 2000 --simulation_days 45 --seed $seed --init_percent_sick 0.001 --config configs/$1 --name $2  --outdir tune/ &
  sleep 5;
done
wait;
