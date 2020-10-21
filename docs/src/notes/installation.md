# Installation
To install the simulator without `ctt`
```
pip install .
```

To install the simulator with [`ctt`](https://github.com/mila-iqia/COVI-ML)
```
pip install '.[ctt]'
```

Please checkout `setup.py` for external dependencies.

## How to run it using command line?
Run the simulator as -
```
python run.py sim --n_people 100 --seed 0
```

The simulator will output a logfile to `output/sim_people-{N_PEOPLE}_days-{SIMULATION_DAYS}_init-{INIT_PERCENT_SICK}_seed-{SEED}_{DATE}-{TIME}/data.zip`. It is a .zip file of `list` of `dict` pickles which contains a log of the mobility activity of a population of humans in `simulator.py`.

Run the risk prediction algorithms as -
```
python risk_prediction.py
```
This file reads in the logs that are output from the simulator, and runs a risk prediction algorithm based on:
 1) The reported symptoms of that individual, given they have the app.
 2) The encounters that individual had (which contain quantized user ids and risk levels).
 3) The risk update messages that are sent when a previously encountered user's risk level changes significantly.

You may select the model which you want to specify, or add your own model in that file.
This file also handles clustering of messages (under the currently understood privacy model), which can be useful for risk prediction algorithms which
depend on knowing how many contacts happened with one individual.

## How to run tests?
Run -
```
python run.py test
```
