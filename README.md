# Covid-19 Spread Simulator for Tracing App

The simulator is built using [`simpy`](!https://simpy.readthedocs.io/en/latest/simpy_intro/index.html).
It simulates human mobility along with infectious disease (COVID-19) spreading in a city, where city has houses, grocery stores, parks, workplaces, and other non-essential establishments.

Human mobility simulation is based on Spatial-EPR model. More details on this model are [here](https://www.nature.com/articles/ncomms9166) and [here](https://www.nature.com/articles/nphys1760).

The infection spread in this simulator is modeled according to what is known about COVID-19.
Our understanding is based on the published research as well as working closely with epidemiologists and other experts.
We plan to update the simulator as more and more about COVID-19 will be known.


## Dependencies

To run all experiments, including Transformer, clone the `machine-learning` repository in addition to this one, and install this simulator with `ctt`:
```
pip install '.[ctt]'
```

If you want to run only the simulator, or the simulator with BCT baselines, you don't need to clone the `machine-learning` repository and don't need to install `ctt`; it is only required for the transformer and other ML baselines. To install the simulator without `ctt`: (*note that this usage is not what we use for paper experiments, and may not be fully supported/explained by this README*)

```
pip install .
```

Please checkout `setup.py` for external dependencies.

## How to run it using command line?
Run the simulator in the 'unmitigated' scenario with console logging of infection progression and various statistics use `tune=True`
```
python run.py tune=True 
```

The simulator will output a logfile and pickle to `output/sim_people-{N_PEOPLE}_days-{SIMULATION_DAYS}_init-{INIT_PERCENT_SICK}_seed-{SEED}_{DATE}-{TIME}/data.zip`. The `.pkl` contains all the activity of a population of humans in `simulator.py`. For a thousand people for 30 days (the default), this takes around 15 minutes on a 2-year old laptop (X1 carbon).

In order to train with a risk predictor in the loop, set the type of intervention (`binary_digital_tracing_order_1`, `binary_digital_tracing_order_2`, `transformer`, `oracle`, `no_intervention`)  and the day that intervention should start. 
```
python run.py tune=True tracing_method=`transformer` INTERVENTION_DAY=5
```
Note that for MLP and linear regression, the tracing method should be set to `transformer`; transformer, MLP, and linear regression use the same intervention codepath but the weights of the inference method are different. In order to change between these you need to modify the following properties (sorry about the naming) in `train_config.yaml`: 

| Property | Transformer | MLP | Linear Regression |
|:--|:--|:--|:--|
| `model.name =`| `MixSetNet`    | `MomentNet` | `MomentNet` |
| `model.kwargs.block_type =` | `sssss`   | `nrrrn` | `l` |

The above commands only produces one run of the simulator. This is useful for debugging, but in order to run multiple simulations with domain randomization, suitable for creating a training dataset, we make use of a config file located at `simulator/src/covid19sim/hydra-configs/search/expriment.yaml` to run 100 train and 20 validation. Note that this takes several hours on a CPU cluster.

```
python random_search.py
```

All experiments are reproducible given a random seed. Use the random seeds given in the config files (don't change anything) in order to reproduce the same results in the paper.


## How to run tests?
From the root of `simulator`, run:
```
pytest
```

### Accessing Simulation Data
Load the output of the simulator as following
```
data = pickle.load(open("output/data.pkl", 'rb'))
```

## How to run it as a function?
Although not designed with this usage in mind one can still call it like this
```
from run import simulate
city, monitors, tracker = simulate(n_stores=100, n_parks=50, n_people=100, n_misc=100, print_progress=False, seed=0)
```

`data` is a `list` of `dict`.


## Semantics of code
`Human` class, located in `simulator.py`, builds people with individual properties, and the `City` and `Location` classes, located in `base.py` build stores, parks, workplaces, households, and non-essential establishments. 
