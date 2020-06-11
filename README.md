# Covid-19 Spread Simulator for Tracing App

The simulator is built using [`simpy`](!https://simpy.readthedocs.io/en/latest/simpy_intro/index.html).
It simulates human mobility along with infectious disease (COVID-19) spreading in a city, where city has houses, grocery stores, parks, workplaces, and other non-essential establishments.

Human mobility simulation is based on Spatial-EPR model. More details on this model are [here](https://www.nature.com/articles/ncomms9166) and [here](https://www.nature.com/articles/nphys1760).

The infection spread in this simulator is modeled according to what is known about COVID-19.
Our understanding is based on the published research as well as interactions with the epidemiologists.
We plan to update the simulator as more and more about COVID-19 will be known.


## Dependencies

To run all experiments, including Transformer, install the simulator with [`ctt`](https://github.com/covi-canada/machine-learning)
```
pip install '.[ctt]'
```


If you want to run only the simulator, or the simulator with BCT baselines, you don't need `ctt`; it is only required for the transformer and other ML baselines. To install the simulator without `ctt`
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

In order to train with a risk predictor in the loop, set the type of intervention (`b`, ,)  and the day that intervention should start:

```
python run.py tune=True tracing_method=`transformer` INTERVENTION_DAY=5
```

The above commands only produces one run of the simulator. In order to run multiple simulations with domain randomization suitable for creating a training dataset, we make use of a config file located at `simulator/src/covid19sim/hydra-configs/search/expriment.yaml`. Note that this takes several hours on a CPU cluster.

```
python random_search.py
```

When invoked, the simulator will read the yaml files found in the `hydra_configs` folder and parametrize the simulation accordingly. Here are a few important config attributes (found in `hydra_configs/base_method.yaml`) :
- n_people: Number of humans in the simulation
- init_percent_sick: Proportion of humans sick at the beginning of the simulation
- simulation_days: Number of days simulated.
- seed: Seed for the random number generators used by the simulator.
- COLLECT_LOGS: If set to True, 
- COLLECT_TRAINING_DATA: If set to True, the simulator will output daily data for each human in the simulation. This day can be used to train an ML model.


## How to run tests?
Run -
```
pytest python run.py test
```


## How to run it as a function?
Although not designed with this usage in mind one, can still call the simulator like this :
```
from run import simulate
monitors = simulate(
    n_people=None,
    init_percent_sick=0.01,
    start_time=datetime.datetime(2020, 2, 28, 0, 0),
    simulation_days=10,
    outfile=None,
    out_chunk_size=None,
    print_progress=False,
    seed=0)
```

## Semantics of code
`Human` class builds people, and `Location` class builds stores, parks, workplaces, households, and non-essential establishments.

## Semantics of Data
`data` is a `list`. Each entry in the `list` is an event represented as a `dict`.
The detailed information about events is in [docs/events.md](docs/src/notes/events.md)

## Contributing
Please get in touch with me at [pgupta@turing.ac.uk](pgupta@turing.ac.uk). There are several people working on it, so it will be the best use of everyone's time and effort if we all work on different aspects of this project.

Some areas that need work are listed [here](docs/src/notes/CONTRIBUTING.md). We track and manage our tasks using [Google Sheets](https://docs.google.com/spreadsheets/d/11t1T66AAVeR6P341nZYP1qwLdvhCkU_EwFwUkyLziLQ/edit?usp=sharing).

## Collaborators
[@marco-gires](https://github.com/marco-gires), [@marie-pellat](https://github.com/mariepellat), [@teganmaharaj](https://github.com/teganmaharaj), [@giancarlok](https://github.com/giancarlok), [@thechange](https://github.com/thechange), [@soundarya98](https://github.com/soundarya98), [@mweiss17](https://github.com/mweiss17)
