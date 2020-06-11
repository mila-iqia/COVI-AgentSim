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


## How to run tests?
Run -
```
python run.py test
```

### Parameters

```
@click.option('--n_people', help='population of the city', type=int, default=100)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--out_chunk_size', help='number of events per dump in outfile', type=int, default=2500, required=False)
@click.option('--outdir', help='the directory to write data to', type=str, default="output", required=False)
@click.option('--seed', help='seed for the process', type=int, default=0)
@click.option('--n_jobs', help='number of parallel procs to query the risk servers with', type=int, default=1)
@click.option('--port', help='which port should we look for inference servers on', type=int, default=6688)
@click.option('--config', help='which experiment config would we like to run with', type=str, default="configs/naive_config.yml")
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
monitors = simulate(n_stores=100, n_parks=50, n_people=100, n_misc=100, print_progress=False, seed=0)
```

`data` is a `list` of `dict`.

## Base SEIR plots
Following will require `cufflinks` and `plotly`.
```
python run.py base
```
It will open a browser window with the plot of [SEIR curves](https://www.idmod.org/docs/hiv/model-seir.html#seir-and-seirs-models).

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
