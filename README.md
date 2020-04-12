# COVID-19 Spread Simulator for Tracing App


This is a sub-project of [Peer-to-Peer AI Tracing App](https://mila.quebec/en/peer-to-peer-ai-tracing-of-covid-19/) delegated by [Prof. Yoshua Bengio](https://yoshuabengio.org/). Read more about the app in Prof. Bengio's [blog post](https://yoshuabengio.org/2020/03/23/peer-to-peer-ai-tracing-of-covid-19/).

The simulator is built using [`simpy`](!https://simpy.readthedocs.io/en/latest/simpy_intro/index.html).
It simulates human mobility along with infectious disease (COVID) spreading in a city, where city has houses, grocery stores, parks, workplaces, and other non-essential establishments.

Human mobility simulation is based on Spatial-EPR model. More details on this model are [here](https://www.nature.com/articles/ncomms9166) and [here](https://www.nature.com/articles/nphys1760).

The infection spread in this simulator is modeled according to what is known about COVID-19.
The assumptions about the COVID-19 spread and mobility implemented in the simulator are in the [Google Doc](https://docs.google.com/document/d/1jn8dOXgmVRX62Ux-jBSuReayATrzrd5XZS2LJuQ2hLs/edit?usp=sharing).
The same document also details our current understanding of COVID-19.
Our understanding is based on the published research as well as interactions with the epidemiologists.
We plan to update the simulator as more and more about COVID-19 will be known.


## Dependencies
Following `python` packages are required (python>=3.6)
```
pip install -r requirements.txt
```

## How to run it using command line?
Run the mobility simulator as -
```
python run.py sim --n_people 100 --n_stores 100 --n_parks 10 --n_misc 100 --init_percent_sick 0.01 --outfile data --seed 0
```

The mobility simulator will output a logfile to `output/data.pkl`. It is a `list` of `dict` which contains a log of the mobility activity of a population of humans in `mobility_simulator.py`.
It also outputs a serialized version of the state of each `Human` object at the end of the simulation to `output/humans.pkl`.

Run the risk prediction algorithms as -
```
python risk_prediction.py
```
This file reads in the logs and human objects that are output from the mobility simulator, and runs a risk prediction algorithm based on: 
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

### Parameters

```
@click.option('--n_people', help='population of the city', type=int, default=100)
@click.option('--n_stores', help='number of grocery stores in the city', type=int, default=100)
@click.option('--n_parks', help='number of parks in the city', type=int, default=20)
@click.option('--n_misc', help='number of non-essential establishments in the city', type=int, default=100)
@click.option('--init_percent_sick', help='% of population initially sick', type=float, default=0.01)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--outfile', help='filename of the output (file format: .pkl)', type=str, required=False)
@click.option('--print_progress', is_flag=True, help='print the evolution of days', default=False)
@click.option('--seed', help='seed for the process', type=int, default=0)
```

### Accessing Simulation Data
Load the output of the simulator as following
```
data = pickle.load(open("output/data.pkl", 'rb'))
```

## How to run it as a function?
Although not designed with this usage in mind one can still call it like this
```
from run import run_simu
monitors = run_simu(n_stores=100, n_parks=50, n_people=100, n_misc=100, init_percent_sick=0.01, print_progress=False, seed=0)
```

`data` is a `list` of `dict`.

## Base SEIR plots
Following will require `cufflinks` and `plotly`.
```
python run.py base --toy_human
```
It will open a browser window with the plot of [SEIR curves](https://www.idmod.org/docs/hiv/model-seir.html#seir-and-seirs-models).

## Semantics of code
`Human` class builds people, and `Location` class builds stores, parks, workplaces, households, and non-essential establishments.

## Semantics of Data
`data` is a `list`. Each entry in the `list` is an event represented as a `dict`.
The detailed information about events is in [docs/events.md](docs/events.md)

## Contributing
Please get in touch with me at [pgupta@turing.ac.uk](pgupta@turing.ac.uk). There are several people working on it, so it will be the best use of everyone's time and effort if we all work on different aspects of this project.

Some areas that need work are listed [here](docs/CONTRIBUTING.md). We track and manage our tasks using [Google Sheets](https://docs.google.com/spreadsheets/d/11t1T66AAVeR6P341nZYP1qwLdvhCkU_EwFwUkyLziLQ/edit?usp=sharing).

## Collaborators
[@marco-gires](https://github.com/marco-gires), [@marie-pellat](https://github.com/mariepellat), [@teganmaharaj](https://github.com/teganmaharaj), [@giancarlok](https://github.com/giancarlok), [@thechange](https://github.com/thechange), [@soundarya98](https://github.com/soundarya98), [@mweiss17](https://github.com/mweiss17)
