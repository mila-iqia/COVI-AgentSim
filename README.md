# COVID-19 Spread Simulator

Simulator is built using [simpy](!https://simpy.readthedocs.io/en/latest/simpy_intro/index.html).
It simulates human mobility along with infectious disease spread in a city, where city has houses, grocery stores, parks, workplaces, and other non-essential establishments.
Human mobility simulation is based on Spatial-EPR model. More details on this model are [here](https://www.nature.com/articles/ncomms9166) and [here](https://www.nature.com/articles/nphys1760).
The infection spread in this simulator is modeled according to what is know about COVID-19.
The assumptions about the spread implemented in the simulator are in [docs/inf_spread_simulator_assumptions.md](docs/inf_spread_simulator_assumptions.md)
We plan to update the simulator as more about COVID-19 will be known.

Our understanding of COVID-19 spread is documented in [docs/inf_spread_known.md](docs/inf_spread_known.md).

## Dependencies
Following `python` packages are required (python>=3.6)
```
simpy
scipy
numpy
```

## How to run it using command line?
Run the simulator as -
```
python simulate.py --n_people 100 --n_stores 100 --n_parks 10 --n_miscs 100 --init_percent_sick 0.01 --outfile data
```

Output will be in `data.pkl`. It is a `list` of `dict`.

### Parameters

```
parser.add_argument( '--n_people', help='population of the city', type=int, default=1000)
parser.add_argument( '--n_stores', help='number of grocery stores in the city', type=int, default=100)
parser.add_argument( '--n_parks', help='number of parks in the city', type=int, default=20)
parser.add_argument( '--n_miscs', help='number of non-essential establishments in the city', type=int, default=100)
parser.add_argument( '--init_percent_sick', help='% of population initially sick', type=float, default=0.01)
parser.add_argument( '--outfile', help='filename of the output (file format: .pkl)', type=str, default="data")
parser.add_argument( '--print_progress', help='print the evolution of days', action='store_tree')
```

### Accessing Simulation Data
Load the output of the simulator as following
```
data = pickle.load(open("data.pkl", 'rb'))
```

## How to run it as a function?
Although not designed with this usage in mind one can still call it like this
```
from simulate import sim
data = sim(n_stores=100, n_parks=50, n_people=100, n_misc=100, init_percent_sick=0.01, print_progress=False)
```

`data` is a `list` of `dict`.

## Semantics of code
`Human` class builds people, and `Location` class builds stores, parks, workplaces, households, and non-essential establishments.

## Semantics of Data
`data` is a `list`. Each entry in the `list` is an event represented as `dict`.

Following events are reported currently -
1. `encounter` - When two `Human`s are present at the same `Location` at the same time, this event is recorded for both the `Human`s.

```
{
 "human_id": 0,
 "time": 240.0,
 "event_type": "encounter",
 "payload": {
  "encounter_human_id": 70,
  "duration": 8.0,
  "distance": 627,
  "lat": 691,
  "lon": 930
 }
}
```
2. `symptom_start` -
3. `contamination` -
4. `test` -

## To Do
