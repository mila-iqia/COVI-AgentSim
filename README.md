# Covid-19 Spread Simulator for Risk Tracing App

This simulator is built using [`simpy`](!https://simpy.readthedocs.io/en/latest/simpy_intro/index.html).
It simulates the spread of Covid-19 in a city block while taking human mobility into account. The "city"
contains houses, workplaces, senior residences, and other non-essential establishments for humans to visit.
The current framework can handle simulations with 1k-10k humans
fairly well, and work is under progress to scale it to the size of real cities.

The simulation is based on age-stratified contact patterns, which are calibrated to yield surveyed data.
More information on agent behavior, interactions, and the transmission model can be found [here](!https://openreview.net/pdf?id=07iDTU-KFK).
The human mobility simulation is based on a Spatial-EPR model. More details on this model are
[here](https://www.nature.com/articles/ncomms9166) and [here](https://www.nature.com/articles/nphys1760).
The infection spread is modeled based on what is currently known about Covid-19. Our understanding is
based on published research as well as working closely with epidemiologists and other experts. We plan
to update and calibrate the simulator as more information about COVID-19 becomes available.


## Installation

For now, we recommend that all users install the simulator's source code as an editable package
in order to properly be able to adjust it to their experimental needs. To do so:
  - Clone the repository to your computer (`git clone https://github.com/mila-iqia/covi-simulator`)
  - Activate your conda/virtualenv environment where the package will be installed
    - Note: the minimal Python verion is 3.7.4!
  - Install the `covid19sim` package (`pip install -e <root_to_covi_simulator>`)
    - Note: this should install all dependencies required for basic (non-ML-enabled) simulations

If you wish to run simulations with the
[ML models for risk inference](https://github.com/mila-iqia/covi-machine-learning), you will have to
install the package for those as well:
  - Clone the repository to your computer (`git clone https://github.com/mila-iqia/covi-machine-learning`)
    - Note: clone it OUTSIDE the `<root_to_covi_simulator>` where the simulator is!
  - Activate the conda/virtualenv environment where the simulator is already installed
  - Install the `ctt` dependencies (`pip install -r <root_to_covi_machine_learning>/requirements-minimal.txt`)
    - The 'minimal' dependencies might not be sufficient to train and export a model, but they should
      be sufficient to use a model for inference in the simulator
  - Install the `ctt` package (`pip install -e <root_to_covi_simulator>`)
    - Note: this should install all dependencies required for basic (non-ML-enabled) simulations


## Running a simulation

To run an "unmitigated" 1k-human simulation for 30 days while logging Covid-19 progression and various
statistics, use:
```
python -m covid19sim.run tune=True intervention=no_intervention n_people=1000 simulation_days=30
```

Note that depending on the size of the simulation and the initial number of infected people, you might see
an explosion in the number of cases, or nothing at all. The latter is possible if the initially infected
human does not go out much, or lives alone in their house. You can try to run the experiment again with
a different seed (it is 0 by default):
```
python -m covid19sim.run tune=True intervention=no_intervention n_people=1000 simulation_days=30 seed=13
```

In any case, the simulator will output some data to the `output` directory in your current folder. In
there, you will find a backup of the configuration used for the simulation (`full_configuration.yaml`),
the simulation log (`log_<current_datetime>.txt`), and a pickle file containing the tracker data
(`tracker_data_n_<nb_people>_seed_<seed>_<current_datetime>.pkl`).

A simulation in an unmitigated scenario will be CPU-bound (and run on a single core due to simpy). With
the default settings (1000 people, 30 days), it should take 3-4 minutes on a modern desktop PC, and require
a maximum of 3GB of RAM (with tracking enabled).

To run a simulation with a risk predictor in the loop, you will have to set the type of intervention
(`bdt1`, `bdt2`, `transformer`, `oracle`,
`heuristicv1`, `heuristicv2`) and the day that intervention should start. For example, to run with the
first version of the heuristic risk prediction, use:
```
python -m covid19sim.run intervention=heuristicv1 INTERVENTION_DAY=5  n_people=1000 simulation_days=30
```

The above commands only run one simulation each. This is useful for debugging, but in order to run
multiple simulations with domain randomization (e.g. to create a training dataset), we make use of
a special config files in (`src/covid19sim/configs/experiment/`, e.g. `app_adoption.yaml`) with a special module
(`covid19sim.job_scripts.experiment.py`) to run over 100 simulations. Note that this takes several
hours on a CPU cluster.

For more information on settings and outputs, we suggest to dig into the code and look at the docstrings.
If you feel lost or are looking into undocumented areas of the code, feel free to contact one of the
developers.


## How to run tests?

From the root of the repository, run:
```
pytest tests/
```

## License

This project is currently distributed under the [Affero GPL (AGPL) license](LICENSE).


## Contributing

If you have an idea to contribute, we suggest that you first sync up with one of the developers working
on the project to minimize potential issues. Then, we will be happy to work on a PR with you.


## About

This simulator is part of an academic research project at Mila. The other useful component of this
project is the machine learning code, located [here](https://github.com/mila-iqia/covi-machine-learning).
