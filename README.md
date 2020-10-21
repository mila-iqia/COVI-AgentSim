# COVI-AgentSim: A testbed for comparing contact tracing apps 

This simulator is an agent-based model (ABM) built in python using [`simpy`](https://simpy.readthedocs.io/en/latest/simpy_intro/index.html).
It simulates the spread of COVID-19 in a population of agents, taking human mobility and indiviual characteristics (e.g. symptoms and pre-existing medical conditions) into account. Agents in the simulation can be assigned one of several types of digital contact tracing (DCT) app, and the ability of each DCT method to control the spread of disease can be compared via cost-benefit analysis.  The level of individual-level detail in our simulator allows for testing a novel type of contact tracing we call Feature-based Contact Tracing (FCT). We provide a simple heuristic baseline FCT method. 

Details of agent behavior, interactions, the transmission model, baselines (including for FCT) and experimental results can be found [here](!https://openreview.net/pdf?id=07iDTU-KFK).

The simulator is modular; you can design, simulate, and benchmark your own DCT method against the baselines provided!
This is the primary intended use of COVI-AgentSim, and the most well-documented. However the simulator can also be used to examine the effects of other types of intervention (e.g. schedules of school or work closure). If you have questions about doing this don't hesitate to contact the developers.

<img src="https://gcovi-simulatorithub.com/mila-iqia/COVI-AgentSim/blob/master/notebooks/GP_r_effective_contacts_mobility_scatter_w_annotations_w_scatter_AR_60.jpg" width="80%">


## Installation

For now, we recommend that all users install the simulator's source code as an editable package
in order to properly be able to adjust it to their experimental needs. To do so:
  - Clone the repository to your computer (`git clone https://github.com/mila-iqia/COVI-AgentSim`)
  - Activate your conda/virtualenv environment where the package will be installed
    - Note: the minimal Python version is 3.7.4!
  - Install the `covid19sim` package (`pip install -e <root_to_covi_simulator>`)


## Running a simulation

To run an "unmitigated" 1k-human simulation for 30 days while logging COVID-19 progression and various
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

To run a simulation with some agents assigned digital contact tracing apps, set the type of intervention to one of the baseline DCT methods we provide: 
(`bdt1`, `bdt2`, `transformer`, `oracle`,
`heuristicv1`, `heuristicv2`) (see [here](!https://openreview.net/pdf?id=07iDTU-KFK) for details on these methods), set the day that intervention should start, and optionally set the app uptake (proportion of smartphone users with the app). For example, to run with the
first version of the heuristic risk prediction, with 40% of the population having the app, use:
```
python -m covid19sim.run intervention=heuristicv1 INTERVENTION_DAY=5  n_people=1000 simulation_days=30 APP_UPTAKE=.5618
```

Note on app adoption:

We model app adoption according to statistics of smartphone usage. The left column is the % of total population with the app, and right column is the uptake by smartphone users.
Percentage of population with app | Uptake required to get that percentage
--- | ---
~1 | ~1.50 
30 | 42.15 
40 | 56.18 
60 | 84.15 
70 | 98.31 

<img src="https://github.com/mila-iqia/COVI-AgentSim/blob/master/notebooks/epi-adoption.png" width="80%">

## Replicating experiments in the paper 
The above commands only run one simulation each. This is useful for debugging, but in order to run
multiple simulations at once (e.g. to average over multiple random seeds),  we make use of
a special config files in (`src/covid19sim/configs/experiment/`, e.g. `app_adoption.yaml`) with a special module
(`covid19sim.job_scripts.experiment.py`) to run over 100 simulations. Note that this takes several
hours on a CPU cluster.

For example, to run the app adoption experiment, use:

```
python experiment.py exp_file=app_adoption base_dir=/your/folder/path/followed_by/output_folder_name track=light env_name=your_env
```

For more examples and details, see [job scripts readme](https://github.com/mila-iqia/COVI-AgentSim/tree/master/src/covid19sim/job_scripts).

To plot the resulting data, use the appropriate notebook, located in `/notebooks`, or use plotting/main.py --help to see the list of available options.



## Running tests

From the root of the repository, run:
```
pytest tests/
```

## License

This project is currently distributed under the [Affero GPL (AGPL) license](LICENSE).


## Contributing

If you have an idea to contribute, please open a github issue or contact the developers; we are happy to collaborate and extend this project!


## About

This simulator has been developed as part of a multi-disciplinary project called COVI, aiming to improve and augment existing contact tracing approaches through evidence-based study. This project is headed by Dr. Yoshua Bengio at Mila.
