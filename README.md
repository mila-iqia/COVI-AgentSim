# COVI-AgentSim: A testbed for comparing contact tracing apps 

This simulator is an agent-based model (ABM) built in python using [`simpy`](https://simpy.readthedocs.io/en/latest/simpy_intro/index.html).
It simulates the spread of COVID-19 in a population of agents, taking human mobility and indiviual characteristics (e.g. symptoms and pre-existing medical conditions) into account. Agents in the simulation can be assigned one of several types of digital contact tracing (DCT) app, and the ability of each DCT method to control the spread of disease can be compared via cost-benefit analysis.  The level of individual-level detail in our simulator allows for testing a novel type of contact tracing we call Feature-based Contact Tracing (FCT). We provide a simple heuristic baseline FCT method. 

Details of agent behavior, interactions, the transmission model, baselines (including for FCT) and experimental results can be found [here](!https://openreview.net/pdf?id=07iDTU-KFK).

The simulator is modular; you can design, simulate, and benchmark your own DCT method against the baselines provided!
This is the primary intended use of COVI-AgentSim, and the most well-documented. However the simulator can also be used to examine the effects of other types of intervention (e.g. schedules of school or work closure). If you have questions about doing this don't hesitate to contact the developers.

<img src="https://github.com/mila-iqia/COVI-AgentSim/blob/master/notebooks/GP_r_effective_contacts_mobility_scatter_w_annotations_w_scatter_AR_60.jpg" width="80%">


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

## Contributing

If you have an idea to contribute, please open a github issue or contact the developers; we are happy to collaborate and extend this project!


## About

This simulator has been developed as part of a multi-disciplinary project called COVI, aiming to improve and augment existing contact tracing approaches through evidence-based study. This project is headed by Dr. Yoshua Bengio at Mila.

## License

Covi Canada has developed a COVID-19 mobile software application and machine learning simulator to help Canadians change the course of the COVID-19 crisis as they go about their daily lives by providing information to navigate social distancing measures and better understand evolving personal and societal risk factors specific to each user’s context (the “Covi Code”). This effort is led by world-renowned AI researcher Yoshua Bengio at Mila and rallies a coalition of researchers, developers and experts across Canada 

MILA is making the Covi Code available to the public on a non-exclusive, royalty-free basis to enable other interested groups to reuse the work products they have created. The Covi Code will be distributed under the terms of the GNU Affero General Public License, Version 3 (“AGPL Licence”). If a copy of the AGPL was not distributed with your software, you can obtain one at https://www.gnu.org/licenses/agpl-3.0.en.html

The AGPL Licence is a free, copyleft license for software and other kinds of works, specifically designed to ensure cooperation with the community in the case of network server software. It is designed specifically to ensure that any modified source code derived from the Covi Code becomes available to the community. Any software that uses code under an AGPL Licence is itself subject to the same AGPL licensing terms. Furthermore, it requires the operator of a network server to provide the source code of the modified version running there to the users of that server. Therefore, public use of a modified version, on a publicly accessible server, gives the public access to the source code of the modified version.

Please note that in compliance with the AGPL Licence and unless prohibited under applicable: (i) Covi Canada is not making or offering any representation, warranty, guarantee or covenant with respect to the Covi Code and provides it to you "as is", without warranty of any kind, written or oral, express or implied, including any implied warranties of merchantability and fitness for a particular purpose or with respect to its condition, quality, conformity, availability, absence of defects, errors and inaccuracies (or their corrections); and (ii) the entire risk including as to the quality and performance of the Covi Code is with you; should the Covi Code or any portion thereof prove defective, you assume the cost of all necessary servicing, repair or correction; and (iii) modifying and adapting the Covi Code shall be at your own risk and any resulting derivative program or code shall be used at your discretion, in accordance with the AGPL Licence, without any liability to Covi Canada whatsoever.

In addition, and unless prohibited under applicable law, the following additional disclaimer of warranty and limitation of liability is hereby incorporated into the terms and conditions of the AGPL Licence for the Covi Code:

- No representations, covenants, guarantees and/or warranties of any kind whatsoever are made as to the results that you will obtain from relying upon the Covi Code (or any information or content obtained by way of the Covi Code), including but not limited to compliance with privacy laws or regulations or clinical care industry standards and protocols.

- Even if the Covi Code and/or the content could be considered (and/or was obtained using) information about individuals’ health/medical condition, the Covi Code and content are not intended to provide medical advice, and accordingly such data shall not: (i) be used for self-diagnosis; (ii) constitute or be construed as an interpretation of any medical condition; (iii) be considered as giving prognostics, opinions, diagnosis or medical recommendations; or (iv) be considered as a substitute for professional advice. Any decision with regard to the appropriateness of treatment, or the validity or reliability of information or content made available by the Covi Code (or other interpretation of the content for medical or related purposes), shall be made by (or in consultation with) health care professionals. Consequently, it is incumbent upon each health care provider to verify all medical history and treatment plans with each patient.

- Unless prohibited under applicable laws, under no circumstances and under no legal theory, whether tort (including negligence), contract, or otherwise, shall Covi Canada, any user, or anyone who distributes any software programs which incorporate in whole or in part the Covi Code as permitted by the license, be liable to you for any direct, indirect, special, incidental, consequential damages of any character including, without limitation, damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other damages or losses, of any nature whatsoever (direct or otherwise) on account of or associated with the use or inability to use the covered content (including, without limitation, the use of information or content made available by the Covi Code, all documentation associated therewith; the foregoing applies as well to any compliance with privacy laws and regulations and/or clinical care industry standards and protocols), is incumbent upon you only even if Covi Canada (or other entity) have been informed of the possibility of such damages. Some jurisdictions do not allow the exclusion or limitation of liability for direct, consequential, incidental or other damages. In such jurisdiction, Covi Canada’s liability is limited to the greatest extent permitted by law, or to $100, whichever is less, unless the foregoing is prohibited in which case said limitation will be inapplicable in that case (but applicable and enforceable to the fullest extent legally permitted against any person and/or in any other circumstances).

- You understand and agree: (i) that the access, use and other processes of the Covi Code and content shall only be made for lawful purposes (and in no event to attempt re-identifying any individual) and further to your own decision and initiative; (ii) that you are deemed to access, rely on, use and/or otherwise process the content at your own risks and based (a) on the abovementioned essential terms, which apply in addition to (and in case of inconsistencies shall have precedence over) any other terms of the AGPL Licence; and (b) on the fact that such access, reliance, use or other process and/or any dispute/proceeding are governed by the laws in force in the Province of Quebec excluding principles and rules that could lead to the application of foreign laws and subject to the exclusive jurisdictions of the courts of that Province; and (iv) that unless prohibited by laws of public order (in certain circumstances), Covi Canada assumes no responsibility whatsoever and shall not be responsible for any claim/damage including those arising/resulting from one of the occurrences referred to in these essential terms.
