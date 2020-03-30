There are several ways to involve in the project.

## Brief overview
The simulator is designed with the aim of generating synthetic data to train ML models.
Specifically, the assumption is that the `Human`s carry bluetooth device with an app that produces data after each event.
Read more about the events [here](events.md).
These devices can communicate restricted information with each other.
The data collected from these devices is use to train ML models.

We have designed a state-based simulator, where the state is defined by location of `Human`s and the information related to them.
A step of simulator involves actions taken by `Human`s and the resulting change in their states.
It will be useful to get yourself familiar with [`sympy`](https://simpy.readthedocs.io/en/latest/)

There are two main  components in the simulator (not independent):
1. Human Mobility - imitate mobility pattern of people in the city
2. COVID spread - imitate the spread of COVID infections among the population

## How to get involved?

There are several ways to improve this simulator -

1. **Implementing sanity checks and tests** - As we keep implementing our assumptions in the simulator, we want to make sure that it is realistic and speaks to the data that has been observed in the past.
Some ways to do this are -
   - Although we are implementing the assumptions in our simulator, we want to make sure that the disease spread follows the [mathematical models of infectious disease](https://en.wikipedia.org/wiki/Mathematical_modelling_of_infectious_disease) all the time. This can be done by using a dynamical models of epidemic like [SEIRs](https://github.com/ryansmcgee/seirsplus). Thus, fitting the COVID data to above mathematical model and cross-checking the simulators performance will be the most useful thing as of now. 
   - Using the simulator to analyze metrics like time spent in house, average number of trips, or reproducibility number(R) of COVID, just to mention some. If the distribution of these metrics is supported by research, it will help us in validating the simulator.
   - Animation of different scenarios using the simulator to help us communicate the assumptions in the simulator as well as understand the shortcomings of it.

2. **Improving the mobility simulator** - We have current list of desired behaviors in [docs/mobility_tasks.md](mobility_tasks.md). Implementing them will be of great help.

3. **Improving the COVID-spread simulator** - We keep the documentation related to our understanding of the COVID disease in document [here](https://docs.google.com/document/d/1jn8dOXgmVRX62Ux-jBSuReayATrzrd5XZS2LJuQ2hLs/edit?usp=sharing). The same document also contains the assumptions that we have built into our simulator. There are many assumptions that we haven't yet implemented. Thus, implementing them in our simulator will be awesome!

4. **Improving features for ML** - Since the simulator is aimed at doing better ML we are constantly looking for useful information that can be passed from the simualtor (or user's app) in the form of `events`. Please read more about events [here](events.md).

5. **Organization of the project** - We are not software engineers. Most of us have machine learning research background, so the organization of the project might not be best aligned for collaboration. Please feel free to reach out/commit changes to make it better.

## Guidelines for contributing

You are very welcome to contribute. Please [follow the guidelines here for contributing](https://gist.github.com/MarcDiethelm/7303312).
