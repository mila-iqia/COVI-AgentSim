# Contributing
There are several ways to involve in the project.

## Brief overview
The simulator is designed with the aim of generating synthetic data to train ML models.
Specifically, the assumption is that the `Human`s carry bluetooth device with an app that produces data after each event.
These devices can communicate restricted information with each other after a delay of about 1 day (for privacy reasons).
The data collected from these devices is used to train ML models.

We have designed a state-based simulator, where the state is defined by location of `Human`s and the information related to them.
A step of simulator involves actions taken by `Human`s and the resulting change in their states.
It will be useful to get yourself familiar with [`sympy`](https://simpy.readthedocs.io/en/latest/)

There are two main  components in the simulator:
1. Human Mobility - imitate mobility pattern of people in the city
2. COVID spread - imitate the spread of COVID infections among the population

## How to get involved?

There are several ways to improve this simulator - you are welcome to reach out and ask for specific ways to help

1. **Implementing tests** - As we keep implementing our assumptions in the simulator, we want to continue to ensure that it is realistic, with few bugs, and speaks to the published literature about Covid-19.
Some ways to do this are -
   - Although we are implementing the assumptions in our simulator, we want to make sure that the disease spread follows the [mathematical models of infectious disease](https://en.wikipedia.org/wiki/Mathematical_modelling_of_infectious_disease) all the time. This can be done by using a dynamical models of epidemic like [SEIRs](https://github.com/ryansmcgee/seirsplus). Thus, fitting the COVID data to above mathematical model and cross-checking the simulators performance will be the most useful thing as of now. 
   - Using the simulator to analyze metrics like time spent in house, average number of trips, or reproducibility number(R) of COVID, just to mention some. If the distribution of these metrics is supported by research, it will help us in validating the simulator.
   - Visualizing different scenarios using the simulator to help us communicate the assumptions in the simulator as well as understand the shortcomings of it.

2. **Improving the mobility model** - We are improving the model of mobility and increase the efficiency of the simulator. 

3. **Organization of the project** - The organization of the project can be improved! Please feel free to reach out/commit changes to make it better.

## Guidelines for contributing

You are very welcome to contribute. Please [follow the guidelines here for contributing](https://gist.github.com/MarcDiethelm/7303312).
