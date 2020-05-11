"""
Samples the viral load models functions and outputs charts showing the course of their
progression
"""
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from matplotlib import pyplot as plt
from utils import _sample_viral_load_gamma, _sample_viral_load_piecewise

VIRAL_LOAD_DIR_PATH = "output/viral_load"
VIRAL_LOAD_PLOT_PATH = os.path.join(VIRAL_LOAD_DIR_PATH, "viral_load.png")

if not os.path.isdir( VIRAL_LOAD_DIR_PATH):
    os.mkdir(VIRAL_LOAD_DIR_PATH)

NUM_DAYS = 30
NUM_PEOPLE = 100
x = np.linspace(1, NUM_DAYS, NUM_DAYS)
rng = np.random.RandomState(1)

def gamma_dist(x, rng, NUM_PEOPLE):
    """
    This function samples the gamma distributed viral_load model

    Args:
        x ([type]): [description]
        rng ([type]): [description]
        NUM_PEOPLE ([type]): [description]

    Returns:
        [type]: [description]
    """
    viral_loads = []
    for i in range(NUM_PEOPLE):
        vals = _sample_viral_load_gamma(rng).pdf(x)
        viral_loads.append(vals)
    viral_loads = np.array(viral_loads)
    return viral_loads

def piecewise_linear(x, rng, NUM_PEOPLE):
    """
    This function samples the piecewise linear viral_load model

    Args:
        x ([type]): [description]
        rng (np.random.RandomState): random number generator
        NUM_PEOPLE (int): [description]

    Returns:
        np.array: [description]
    """
    viral_loads = []

    for person in range(NUM_PEOPLE):
        plateau_height, plateau_start, plateau_end, recovered = _sample_viral_load_piecewise(rng)
        viral_load = []
        for time_sample in x:
            if time_sample < plateau_start:
                cur_viral_load = plateau_height * time_sample / plateau_start
            elif time_sample < plateau_end:
                cur_viral_load = plateau_height
            else:
                cur_viral_load = plateau_height - plateau_height * (time_sample - plateau_end) / (recovered - plateau_end)
            if cur_viral_load < 0:
                cur_viral_load = np.array([0.])

            viral_load.append(cur_viral_load)

        viral_loads.append(np.array(viral_load, dtype=float).flatten())
    viral_loads = np.array(viral_loads)
    return viral_loads


if __name__ == "__main__":
    # Sample the models
    viral_loads_gamma = gamma_dist(x, rng, NUM_PEOPLE)
    viral_loads_piecewise = piecewise_linear(x, rng, NUM_PEOPLE)

    # Plot the results
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(x, viral_loads_gamma.mean(axis=0), yerr=viral_loads_gamma.std(axis=0), lw=2, label='noisy gamma viral load model')
    old_viral_dist = np.zeros(NUM_DAYS)
    old_viral_dist[:9] =[0.05, 0.1, 0.2, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    ax.plot(x, old_viral_dist, lw=2, label='old viral load dist')
    ax.errorbar(x, viral_loads_piecewise.mean(axis=0), yerr=viral_loads_piecewise.std(axis=0), lw=2, label='noisy piecewise viral load model')
    plt.legend()
    plt.xlabel("Days since infection")
    plt.ylabel("Viral load")
    plt.title("Viral Load")
    plt.savefig(VIRAL_LOAD_PLOT_PATH)


    # Here, we sample the piecewise linear model for 10 people, and plot them as individuals
    x = np.linspace(1, NUM_DAYS, 10 * NUM_DAYS)
    num_people = 10
    viral_loads_piecewise = piecewise_linear(x, rng, 10)

    # Plot the individuals
    fig, ax = plt.subplots(1, 1)
    for i in range(viral_loads_piecewise.shape[0]):
        ax.plot(x, viral_loads_piecewise[i], lw=2, label='noisy piecewise viral load model')
    plt.xlabel("Days since infection")
    plt.ylabel("Viral load")
    plt.title("Viral Load (individuals)")
    plt.savefig(os.path.join(VIRAL_LOAD_DIR_PATH, "viral_load_individuals.png"))

