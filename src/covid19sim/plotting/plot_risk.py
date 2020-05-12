"""
[summary]
"""

import os
import sys
sys.path.append(os.getcwd())
from config import RISK_TRANSMISSION_PROBA
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
""" This file plots the predicted risk for infected and uninfected people at one snapshot in time"""

matplotlib.use('Agg')

def hist_plot(risks, path_to_plot):
    """
    [summary]

    Args:
        risks ([type]): [description]
        path_to_plot ([type]): [description]
    """
    plt.figure()
    uninfectious_risks = []
    infectious_risks = []
    for risk, is_infectious, name in risks:
        if is_infectious:
            infectious_risks.append(risk)
        else:
            uninfectious_risks.append(risk)

    print(f"num uninfected: {len(uninfectious_risks)}, mean risk:  {np.mean(uninfectious_risks)}")
    print(f"num infectious: {len(infectious_risks)}, mean risk {np.mean(infectious_risks)}")

    if any(infectious_risks):
        plt.hist(
            infectious_risks,
            density=True,
            label="infectious",
            bins=10,
            alpha=0.7,
        )
    if any(uninfectious_risks):
        plt.hist(
            uninfectious_risks,
            density=True,
            label="not infectious",
            bins=10,
            alpha=0.7,
        )
    plt.xlim(left=-0.1, right=1.1)
    plt.ylim(bottom=-0.1, top=30)
    plt.xlabel("Risk")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Hist of day {path_to_plot}, Risk Transmission Proba = {RISK_TRANSMISSION_PROBA}")
    plt.savefig(path_to_plot)
    plt.close()


def dist_plot(risks, path_to_plot):
    """
    [summary]

    Args:
        risks ([type]): [description]
        path_to_plot ([type]): [description]
    """
    plt.figure()
    uninfectious_risks = []
    infectious_risks = []
    for risk, is_infectious, name in risks:
        if is_infectious:
            infectious_risks.append(risk)
        else:
            uninfectious_risks.append(risk)

    print(f"num uninfected: {len(uninfectious_risks)}, mean risk:  {np.mean(uninfectious_risks)}")
    print(f"num infectious: {len(infectious_risks)}, mean risk {np.mean(infectious_risks)}")

    sns.distplot(
        infectious_risks,
        kde=False,
        axlabel="infectious",
        hist=True,
    )
    sns.distplot(
        uninfectious_risks,
        kde=False,
        axlabel="not infectious",
        hist=True,
    )
    plt.xlabel("Risk")
    plt.ylabel("Number of risk readings")
    plt.legend(['infectious', 'not infectious'])
    plt.title(f"Risk Transmission Proba = {RISK_TRANSMISSION_PROBA}")
    print("Saving figure...")
    plt.savefig(path_to_plot)
    plt.close()
