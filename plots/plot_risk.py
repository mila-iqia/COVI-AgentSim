import pickle
from base import Event
import config as cfg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
""" This file plots the predicted risk for infected and uninfected people at one snapshot in time"""

sns.set()


def hist_plot(risks, PATH_TO_PLOT):
    plt.figure()
    infectious_unquarantined_risks = [risk for risk, is_infectious, is_infected, is_quarantined, name in risks if (is_infectious and not is_quarantined)]
    print(f"infectious_risks: {np.mean(infectious_unquarantined_risks)}")
    quarantined_risks = [risk for risk, is_infectious, is_infected, is_quarantined, name in risks if is_quarantined]
    print(f"is_quarantined risks: {np.mean(quarantined_risks)}")
    uninfectious_risks = [risk for risk, is_infectious, is_infected, is_quarantined, name in risks if not is_infected]
    print(f"uninfected_risks: {np.mean(uninfectious_risks)}")
    infected_risks = [risk for risk, is_infectious, is_infected, is_quarantined, name in risks if is_infected]
    print(f"infected_risks: {np.mean(uninfectious_risks)}")

    if any(infected_risks):
        plt.hist(
            infected_risks,
            density=True,
            label="Risk-levels of anyone who is infected",
            bins=10,
            alpha=0.7,
        )
    if any(infectious_unquarantined_risks):
        plt.hist(
            infectious_unquarantined_risks,
            density=True,
            label="Risk-levels of anyone who is infected and not quarantined",
            bins=10,
            alpha=0.7,
        )
    if any(quarantined_risks):
        plt.hist(
            quarantined_risks,
            density=True,
            label="Risk-levels of anyone who is quarantined",
            bins=10,
            alpha=0.7,
        )
    if any(uninfectious_risks):
        plt.hist(
            uninfectious_risks,
            density=True,
            label="Risk-levels of anyone who is not infected",
            bins=10,
            alpha=0.7,
        )
    plt.xlim(left=-0.1, right=1.1)
    plt.ylim(bottom=-0.1, top=30)
    plt.xlabel("Risk")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Hist of day {PATH_TO_PLOT}, Risk Transmission Proba = {cfg.RISK_TRANSMISSION_PROBA}")
    plt.savefig(PATH_TO_PLOT)
    plt.close()


def dist_plot(risk_vs_infected, PATH_TO_PLOT):
    plt.figure()
    sns.distplot(
        [risk for risk, infected in risk_vs_infected if infected],
        kde=False,
        axlabel="infected",
        hist=True,
    )
    sns.distplot(
        [risk for risk, infected in risk_vs_infected if not infected],
        kde=False,
        axlabel="not infected",
        hist=True,
    )
    plt.xlabel("Risk")
    plt.ylabel("Density")
    plt.legend(['infected', 'not infected'])
    plt.title(f"Risk Transmission Proba = {cfg.RISK_TRANSMISSION_PROBA}")
    print("Saving figure...")
    plt.savefig(PATH_TO_PLOT)
    plt.close()
