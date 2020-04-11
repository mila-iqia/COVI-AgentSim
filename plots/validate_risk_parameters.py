import pickle
from base import Event
import config as cfg
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def hist_plot(risk_vs_infected, PATH_TO_PLOT):
    plt.figure()
    plt.hist(
        [risk for risk, infected in risk_vs_infected if infected],
        density=True,
        label="infected",
        bins=20,
        alpha=0.7,
    )
    plt.hist(
        [risk for risk, infected in risk_vs_infected if not infected],
        density=True,
        label="not infected",
        bins=20,
        alpha=0.7,
    )

    plt.xlabel("Risk")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Risk Transmission Proba = {cfg.RISK_TRANSMISSION_PROBA}")
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


if __name__ == "__main__":
    PATH_TO_DATA = "../data.pkl"
    PATH_TO_PLOT = "./infected_dist.png"

    with open(PATH_TO_DATA, "rb") as f:
        logs = pickle.load(f)
    enc_logs = [l for l in logs if l["event_type"] == Event.encounter]
    risk_vs_infected = [
        (
            l["payload"]["unobserved"]["risk"],
            l["payload"]["unobserved"]["human1"]["is_infected"],
        )
        for l in enc_logs
    ]

    dist_plot(risk_vs_infected)
