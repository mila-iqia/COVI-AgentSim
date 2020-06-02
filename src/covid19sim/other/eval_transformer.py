from pathlib import Path
import argparse
import pickle
import numpy as np
from collections import defaultdict
import datetime

def compute_early_warning(data) -> dict:
    """
    Computes metric to quantify how early signal is reached to the user who is infected during the simulation.
    """
    infection_timestamps = {}
    for x in data['infection_monitor']:
        infection_timestamps[x['to']] = x['infection_timestamp']

    early_warning, early_warning_infected = {}, {}
    risk_attributes = sorted(data['risk_attributes'], lambda x:x['timestamp'])
    for attr in risk_attributes:
        name = attr['name']
        rec_level = attr['rec_level']
        infected = attr['is_infectious'] or attr['is_exposed']
        infection_timestamp = infection_timestamps.get(name, None)
        symptoms = attr['symptoms']

        if infection_timestamp is None:
            continue

        timestamp = attr['timestamp']
        if rec_level != 0 and name not in early_warning:
            early_warning[name] = (timestamp - infection_timestamp).total_seconds() / 86400

        if infected and rec_level != 0 and name not in early_warning_infected:
            early_warning_infected[name] = (timestamp - infection_timestamp).total_seconds() / 86400

    return {
                'early_warning_all_mean': np.mean(early_warning.values()),
                'early_warning_all_median': np.median(early_warning.values()),
                'early_warning_infected_mean': np.mean(early_warning_infected.values()),
                'early_warning_infected_median': np.median(early_warning_infected.values())
            }


def compute_true_false_quarantine(data) -> tuple:
    """
    Calculates number of people correctly (incorrectly) identified for quarantining
    """
    true_quarantine, false_quarantine = 0, 0
    for attr in data['risk_attributes']:
        infected = attr['is_infectious'] or attr['is_exposed']
        rec_level = attr['rec_level']
        if rec_level == 3:
            if infected:
                true_quarantine += 1
            else:
                false_quarantine += 1
    return true_quarantine, false_quarantine


def compute_presymptomatic_warning_signals(data) -> float:
    """
    Of all the infected people, how many got a signal (rec_level > 0) before they became symptomatic or infectious but after they were exposed
    """
    infection_timestamps = {}
    for x in data['infection_monitor']:
        infection_timestamps[x['to']] = x['infection_timestamp']

    early_warning, early_warning_infected = {}, {}
    human_risk_attributes = defaultdict(list)
    risk_attributes = sorted(data['risk_attributes'], lambda x:x['timestamp'])
    for attr in risk_attributes:
        human_risk_attributes[attr['name']].append(attr)

    presymp, preinf = 0, 0
    for human, infection_timestamp in infection_timestamps.items():
        ras = human_risk_attributes[human]
        t_s, t_rec, t_inf = None, None, None
        for x in ras:
            if x['symptoms'] > 0 and t_s is None:
                t_s = x['timestamp']

            if x['rec_level'] != 0 and t_rec is None:
                t_rec = x['timestamp']

            if x['infectious'] and t_inf is None:
                t_inf = x['timestamp']

            if (
                t_s is not None
                and t_rec is not None
                and t_inf is not None
            ):
                break
        if t_s is None:
            t_s = datetime.datetime.max

        assert t_inf is not None, "Infection timestamp is not None but is_infectious is None"
        presymp += infection_timestamp <= t_rec <= t_s
        preinf += infection_timestamp <= t_rec <= t_inf

    total = len(infection_timetamps)
    return presymp, preinf, total


def get_metrics(data) -> dict:
    # early warning: how much time after exposure put in non-green bucket
    early_warning_per_rec_level: dict = compute_early_warning(data)
    # red but not even exposed (not infected nor infectious) and red and truely infectious
    true_quarantine, false_quarantine = compute_true_false_quarantine(data)
    # was the rec_level increased between exposure and infectiousness (good) or not (bad)
    exposed_to_infectious_interval = compute_exposed_to_infectious_interval(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type="str", default=".", help="experimental directory")
    opts = parser.parse_args()

    path = Path(opts.path).resolve()
    model = path.parent.name

    runs = [d for d in path.iterdir() if d.is_dir()]

    for run in runs:
        data_path = list(run.glob("tracker*.pkl"))[0]
        with data_path.open("rb") as f:
            data = pickle.load(f)
            metrics: dict = get_metrics(data)
