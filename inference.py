import pickle
import json
from base import Event
import numpy as np
from config import *
from utils import binary_to_float
import matplotlib.pyplot as plt
from utils import _encode_message, _decode_message
from bitarray import bitarray
import operator
from collections import defaultdict
import datetime
from tqdm import tqdm
from plots.validate_risk_parameters import dist_plot, hist_plot


""" This file contains the core of the side simulation, which is run on the output encounters from the main simulation.
It's primary functionality is to run the message clustering and risk prediction algorithms.
"""


def risk_for_symptoms(human):
    """ This function calculates a risk score based on the person's symptoms."""
    # if they have a positive test, they have a risk of 1.
    if human.test_results == 'positive':
        return 1.

    symptoms = human.reported_symptoms_for_sickness()
    if 'severe' in symptoms:
        return 0.75
    if 'moderate' in symptoms:
        return 0.5
    if 'mild' in symptoms:
        return 0.25
    return 0.


if __name__ == "__main__":

    # TODO: add as args that can be called from cmdline
    PLOT_DAILY = True
    PATH_TO_DATA = "data.pkl"
    PATH_TO_HUMANS = "humans.pkl"
    RISK_MODEL = 'yoshua'  # options: ['yoshua', 'lenka', 'eilif']
    METHOD_CLUSTERING_MAP = {"eilif": True, "yoshua": False, "lenka": False}
    LOGS_SUBSET_SIZE = 10000000

    # read and filter the pickles
    with open(PATH_TO_DATA, "rb") as f:
        logs = pickle.load(f)
    with open(PATH_TO_HUMANS, "rb") as f:
        humans = pickle.load(f)
    enc_logs = [l for l in logs if l["event_type"] == Event.encounter]

    # A hack for using the re-inflated human objects
    class dummy_env:
        def __init__(self):
            return

    # re-inflate the humans
    hd = {}
    for human in humans:
        try:
            human.infection_timestamp = datetime.datetime.strptime(human.infection_timestamp, '%Y-%m-%d %H:%M:%S')
        except Exception:
            human.infection_timestamp = None
        try:
            human.recovered_timestamp = datetime.datetime.strptime(human.recovered_timestamp, '%Y-%m-%d %H:%M:%S')
        except Exception:
            human.recovered_timestamp = None

        env = dummy_env()
        env.timestamp = datetime.datetime(2020, 2, 28, 0, 0)
        human.env = env
        human.rng = np.random.RandomState(0)
        human.update_uid()
        hd[human.name] = human

    # TODO: add a way to process only a fraction of the log files (some methods can be slow)
    risks = []
    risk_vs_infected = []

    # Sort by time and then by human id
    enc_logs = sorted(enc_logs, key=operator.itemgetter('time', 'human_id'))
    enc_logs = sorted(enc_logs, key=lambda k: (k['time'], k['human_id']))
    enc_logs = sorted(enc_logs, key=lambda k: (k['time'], -k['human_id']))

    log_idx = 0
    plot_day = enc_logs[0]['time'].day
    plot_num = 0
    for log in tqdm(enc_logs):
        now = log['time']
        unobs = log['payload']['unobserved']
        h1 = unobs['human1']['human_id']
        h2 = unobs['human2']['human_id']
        this_human = hd[h1]
        other_human = hd[h2]
        this_human.env.timestamp = now
        other_human.env.timestamp = now

        # if it's a new day in this log, send this log
        if this_human.cur_day != now.day:
            this_human.cur_day = now.day
            this_human.update_uid()
            cur_risk = this_human.risk
            # if the person's infected, look at their symptoms once per day and calculate a risk
            # TODO: model false report of symptoms
            if this_human.infection_timestamp and (this_human.env.timestamp - this_human.infection_timestamp).days >= 0:
                # if (this_human.env.timestamp - this_human.infection_timestamp).days > 6:
                #     import pdb; pdb.set_trace()
                this_human.risk = risk_for_symptoms(this_human)
            # update risk based on that day's messages
            for j in range(len(this_human.pending_messages)):
                m_j = this_human.pending_messages.pop()
                if METHOD_CLUSTERING_MAP[RISK_MODEL]:
                    this_human.handle_message(m_j)
                this_human.update_risk_encounter(m_j, RISK_MODEL)
            risk_vs_infected.append((this_human.risk, this_human.is_infectious))
            if PLOT_DAILY and plot_day != now.day:
                plot_num += 1
                plot_day = now.day
                # plot the resulting
                hist_plot(risk_vs_infected, f"plots/infected_dist/infected_dist_day_{str(plot_num).zfill(3)}.png")
                risk_vs_infected = []
            # TODO: if risk changed substantially, send update messages for all of my messages in a rolling 14 day window
            # if cur_risk != this_human.risk:
            #     print("should be sending risk update messages")
                # import pdb; pdb.set_trace()

        this_human.pending_messages.append(other_human.cur_message(now))

        # sometimes we only want to read a subset of the logs, for development
        log_idx += 1
        if log_idx > LOGS_SUBSET_SIZE:
            break

    # write out the clusters to be processed by privacy_plots
    if METHOD_CLUSTERING_MAP[RISK_MODEL]:
        clusters = []
        for human in humans:
            clusters.append(human.M)
        json.dump(clusters, open('clusters.json', 'w'))


