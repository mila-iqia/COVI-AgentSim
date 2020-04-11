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
from plots.validate_risk_parameters import dist_plot


""" This file contains the core of the side simulation, which is run on the output encounters from the main simulation.
It's primary functionality is to run the message clustering and risk prediction algorithms.
"""

if __name__ == "__main__":

    # TODO: put these in args
    PATH_TO_DATA = "data.pkl"
    PATH_TO_HUMANS = "humans.pkl"
    PATH_TO_PLOT = "plots/infected_dist.png"

    # TODO: refactor this so that clustering only happens for risk methods which require message clustering
    DO_CLUSTER = True

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
    for log in tqdm(enc_logs):
        now = log['time']
        unobs = log['payload']['unobserved']
        h1 = unobs['human1']['human_id']
        h2 = unobs['human2']['human_id']
        this_human = hd[h1]
        other_human = hd[h2]
        this_human.env.timestamp = now
        other_human.env.timestamp = now
        this_human.pending_messages.append(other_human.cur_message(now))

        if this_human.cur_day != now.day:
            this_human.cur_day = now.day
            this_human.update_uid()
            for j in range(len(this_human.pending_messages)):
                m_j = this_human.pending_messages.pop()
                if DO_CLUSTER:
                    this_human.handle_message(m_j)
                this_human.update_risk(m_j)
            risk_vs_infected.append((this_human.risk, this_human.is_infectious))

    # plot the resulting
    dist_plot(risk_vs_infected, PATH_TO_PLOT)

    # write out the clusters to be processed by privacy_plots
    if DO_CLUSTER:
        clusters = []
        for human in humans:
            clusters.append(human.A)
        json.dump(clusters, open('clusters.json', 'w'))


