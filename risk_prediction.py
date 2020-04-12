import pickle
import json
from base import Event
import subprocess
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
from plots.plot_risk import dist_plot, hist_plot


""" This file contains the core of the side simulation, which is run on the output encounters from the main simulation.
It's primary functionality is to run the message clustering and risk prediction algorithms.
"""


def risk_for_symptoms(human):
    """ This function calculates a risk score based on the person's symptoms."""
    # if they get tested, it takes TEST_DAYS to get the result, and they are quarantined for QUARANTINE_DAYS.
    # The test_timestamp is set to datetime.min, unless they get a positive test result.
    # Basically, once they know they have a positive test result, they have a risk of 1 until after quarantine days.
    # if human.is_infectious:
    #     import pdb; pdb.set_trace()
    if human.name == 2 and datetime.datetime(2020, 3, 15, 0, 0) < human.env.timestamp:
        import pdb; pdb.set_trace()
    if human.is_quarantined:
        return 1.

    symptoms = human.reported_symptoms_for_sickness()
    if 'severe' in symptoms:
        return 0.75
    if 'moderate' in symptoms:
        return 0.5
    if 'mild' in symptoms:
        return 0.25
    return 0.0


def add_message_to_cluster(human, m_i):
    """ This function clusters new messages by scoring them against old messages in a sort of naive nearest neighbors approach"""
    # TODO: include risk level in clustering, currently only uses quantized uid
    # TODO: refactor to compare multiple clustering schemes
    # TODO: check for mutually exclusive messages in order to break up a group and re-run nearest neighbors
    m_i_enc = _encode_message(m_i)
    m_risk = binary_to_float("".join([str(x) for x in np.array(m_i[1].tolist()).astype(int)]), 0, 4)

    # otherwise score against previous messages
    scores = {}
    for m_enc, _ in human.M.items():
        m = _decode_message(m_enc)
        if m_i[0] == m[0] and m_i[2].day == m[2].day:
            scores[m_enc] = 3
        elif m_i[0][:3] == m[0][:3] and m_i[2].day - 1 == m[2].day:
            scores[m_enc] = 2
        elif m_i[0][:2] == m[0][:2] and m_i[2].day - 2 == m[2].day:
            scores[m_enc] = 1
        elif m_i[0][:1] == m[0][:1] and m_i[2].day - 2 == m[2].day:
            scores[m_enc] = 0

    if scores:
        max_score_message = max(scores.items(), key=operator.itemgetter(1))[0]
        human.M[m_i_enc] = {'assignment': human.M[max_score_message]['assignment'], 'previous_risk': m_risk, 'carry_over_transmission_proba': RISK_TRANSMISSION_PROBA}
    # if it's either the first message
    elif len(human.M) == 0:
        human.M[m_i_enc] = {'assignment': 0, 'previous_risk': m_risk, 'carry_over_transmission_proba': RISK_TRANSMISSION_PROBA}
    # if there was no nearby neighbor
    else:
        new_group = max([v['assignment'] for k, v in human.M.items()]) + 1
        human.M[m_i_enc] = {'assignment': new_group, 'previous_risk': m_risk, 'carry_over_transmission_proba': RISK_TRANSMISSION_PROBA}


def update_risk_encounter(this_human, message, RISK_MODEL):
    """ This function updates an individual's risk based on the receipt of a new message"""

    # if the person has recovered, their risk is 0
    # TODO: This leaks information. We should not set their risk to zero just because their symptoms went away and they have "recovered".
    if this_human.recovered_timestamp and (this_human.env.timestamp - this_human.recovered_timestamp).days >= 0:
        this_human.risk = 0

    # Get the binarized contact risk
    m_risk = binary_to_float("".join([str(x) for x in np.array(message[1].tolist()).astype(int)]), 0, 4)

    # select your contact risk prediction model
    update = 0
    if RISK_MODEL == 'yoshua':
        if this_human.risk < m_risk:
            pass
        #     update = (m_risk - m_risk * this_human.risk) * RISK_TRANSMISSION_PROBA
    elif RISK_MODEL == 'lenka':
        update = m_risk * RISK_TRANSMISSION_PROBA

    elif RISK_MODEL == 'eilif':
        msg_enc = _encode_message(message)
        if msg_enc not in this_human.M:
            # update is delta_risk
            update = m_risk * RISK_TRANSMISSION_PROBA
        else:
            previous_risk = this_human.M[msg_enc]['previous_risk']
            carry_over_transmission_proba = this_human.M[msg_enc]['carry_over_transmission_proba']
            update = ((m_risk - previous_risk) * RISK_TRANSMISSION_PROBA + previous_risk * carry_over_transmission_proba)

        # Update contact history
        this_human.M[msg_enc]['previous_risk'] = m_risk
        this_human.M[msg_enc]['carry_over_transmission_proba'] = RISK_TRANSMISSION_PROBA * (1 - update)

    # this_human.risk += update

    # some models require us to clip the risk at one (like 'lenka' and 'eilif')
    if CLIP_RISK:
        this_human.risk = min(this_human.risk, 1.)


if __name__ == "__main__":

    # TODO: add as args that can be called from cmdline
    PLOT_DAILY = True
    PATH_TO_DATA = "output/data.pkl"
    PATH_TO_HUMANS = "output/humans.pkl"
    CLUSTER_PATH = "output/clusters.json"
    PATH_TO_PLOT = "plots/risk/"
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
        try:
            human.test_timestamp = datetime.datetime.strptime(human.test_timestamp, '%Y-%m-%d %H:%M:%S')
            print(f"{human.name}: {human.test_timestamp}")
        except Exception:
            human.test_timestamp = None

        env = dummy_env()
        env.timestamp = datetime.datetime(2020, 2, 28, 0, 0)
        human.env = env
        human.rng = np.random.RandomState(0)
        human.update_uid()
        hd[human.name] = human

    risks = []
    risk_vs_infected = []

    # Sort by time and then by human id
    enc_logs = sorted(enc_logs, key=operator.itemgetter('time', 'human_id'))
    enc_logs = sorted(enc_logs, key=lambda k: (k['time'], k['human_id']))
    enc_logs = sorted(enc_logs, key=lambda k: (k['time'], -k['human_id']))

    log_idx = 0
    plot_day = enc_logs[0]['time'].day
    plot_num = 0
    for day in range(enc_logs[0]['time'].day + datetime.timedelta(days=100)):

        for log in tqdm(enc_logs):
            # manage time logic
            now = log['time']
            if now.day != day:
                continue

            # extract variables and update the humans and sim
            unobs = log['payload']['unobserved']
            h1 = unobs['human1']['human_id']
            h2 = unobs['human2']['human_id']
            this_human = hd[h1]
            other_human = hd[h2]
            this_human.env.timestamp = now
            other_human.env.timestamp = now
            # handle messaging for that day
            handle_message_log()

            # if it's a new day in this log, send this log
            if this_human.cur_day != now.day:
                this_human.cur_day = now.day
                this_human.update_uid()
                cur_risk = this_human.risk

                # if the person's infected, look at their symptoms once per day and calculate a risk
                this_human.risk = risk_for_symptoms(this_human)

                # update risk based on that day's messages
                for j in range(len(this_human.pending_messages)):
                    m_j = this_human.pending_messages.pop()
                    if METHOD_CLUSTERING_MAP[RISK_MODEL]:
                        add_message_to_cluster(this_human, m_j)
                    update_risk_encounter(this_human, m_j, RISK_MODEL)

                # append the updated risk for this person and whether or not they are actually infectious
                risk_vs_infected.append((this_human.risk, this_human.is_infectious, this_human.is_infected, this_human.is_quarantined, this_human.name))

                if PLOT_DAILY and plot_day != now.day:
                    plot_num += 1
                    print(f"plot_day: {plot_day}, plt_num: {plot_num}, env.time: {this_human.env.timestamp}")
                    plot_day = now.day
                    for human in humans:
                        found = False
                        for (risk, is_infectious, is_infected, is_quarantined, name) in risk_vs_infected:
                            if human.name == name:
                                found = True
                        if not found:
                            this_human.risk = risk_for_symptoms(this_human)
                            risk_vs_infected.append((this_human.risk, this_human.is_infectious, this_human.is_infected, this_human.is_quarantined, this_human.name))
                    hist_plot(risk_vs_infected, f"{PATH_TO_PLOT}day_{str(plot_num).zfill(3)}.png")
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

    # make a gif of the dist output
    process = subprocess.Popen(f"convert -delay 50 -loop 0 {PATH_TO_PLOT}/*.png {PATH_TO_PLOT}/risk.gif".split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # write out the clusters to be processed by privacy_plots
    if METHOD_CLUSTERING_MAP[RISK_MODEL]:
        clusters = []
        for human in humans:
            clusters.append(human.M)
        json.dump(clusters, open(CLUSTER_PATH, 'w'))


