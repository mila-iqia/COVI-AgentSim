import pickle
import json
from base import Event
import subprocess
import numpy as np
from config import *
from utils import binary_to_float
import matplotlib.pyplot as plt
from utils import _encode_message, _decode_message, float_to_binary, binary_to_float
from bitarray import bitarray
import operator
from collections import defaultdict, namedtuple
import datetime
from plots.plot_risk import dist_plot, hist_plot


""" This file contains the core of the side simulation, which is run on the output encounters from the main simulation.
It's primary functionality is to run the message clustering and risk prediction algorithms.
"""


# A utility class for re-inflating human objects with just the stuff we need for message passing / risk prediction
class dummy_human:
    def __init__(self, name=None, timestamp=None, rng=None):
        self.name = name
        self.M = {}
        self.messages = []
        self.risk = 0
        self.rng = rng
        self.all_reported_symptoms = []
        self.test_logs = []
        self.timestamp = timestamp
        self._uid = None
        self.is_infectious = False

    @property
    def message_risk(self):
        """quantizes the risk in order to be used in a message"""
        if self.risk == 1.0:
            return bitarray('1111')
        return bitarray(float_to_binary(self.risk, 0, 4))

    def cur_message(self, time):
        """creates the current message for this user"""
        Message = namedtuple('message', 'uid risk time unobs_id')
        message = Message(self.uid, self.message_risk, time, self.name)
        return message

    @property
    def uid(self):
        return self._uid

    def update_uid(self):
        try:
            self._uid.pop()
            self._uid.extend([self.rng.choice([True, False])])
        except AttributeError:
            self._uid = bitarray()
            self._uid.extend(self.rng.choice([True, False], 4))  # generate a random 4-bit code

    def reported_symptoms_at_time(self, now):
        # TODO: this is kind of a lossy way to take into account the information currently available
        try:
            sickness_day = (self.symptoms_start - now).days
            all_reported_symptoms_till_day = []
            for day in range(sickness_day+1):
                all_reported_symptoms_till_day.extend(self.all_reported_symptoms[sickness_day])
            return all_reported_symptoms_till_day
        except Exception as e:
            return []

    def update_risk_encounter(self, message, RISK_MODEL):
        """ This function updates an individual's risk based on the receipt of a new message"""

        # Get the binarized contact risk
        m_risk = binary_to_float("".join([str(x) for x in np.array(message.risk.tolist()).astype(int)]), 0, 4)

        # select your contact risk prediction model
        update = 0
        if RISK_MODEL == 'yoshua':
            if self.risk < m_risk:
                update = (m_risk - m_risk * self.risk) * RISK_TRANSMISSION_PROBA
        elif RISK_MODEL == 'lenka':
            update = m_risk * RISK_TRANSMISSION_PROBA

        elif RISK_MODEL == 'eilif':
            msg_enc = _encode_message(message)
            if msg_enc not in self.M:
                # update is delta_risk
                update = m_risk * RISK_TRANSMISSION_PROBA
            else:
                previous_risk = self.M[msg_enc]['previous_risk']
                carry_over_transmission_proba = self.M[msg_enc]['carry_over_transmission_proba']
                update = ((m_risk - previous_risk) * RISK_TRANSMISSION_PROBA + previous_risk * carry_over_transmission_proba)

            # Update contact history
            self.M[msg_enc]['previous_risk'] = m_risk
            self.M[msg_enc]['carry_over_transmission_proba'] = RISK_TRANSMISSION_PROBA * (1 - update)
        print(f"self.risk: {self.risk}, m_risk: {m_risk}, update: {update}")

        self.risk += update

        # some models require us to clip the risk at one (like 'lenka' and 'eilif')
        if CLIP_RISK:
            self.risk = min(self.risk, 1.)


    def update_risk_local(self, now):
        """ This function calculates a risk score based on the person's symptoms."""
        # if they get tested, it takes TEST_DAYS to get the result, and they are quarantined for QUARANTINE_DAYS.
        # The test_timestamp is set to datetime.min, unless they get a positive test result.
        # Basically, once they know they have a positive test result, they have a risk of 1 until after quarantine days.
        if self.time_of_recovery < now:
            return 0.
        if self.time_of_death < now:
            return 0.
        if self.test_logs[1] and self.test_logs[0] < now + datetime.timedelta(days=2):
            return 1.

        reported_symptoms = self.reported_symptoms_at_time(now)
        if 'severe' in reported_symptoms:
            return 0.75
        if 'moderate' in reported_symptoms:
            return 0.5
        if 'mild' in reported_symptoms:
            return 0.25
        if len(reported_symptoms) > 3:
            return 0.25
        if len(reported_symptoms) > 1:
            return 0.1
        if len(reported_symptoms) > 0:
            return 0.05
        return 0.0


    def add_message_to_cluster(self, m_i):
        """ This function clusters new messages by scoring them against old messages in a sort of naive nearest neighbors approach"""
        # TODO: include risk level in clustering, currently only uses quantized uid
        # TODO: refactor to compare multiple clustering schemes
        # TODO: check for mutually exclusive messages in order to break up a group and re-run nearest neighbors
        m_i_enc = _encode_message(m_i)
        m_risk = binary_to_float("".join([str(x) for x in np.array(m_i[1].tolist()).astype(int)]), 0, 4)

        # otherwise score against previous messages
        scores = {}
        for m_enc, _ in self.M.items():
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
            self.M[m_i_enc] = {'assignment': self.M[max_score_message]['assignment'], 'previous_risk': m_risk, 'carry_over_transmission_proba': RISK_TRANSMISSION_PROBA}
        # if it's either the first message
        elif len(self.M) == 0:
            self.M[m_i_enc] = {'assignment': 0, 'previous_risk': m_risk, 'carry_over_transmission_proba': RISK_TRANSMISSION_PROBA}
        # if there was no nearby neighbor
        else:
            new_group = max([v['assignment'] for k, v in self.M.items()]) + 1
            self.M[m_i_enc] = {'assignment': new_group, 'previous_risk': m_risk, 'carry_over_transmission_proba': RISK_TRANSMISSION_PROBA}



if __name__ == "__main__":
    # TODO: add as args that can be called from cmdline
    PLOT_DAILY = False
    PATH_TO_DATA = "output/data.pkl"
    PATH_TO_HUMANS = "output/humans.pkl"
    CLUSTER_PATH = "output/clusters.json"
    PATH_TO_PLOT = "plots/risk/"
    RISK_MODEL = 'yoshua'  # options: ['yoshua', 'lenka', 'eilif']
    METHOD_CLUSTERING_MAP = {"eilif": True, "yoshua": False, "lenka": False}
    LOGS_SUBSET_SIZE = 10000000
    seed = 0

    rng = np.random.RandomState(seed)

    # read and filter the pickles
    with open(PATH_TO_DATA, "rb") as f:
        logs = pickle.load(f)
    human_ids = set()
    enc_logs = []
    symp_logs = []
    test_logs = []
    recovered_logs = []

    start = logs[0]['time']
    for log in logs:
        human_ids.add(log['human_id'])
        if log['event_type'] == Event.encounter:
            enc_logs.append(log)
        elif log['event_type'] == Event.symptom_start:
            symp_logs.append(log)
        elif log['event_type'] == Event.recovered:
            recovered_logs.append(log)
        elif log['event_type'] == Event.test:
            test_logs.append(log)

    # create some dummy humans
    hd = {}
    for human_id in human_ids:
        hd[human_id] = dummy_human(name=human_id, timestamp=start, rng=rng)

    # Sort encounter logs by time and then by human id
    enc_logs = sorted(enc_logs, key=operator.itemgetter('time'))
    logs = defaultdict(list)
    def hash_id_day(hid, day):
        return str(hid) + "-" + str(day)

    for log in enc_logs:
        day_since_epoch = (log['time'] - start).days
        logs[hash_id_day(log['human_id'], day_since_epoch)].append(log)

    for log in symp_logs:
        hd[log['human_id']].symptoms_start = log['time']
        hd[log['human_id']].infectiousness_start = log['time'] - datetime.timedelta(days=3)
        hd[log['human_id']].all_reported_symptoms = log['payload']['observed']['reported_symptoms']

    for log in recovered_logs:
        if log['payload']['unobserved']['death']:
            hd[log['human_id']].time_of_death = log['time']
            hd[log['human_id']].time_of_recovery = datetime.datetime.max
        else:
            hd[log['human_id']].time_of_recovery = log['time']
            hd[log['human_id']].time_of_death = datetime.datetime.max


    test_logs_proc = {}
    for log in test_logs:
        hd[log['human_id']].test_logs = (log['time'], log['payload']['observed']['result'])

    all_risks = []
    daily_risks = []
    days = (enc_logs[-1]['time'] - enc_logs[0]['time']).days
    for current_day in range(days):
        for hid, human in hd.items():
            start_risk = human.risk
            todays_date = start + datetime.timedelta(days=current_day)

            # update your quantized uid
            human.update_uid()

            # check if you have new reported symptoms
            human.risk = human.update_risk_local(todays_date)
            if todays_date > human.infectiousness_start:
                human.is_infectious = True
            if human.time_of_recovery < todays_date or human.time_of_death < todays_date:
                human.is_infectious = False

            # read your old messages
            for m_i in human.messages:
                human.timestamp = m_i[0]
                # update risk based on that day's messages
                if METHOD_CLUSTERING_MAP[RISK_MODEL]:
                    human.add_message_to_cluster(m_i)
                human.update_risk_encounter(m_i, RISK_MODEL)

            # go about your day and accrue encounters
            encounters = logs[hash_id_day(human.name, current_day)]
            for encounter in encounters:
                # extract variables from log
                encounter_time = encounter['time']
                unobs = encounter['payload']['unobserved']
                encountered_human = hd[unobs['human2']['human_id']]
                human.messages.append(encountered_human.cur_message(encounter_time))


            if start_risk > human.risk + 0.1 or start_risk < human.risk - 0.1:
                for m in human.messages:
                    # if the encounter happened within the last 14 days, and your symptoms started at most 3 days after your contact
                    if todays_date - m.time < datetime.timedelta(days=14) and human.symptoms_start < m.time + datetime.timedelta(days=3):
                        hd[m.unobs_id].messages.append(human.cur_message(encounter_time))

            # append the updated risk for this person and whether or not they are actually infectious
            daily_risks.append((human.risk, human.is_infectious, human.name))
        if PLOT_DAILY:
            hist_plot(daily_risks, f"{PATH_TO_PLOT}day_{str(current_day).zfill(3)}.png")
        all_risks.extend(daily_risks)
        daily_risks = []
    dist_plot(all_risks,  f"{PATH_TO_PLOT}all_risks.png")

    # make a gif of the dist output
    process = subprocess.Popen(f"convert -delay 50 -loop 0 {PATH_TO_PLOT}/*.png {PATH_TO_PLOT}/risk.gif".split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # write out the clusters to be processed by privacy_plots
    if METHOD_CLUSTERING_MAP[RISK_MODEL]:
        clusters = []
        for human in humans:
            clusters.append(human.M)
        json.dump(clusters, open(CLUSTER_PATH, 'w'))


