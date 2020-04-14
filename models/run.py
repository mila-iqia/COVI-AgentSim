import sys
import os
sys.path.append(os.getcwd())
import pickle
import json
import argparse
import subprocess
import numpy as np
import operator
import datetime
from collections import defaultdict
from base import Event
from dummy_human import DummyHuman
from risk_models import RiskModelYoshua, RiskModelLenka, RiskModelEilif
from plots.plot_risk import dist_plot, hist_plot


def main(args):
    # Define constants
    SIGNIFICANT_RISK_LEVEL_CHANGE = 0.1

    # read and filter the pickles
    with open(args.data_path, "rb") as f:
        logs = pickle.load(f)

    # select the risk model
    if args.risk_model == 'yoshua':
        RiskModel = RiskModelYoshua
    elif args.risk_model == 'lenka':
        RiskModel = RiskModelLenka
    elif args.risk_model == 'eilif':
        RiskModel = RiskModelEilif

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
    rng = np.random.RandomState(args.seed)
    hd = {}
    for human_id in human_ids:
        hd[human_id] = DummyHuman(name=human_id, timestamp=start, rng=rng)

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
            human.risk = RiskModel.update_risk_local(human, todays_date)
            if todays_date > human.infectiousness_start:
                human.is_infectious = True
            if human.time_of_recovery < todays_date or human.time_of_death < todays_date:
                human.is_infectious = False

            # read your old messages
            for m_i in human.messages:
                human.timestamp = m_i[0]
                # update risk based on that day's messages
                RiskModel.update_risk_encounter(human, m_i)

            # go about your day and accrue encounters
            encounters = logs[hash_id_day(human.name, current_day)]
            for encounter in encounters:
                # extract variables from log
                encounter_time = encounter['time']
                unobs = encounter['payload']['unobserved']
                encountered_human = hd[unobs['human2']['human_id']]
                human.messages.append(encountered_human.cur_message(encounter_time))


            if start_risk > human.risk + SIGNIFICANT_RISK_LEVEL_CHANGE or start_risk < human.risk - SIGNIFICANT_RISK_LEVEL_CHANGE:
                for m in human.messages:
                    # if the encounter happened within the last 14 days, and your symptoms started at most 3 days after your contact
                    if todays_date - m.time < datetime.timedelta(days=14) and human.symptoms_start < m.time + datetime.timedelta(days=3):
                        hd[m.unobs_id].messages.append(human.cur_message(encounter_time))

            # append the updated risk for this person and whether or not they are actually infectious
            daily_risks.append((human.risk, human.is_infectious, human.name))
        if args.plot_daily:
            hist_plot(daily_risks, f"{args.plot_path}day_{str(current_day).zfill(3)}.png")
        all_risks.extend(daily_risks)
        daily_risks = []
    dist_plot(all_risks,  f"{args.plot_path}all_risks.png")

    # make a gif of the dist output
    process = subprocess.Popen(f"convert -delay 50 -loop 0 {args.plot_path}/*.png {args.plot_path}/risk.gif".split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # write out the clusters to be processed by privacy_plots
    if hd[0].M:
        clusters = []
        for human in hd.values():
            clusters.append(human.M)
        json.dump(clusters, open(args.cluster_path, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Risk Models and Plot results')
    parser.add_argument('--plot_path', type=str, default="plots/risk/")
    parser.add_argument('--data_path', type=str, default="output/data.pkl")
    parser.add_argument('--cluster_path', type=str, default="output/clusters.json")
    parser.add_argument('--plot_daily', type=bool, default=False)
    parser.add_argument('--risk_model', type=str, default="yoshua", choices=['yoshua', 'lenka', 'eilif'])
    parser.add_argument('--seed', type=int, default="0")
    args = parser.parse_args()
    main(args)
