import sys
import os
sys.path.append(os.getcwd())
import pickle
import json
import zipfile
import argparse
import subprocess
import numpy as np
import operator
import datetime
import time
from tqdm import tqdm
from collections import defaultdict
from event import Event
from models.dummy_human import DummyHuman
from models.risk_models import RiskModelYoshua, RiskModelLenka, RiskModelEilif, RiskModelTristan
from plots.plot_risk import dist_plot, hist_plot
from models.helper import messages_to_np, symptoms_to_np, candidate_exposures, rolling_infectiousness
from models.utils import encode_message, update_uid, create_new_uid
from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(description='Run Risk Models and Plot results')
    parser.add_argument('--plot_path', type=str, default="output/plots/risk/")
    parser.add_argument('--data_path', type=str, default="output/data.pkl")
    parser.add_argument('--cluster_path', type=str, default="output/clusters.json")
    parser.add_argument('--output_file', type=str, default='output/output.pkl')
    parser.add_argument('--plot_daily', action="store_true")
    parser.add_argument('--risk_model', type=str, default="tristan", choices=['yoshua', 'lenka', 'eilif', 'tristan'])
    parser.add_argument('--seed', type=int, default="0")
    parser.add_argument('--save_training_data', action="store_true")
    parser.add_argument('--n_jobs', type=int, default=1, help="Default is no parallelism, jobs = 1")
    parser.add_argument('--max_num_days', type=int, default=10000, help="Default is to run for all days")
    args = parser.parse_args()
    return args

def hash_id_day(hid, day):
    return str(hid) + "-" + str(day)


def proc_human(params):
    """This function can be parallelized across CPUs. Currently, we only check for messages once per day, so this can be run in parallel"""
    start, current_day, RiskModel, encounters, rng, all_possible_symptoms, human, save_training_data = params.values()
    human.start_risk = human.risk
    todays_date = start + datetime.timedelta(days=current_day)
    import time
    # check if you have new reported symptoms
    human.risk = RiskModel.update_risk_daily(human, todays_date)
    print(f"len(human.messages): {len(human.messages)}")
    start1 = time.time()
    # read your old messages
    for m_i in human.messages:
        # update risk based on that day's messages
        RiskModel.update_risk_encounter(human, m_i)
        human.clusters.add_message(m_i)
    print(f"read old messages and cluster: {time.time()- start1}")

    start2 = time.time()
    human.clusters.update_records(human.update_messages)
    print(f"update records: {time.time()- start2}")
    start3 = time.time()
    human.clusters.purge(current_day)
    print(f"purge: {time.time() - start3}")
    # for each sim day, for each human, save an output training example
    daily_output = {}
    if save_training_data:
        is_exposed, exposure_day = human.is_exposed(todays_date)
        is_infectious, infectious_day = human.is_infectious(todays_date)
        is_recovered, recovery_day = human.is_recovered(todays_date)
        candidate_encounters, exposure_encounter, candidate_locs, exposed_locs = candidate_exposures(human, todays_date)
        infectiousness = rolling_infectiousness(start, todays_date, human)
        daily_output = {"current_day": current_day,
                        "observed":
                            {
                                "reported_symptoms": symptoms_to_np(
                                    (todays_date - human.symptoms_start).days,
                                    human.symptoms_at_time(todays_date, human.all_reported_symptoms),
                                    all_possible_symptoms),
                                "candidate_encounters": candidate_encounters,
                                "candidate_locs": candidate_locs,
                                "test_results": human.get_test_result_array(todays_date),
                            },
                        "unobserved":
                            {
                                "true_symptoms": symptoms_to_np((todays_date - human.symptoms_start).days,
                                                                human.symptoms_at_time(todays_date,
                                                                                       human.all_symptoms),
                                                                all_possible_symptoms),
                                "is_exposed": is_exposed,
                                "exposure_day": exposure_day,
                                "is_infectious": is_infectious,
                                "infectious_day": infectious_day,
                                "is_recovered": is_recovered,
                                "recovery_day": recovery_day,
                                "exposed_locs": exposed_locs,
                                "exposure_encounter": exposure_encounter,
                                "infectiousness": infectiousness,
                            }
                        }
    return {human.name: daily_output, "human": human}

def main(args=None):
    if not args:
        args = parse_args()

    # check that the plot_dir exists:
    if args.plot_path and not os.path.isdir(args.plot_path):
        os.mkdir(args.plot_path)

    # read and filter the pickles
    logs = []
    with zipfile.ZipFile(args.data_path, 'r') as zf:
        for pkl in zf.namelist():
            logs.extend(pickle.load(zf.open(pkl, 'r')))

    # Sort the logs
    logs.sort(key=lambda e: (int(e['human_id'][6:]), e['time']))

    # select the risk model
    if args.risk_model == 'yoshua':
        RiskModel = RiskModelYoshua
    elif args.risk_model == 'lenka':
        RiskModel = RiskModelLenka
    elif args.risk_model == 'eilif':
        RiskModel = RiskModelEilif
    elif args.risk_model == 'tristan':
        RiskModel = RiskModelTristan

    human_ids = set()
    enc_logs = []
    symp_logs = []
    test_logs = []
    recovered_logs = []
    contamination_logs = []
    static_info_logs = []
    visit_logs = []
    daily_logs = []
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
        elif log['event_type'] == Event.contamination:
            contamination_logs.append(log)
        elif log['event_type'] == Event.static_info:
            static_info_logs.append(log)
        elif log['event_type'] == Event.visit:
            visit_logs.append(log)
        elif log['event_type'] == Event.daily:
            daily_logs.append(log)
        else:
            print(log['event_type'])
            raise "NotImplemented Log"

    # create some dummy humans
    rng = np.random.RandomState(args.seed)
    hd = {}
    for human_id in human_ids:
        hd[human_id] = DummyHuman(name=human_id)
        hd[human_id]._uid = create_new_uid(rng)

    # Sort encounter logs by time and then by human id
    enc_logs = sorted(enc_logs, key=operator.itemgetter('time'))
    logs = defaultdict(list)

    for log in enc_logs:
        day_since_epoch = (log['time'] - start).days
        logs[hash_id_day(log['human_id'], day_since_epoch)].append(log)

    all_possible_symptoms = set()
    for log in symp_logs:
        hd[log['human_id']].symptoms_start = log['time']
        hd[log['human_id']].all_reported_symptoms = log['payload']['observed']['reported_symptoms']
        hd[log['human_id']].all_symptoms = log['payload']['unobserved']['all_symptoms']
        for symptoms in hd[log['human_id']].all_symptoms:
            for symptom in symptoms:
                all_possible_symptoms.add(symptom)

    for log in recovered_logs:
        if log['payload']['unobserved']['death']:
            hd[log['human_id']].time_of_death = log['time']
        else:
            hd[log['human_id']].time_of_recovery = log['time']

    for log in test_logs:
        hd[log['human_id']].test_time = log['time']

    for log in contamination_logs:
        hd[log['human_id']].time_of_exposure = log['time']
        hd[log['human_id']].infectiousness_start_time = log['payload']['unobserved']['infectiousness_start_time']
        hd[log['human_id']].exposure_source = log['payload']['unobserved']['source']

    for log in visit_logs:
        if not hd[log['human_id']].locations_visited.get(log['payload']['observed']['location_name']):
            hd[log['human_id']].locations_visited[log['payload']['observed']['location_name']] = log['time']

    for log in daily_logs:
        hd[log['human_id']].infectiousness[(log['time'] - start).days] = log['payload']['unobserved']['infectiousness']

    for log in static_info_logs:
        hd[log['human_id']].obs_preexisting_conditions = log['payload']['observed']['obs_preexisting_conditions']
        hd[log['human_id']].preexisting_conditions = log['payload']['unobserved']['preexisting_conditions']

    all_outputs = []
    all_risks = []
    days = (enc_logs[-1]['time'] - enc_logs[0]['time']).days
    for current_day in tqdm(range(days)):
        if args.max_num_days < current_day:
            break
        start1 = time.time()
        daily_risks = []
        all_params = []

        for human in hd.values():
            # update the uid
            human._uid = update_uid(human._uid, rng)

            # get list of encounters for that day
            encounters = logs[hash_id_day(human.name, current_day)]

            # setup parameters for parallelization
            all_params.append(
                {"start": start, "current_day": current_day, "RiskModel": RiskModel, "encounters": encounters,
                 "rng": rng, "all_possible_symptoms": all_possible_symptoms, "human": human,
                 "save_training_data": args.save_training_data})

            # caluclate the messages to be sent during each encounter
            # go about your day accruing encounters and clustering them
            for idx, encounter in enumerate(encounters):
                encounter_time = encounter['time']
                unobs = encounter['payload']['unobserved']
                encountered_human = hd[unobs['human2']['human_id']]
                message = encountered_human.cur_message(current_day, RiskModel)
                encountered_human.sent_messages[
                    str(unobs['human1']['human_id']) + "_" + str(encounter_time)] = message
                human.messages.append(message)
                got_exposed = encounter['payload']['unobserved']['human1']['got_exposed']
                if got_exposed:
                    human.exposure_message = encode_message(message)

            # if the encounter happened within the last 14 days, and your symptoms started at most 3 days after your contact
            if RiskModel.quantize_risk(human.start_risk) != RiskModel.quantize_risk(human.risk):
                sent_at = start + datetime.timedelta(days=current_day, minutes=rng.randint(low=0, high=1440))
                for k, m in human.sent_messages.items():
                    if current_day - m.day < 14:
                        # using encounter
                        hd[m.unobs_id].update_messages.append(human.cur_message_risk_update(m.day, m.risk, sent_at, RiskModel))

        with Parallel(n_jobs=args.n_jobs, batch_size='auto', verbose=10) as parallel:
            # in parallel, cluster received messages and predict risks
            daily_output = parallel((delayed(proc_human)(params) for params in all_params))

        # handle the output of the parallel processes
        for idx, output in enumerate(daily_output):
            hd[output['human'].name] = output['human']
            del daily_output[idx]['human']
        all_outputs.append(daily_output)

        print(f"mainloop {time.time() - start1}")

        # add risks for plotting
        todays_date = start + datetime.timedelta(days=current_day)
        daily_risks.extend([(np.e ** human.risk, human.is_infectious(todays_date)[0], human.name) for human in hd.values()])
        if args.plot_daily:
            hist_plot(daily_risks, f"{args.plot_path}day_{str(current_day).zfill(3)}.png")
        all_risks.extend(daily_risks)

    if args.save_training_data:
        pickle.dump(all_outputs, open(args.output_file, 'wb'))

    dist_plot(all_risks,  f"{args.plot_path}all_risks.png")

    # make a gif of the dist output
    process = subprocess.Popen(f"convert -delay 50 -loop 0 {args.plot_path}/*.png {args.plot_path}/risk.gif".split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # write out the clusters to be processed by privacy_plots
    clusters = []
    for human in hd.values():
        clusters.append(human.clusters.clusters)
    json.dump(clusters, open(args.cluster_path, 'w'))


if __name__ == "__main__":
    main()
