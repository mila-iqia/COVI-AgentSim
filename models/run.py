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
from base import Event
from models.dummy_human import DummyHuman
from models.risk_models import RiskModelYoshua, RiskModelLenka, RiskModelEilif, RiskModelTristan
from plots.plot_risk import dist_plot, hist_plot
from models.helper import messages_to_np, symptoms_to_np, candidate_exposures, rolling_infectiousness
from utils import _encode_message
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
    parser.add_argument('--max_pickles', type=int, default=1000000, help="If you don't want to load the whole dataset")
    args = parser.parse_args()
    return args

def hash_id_day(hid, day):
    return str(hid) + "-" + str(day)


def proc_human(params):
    """This function can be parallelized across CPUs. Currently, we only check for messages once per day, so this can be run in parallel"""
    start, current_day, RiskModel, encounters, rng, all_possible_symptoms, human, save_training_data = params.values()
    human.start_risk = human.risk
    todays_date = start + datetime.timedelta(days=current_day)

    # update your quantized uid and shuffle the messages (following privacy protocol)
    human.update_uid()

    # check if you have new reported symptoms
    human.risk = RiskModel.update_risk_daily(human, todays_date)

    # read your old messages
    for m_i in human.messages:
        # update risk based on that day's messages
        RiskModel.update_risk_encounter(human, m_i)
        RiskModel.add_message_to_cluster(human, m_i, rng)

    # check your update messages
    for m_i in human.update_messages:
        RiskModel.update_risk_risk_update(human, m_i, rng)

    # append the updated risk for this person and whether or not they are actually infectious
    human.purge_messages(current_day)

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

def init_humans(params):
    pkl_name = params['pkl_name']
    start = params['start']
    data_path = params['data_path']
    rng = params['rng']
    # read and filter the pickles
    hd = {}
    human_ids = set()
    all_possible_symptoms = set()
    with zipfile.ZipFile(data_path, 'r') as zf:
        logs = pickle.load(zf.open(pkl_name, 'r'))
        for log in logs:
            # check if we have a human object for this log, if not create it
            human_id = log['human_id']
            if human_id not in human_ids:
                human_ids.add(human_id)
                hd[human_id] = DummyHuman(name=human_id, rng=rng)
                hd[human_id].update_uid()

            if log['event_type'] == Event.symptom_start:
                hd[log['human_id']].symptoms_start = log['time']
                hd[log['human_id']].all_reported_symptoms = log['payload']['observed']['reported_symptoms']
                hd[log['human_id']].all_symptoms = log['payload']['unobserved']['all_symptoms']
                for symptoms in hd[log['human_id']].all_symptoms:
                    for symptom in symptoms:
                        all_possible_symptoms.add(symptom)
            elif log['event_type'] == Event.recovered:
                if log['payload']['unobserved']['death']:
                    hd[log['human_id']].time_of_death = log['time']
                else:
                    hd[log['human_id']].time_of_recovery = log['time']
            elif log['event_type'] == Event.test:
                hd[log['human_id']].test_time = log['time']
            elif log['event_type'] == Event.contamination:
                hd[log['human_id']].time_of_exposure = log['time']
                hd[log['human_id']].infectiousness_start_time = log['payload']['unobserved'][
                    'infectiousness_start_time']
                hd[log['human_id']].exposure_source = log['payload']['unobserved']['source']
            elif log['event_type'] == Event.static_info:
                hd[log['human_id']].obs_preexisting_conditions = log['payload']['observed'][
                    'obs_preexisting_conditions']
                hd[log['human_id']].preexisting_conditions = log['payload']['unobserved']['preexisting_conditions']
            elif log['event_type'] == Event.visit:
                if not hd[log['human_id']].locations_visited.get(log['payload']['observed']['location_name']):
                    hd[log['human_id']].locations_visited[log['payload']['observed']['location_name']] = log['time']
            elif log['event_type'] == Event.daily:
                hd[log['human_id']].infectiousness[(log['time'] - start).days] = log['payload']['unobserved'][
                    'infectiousness']

    return hd, all_possible_symptoms


def pick_risk_model(risk_model):
    # select the risk model
    if risk_model == 'yoshua':
        return RiskModelYoshua
    elif risk_model == 'lenka':
        return RiskModelLenka
    elif risk_model == 'eilif':
        return RiskModelEilif
    elif risk_model == 'tristan':
        return RiskModelTristan
    raise "unknown risk model"


def get_days_worth_of_logs(data_path, start, start_pkl, cur_day):
    to_return = defaultdict(list)
    started = False
    with zipfile.ZipFile(data_path, 'r') as zf:
        for pkl in zf.namelist():
            if not started:
                if pkl != start_pkl:
                    continue
            started = True
            start_pkl = pkl
            logs = pickle.load(zf.open(pkl, 'r'))
            for log in logs:
                if log['event_type'] == Event.encounter:
                    day_since_epoch = (log['time'] - start).days
                    if day_since_epoch == cur_day:
                        to_return[log['human_id']].append(log)
                    elif day_since_epoch > cur_day:
                        return to_return, start_pkl
    return to_return, start_pkl

def merge_humans(humans):
    merged_human = humans[0]
    for human in humans:
        merged_human.merge(human)
    return merged_human

def main(args=None):
    if not args:
        args = parse_args()
    rng = np.random.RandomState(args.seed)

    # iterate the logs and init people
    with zipfile.ZipFile(args.data_path, 'r') as zf:
        start_logs = pickle.load(zf.open(zf.namelist()[0], 'r'))
        end_logs = pickle.load(zf.open(zf.namelist()[-1], 'r'))
        start = start_logs[0]['time']
        end = end_logs[-1]['time']
        total_days = (end - start).days
        all_params = []
        for idx, pkl in enumerate(zf.namelist()):
            if idx > args.max_pickles:
                break
            all_params.append({"pkl_name": pkl, "start": start, "data_path": args.data_path, "rng": rng})
    print("initializing humans from logs.")
    with Parallel(n_jobs=args.n_jobs, batch_size='auto', verbose=10) as parallel:
        results = parallel((delayed(init_humans)(params) for params in all_params))
    import pdb; pdb.set_trace()
    humans = defaultdict(list)
    all_possible_symptoms = set()
    for result in results:
        for hid, human in result[0].items():
            humans[hid].append(human)
        for symp in result[1]:
            all_possible_symptoms.add(symp)

    hd = {}
    for hid, humans in humans.items():
        import pdb; pdb.set_trace()
        hd[hid] = merge_humans(humans)

    # select the risk prediction model to embed in messaging protocol
    RiskModel = pick_risk_model(args.risk_model)

    all_outputs = []
    all_risks = []
    with zipfile.ZipFile(args.data_path, 'r') as zf:
        start_pkl = zf.namelist()[0]
    for current_day in range(total_days):
        print(f"day {current_day} of {total_days}")
        days_logs, start_pkl = get_days_worth_of_logs(args.data_path, start, start_pkl, current_day)
        start1 = time.time()
        daily_risks = []

        all_params = []
        for human in hd.values():
            encounters = days_logs[human.name]
            all_params.append({"start": start, "current_day": current_day, "RiskModel": RiskModel, "encounters": encounters, "rng": rng, "all_possible_symptoms": all_possible_symptoms, "human": human, "save_training_data": args.save_training_data})
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
                    human.exposure_message = _encode_message(message)

            # if the encounter happened within the last 14 days, and your symptoms started at most 3 days after your contact
            if RiskModel.quantize_risk(human.start_risk) != RiskModel.quantize_risk(human.risk):
                for k, m in human.sent_messages.items():
                    if current_day - m.day < 14:
                        hd[m.unobs_id].update_messages.append(human.cur_message_risk_update(m.day, m.risk, RiskModel))

        with Parallel(n_jobs=args.n_jobs, batch_size='auto', verbose=10) as parallel:
            daily_output = parallel((delayed(proc_human)(params) for params in all_params))

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
        clusters.append(human.M)
    json.dump(clusters, open(args.cluster_path, 'w'))


if __name__ == "__main__":
    main()
