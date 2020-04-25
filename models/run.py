import os
import pickle
import json
import zipfile
import argparse
import numpy as np
import datetime
import pathlib
import time
from collections import defaultdict
from plots.plot_risk import hist_plot
from models.risk_models import RiskModelTristan
from models.helper import conditions_to_np, symptoms_to_np, candidate_exposures, \
    encode_age, encode_sex
from models.utils import encode_message, update_uid, create_new_uid, encode_update_message, decode_message
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Run Risk Models and Plot results')
parser.add_argument('--plot_path', type=str, default="output/plots/risk/")
parser.add_argument('--data_path', type=str, default="output/data.pkl")
parser.add_argument('--cluster_path', type=str, default="output/clusters.json")
parser.add_argument('--output_file', type=str, default='output/output.pkl')
parser.add_argument('--plot_daily', action="store_true")
parser.add_argument('--risk_model', type=str, default="tristan", choices=[ 'tristan'])
parser.add_argument('--seed', type=int, default="0")
parser.add_argument('--save_training_data', action="store_true")
parser.add_argument('--n_jobs', type=int, default=1, help="Default is no parallelism, jobs = 1")
parser.add_argument('--max_num_days', type=int, default=10000, help="Default is to run for all days")
parser.add_argument('--max_pickles', type=int, default=1000000, help="If you don't want to load the whole dataset")
parser.add_argument('--mp_backend', type=str, default="loky", help="which joblib backend to use")
parser.add_argument('--mp_batchsize', type=int, default=-1, help="-1 is converted to auto batchsize, otherwise it's the integer you provide")
parser.add_argument('--random_clusters', action="store_true", help="make random cluster assignments")


def hash_id_day(hid, day):
    return str(hid) + "-" + str(day)


def proc_human(params):
    """This function can be parallelized across CPUs. Currently, we only check for messages once per day, so this can be run in parallel"""
    start, current_day, encounters, rng, all_possible_symptoms, human, save_training_data, log_path, random_clusters = params.values()

    RiskModel = RiskModelTristan
    human.start_risk = human.risk
    todays_date = start + datetime.timedelta(days=current_day)

    # check if you have new reported symptoms
    human.risk = RiskModel.update_risk_daily(human, todays_date)

    # update risk based on that day's messages
    RiskModel.update_risk_encounters(human, human.messages)


    # add them to clusters
    human.clusters.add_messages(human.messages, current_day, rng)
    human.messages = []

    human.clusters.update_records(human.update_messages, human)

    human.update_messages = []
    human.clusters.purge(current_day)

    # for each sim day, for each human, save an output training example
    if save_training_data:
        is_exposed, exposure_day = human.exposure_array(todays_date)
        is_recovered, recovery_day = human.recovered_array(todays_date)
        candidate_encounters, exposure_encounter = candidate_exposures(human, todays_date)
        reported_symptoms = symptoms_to_np(human.symptoms_at_time(todays_date, human.all_reported_symptoms), all_possible_symptoms)
        true_symptoms = symptoms_to_np(human.symptoms_at_time(todays_date, human.all_symptoms), all_possible_symptoms)
        daily_output = {"current_day": current_day,
                        "observed":
                            {
                                "reported_symptoms": reported_symptoms,
                                "candidate_encounters": candidate_encounters,
                                "test_results": human.get_test_result_array(todays_date),
                                "preexisting_conditions": conditions_to_np(human.obs_preexisting_conditions),
                                "age": encode_age(human.obs_age),
                                "sex": encode_sex(human.obs_sex)
                            },
                        "unobserved":
                            {
                                "true_symptoms": true_symptoms,
                                "is_exposed": is_exposed,
                                "exposure_day": exposure_day,
                                "is_recovered": is_recovered,
                                "recovery_day": recovery_day,
                                "infectiousness": np.array(human.infectiousnesses),
                                "true_preexisting_conditions": conditions_to_np(human.preexisting_conditions),
                                "true_age": encode_age(human.age),
                                "true_sex": encode_sex(human.sex)
                            }
                        }
        if not os.path.isdir(log_path):
            pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(log_path, f"daily_human.pkl")
        log_file = open(path, 'wb')
        pickle.dump(daily_output, log_file)

    return human


def get_days_worth_of_logs(data_path, start, cur_day):
    to_return = defaultdict(list)
    started = False

    with zipfile.ZipFile(data_path, 'r') as zf:
        start_pkl = zf.namelist()[0]
        for pkl in zf.namelist():
            if not started:
                if pkl != start_pkl:
                    continue
            started = True
            start_pkl = pkl
            logs = pickle.load(zf.open(pkl, 'r'))
            from base import Event

            for log in logs:
                if log['event_type'] == Event.encounter:
                    day_since_epoch = (log['time'] - start).days
                    if day_since_epoch == cur_day:
                        to_return[log['human_id']].append(log)
                    elif day_since_epoch > cur_day:
                        return to_return, start_pkl
    return to_return, start_pkl

def integrated_risk_pred(humans, data_path, start, current_day, all_possible_symptoms, mp_batchsize="auto", mp_backend="loky", n_jobs=1):
    RiskModel = RiskModelTristan

    days_logs, start_pkl = get_days_worth_of_logs(data_path + ".zip", start, current_day)
    all_params = []
    hd = {human.name: human for human in humans}
    for human in humans:
        encounters = days_logs[human.name]
        log_path = f'{os.path.dirname(data_path)}/daily_outputs/{current_day}/{human.name[6:]}/'
        # go about your day accruing encounters and clustering them
        for encounter in encounters:
            encounter_time = encounter['time']
            unobs = encounter['payload']['unobserved']
            encountered_human = hd[unobs['human2']['human_id']]
            message = encode_message(encountered_human.cur_message(current_day, RiskModel))
            encountered_human.sent_messages[str(unobs['human1']['human_id']) + "_" + str(encounter_time)] = message
            human.messages.append(message)

            got_exposed = encounter['payload']['unobserved']['human1']['got_exposed']
            if got_exposed:
                human.exposure_message = message

        # if the encounter happened within the last 14 days, and your symptoms started at most 3 days after your contact
        if RiskModel.quantize_risk(human.start_risk) != RiskModel.quantize_risk(human.risk):
            sent_at = start + datetime.timedelta(days=current_day, minutes=human.rng.randint(low=0, high=1440))
            for k, m in human.sent_messages.items():
                message = decode_message(m)
                if current_day - message.day < 14:
                    # add the update message to the receiver's inbox
                    update_message = encode_update_message(
                        human.cur_message_risk_update(message.day, message.risk, sent_at, RiskModel))
                    hd[k.split("_")[0]].update_messages.append(update_message)
            human.sent_messages = {}
        all_params.append({"start": start, "current_day": current_day, "encounters": encounters, "rng": human.rng,
                           "all_possible_symptoms": all_possible_symptoms, "human": human,
                           "save_training_data": True, "log_path": log_path,
                           "random_clusters": False})
        human.uid = update_uid(human.uid, human.rng)
    if current_day == 10:
        import pdb; pdb.set_trace()
    with Parallel(n_jobs=n_jobs, batch_size=mp_batchsize, backend=mp_backend, verbose=10) as parallel:
        humans = parallel((delayed(proc_human)(params) for params in all_params))

    return humans

def main(args=None):
    if args is None:
        args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    # check that the plot_dir exists:
    if args.plot_path and not os.path.isdir(args.plot_path):
        os.mkdir(args.plot_path)

    # joblib sometimes takes a string and sometimes an int
    if args.mp_batchsize == -1:
        mp_batchsize = "auto"
    else:
        mp_batchsize = args.mp_batchsize

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
            all_params.append({"pkl_name": pkl, "start": start, "data_path": args.data_path})

    print("initializing humans from logs.")
    with Parallel(n_jobs=args.n_jobs, batch_size=mp_batchsize, backend=args.mp_backend, verbose=10) as parallel:
        results = parallel((delayed(init_humans)(params) for params in all_params))

    humans = defaultdict(list)
    all_possible_symptoms = set()
    for result in results:
        for human in result[0]:
            humans[human['name']].append(human)
        for symp in result[1]:
            all_possible_symptoms.add(symp)

    hd = {}
    for hid, human_splits in humans.items():
        merged_human = DummyHuman(name=human_splits[0]['name'])
        for human in human_splits:
            merged_human.merge(human)
        merged_human.uid = create_new_uid(rng)
        hd[hid] = merged_human

    # select the risk prediction model to embed in messaging protocol

    RiskModel = pick_risk_model(args.risk_model)

    with zipfile.ZipFile(args.data_path, 'r') as zf:
        start_pkl = zf.namelist()[0]
    for current_day in range(total_days):
        if args.max_num_days <= current_day:
            break

        print(f"day {current_day} of {total_days}")
        days_logs, start_pkl = get_days_worth_of_logs(args.data_path, start, start_pkl, current_day)

        all_params = []
        for human in hd.values():
            encounters = days_logs[human.name]
            log_path = f'{os.path.dirname(args.data_path)}/daily_outputs/{current_day}/{human.name[6:]}/'
            all_params.append({"start": start, "current_day": current_day, "encounters": encounters, "rng": rng, "all_possible_symptoms": all_possible_symptoms, "human": human.__dict__, "save_training_data": args.save_training_data, "log_path": log_path, "random_clusters": args.random_clusters})
            # go about your day accruing encounters and clustering them
            for encounter in encounters:
                encounter_time = encounter['time']
                unobs = encounter['payload']['unobserved']
                encountered_human = hd[unobs['human2']['human_id']]
                message = encode_message(encountered_human.cur_message(current_day, RiskModel))
                encountered_human.sent_messages[str(unobs['human1']['human_id']) + "_" + str(encounter_time)] = message
                human.messages.append(message)

                got_exposed = encounter['payload']['unobserved']['human1']['got_exposed']
                if got_exposed:
                    human.exposure_message = message

            # if the encounter happened within the last 14 days, and your symptoms started at most 3 days after your contact
            if RiskModel.quantize_risk(human.start_risk) != RiskModel.quantize_risk(human.risk):
                sent_at = start + datetime.timedelta(days=current_day, minutes=rng.randint(low=0, high=1440))
                for k, m in human.sent_messages.items():
                    message = decode_message(m)
                    if current_day - message.day < 14:
                        # add the update message to the receiver's inbox
                        update_message = encode_update_message(human.cur_message_risk_update(message.day, message.risk, sent_at, RiskModel))
                        hd[k.split("_")[0]].update_messages.append(update_message)
                human.sent_messages = {}
            human.uid = update_uid(human.uid, rng)

        with Parallel(n_jobs=args.n_jobs, batch_size=mp_batchsize, backend=args.mp_backend, verbose=10) as parallel:
            human_dicts = parallel((delayed(proc_human)(params) for params in all_params))

        for human_dict in human_dicts:
            human = DummyHuman(name=human_dict['name']).merge(human_dict)
            hd[human.name] = human
        if args.plot_daily:
            daily_risks = [(np.e ** human.risk, human.is_infectious(encounter['time'])[0], human.name) for human in hd.values()]
            hist_plot(daily_risks, f"{args.plot_path}day_{str(current_day).zfill(3)}.png")

    # print out the clusters
    clusters = []
    for human in hd.values():
        clusters.append(dict(human.clusters.clusters))
    json.dump(clusters, open(args.cluster_path, 'w'))

if __name__ == "__main__":
    main()
