import os
import pickle
import json
import zipfile
import numpy as np
import datetime
import pathlib
from collections import defaultdict
from joblib import Parallel, delayed
import config
from plots.plot_risk import hist_plot
from models.risk_models import RiskModelTristan
from frozen.helper import conditions_to_np, symptoms_to_np, candidate_exposures, encode_age, encode_sex
from frozen.utils import encode_message, update_uid, encode_update_message, decode_message



def proc_human(params):
    """This function can be parallelized across CPUs. Currently, we only check for messages once per day, so this can be run in parallel"""
    start, current_day, encounters, all_possible_symptoms, human, save_training_data, log_path, random_clusters = params.values()
    todays_date = start + datetime.timedelta(days=current_day)

    # add them to clusters
    human.clusters.add_messages(human.messages, current_day, human.rng)
    human.messages = []

    human.clusters.update_records(human.update_messages, human)
    human.update_messages = []
    human.clusters.purge(current_day)

    # save an output training example
    is_exposed, exposure_day = human.exposure_array(todays_date)
    is_recovered, recovery_day = human.recovered_array(todays_date)
    candidate_encounters, exposure_encounter = candidate_exposures(human, todays_date)
    reported_symptoms = symptoms_to_np(human.all_reported_symptoms, all_possible_symptoms)
    true_symptoms = symptoms_to_np(human.all_symptoms, all_possible_symptoms)
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

    # TODO: read in the correct risk model here
    RiskModel = RiskModelTristan
    human.start_risk = human.risk

    # check if you have new reported symptoms
    human.risk = RiskModel.update_risk_daily(human, todays_date)

    # update risk based on that day's messages
    RiskModel.update_risk_encounters(human, human.messages)

    if config.COLLECT_LOGS:
        if not os.path.isdir(log_path):
            pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(log_path, f"daily_human.pkl")
        log_file = open(path, 'wb')
        pickle.dump(daily_output, log_file)
    return (human.name, human.risk, human.clusters)


def get_days_worth_of_logs(data_path, start, cur_day, start_pkl):
    to_return = defaultdict(list)
    started = False
    try:
        with zipfile.ZipFile(data_path, 'r') as zf:
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
    except Exception:
        pass
    return to_return, start_pkl

def integrated_risk_pred(humans, data_path, start, current_day, all_possible_symptoms, start_pkl, n_jobs=1):
    if config.RISK_MODEL:
        RiskModel = RiskModelTristan

    # check that the plot_dir exists:
    if config.PLOT_RISK and not os.path.isdir(config.RISK_PLOT_PATH):
        os.mkdir(config.RISK_PLOT_PATH)

    days_logs, start_pkl = get_days_worth_of_logs(data_path + ".zip", start, current_day, start_pkl)
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
        all_params.append({"start": start, "current_day": current_day, "encounters": encounters,
                           "all_possible_symptoms": all_possible_symptoms, "human": human,
                           "save_training_data": True, "log_path": log_path,
                           "random_clusters": False})
        human.uid = update_uid(human.uid, human.rng)
    with Parallel(n_jobs=n_jobs, batch_size=config.MP_BATCHSIZE, backend=config.MP_BACKEND, verbose=10) as parallel:
        results = parallel((delayed(proc_human)(params) for params in all_params))

    for name, risk, clusters in results:
        hd[name].risk = risk
        hd[name].clusters = clusters

    if config.PLOT_RISK and config.COLLECT_LOGS:
        daily_risks = [(np.e ** human.risk, human.is_infectious, human.name) for human in hd.values()]
        hist_plot(daily_risks, f"{config.RISK_PLOT_PATH}day_{str(current_day).zfill(3)}.png")

    # print out the clusters
    if config.DUMP_CLUSTERS and config.COLLECT_LOGS:
        clusters = []
        for human in hd.values():
            clusters.append(dict(human.clusters.clusters))
        json.dump(clusters, open(config.CLUSTER_PATH, 'w'))

    return humans, start_pkl
