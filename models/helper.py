import numpy as np
import datetime
from collections import Counter
from models.utils import Message, decode_message

def messages_to_np(human):
    ms_enc = np.zeros((human.clusters.num_messages, 3))
    idx = 0
    for assignment, messages in human.clusters:
        for message in messages:
            obs_uid, risk, day, unobs_uid = decode_message(message)
            message = Message(obs_uid, risk, day, unobs_uid)
            m_enc = np.array([assignment, message.risk, day])
            ms_enc[idx] = m_enc
            idx += 1
    return ms_enc

def candidate_exposures(human, date):
    candidate_locs = list(human.locations_visited.keys())
    exposed_locs = np.zeros(len(candidate_locs))
    if human.exposure_source in candidate_locs:
        exposed_locs[candidate_locs.index(human.exposure_source)] = 1.
    candidate_encounters = list(messages_to_np(human))
    exposed_encounters = np.zeros(len(candidate_encounters))
    if human.exposure_message and human.exposure_message in human.clusters.all_messages:
        idx = human.clusters.all_messages.index(human.exposure_message)
        exposed_encounters[idx] = 1.
    return np.array(candidate_encounters), exposed_encounters, candidate_locs, exposed_locs


def symptoms_to_np(symptoms_day, all_symptoms, all_possible_symptoms):
    rolling_window = 14
    aps = list(all_possible_symptoms)
    symptoms_enc = np.zeros((rolling_window, len(all_possible_symptoms)+1))
    for day, symptoms in enumerate(all_symptoms[:14]):
        for symptom in symptoms:
            symptoms_enc[day, aps.index(symptom)] = 1.
    return symptoms_enc

def group_to_majority_id(all_groups):
    all_new_groups = []
    for group_idx, groups in enumerate(all_groups):
        new_groups = {}
        for group, uids in groups.items():
            cnt = Counter()
            for idx, uid in enumerate(uids):
                cnt[uid] += 1
            for i in range(len(cnt)):
                new_groups[cnt.most_common()[i][0]] = uids
                break
        all_new_groups.append(new_groups)
    return all_new_groups

def rolling_infectiousness(start, date, human):
    rolling_window = 14
    rolling = np.zeros(rolling_window)
    if human.infectiousness_start_time == datetime.datetime.max:
        return rolling
    cur_day = (date - start).days

    hinf = []
    for v in human.infectiousness.values():
        if type(v) == float:
            hinf.append(v)
        elif type(v) == np.ndarray:
            hinf.append(v[0])

    if not human.infectiousness:
        return rolling

    rollings = []
    for end in range(1, len(hinf) + 1):
        if end - rolling_window > 0:
            start = end - rolling_window
            rolling = np.flip(hinf[start:end])
        else:
            rolling = np.flip(hinf[:end])
        rolling = np.pad(rolling, (0, rolling_window - len(rolling)))
        rollings.append(rolling)
    human.rolling_infectiousness_array = rollings

    try:
        return rollings[cur_day]
    except Exception:
        return rolling
