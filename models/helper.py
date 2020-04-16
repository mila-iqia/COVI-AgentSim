import numpy as np
from utils import _decode_message
from collections import Counter

def messages_to_np(human):
    ms_enc = np.zeros((len(human.M), 3))
    idx = 0
    for m_enc, assignment in human.M.items():
        obs_uid, risk, day, unobs_uid = _decode_message(m_enc)
        message = human.Message(obs_uid, risk, day, unobs_uid)

        m_enc = np.array([assignment, message.risk, day])
        ms_enc[idx] = m_enc
        idx += 1
    return ms_enc

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
