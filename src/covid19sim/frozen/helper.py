import numpy as np

from covid19sim.frozen.clustering.base import ClusterManagerBase
from covid19sim.frozen.utils import decode_message, convert_message_to_new_format

# NOTE: THIS MAP SHOULD ALWAYS MATCH THE NAME/IDS PROVIDED IN utils.py
PREEXISTING_CONDITIONS_META = {
    'smoker': 5,
    'diabetes': 1,
    'heart_disease': 2,
    'cancer': 6,
    'COPD': 3,
    'asthma': 4,
    'stroke': 7,
    'immuno-suppressed': 0,
    'lung_disease': 8,
    'pregnant': 9,
}

# NOTE: THIS MAP SHOULD ALWAYS MATCH THE NAME/IDS PROVIDED IN utils.py
SYMPTOMS_META = {
    'mild': 1,
    'moderate': 0,
    'severe': 2,
    'extremely-severe': 3,
    'fever': 4,
    'chills': 5,
    'gastro': 6,
    'diarrhea': 7,
    'nausea_vomiting': 8,
    'fatigue': 9,
    'unusual': 10,
    'hard_time_waking_up': 11,
    'headache': 12,
    'confused': 13,
    'lost_consciousness': 14,
    'trouble_breathing': 15,
    'sneezing': 16,
    'cough': 17,
    'runny_nose': 18,
    'sore_throat': 20,
    'severe_chest_pain': 21,
    'light_trouble_breathing': 24,
    'mild_trouble_breathing': 23,
    'moderate_trouble_breathing': 25,
    'heavy_trouble_breathing': 26,
    'loss_of_taste': 22,
    'aches': 19
}


def exposure_array(human_infection_timestamp, date):
    # identical to human.exposure_array
    exposed = False
    exposure_day = None
    if human_infection_timestamp:
        exposure_day = (date - human_infection_timestamp).days
        if exposure_day >= 0 and exposure_day < 14:
            exposed = True
        else:
            exposure_day = None
    return exposed, exposure_day


def recovered_array(human_recovered_timestamp, date):
    # identical to human.recovered_array
    is_recovered = False
    recovery_day = (date - human_recovered_timestamp).days
    if recovery_day >= 0 and recovery_day < 14:
        is_recovered = True
    else:
        recovery_day = None
    return is_recovered, recovery_day


def get_test_result_array(human_test_time, date):
    # identical to human.get_test_result_array
    results = np.zeros(14)
    result_day = (date - human_test_time).days
    if result_day >= 0 and result_day < 14:
        # TODO: add if human test results "negative" -1
        results[result_day] = 1
    return results


def messages_to_np(human):
    if isinstance(human["clusters"], ClusterManagerBase):
        return human["clusters"].get_embeddings_array()
    else:
        ms_enc = []
        for day, clusters in human["clusters"].clusters_by_day.items():
            for cluster_id, messages in clusters.items():
                # TODO: take an average over the risks for that day
                if not any(messages):
                    continue
                ms_enc.append([cluster_id, decode_message(messages[0]).risk, len(messages), day])
        return np.array(ms_enc)


def candidate_exposures(human, date):
    candidate_encounters = messages_to_np(human)
    if isinstance(human["clusters"], ClusterManagerBase):
        exposed_encounters = human["clusters"]._get_expositions_array()
    else:
        exposed_encounters = np.zeros(len(candidate_encounters))
        if human["exposure_message"]:
            idx = 0
            for day, clusters in human["clusters"].clusters_by_day.items():
                for cluster_id, messages in clusters.items():
                    for message in messages:
                        if message == human["exposure_message"]:
                            exposed_encounters[idx] = 1.
                            break
                    if sum(exposed_encounters) == 1:
                        break
                    if any(messages):
                        idx += 1
                if sum(exposed_encounters) == 1:
                    break

    return candidate_encounters, exposed_encounters


def conditions_to_np(conditions):
    conditions_encs = np.zeros((len(PREEXISTING_CONDITIONS_META),))
    for condition in conditions:
        conditions_encs[PREEXISTING_CONDITIONS_META[condition]] = 1
    return conditions_encs


def symptoms_to_np(all_symptoms, all_possible_symptoms):
    rolling_window = 14
    symptoms_enc = np.zeros((rolling_window, len(all_possible_symptoms) + 1))
    for day, symptoms in zip(range(rolling_window), all_symptoms):
        for symptom in symptoms:
            symptoms_enc[day, all_possible_symptoms.index(symptom)] = 1.
    return symptoms_enc


def encode_age(age):
    if age is None:
        return -1
    else:
        return age


def encode_sex(sex):
    if not sex:
        return -1
    sex = sex.lower()
    if sex.startswith('f'):
        return 1
    elif sex.startswith('m'):
        return 2
    else:
        return 0
