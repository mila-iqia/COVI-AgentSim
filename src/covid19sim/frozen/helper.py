import numpy as np

from covid19sim.frozen.clustering.base import ClusterManagerBase

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

# Index SYMPTOMS_META by ID
SYMPTOMS_META_IDMAP = [""] * len(SYMPTOMS_META)
for k, v in SYMPTOMS_META.items():
    SYMPTOMS_META_IDMAP[v] = k

def exposure_array(human_infection_timestamp, date, conf):
    # identical to human.exposure_array
    exposed = False
    exposure_day = None
    if human_infection_timestamp:
        exposure_day = (date - human_infection_timestamp).days
        if exposure_day >= 0 and exposure_day < conf.get("TRACING_N_DAYS_HISTORY"):
            exposed = True
        else:
            exposure_day = None
    return exposed, exposure_day


def recovered_array(human_recovered_timestamp, date, conf):
    # identical to human.recovered_array
    is_recovered = False
    recovery_day = (date - human_recovered_timestamp).days
    if recovery_day >= 0 and recovery_day < conf.get("TRACING_N_DAYS_HISTORY"):
        is_recovered = True
    else:
        recovery_day = None
    return is_recovered, recovery_day


def get_test_result_array(test_results, date, conf):
    # identical to human.get_test_result_array
    results = np.zeros(conf.get("TRACING_N_DAYS_HISTORY"))
    for test_result, test_time in test_results:
        result_day = (date - test_time).days
        if result_day >= 0 and result_day < conf.get("TRACING_N_DAYS_HISTORY"):
            results[result_day] = test_result
    return results


# TODO: negative should probably return 0 and None return -1 to be consistent with encoded age and sex
def encode_test_result(test_result):
    if test_result is None:
        return 0
    if test_result.lower() == 'positive':
        return 1
    elif test_result.lower() == 'negative':
        return -1


def messages_to_np(cluster_mgr):
    return cluster_mgr.get_embeddings_array()


def candidate_exposures(cluster_mgr):
    candidate_encounters = messages_to_np(cluster_mgr)
    assert isinstance(cluster_mgr, ClusterManagerBase)
    exposed_encounters = cluster_mgr._get_expositions_array()
    return candidate_encounters, exposed_encounters


def conditions_to_np(conditions):
    conditions_encs = np.zeros((len(PREEXISTING_CONDITIONS_META),))
    for condition in conditions:
        conditions_encs[PREEXISTING_CONDITIONS_META[condition]] = 1
    return conditions_encs


def symptoms_to_np(all_symptoms, conf):
    rolling_window = conf.get("TRACING_N_DAYS_HISTORY")
    symptoms_enc = np.zeros((rolling_window, len(SYMPTOMS_META) + 1))
    for day, symptoms in zip(range(rolling_window), all_symptoms):
        for symptom in symptoms:
            symptoms_enc[day, SYMPTOMS_META[symptom]] = 1.
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
