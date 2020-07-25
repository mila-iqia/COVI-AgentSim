import numpy as np

from covid19sim.epidemiology.symptoms import SYMPTOMS
from covid19sim.inference.clustering.base import ClusterManagerBase

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
    'allergies': 10,
}

def exposure_array(human_infection_timestamp, date, conf):
    # identical to human.exposure_array
    # TODO FIXME: THIS IS NOT AN "EXPOSURE DAY", THIS IS A TIME SINCE EXPOSURE. TERRIBLE
    #   VARIABLE NAMES NEED TO BE UPDATED.
    exposed = False
    exposure_day = None
    if human_infection_timestamp:
        exposure_day = (date - human_infection_timestamp).days
        # TODO FIXME: WHY ARE WE THRESHOLDING WITH N_DAYS_HISTORY? WE IF WANT TO DETERMINE
        #   WHETHER THE USER IS STILL INFECTIOUS OR HAS RECOVERED, WE SHOULD USE 'RECOVERY_DAYS'
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
    symptoms_enc = np.zeros((rolling_window, len(SYMPTOMS)))
    for day, symptoms in zip(range(rolling_window), all_symptoms):
        for symptom in symptoms:
            symptoms_enc[day, symptom.id] = 1.
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
