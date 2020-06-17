"""
[summary]
"""
import datetime
import math
import os
import pathlib
import shutil
import subprocess
import time
import typing
import zipfile
from collections import OrderedDict, namedtuple
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import time
import dill
import numpy as np
import requests
import yaml
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from scipy.stats import gamma, norm, truncnorm
from scipy.optimize import linprog

P_SEVERE_ALLERGIES = 0.02

SymptomProbability = namedtuple('SymptomProbability', ['name', 'id', 'probabilities'])
SymptomProbability.__doc__ = '''A symptom probabilities collection given contexts

Attributes
    ----------
    name : str
        name of the symptom
    id : positive int
        id of the symptom
        This attribute should never change once set. It is used to define the
        position of the symptom in a multi-hot encoding
    probabilities : dict
        probabilities of the symptom per context
        A probability of `-1` is assigned when it is heavily variable given
        multiple factors and is handled entirely in the code
        A probability of `None` is assigned when the symptom can be skipped
        entirely in the context. The context can also be removed from the dict
'''
ConditionProbability = namedtuple('ConditionProbability', ['name', 'id', 'age', 'sex', 'probability'])
ConditionProbability.__doc__ = '''A pre-condition probability given an age and sex

Attributes
    ----------
    name : str
        name of the condition
    id : positive int
        id of the condition
        This attribute should never change once set. It is used to define the
        position of the condition in a multi-hot encoding
    age : int
        exclusive maximum age for which this probability is effective
        An age of `1000` is assigned when no check on age is needed
        An age of `-1` is assigned when it is handled entirely in the code
    sex : char
        single lower case char representing the sex for which this probability
        is effective. Possible values are: `'f'`, `'m'`, `'a'`
        An `'f'` sex is assigned when the probability is related to females
        An `'m'` sex is assigned when the probability is related to males
        An `'a'` sex is assigned when no check on sex is needed
    probability : float
        probability of the condition
        A probability of `-1` is assigned when it is handled entirely in the code
'''


# FIXME: covid_pre_plateau should be covid_incubation and there should be no symptoms in this phase
SYMPTOMS_CONTEXTS = {'covid': {0: 'covid_incubation', 1: 'covid_onset', 2: 'covid_plateau',
                               3: 'covid_post_plateau_1', 4: 'covid_post_plateau_2'},
                     'allergy': {0: 'allergy'},
                     'cold': {0: 'cold', 1: 'cold_last_day'},
                     'flu': {0: 'flu_first_day', 1: 'flu', 2: 'flu_last_day'}}

SYMPTOMS = OrderedDict([
    # Sickness severity
    # A lot of symptoms are dependent on the sickness severity so severity
    # level needs to be first
    (
        'mild',
        SymptomProbability('mild', 1, {'covid_incubation': 0.0,
                                       'covid_onset': -1,
                                       'covid_plateau': -1,
                                       'covid_post_plateau_1': -1,
                                       'covid_post_plateau_2': -1,
                                       'cold': -1,
                                       'cold_last_day': 1.0,
                                       'flu_first_day': 1.0,
                                       'flu': -1,
                                       'flu_last_day': 1.0})
    ),
    (
        'moderate',
        SymptomProbability('moderate', 0, {'covid_incubation': 0.0,
                                           'covid_onset': -1,
                                           'covid_plateau': -1,
                                           'covid_post_plateau_1': -1,
                                           'covid_post_plateau_2': -1,
                                           'cold': -1,
                                           'cold_last_day': 0.0,
                                           'flu_first_day': 0.0,
                                           'flu': -1,
                                           'flu_last_day': 0.0})
    ),
    (
        'severe',
        SymptomProbability('severe', 2, {'covid_incubation': 0.0,
                                         'covid_onset': 0.0,
                                         'covid_plateau': -1,
                                         'covid_post_plateau_1': -1,
                                         'covid_post_plateau_2': 0.0})
    ),
    (
        'extremely-severe',
        SymptomProbability('extremely-severe', 3, {'covid_incubation': 0.0,
                                                   'covid_onset': 0.0,
                                                   'covid_plateau': -1,
                                                   'covid_post_plateau_1': 0.0,
                                                   'covid_post_plateau_2': 0.0})
    ),

    # Symptoms

    (
        'fever',
        SymptomProbability('fever', 4, {'covid_incubation': 0.0,
                                        'covid_onset': 0.2,
                                        'covid_plateau': 0.8,
                                        'covid_post_plateau_1': 0.0,
                                        'covid_post_plateau_2': 0.0,
                                        'flu_first_day': 0.7,
                                        'flu': 0.7,
                                        'flu_last_day': 0.3})
    ),
    # 'fever' is a dependency of 'chills' so it needs to be inserted before
    # this position
    (
        'chills',
        SymptomProbability('chills', 5, {'covid_incubation': 0.0,
                                         'covid_onset': 0.8,
                                         'covid_plateau': 0.5,
                                         'covid_post_plateau_1': 0.0,
                                         'covid_post_plateau_2': 0.0})
    ),

    (
        'gastro',
        SymptomProbability('gastro', 6, {'covid_incubation': 0.0,
                                         'covid_onset': -1,
                                         'covid_plateau': -1,
                                         'covid_post_plateau_1': -1,
                                         'covid_post_plateau_2': -1,
                                         'flu_first_day': 0.7,
                                         'flu': 0.7,
                                         'flu_last_day': 0.2})
    ),
    # 'gastro' is a dependency of 'diarrhea' so it needs to be inserted before
    # this position
    (
        'diarrhea',
        SymptomProbability('diarrhea', 7, {'covid_incubation': 0.0,
                                           'covid_onset': 0.9,
                                           'covid_plateau': 0.9,
                                           'covid_post_plateau_1': 0.9,
                                           'covid_post_plateau_2': 0.9,
                                           'flu_first_day': 0.5,
                                           'flu': 0.5,
                                           'flu_last_day': 0.5})
    ),
    # 'gastro' is a dependency of 'nausea_vomiting' so it needs to be inserted
    # before this position
    (
        'nausea_vomiting',
        SymptomProbability('nausea_vomiting', 8, {'covid_incubation': 0.0,
                                                  'covid_onset': 0.7,
                                                  'covid_plateau': 0.7,
                                                  'covid_post_plateau_1': 0.7,
                                                  'covid_post_plateau_2': 0.7,
                                                  'flu_first_day': 0.5,
                                                  'flu': 0.5,
                                                  'flu_last_day': 0.25})
    ),

    # Age based lethargies
    # 'gastro' is a dependency of so it needs to be inserted before this
    # position
    (
        'fatigue',
        SymptomProbability('fatigue', 9, {'covid_incubation': 0.0,
                                          'covid_onset': -1,
                                          'covid_plateau': -1,
                                          'covid_post_plateau_1': -1,
                                          'covid_post_plateau_2': -1,
                                          'allergy': 0.2,
                                          'cold': 0.8,
                                          'cold_last_day': 0.8,
                                          'flu_first_day': 0.4,
                                          'flu': 0.8,
                                          'flu_last_day': 0.8})
    ),
    (
        'unusual',
        SymptomProbability('unusual', 10, {'covid_incubation': 0.0,
                                           'covid_onset': 0.2,
                                           'covid_plateau': 0.5,
                                           'covid_post_plateau_1': 0.5,
                                           'covid_post_plateau_2': 0.5})
    ),
    (
        'hard_time_waking_up',
        SymptomProbability('hard_time_waking_up', 11, {'covid_incubation': 0.0,
                                                       'covid_onset': 0.6,
                                                       'covid_plateau': 0.6,
                                                       'covid_post_plateau_1': 0.6,
                                                       'covid_post_plateau_2': 0.6,
                                                       'allergy': 0.3,
                                                       'flu_first_day': 0.3,
                                                       'flu': 0.5,
                                                       'flu_last_day': 0.4})
    ),
    (
        'headache',
        SymptomProbability('headache', 12, {'covid_incubation': 0.0,
                                            'covid_onset': 0.5,
                                            'covid_plateau': 0.5,
                                            'covid_post_plateau_1': 0.5,
                                            'covid_post_plateau_2': 0.5,
                                            'allergy': 0.6})
    ),
    (
        'confused',
        SymptomProbability('confused', 13, {'covid_incubation': 0.0,
                                            'covid_onset': 0.1,
                                            'covid_plateau': 0.1,
                                            'covid_post_plateau_1': 0.1,
                                            'covid_post_plateau_2': 0.1})
    ),
    (
        'lost_consciousness',
        SymptomProbability('lost_consciousness', 14, {'covid_incubation': 0.0,
                                                      'covid_onset': 0.1,
                                                      'covid_plateau': 0.1,
                                                      'covid_post_plateau_1': 0.1,
                                                      'covid_post_plateau_2': 0.1})
    ),

    # Respiratory symptoms
    # 'trouble_breathing' is a dependency of all this category so it should be
    # inserted before them
    (
        'trouble_breathing',
        SymptomProbability('trouble_breathing', 15, {'covid_incubation': 0.0,
                                                     'covid_onset': -1,
                                                     'covid_plateau': -1,
                                                     'covid_post_plateau_1': -1,
                                                     'covid_post_plateau_2': -1})
    ),
    (
        'sneezing',
        SymptomProbability('sneezing', 16, {'covid_incubation': 0.0,
                                            'covid_onset': 0.2,
                                            'covid_plateau': 0.3,
                                            'covid_post_plateau_1': 0.3,
                                            'covid_post_plateau_2': 0.3,
                                            'allergy': 1.0,
                                            'cold': 0.4,
                                            'cold_last_day': 0.0})
    ),
    (
        'cough',
        SymptomProbability('cough', 17, {'covid_incubation': 0.0,
                                         'covid_onset': 0.6,
                                         'covid_plateau': 0.9,
                                         'covid_post_plateau_1': 0.9,
                                         'covid_post_plateau_2': 0.9,
                                         'cold': 0.8,
                                         'cold_last_day': 0.8})
    ),
    (
        'runny_nose',
        SymptomProbability('runny_nose', 18, {'covid_incubation': 0.0,
                                              'covid_onset': 0.1,
                                              'covid_plateau': 0.2,
                                              'covid_post_plateau_1': 0.2,
                                              'covid_post_plateau_2': 0.2,
                                              'cold': 0.8,
                                              'cold_last_day': 0.8})
    ),
    (
        'sore_throat',
        SymptomProbability('sore_throat', 20, {'covid_incubation': 0.0,
                                               'covid_onset': 0.5,
                                               'covid_plateau': 0.8,
                                               'covid_post_plateau_1': 0.8,
                                               'covid_post_plateau_2': 0.8,
                                               'allergy': 0.3,
                                               'cold': 0.0,
                                               'cold_last_day': 0.6})
    ),
    (
        'severe_chest_pain',
        SymptomProbability('severe_chest_pain', 21, {'covid_incubation': 0.0,
                                                     'covid_onset': 0.4,
                                                     'covid_plateau': 0.5,
                                                     'covid_post_plateau_1': 0.15,
                                                     'covid_post_plateau_2': 0.15})
    ),

    # 'trouble_breathing' is a dependency of any '*_trouble_breathing' so it
    # needs to be inserted before this position
    (
        'light_trouble_breathing',
        SymptomProbability('light_trouble_breathing', 24, {'covid_incubation': 0.0,
                                                           'covid_onset': -1,
                                                           'covid_plateau': -1,
                                                           'covid_post_plateau_1': -1,
                                                           'covid_post_plateau_2': -1})
    ),
    (
        'mild_trouble_breathing',
        SymptomProbability('mild_trouble_breathing', 23, {'allergy': P_SEVERE_ALLERGIES})
    ),
    (
        'moderate_trouble_breathing',
        SymptomProbability('moderate_trouble_breathing', 25, {'covid_incubation': 0.0,
                                                              'covid_onset': -1,
                                                              'covid_plateau': -1,
                                                              'covid_post_plateau_1': -1,
                                                              'covid_post_plateau_2': -1})
    ),
    (
        'heavy_trouble_breathing',
        SymptomProbability('heavy_trouble_breathing', 26, {'covid_incubation': 0.0,
                                                           'covid_onset': 0,
                                                           'covid_plateau': -1,
                                                           'covid_post_plateau_1': -1,
                                                           'covid_post_plateau_2': -1})
    ),

    (
        'loss_of_taste',
        SymptomProbability('loss_of_taste', 22, {'covid_incubation': 0.0,
                                                 'covid_onset': 0.25,
                                                 'covid_plateau': 0.35,
                                                 'covid_post_plateau_1': 0.0,
                                                 'covid_post_plateau_2': 0.0})
    ),

    (
        'aches',
        SymptomProbability('aches', 19, {'flu_first_day': 0.3,
                                         'flu': 0.5,
                                         'flu_last_day': 0.8})
    )

    # commented out because these are not used elsewhere for now
    # (
    #     'hives',
    #     SymptomProbability('hives', __, {'allergy': 0.4})
    # ),
    # (
    #     'swelling',
    #     SymptomProbability('swelling', __, {'allergy': 0.3})
    # )
])

# NOTE: THE PREEXISTING CONDITION NAMES/IDs BELOW MUST MATCH THOSE IN frozen/helper.py

PREEXISTING_CONDITIONS = OrderedDict([
    ('smoker', [
        ConditionProbability('smoker', 5, 12, 'a', 0.0),
        ConditionProbability('smoker', 5, 18, 'a', 0.03),
        ConditionProbability('smoker', 5, 65, 'a', 0.185),
        ConditionProbability('smoker', 5, 1000, 'a', 0.09)
    ]),
    ('diabetes', [
        ConditionProbability('diabetes', 1, 18, 'a', 0.005),
        ConditionProbability('diabetes', 1, 35, 'a', 0.009),
        ConditionProbability('diabetes', 1, 50, 'a', 0.039),
        ConditionProbability('diabetes', 1, 75, 'a', 0.13),
        ConditionProbability('diabetes', 1, 1000, 'a', 0.179)
    ]),
    # 'smoker' and 'diabetes' are dependencies of 'heart_disease' so they
    # need to be inserted before this position
    ('heart_disease', [
        ConditionProbability('heart_disease', 2, 20, 'a', 0.001),
        ConditionProbability('heart_disease', 2, 35, 'a', 0.005),
        ConditionProbability('heart_disease', 2, 50, 'f', 0.013),
        ConditionProbability('heart_disease', 2, 50, 'm', 0.021),
        ConditionProbability('heart_disease', 2, 50, 'a', 0.017),
        ConditionProbability('heart_disease', 2, 75, 'f', 0.13),
        ConditionProbability('heart_disease', 2, 75, 'm', 0.178),
        ConditionProbability('heart_disease', 2, 75, 'a', 0.15),
        ConditionProbability('heart_disease', 2, 100, 'f', 0.311),
        ConditionProbability('heart_disease', 2, 100, 'm', 0.44),
        ConditionProbability('heart_disease', 2, 1000, 'a', 0.375)
    ]),
    # 'smoker' is a dependency of 'cancer' so it needs to be inserted
    # before this position
    ('cancer', [
        ConditionProbability('cancer', 6, 30, 'a', 0.00029),
        ConditionProbability('cancer', 6, 60, 'a', 0.0029),
        ConditionProbability('cancer', 6, 90, 'a', 0.029),
        ConditionProbability('cancer', 6, 1000, 'a', 0.05)
    ]),
    # 'smoker' is a dependency of 'COPD' so it needs to be inserted
    # before this position
    ('COPD', [
        ConditionProbability('COPD', 3, 35, 'a', 0.0),
        ConditionProbability('COPD', 3, 50, 'a', 0.015),
        ConditionProbability('COPD', 3, 65, 'f', 0.037),
        ConditionProbability('COPD', 3, 1000, 'a', 0.075)
    ]),
    ('asthma', [
        ConditionProbability('asthma', 4, 10, 'f', 0.07),
        ConditionProbability('asthma', 4, 10, 'm', 0.12),
        ConditionProbability('asthma', 4, 10, 'a', 0.09),
        ConditionProbability('asthma', 4, 25, 'f', 0.15),
        ConditionProbability('asthma', 4, 25, 'm', 0.19),
        ConditionProbability('asthma', 4, 25, 'a', 0.17),
        ConditionProbability('asthma', 4, 75, 'f', 0.11),
        ConditionProbability('asthma', 4, 75, 'm', 0.06),
        ConditionProbability('asthma', 4, 75, 'a', 0.08),
        ConditionProbability('asthma', 4, 1000, 'f', 0.12),
        ConditionProbability('asthma', 4, 1000, 'm', 0.08),
        ConditionProbability('asthma', 4, 1000, 'a', 0.1)
    ]),
    # All conditions above are dependencies of 'stroke' so they need to be
    # inserted before this position
    ('stroke', [
        ConditionProbability('stroke', 7, 20, 'a', 0.0),
        ConditionProbability('stroke', 7, 40, 'a', 0.01),
        ConditionProbability('stroke', 7, 60, 'a', 0.03),
        ConditionProbability('stroke', 7, 80, 'a', 0.04),
        ConditionProbability('stroke', 7, 1000, 'a', 0.07)
    ]),
    # 'cancer' is a dependency of 'immuno-suppressed' so it needs to be inserted
    # before this position
    ('immuno-suppressed', [  # (3.6% on average)
        ConditionProbability('immuno-suppressed', 0, 40, 'a', 0.005),
        ConditionProbability('immuno-suppressed', 0, 65, 'a', 0.036),
        ConditionProbability('immuno-suppressed', 0, 85, 'a', 0.045),
        ConditionProbability('immuno-suppressed', 0, 1000, 'a', 0.20)
    ]),
    ('lung_disease', [
        ConditionProbability('lung_disease', 8, -1, 'a', -1)
    ]),
    ('pregnant', [
        ConditionProbability('pregnant', 9, -1, 'f', -1)
    ])
])


def log(str, logfile=None, timestamp=False):
    """
    [summary]

    Args:
        str ([type]): [description]
        logfile ([type], optional): [description]. Defaults to None.
        timestamp (bool, optional): [description]. Defaults to False.
    """
    if timestamp:
        str = f"[{datetime.datetime.now()}] {str}"

    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def _sample_viral_load_gamma(rng, shape_mean=4.5, shape_std=.15, scale_mean=1., scale_std=.15):
    """
    This function samples the shape and scale of a gamma distribution, then returns it

    Args:
        rng ([type]): [description]
        shape_mean (float, optional): [description]. Defaults to 4.5.
        shape_std (float, optional): [description]. Defaults to .15.
        scale_mean ([type], optional): [description]. Defaults to 1..
        scale_std (float, optional): [description]. Defaults to .15.

    Returns:
        [type]: [description]
    """
    shape = rng.normal(shape_mean, shape_std)
    scale = rng.normal(scale_mean, scale_std)
    return gamma(shape, scale=scale)

def _get_inflammatory_disease_level(rng, preexisting_conditions, inflammatory_conditions):
    cond_count = 0
    for cond in inflammatory_conditions:
        if cond in preexisting_conditions:
          cond_count += 1
    if cond_count > 3:
        cond_count = 3
    return cond_count

def _get_disease_days(rng, conf, age, inflammatory_disease_level):
    """
    Defines viral load curve parameters.
    It is based on the study here https://www.medrxiv.org/content/10.1101/2020.04.10.20061325v2.full.pdf (Figure 1).

    We have used the same scale for the gamma distribution for all the parameters as fitted in the study here
        https://www.acpjournals.org/doi/10.7326/M20-0504 (Appendix Table 2)

    NOTE: Using gamma for all paramters is for the ease of computation.
    NOTE: Gamma distribution is only well supported in literature for incubation days

    Args:
        rng (np.random.RandomState): random number generator
        conf (dict): configuration dictionary
        age (float): age of human
        inflammatory_disease_level (int): based on count of inflammatory conditions.
    """
    # NOTE: references are in core.yaml alongside above parameters
    # All days count FROM EXPOSURE i.e. infection_timestamp

    PLATEAU_DURATION_CLIP_HIGH = conf.get("PLATEAU_DURATION_CLIP_HIGH")
    PLATEAU_DURATION_CLIP_LOW = conf.get("PLATEAU_DURATION_CLIP_LOW")
    PLATEAU_DURATION_MEAN = conf.get("PLATEAU_DURATION_MEAN")
    PLATEAU_DURATION_STD = conf.get("PLATEAU_DURATION_STD")

    INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG = conf.get("INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET_AVG")
    INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_STD = conf.get("INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET_STD")
    INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_CLIP_LOW = conf.get("INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET_CLIP_LOW")
    INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_CLIP_HIGH = conf.get("INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET_CLIP_HIGH")

    INFECTIOUSNESS_PEAK_AVG = conf.get("INFECTIOUSNESS_PEAK_AVG")
    INFECTIOUSNESS_PEAK_STD = conf.get("INFECTIOUSNESS_PEAK_STD")
    INFECTIOUSNESS_PEAK_CLIP_HIGH = conf.get("INFECTIOUSNESS_PEAK_CLIP_HIGH")
    INFECTIOUSNESS_PEAK_CLIP_LOW = conf.get("INFECTIOUSNESS_PEAK_CLIP_LOW")

    RECOVERY_DAYS_AVG = conf.get("RECOVERY_DAYS_AVG")
    RECOVERY_STD = conf.get("RECOVERY_STD")
    RECOVERY_CLIP_LOW = conf.get("RECOVERY_CLIP_LOW")
    RECOVERY_CLIP_HIGH = conf.get("RECOVERY_CLIP_HIGH")

    # days after exposure when symptoms show up
    incubation_days = rng.gamma(
        shape=conf['INCUBATION_DAYS_GAMMA_SHAPE'],
        scale=conf['INCUBATION_DAYS_GAMMA_SCALE']
    )
    # (no-source) assumption is that there is at least two days to remain exposed
    # Comparitively, we set infectiousness_onset_days to be at least one day to remain exposed
    incubation_days = max(2.0, incubation_days)

    # days after exposure when viral shedding starts, i.e., person is infectious
    infectiousness_onset_days = \
        incubation_days - \
        truncnorm((INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_CLIP_LOW - INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG) /
                  INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_STD,
                  (INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_CLIP_HIGH - INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG) /
                  INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_STD,
                  loc=INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG,
                  scale=INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_STD).rvs(1, random_state=rng).item()

    # (no-source) assumption is that there is at least one day to remain exposed
    infectiousness_onset_days = max(1.0, infectiousness_onset_days)

    # viral load peaks INFECTIOUSNESS_PEAK_AVG days before incubation days
    viral_load_peak_wrt_incubation_days = \
        truncnorm((INFECTIOUSNESS_PEAK_CLIP_LOW - INFECTIOUSNESS_PEAK_AVG) /
                  INFECTIOUSNESS_PEAK_STD,
                  (INFECTIOUSNESS_PEAK_CLIP_HIGH - INFECTIOUSNESS_PEAK_AVG) /
                  INFECTIOUSNESS_PEAK_STD,
                  loc=INFECTIOUSNESS_PEAK_AVG,
                  scale=INFECTIOUSNESS_PEAK_STD).rvs(1, random_state=rng).item()

    viral_load_peak = incubation_days - viral_load_peak_wrt_incubation_days

    # (no-source) assumption is that there is at least half a day after the infectiousness_onset_days
    viral_load_peak = max(infectiousness_onset_days + 0.5, viral_load_peak)

    viral_load_peak_wrt_incubation_days = incubation_days - viral_load_peak

    # (no-source) We assume that plateau start is equi-distant from the peak
    # infered from the curves in Figure 1 of the reference above
    plateau_start = incubation_days + viral_load_peak_wrt_incubation_days

    # (no-source) plateau duration is assumed to be of avg PLATEAU_DRATION_MEAN
    plateau_end = \
        plateau_start + \
        truncnorm((PLATEAU_DURATION_CLIP_LOW - PLATEAU_DURATION_MEAN) /
                  PLATEAU_DURATION_STD,
                  (PLATEAU_DURATION_CLIP_HIGH - PLATEAU_DURATION_MEAN) /
                  PLATEAU_DURATION_STD,
                  loc=PLATEAU_DURATION_MEAN,
                  scale=PLATEAU_DURATION_STD).rvs(1, random_state=rng).item()

    # recovery is often quoted with respect to the incubation days
    # so we add it here with respect to the plateau end.
    RECOVERY_WRT_PLATEAU_END_AVG = RECOVERY_DAYS_AVG - PLATEAU_DURATION_MEAN - INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG
    recovery_days = \
        plateau_end + \
        truncnorm((RECOVERY_CLIP_LOW - RECOVERY_WRT_PLATEAU_END_AVG) /
                  RECOVERY_STD,
                  (RECOVERY_CLIP_HIGH - RECOVERY_WRT_PLATEAU_END_AVG) /
                  RECOVERY_STD,
                  loc=RECOVERY_WRT_PLATEAU_END_AVG,
                  scale=RECOVERY_STD).rvs(1, random_state=rng).item()

    # Time to recover is proportional to age
    # based on hospitalization data (biased towards older people) https://pubs.rsna.org/doi/10.1148/radiol.2020200370
    # (no-source) it adds dependency of recovery days on age
    recovery_days += age/40

    # viral load height. There are two parameters here -
    # peak - peak of the viral load curve
    # plateau - plateau of the viral load curve
    # max: 130/200 + 3/3.5 = 2.5, scales the base to [0-1]
    # Older people and those with inflammatory diseases have higher viral load
    # https://www.medrxiv.org/content/10.1101/2020.04.10.20061325v2.full.pdf
    # TODO : make it dependent on initial viral load
    # (no-source) dependence on age vs inflammatory_disease_count
    # base = conf['AGE_FACTOR_VIRAL_LOAD_HEIGHT'] * age/200 + conf['INFLAMMATORY_DISEASE_FACTOR_VIRAL_LOAD_HEIGHT'] * np.exp(-inflammatory_disease_level/3)
    base = 1.0

    # as long as min and max viral load are [0-1], this will be [0-1]
    peak_height = rng.uniform(conf['MIN_VIRAL_LOAD_PEAK_HEIGHT'], conf['MAX_VIRAL_LOAD_PEAK_HEIGHT']) * base

    # as long as min and max viral load are [0-1], this will be [0-1]
    plateau_height = peak_height * rng.uniform(conf['MIN_MULTIPLIER_PLATEAU_HEIGHT'], conf['MAX_MULTIPLIER_PLATEAU_HEIGHT'])

    assert peak_height != 0, f"viral load of peak of 0 sampled age:{age}"
    return infectiousness_onset_days, viral_load_peak, incubation_days, plateau_start, plateau_end, recovery_days, peak_height, plateau_height

def _get_disease_days_v1(rng, conf, age, inflammatory_disease_level):
    """
    Defines viral load curve parameters.
    It is based on the study here https://www.medrxiv.org/content/10.1101/2020.04.10.20061325v2.full.pdf (Figure 1).

    We have used the same scale for the gamma distribution for all the parameters as fitted in the study here
        https://www.acpjournals.org/doi/10.7326/M20-0504 (Appendix Table 2)

    NOTE: Using gamma for all paramters is for the ease of computation.
    NOTE: Gamma distribution is only well supported in literature for incubation days

    Args:
        rng (np.random.RandomState): random number generator
        conf (dict): configuration dictionary
        age (float): age of human
        inflammatory_disease_level (int): based on count of inflammatory conditions.
    """
    # NOTE: references are in core.yaml alongside above parameters
    # All days count FROM EXPOSURE i.e. infection_timestamp
    # days are sampled additively to result in a gamma distribution for known paramters

    # person starts being infectious some days before symptom onset
    INFECTIOUSNESS_ONSET_DAYS_AVG = conf['INCUBATION_DAYS_GAMMA_SHAPE'] - conf['INFECTIOUSNESS_ONSET_DAYS_WRT_INCUBATION_AVG']
    infectiousness_onset_days = rng.gamma(
                                    shape=INFECTIOUSNESS_ONSET_DAYS_AVG,
                                    scale=conf['INCUBATION_DAYS_GAMMA_SCALE']
                                )

    # peak of viral load is reached after that
    VIRAL_LOAD_PEAK_WRT_INFECTIOUSNESS_AVG = conf['INCUBATION_DAYS_GAMMA_SHAPE'] \
                     - INFECTIOUSNESS_ONSET_DAYS_AVG \
                     - conf['INFECTIOUSNESS_PEAK_WRT_INCUBATION_AVG']

    viral_load_peak  = infectiousness_onset_days +  rng.gamma(
                                                shape=VIRAL_LOAD_PEAK_WRT_INFECTIOUSNESS_AVG,
                                                scale=conf['INCUBATION_DAYS_GAMMA_SCALE']
                                                )
    # symptoms start after that; incubation period is from the day of exposure to symptom onset
    # it is modeled additively so that final incubation days is a gamma distribution
    incubation_days = infectiousness_onset_days + viral_load_peak + \
                     + rng.gamma(
                                shape=conf['INFECTIOUSNESS_PEAK_WRT_INCUBATION_AVG'],
                                scale=conf['INCUBATION_DAYS_GAMMA_SCALE']
                        )

    # (no-source) We assume that plateau start is equi-distant from the peak
    # infered from the curves in Figure 1 of the reference above
    plateau_start = incubation_days + (incubation_days - viral_load_peak)

    # (no-source) plateau duration is assumed to be of avg PLATEAU_DRATION_MEAN
    plateau_end = plateau_start + rng.gamma(
                                shape=conf['PLATEAU_DURATION_MEAN'],
                                scale=conf['INCUBATION_DAYS_GAMMA_SCALE']
                    )

    # recovery is often quoted with respect to the incubation days
    # so we add it here with respect to the plateau end. It results in gamma distribution for recovery days.
    # Gamma distribution is an assumption while maintaining the mean values.
    RECOVERY_WRT_PLATEAU_END_AVG = conf['RECOVERY_DAYS_AVG'] - conf['PLATEAU_DURATION_MEAN'] - conf['INFECTIOUSNESS_PEAK_WRT_INCUBATION_AVG']
    recovery_days = plateau_end + rng.gamma(
                                shape=RECOVERY_WRT_PLATEAU_END_AVG,
                                scale=conf['INCUBATION_DAYS_GAMMA_SCALE']
                    )

    # Time to recover is proportional to age
    # based on hospitalization data (biased towards older people) https://pubs.rsna.org/doi/10.1148/radiol.2020200370
    # (no-source) it adds dependency of recovery days on age
    recovery_days += age/40

    # viral load height. There are two parameters here -
    # peak - peak of the viral load curve
    # plateau - plateau of the viral load curve
    # max: 130/200 + 1 + 3/3.5 = 2.5, scales the base to [0-1]
    # Older people and those with inflammatory diseases have higher viral load
    # https://www.medrxiv.org/content/10.1101/2020.04.10.20061325v2.full.pdf
    # TODO : make it dependent on initial viral load
    # (no-source) dependence on age vs inflammatory_disease_count
    base = (age/200 + inflammatory_disease_level/3.5)/1.5

    # as long as min and max viral load are [0-1], this will be [0-1]
    peak_height = rng.uniform(conf['MIN_VIRAL_LOAD_PEAK_HEIGHT'], conf['MAX_VIRAL_LOAD_PEAK_HEIGHT']) * base

    # as long as min and max viral load are [0-1], this will be [0-1]
    plateau_height = peak_height * rng.uniform(conf['MIN_MULTIPLIER_PLATEAU_HEIGHT'], conf['MAX_MULTIPLIER_PLATEAU_HEIGHT'])

    assert peak_height != 0, f"viral load of peak of 0 sampled age:{age}"
    return infectiousness_onset_days, viral_load_peak, incubation_days, plateau_start, plateau_end, recovery_days, peak_height, plateau_height

def _sample_viral_load_piecewise(rng, plateau_start, initial_viral_load=0, age=40, conf={}):
    """
    This function samples a piece-wise linear viral load model which increases, plateaus, and drops.

    Args:
        rng (np.random.RandomState): random number generator
        plateau_start: start of the plateau with respect to infectiousness_onset_days
        initial_viral_load (int, optional): unused
        age (int, optional): age of the person. Defaults to 40.

    Returns:
        plateau_height (float): height of the plateau, i.e., viral load at its peak
        plateau_end (float): days after beign infectious when the plateau ends
        recovered (float): days after being infectious when the viral load is assumed to be ineffective (not necessarily 0)
    """
    # https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
	# https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30196-1/fulltext

    MAX_VIRAL_LOAD = conf.get("MAX_VIRAL_LOAD")
    MIN_VIRAL_LOAD = conf.get("MIN_VIRAL_LOAD")
    PLATEAU_DURATION_CLIP_HIGH = conf.get("PLATEAU_DURATION_CLIP_HIGH")
    PLATEAU_DURATION_CLIP_LOW = conf.get("PLATEAU_DURATION_CLIP_LOW")
    PLATEAU_DURATION_MEAN = conf.get("PLATEAU_DURATION_MEAN")
    PLATEAU_START_CLIP_HIGH = conf.get("PLATEAU_START_CLIP_HIGH")
    PLATEAU_START_CLIP_LOW = conf.get("PLATEAU_START_CLIP_LOW")
    PLATEAU_START_MEAN = conf.get("PLATEAU_START_MEAN")
    PLATEAU_START_STD = conf.get("PLATEAU_START_STD")
    PLATEAU_DURATION_STD = conf.get("PLATEAU_DURATION_STD")
    RECOVERY_CLIP_HIGH = conf.get("RECOVERY_CLIP_HIGH")
    RECOVERY_CLIP_LOW = conf.get("RECOVERY_CLIP_LOW")
    RECOVERY_MEAN = conf.get("RECOVERY_MEAN")
    RECOVERY_STD = conf.get("RECOVERY_STD")
    VIRAL_LOAD_RECOVERY_FACTOR = conf.get("VIRAL_LOAD_RECOVERY_FACTOR")

    # plateau_start = truncnorm((PLATEAU_START_CLIP_LOW - PLATEAU_START_MEAN)/PLATEAU_START_STD, (PLATEAU_START_CLIP_HIGH - PLATEAU_START_MEAN) / PLATEAU_START_STD, loc=PLATEAU_START_MEAN, scale=PLATEAU_START_STD).rvs(1, random_state=rng)
    plateau_end = plateau_start + truncnorm((PLATEAU_DURATION_CLIP_LOW - PLATEAU_DURATION_MEAN)/PLATEAU_DURATION_STD,
                                            (PLATEAU_DURATION_CLIP_HIGH - PLATEAU_DURATION_MEAN) / PLATEAU_DURATION_STD,
                                            loc=PLATEAU_DURATION_MEAN, scale=PLATEAU_DURATION_STD).rvs(1, random_state=rng)

    recovered = plateau_end + ((age/10)-1) # age is a determining factor for the recovery time
    recovered = recovered + initial_viral_load * VIRAL_LOAD_RECOVERY_FACTOR \
                          + truncnorm((RECOVERY_CLIP_LOW - RECOVERY_MEAN) / RECOVERY_STD,
                                        (RECOVERY_CLIP_HIGH - RECOVERY_MEAN) / RECOVERY_STD,
                                        loc=RECOVERY_MEAN, scale=RECOVERY_STD).rvs(1, random_state=rng)

    base = age/200 # peak viral load varies linearly with age
    # plateau_mean =  initial_viral_load - (base + MIN_VIRAL_LOAD) / (base + MIN_VIRAL_LOAD, base + MAX_VIRAL_LOAD) # transform initial viral load into a range
    # plateau_height = rng.normal(plateau_mean, 1)
    plateau_height = rng.uniform(base + MIN_VIRAL_LOAD, base + MAX_VIRAL_LOAD)
    return plateau_height, plateau_end.item(), recovered.item()


def _normalize_scores(scores):
    """
    [summary]

    Args:
        scores ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.array(scores)/np.sum(scores)

# &canadian-demgraphics
def _get_random_age(rng):
    """
    [summary]

    Args:
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    # random normal centered on 50 with stdev 25
    draw = rng.normal(50, 25, 1)
    if draw < 0:
        # if below 0, shift to a bump centred around 30
        age = round(30 + rng.normal(0, 4))
    else:
        age = round(float(draw))
    return age


# &sex
def _get_random_sex(rng):
    """
    This function returns the sex at birth of the person.
    Other is associated with 'prefer not to answer' for the CanStats census.
    TODO: the proportion parameters should be in a config file.

    Args:
        rng (): A random number generator

    Returns:
        [str]: Possible choices of sex {female, male, other}
    """
    p = rng.rand()
    if p < .45:
        return 'female'
    elif p < .90:
        return 'male'
    else:
        return 'other'


def _get_get_really_sick(age, sex, rng):
    """
    [summary]

    Args:
        age ([type]): [description]
        sex ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    if sex.lower().startswith('f'):
        if age < 10:
            return rng.rand() < 0.02
        if age < 20:
            return rng.rand() < 0.002
        if age < 40:
            return rng.rand() < 0.05
        if age < 50:
            return rng.rand() < 0.13
        if age < 60:
            return rng.rand() < 0.18
        if age < 70:
            return rng.rand() < 0.16
        if age < 80:
            return rng.rand() < 0.24
        if age < 90:
            return rng.rand() < 0.17
        else:
            return rng.rand() < 0.03

    elif sex.lower().startswith('m'):
        if age < 10:
            return rng.rand() < 0.002
        if age < 20:
            return rng.rand() < 0.02
        if age < 30:
            return rng.rand() < 0.03
        if age < 40:
            return rng.rand() < 0.07
        if age < 50:
            return rng.rand() < 0.13
        if age < 60:
            return rng.rand() < 0.17
        if age < 80:
            return rng.rand() < 0.22
        if age < 90:
            return rng.rand() < 0.15
        else:
            return rng.rand() < 0.03

    else:
        if age < 20:
            return rng.rand() < 0.02
        if age < 30:
            return rng.rand() < 0.04
        if age < 40:
            return rng.rand() < 0.07
        if age < 50:
            return rng.rand() < 0.13
        if age < 60:
            return rng.rand() < 0.18
        if age < 80:
            return rng.rand() < 0.24
        if age < 90:
            return rng.rand() < 0.18
        else:
            return rng.rand() < 0.03

# 2D Array of symptoms; first axis is days after exposure (infection), second is an array of symptoms
def _get_covid_progression(initial_viral_load, viral_load_plateau_start, viral_load_plateau_end,
                           recovery_days, age, incubation_days, infectiousness_onset_days,
                           really_sick, extremely_sick, rng, preexisting_conditions, carefulness):
    """
    [summary]

    Args:
        initial_viral_load ([type]): [description]
        viral_load_plateau_start ([type]): [description]
        viral_load_plateau_end ([type]): [description]
        recovery_days (float): time to recover
        age ([type]): [description]
        incubation_days ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]
        rng ([type]): [description]
        preexisting_conditions ([type]): [description]
        carefulness ([type]): [description]

    Returns:
        [type]: [description]
    """
    symptoms_contexts = SYMPTOMS_CONTEXTS['covid']
    progression = []
    symptoms_per_phase = [[] for i in range(len(symptoms_contexts))]

    # Phase 0 - Before onset of symptoms (incubation)
    # ====================================================
    phase_i = 0
    symptoms_per_phase[phase_i]= []
    # for day in range(math.ceil(incubation_days)):
    #     progression.append([])

    # Phase 1 of plateau
    # ====================================================
    phase_i = 1
    phase = symptoms_contexts[phase_i]

    p_fever = SYMPTOMS['fever'].probabilities[phase]
    if really_sick or extremely_sick or len(preexisting_conditions)>2 or initial_viral_load > 0.6:
        symptoms_per_phase[phase_i].append('moderate')
        p_fever = 0.4
    else:
        symptoms_per_phase[phase_i].append('mild')

    if rng.rand() < p_fever:
        symptoms_per_phase[phase_i].append('fever')
        if extremely_sick:
            if rng.rand() < 0.8:
                symptoms_per_phase[phase_i].append('chills')

    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later
    p_gastro = initial_viral_load - .15
    if rng.rand() < p_gastro:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    p_lethargy = (age/200) + initial_viral_load*0.6 - carefulness/2
    if rng.rand() < p_lethargy:
        symptoms_per_phase[phase_i].append('fatigue')

        if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('unusual')
        if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
                rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('lost_consciousness')

        for symptom in ('hard_time_waking_up', 'headache', 'confused'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # respiratory symptoms not so common at this stage
    p_respiratory = (0.5 * initial_viral_load) - (carefulness * 0.25) # e.g. 0.5*0.5 - 0.7*0.25 = 0.25-0.17
    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    if rng.rand() < p_respiratory:
        symptoms_per_phase[phase_i].append('trouble_breathing')

        if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('severe_chest_pain')

        for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    if rng.rand() < SYMPTOMS['loss_of_taste'].probabilities[phase]:
        symptoms_per_phase[phase_i].append('loss_of_taste')

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')

    # TODO: Delete me
    # # During the plateau of symptoms Part 1
    # # ====================================================
    # phase_i = 1
    # phase = symptoms_contexts[phase_i]
    #
    # if extremely_sick:
    #     symptoms_per_phase[phase_i].append('extremely-severe')
    # elif really_sick or len(preexisting_conditions) >2 or 'moderate' in symptoms_per_phase[phase_i-1] or \
    #         initial_viral_load > 0.6:
    #     symptoms_per_phase[phase_i].append('severe')
    # elif rng.rand() < p_gastro:
    #     symptoms_per_phase[phase_i].append('moderate')
    # else:
    #     symptoms_per_phase[phase_i].append('mild')
    #
    # if 'fever' in symptoms_per_phase[phase_i-1] or initial_viral_load > 0.8 or \
    #         rng.rand() < SYMPTOMS['fever'].probabilities[phase]:
    #     symptoms_per_phase[phase_i].append('fever')
    #     if rng.rand() < SYMPTOMS['chills'].probabilities[phase]:
    #         symptoms_per_phase[phase_i].append('chills')
    #
    #
    # # gastro symptoms are more likely to be earlier and are more
    # # likely to show extreme symptoms later
    # if 'gastro' in symptoms_per_phase[phase_i-1] or rng.rand() < p_gastro *.5:
    #     symptoms_per_phase[phase_i].append('gastro')
    #
    #     for symptom in ('diarrhea', 'nausea_vomiting'):
    #         rand = rng.rand()
    #         if rand < SYMPTOMS[symptom].probabilities[phase]:
    #             symptoms_per_phase[phase_i].append(symptom)
    #
    # # fatigue and unusual symptoms are more heavily age-related
    # # but more likely later, and less if you're careful/taking care
    # # of yourself
    # if rng.rand() < p_lethargy + (p_gastro/2): #if you had gastro symptoms before more likely to be lethargic now
    #     symptoms_per_phase[phase_i].append('fatigue')
    #
    #     if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
    #         symptoms_per_phase[phase_i].append('unusual')
    #     if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
    #             rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
    #         symptoms_per_phase[phase_i].append('lost_consciousness')
    #
    #     for symptom in ('hard_time_waking_up', 'headache', 'confused'):
    #         rand = rng.rand()
    #         if rand < SYMPTOMS[symptom].probabilities[phase]:
    #             symptoms_per_phase[phase_i].append(symptom)
    #
    # # respiratory symptoms more common at this stage
    # p_respiratory = initial_viral_load - (carefulness * 0.25) # e.g. 0.5 - 0.7*0.25 = 0.5-0.17
    # if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
    #     p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    # if rng.rand() < p_respiratory:
    #     symptoms_per_phase[phase_i].append('trouble_breathing')
    #
    #     if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
    #         symptoms_per_phase[phase_i].append('severe_chest_pain')
    #
    #     for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
    #         rand = rng.rand()
    #         if rand < SYMPTOMS[symptom].probabilities[phase]:
    #             symptoms_per_phase[phase_i].append(symptom)
    #
    # if 'loss_of_taste' in symptoms_per_phase[phase_i-1] or \
    #         rng.rand() < SYMPTOMS['loss_of_taste'].probabilities[phase]:
    #     symptoms_per_phase[phase_i].append('loss_of_taste')
    #
    # if 'mild' in symptoms_per_phase[phase_i] and \
    #         'trouble_breathing' in symptoms_per_phase[phase_i]:
    #     symptoms_per_phase[phase_i].append('light_trouble_breathing')
    # if 'moderate' in symptoms_per_phase[phase_i] and \
    #         'trouble_breathing' in symptoms_per_phase[phase_i]:
    #     symptoms_per_phase[phase_i].append('moderate_trouble_breathing')
    # if ('severe' in symptoms_per_phase[phase_i] or 'extremely-severe' in symptoms_per_phase[phase_i]) and \
    #         'trouble_breathing' in symptoms_per_phase[phase_i]:
    #     symptoms_per_phase[phase_i].append('heavy_trouble_breathing')
    #

    # During the symptoms plateau Part 2 (worst part of the disease)
    # ====================================================
    phase_i = 2
    phase = symptoms_contexts[phase_i]

    if extremely_sick:
        symptoms_per_phase[phase_i].append('extremely-severe')
    elif really_sick or len(preexisting_conditions) >2 or 'severe' in symptoms_per_phase[phase_i-1] or \
            initial_viral_load > 0.6:
        symptoms_per_phase[phase_i].append('severe')
    elif rng.rand() < p_gastro:
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    if 'fever' in symptoms_per_phase[phase_i-1] or initial_viral_load > 0.6 or \
            rng.rand() < SYMPTOMS['fever'].probabilities[phase]:
        symptoms_per_phase[phase_i].append('fever')
        if rng.rand() < SYMPTOMS['chills'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('chills')


    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later (p gastro reduced compared to part1)
    if 'gastro' in symptoms_per_phase[phase_i-1] or rng.rand() < p_gastro *.25:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    if rng.rand() < (p_lethargy + p_gastro): #if you had gastro symptoms before more likely to be lethargic now
        symptoms_per_phase[phase_i].append('fatigue')

        if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('unusual')
        if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
                rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('lost_consciousness')

        for symptom in ('hard_time_waking_up', 'headache', 'confused'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # respiratory symptoms more common at this stage
    p_respiratory = 2*(initial_viral_load - (carefulness * 0.25)) # e.g. 2* (0.5 - 0.7*0.25) = 2*(0.5-0.17)
    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    if rng.rand() < p_respiratory:
        symptoms_per_phase[phase_i].append('trouble_breathing')

        if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('severe_chest_pain')

        for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')
    if ('severe' in symptoms_per_phase[phase_i] or 'extremely-severe' in symptoms_per_phase[phase_i]) and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('heavy_trouble_breathing')

    if 'loss_of_taste' in symptoms_per_phase[phase_i-1] or \
            rng.rand() < SYMPTOMS['loss_of_taste'].probabilities[phase]:
        symptoms_per_phase[phase_i].append('loss_of_taste')


    # After the plateau (recovery part 1)
    # ====================================================
    phase_i = 3
    phase = symptoms_contexts[phase_i]

    if extremely_sick:
        symptoms_per_phase[phase_i].append('severe')
    elif really_sick:
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later (p gastro reduced compared to part1)
    if 'gastro' in symptoms_per_phase[phase_i-1] or rng.rand() < p_gastro *.1:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    if rng.rand() < (p_lethargy*1.5 + p_gastro): #if you had gastro symptoms before more likely to be lethargic now
        symptoms_per_phase[phase_i].append('fatigue')

        if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('unusual')
        if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
                rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('lost_consciousness')

        for symptom in ('hard_time_waking_up', 'headache', 'confused'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # respiratory symptoms more common at this stage but less than plateau
    p_respiratory = (initial_viral_load - (carefulness * 0.25)) # e.g. 2* (0.5 - 0.7*0.25) = 2*(0.5-0.17)
    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    if rng.rand() < p_respiratory:
        symptoms_per_phase[phase_i].append('trouble_breathing')

        if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('severe_chest_pain')

        for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')
    if ('severe' in symptoms_per_phase[phase_i] or 'extremely-severe' in symptoms_per_phase[phase_i]) and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('heavy_trouble_breathing')


    # After the plateau (recovery part 2)
    # ====================================================
    phase_i = 4
    phase = symptoms_contexts[phase_i]

    if extremely_sick:
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later (p gastro reduced compared to part1)
    if 'gastro' in symptoms_per_phase[phase_i-1] or rng.rand() < p_gastro *.1:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    if rng.rand() < (p_lethargy*2 + p_gastro): #if you had gastro symptoms before more likely to be lethargic now
        symptoms_per_phase[phase_i].append('fatigue')

        if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('unusual')
        if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
                rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('lost_consciousness')

        for symptom in ('hard_time_waking_up', 'headache', 'confused'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # respiratory symptoms getting less common
    p_respiratory = 0.5 * (initial_viral_load - (carefulness * 0.25)) # e.g. (0.5 - 0.7*0.25) = 0.5*(0.5-0.17)
    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    if rng.rand() < p_respiratory:
        symptoms_per_phase[phase_i].append('trouble_breathing')

        if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('severe_chest_pain')

        for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')
    if ('severe' in symptoms_per_phase[phase_i] or 'extremely-severe' in symptoms_per_phase[phase_i]) and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('heavy_trouble_breathing')

    viral_load_plateau_duration = math.ceil(viral_load_plateau_end - viral_load_plateau_start)
    recovery_duration = math.ceil(recovery_days - viral_load_plateau_end)
    incubation_days_wrt_infectiousness_onset_days = incubation_days - infectiousness_onset_days

    # same delay in symptom plateau as there was in symptom onset
    incubation_duration = math.ceil(incubation_days)
    covid_onset_duration = math.ceil(viral_load_plateau_start -
                                     incubation_days_wrt_infectiousness_onset_days) + \
                           viral_load_plateau_duration // 3
    plateau_duration = math.ceil(viral_load_plateau_duration * 2/3)
    post_plateau_1_duration = recovery_duration // 2
    post_plateau_2_duration = recovery_duration - post_plateau_1_duration

    assert viral_load_plateau_start >= incubation_days_wrt_infectiousness_onset_days

    for duration, symptoms in zip((incubation_duration, covid_onset_duration, plateau_duration,
                                   post_plateau_1_duration, post_plateau_2_duration),
                                  symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression


def _get_allergy_progression(rng):
    """
    [summary]

    Args:
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    symptoms_contexts = SYMPTOMS_CONTEXTS['allergy']
    phase_i = 0
    phase = symptoms_contexts[phase_i]

    symptoms = []
    for symptom in ('sneezing', 'mild_trouble_breathing', 'sore_throat', 'fatigue',
                    'hard_time_waking_up', 'headache'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms.append(symptom)

            # commented out because these are not used elsewhere for now
            # if symptom == 'mild_trouble_breathing':
            #     for symptom in ('hives', 'swelling'):
            #         rand = rng.rand()
            #         if rand < SYMPTOMS[symptom].probabilities[phase]:
            #             symptoms.append(symptom)
    progression = [symptoms]
    return progression

def _get_flu_progression(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick, AVG_FLU_DURATION):
    """
    [summary]

    Args:
        age ([type]): [description]
        rng ([type]): [description]
        carefulness ([type]): [description]
        preexisting_conditions ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]

    Returns:
        [type]: [description]
    """
    symptoms_contexts = SYMPTOMS_CONTEXTS['flu']
    symptoms_per_phase = [[] for _ in range(len(symptoms_contexts))]

    # Day 1 symptoms:
    phase_i = 0
    phase = symptoms_contexts[phase_i]

    symptoms_per_phase[phase_i].append('mild')

    for symptom in ('fatigue', 'fever', 'aches', 'hard_time_waking_up', 'gastro'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == 'gastro':
                for symptom in ('diarrhea', 'nausea_vomiting'):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    # Day 2-4ish if it's a longer flu, if 2 days long this doesn't get added
    phase_i = 1
    phase = symptoms_contexts[phase_i]

    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    for symptom in ('fatigue', 'fever', 'aches', 'hard_time_waking_up', 'gastro'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == 'gastro':
                for symptom in ('diarrhea', 'nausea_vomiting'):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    # Last day
    phase_i = 2
    phase = symptoms_contexts[phase_i]

    symptoms_per_phase[phase_i].append('mild')

    for symptom in ('fatigue', 'fever', 'aches', 'hard_time_waking_up', 'gastro'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == 'gastro':
                for symptom in ('diarrhea', 'nausea_vomiting'):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = AVG_FLU_DURATION + 2 - 2 * carefulness
    else:
        mean = AVG_FLU_DURATION - 2 * carefulness

    len_flu = rng.normal(mean,3)

    if len_flu < 2:
        len_flu = 3
    else:
        len_flu = round(len_flu)

    progression = []
    for duration, symptoms in zip((1, len_flu - 2, 1),
                                  symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression


def _get_cold_progression(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick):
    """
    [summary]

    Args:
        age ([type]): [description]
        rng ([type]): [description]
        carefulness ([type]): [description]
        preexisting_conditions ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]

    Returns:
        [type]: [description]
    """
    symptoms_contexts = SYMPTOMS_CONTEXTS['cold']

    symptoms_per_phase = [[] for _ in range(len(symptoms_contexts))]

    # Day 2-4ish if it's a longer cold, if 2 days long this doesn't get added
    phase_i = 0
    phase = symptoms_contexts[phase_i]

    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    for symptom in ('runny_nose', 'cough', 'fatigue', 'sneezing'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

    # Last day
    phase_i = 1
    phase = symptoms_contexts[phase_i]

    symptoms_per_phase[phase_i].append('mild')

    for symptom in ('runny_nose', 'cough', 'fatigue', 'sore_throat'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = 4 - round(carefulness)
    else:
        mean = 3 - round(carefulness)

    len_cold = rng.normal(mean,3)
    if len_cold < 1:
        len_cold = 1
    else:
        len_cold = math.ceil(len_cold)

    progression = [[]]
    for duration, symptoms in zip((len_cold - 1, 1),
                                  symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression

def _reported_symptoms(all_symptoms, rng, carefulness):
    all_reported_symptoms = []
    for symptoms in all_symptoms:
        reported_symptoms = []
        # miss a day of symptoms
        if rng.rand() < carefulness:
            continue
        for symptom in symptoms:
            if rng.rand() < carefulness:
                continue
            reported_symptoms.append(symptom)
        all_reported_symptoms.append(reported_symptoms)
    return all_reported_symptoms


# &preexisting-conditions
def _get_preexisting_conditions(age, sex, rng):
    """
    [summary]

    Args:
        age ([type]): [description]
        sex ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    #if rng.rand() < 0.6 + age/200:
    #    conditions = None
    #else:
    conditions = []

    # Conditions in PREEXISTING_CONDITIONS are ordered to fulfil dependencies
    for c_name, c_prob in PREEXISTING_CONDITIONS.items():
        rand = rng.rand()
        modifier = 1.
        # 'diabetes' and 'smoker' are dependencies of 'heart_disease'
        if c_name == 'heart_disease':
            if 'diabetes' in conditions or 'smoker' in conditions:
                modifier = 2
            else:
                modifier = 0.5
        # 'smoker' is a dependencies of 'cancer' and 'COPD' so it's execution
        # needs to be already done at this point
        if c_name in ('cancer', 'COPD'):
            if 'smoker' in conditions:
                modifier = 1.3
            else:
                modifier = 0.95
        # TODO: 'immuno-suppressed' condiction is currently excluded when
        #  setting the 'stroke' modifier value. Is that wanted?
        if c_name == 'stroke':
            modifier = len(conditions)
        if c_name == 'immuno-suppressed':
            if 'cancer' in conditions:
                modifier = 1.2
            else:
                modifier = 0.98
        for p in c_prob:
            if age < p.age:
                if p.sex == 'a' or sex.lower().startswith(p.sex):
                    if rand < modifier * p.probability:
                        conditions.append(p.name)
                    break

    # TODO PUT IN QUICKLY WITHOUT VERIFICATION OF NUMBERS
    if 'asthma' in conditions or 'COPD' in conditions:
        conditions.append('lung_disease')

    if sex.lower().startswith('f') and age > 18 and age < 50:
        p_pregnant = rng.normal(27, 5)
        if rng.rand() < p_pregnant:
            conditions.append('pregnant')

    return conditions

# &canadian-demgraphics
def _get_random_age_multinomial(AGE_DISTRIBUTION, rng):
    """
    [summary]

    Args:
        AGE_DISTRIBUTION ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    x = list(zip(*AGE_DISTRIBUTION.items()))
    idx = rng.choice(range(len(x[0])), p=x[1])
    age_group = x[0][idx]
    return rng.uniform(age_group[0], age_group[1])

def _get_random_area(num, total_area, rng):
    """
    Using Dirichlet distribution since it generates a "distribution of probabilities"
    which will ensure that the total area allotted to a location type remains conserved
    while also maintaining a uniform distribution

    Args:
        num ([type]): [description]
        total_area ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Keeping max at area/2 to ensure no location is allocated more than half of the total area allocated to its location type
    # area = rng.dirichlet(np.ones(math.ceil(num/2)))*(total_area/2)
    # area = np.append(area, rng.dirichlet(np.ones(math.floor(num/2)))*(total_area/2))
    area = np.array([total_area/num for _ in range(num)])
    return area

def _draw_random_discreet_gaussian(avg, scale, rng):
    """
    [summary]

    Args:
        avg ([type]): [description]
        scale ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    # https://stackoverflow.com/a/37411711/3413239
    irange, normal_pdf = _get_integer_pdf(avg, scale, 2)
    return int(rng.choice(irange, size=1, p=normal_pdf))

def _json_serialize(o):
    """
    [summary]

    Args:
        o ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(o, datetime.datetime):
        return o.__str__()

def compute_distance(loc1, loc2):
    """
    [summary]

    Args:
        loc1 ([type]): [description]
        loc2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.sqrt((loc1.lat - loc2.lat) ** 2 + (loc1.lon - loc2.lon) ** 2)


@lru_cache(500)
def _get_integer_pdf(avg, scale, num_sigmas=2):
    """
    [summary]

    Args:
        avg ([type]): [description]
        scale ([type]): [description]
        num_sigmas (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    irange = np.arange(avg - num_sigmas * scale, avg + num_sigmas * scale + 1)
    normal_pdf = norm.pdf(irange - avg)
    normal_pdf /= normal_pdf.sum()
    return irange, normal_pdf


def probas_to_risk_mapping(probas,
                           num_bins,
                           lower_cutoff=None,
                           upper_cutoff=None):
    """
    Create a mapping from probabilities returned by the model to discrete
    risk levels, with a number of predictions in each bins being approximately
    equivalent.

    Args:
        probas (np.ndarray): The array of probabilities returned by the model.
        num_bins (int): The number of bins. For example, `num_bins=16` for risk
            messages on 4 bits.
        lower_cutoff (float, optional): Ignore values smaller than `lower_cutoff`
            in the creation of the bins. This avoids any bias towards values which
            are too close to 0. If `None`, then do not cut off the small probabilities.
            Defaults to None.
        upper_cutoff (float, optional): Ignore values larger than `upper_cutoff` in the
            creation of the bins. This avoids any bias towards values which are too
            close to 1. If `None`, then do not cut off the large probabilities.
            Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: The mapping from probabilities to discrete risk levels. This mapping has
        size `num_bins + 1`, with the first values always being 0, and the last
        always being 1.
    """
    if (lower_cutoff is not None) and (upper_cutoff is not None):
        if lower_cutoff >= upper_cutoff:
            raise ValueError('The lower cutoff must have a value which is '
                             'smaller than the upper cutoff, got `lower_cutoff='
                             '{0}` and `upper_cutoff={1}`.'.format(
                             lower_cutoff, upper_cutoff))
    mask = np.ones_like(probas, dtype=np.bool_)
    num_percentiles = num_bins + 1
    # First value is always 0, last value is always 1
    cutoffs = np.zeros((num_bins + 1,), dtype=probas.dtype)
    cutoffs[-1] = 1.

    # Remove probabilities close to 0
    lower_idx = 1 if (lower_cutoff is None) else None
    if lower_cutoff is not None:
        mask = np.logical_and(mask, probas > lower_cutoff)
        num_percentiles -= 1

    # Remove probabilities close to 1
    upper_idx = -1 if (upper_cutoff is None) else None
    if upper_cutoff is not None:
        mask = np.logical_and(mask, probas <= upper_cutoff)
        num_percentiles -= 1

    percentiles = np.linspace(0, 100, num_percentiles)
    cutoffs[1:-1] = np.percentile(probas[mask],
                                  q=percentiles[lower_idx:upper_idx])

    return cutoffs

def proba_to_risk_fn(mapping):
    """
    Create a callable, based on a mapping, that takes probabilities (in
    [0, 1]) and returns a discrete risk level (in [0, num_bins - 1]).

    Args:
        mapping (np.ndarray): The mapping from probabilities to discrete risk levels.
        See `probas_to_risk_mapping`.

    Returns:
        callable: Function taking probabilities and returning discrete risk levels.
    """
    def _proba_to_risk(probas):
        return np.maximum(np.searchsorted(mapping, probas, side='left') - 1, 0)

    return _proba_to_risk

def calculate_average_infectiousness(human):
    cur_infectiousness = human.get_infectiousness_for_day(human.env.timestamp, human.is_infectious)
    is_infectious_tomorrow = True if human.infection_timestamp and human.env.timestamp - human.infection_timestamp + datetime.timedelta(days=1) >= datetime.timedelta(days=human.infectiousness_onset_days) else False
    tomorrows_infectiousness = human.get_infectiousness_for_day(human.env.timestamp + datetime.timedelta(days=1),
                                                                is_infectious_tomorrow)
    return (cur_infectiousness + tomorrows_infectiousness) / 2


def filter_open(locations):
    """Given an iterable of locations, returns a list of those that are open for business.

    Args:
        locations (iterable): a list of objects inheriting from the covid19sim.base.Location class

    Returns:
        list
    """
    return [loc for loc in locations if loc.is_open_for_business]

def filter_queue_max(locations, max_len):
    """Given an iterable of locations, will return a list of those
    with queues that are not too long.

    Args:
        locations (iterable): a list of objects inheriting from the covid19sim.base.Location class

    Returns:
        list
    """
    return [loc for loc in locations if len(loc.queue) <= max_len]


def download_file_from_google_drive(
        gdrive_file_id: typing.AnyStr,
        destination: typing.AnyStr,
        chunk_size: int = 32768
) -> typing.AnyStr:
    """
    Downloads a file from google drive, bypassing the confirmation prompt.

    Args:
        gdrive_file_id: ID string of the file to download from google drive.
        destination: where to save the file.
        chunk_size: chunk size for gradual downloads.

    Returns:
        The path to the downloaded file.
    """
    # taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': gdrive_file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': gdrive_file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return destination


def download_exp_data_if_not_exist(
        exp_data_url: typing.AnyStr,
        exp_data_destination: typing.AnyStr,
) -> typing.AnyStr:
    """
    Downloads & extract config/weights for a model if the provided destination does not exist.

    The provided URL will be assumed to be a Google Drive download URL. The download will be
    skipped entirely if the destination folder already exists. This function will return the
    path to the existing folder, or to the newly created folder.

    Args:
        exp_data_url: the zip URL (under the `https://drive.google.com/file/d/ID` format).
        exp_data_destination: where to extract the model data.

    Returns:
        The path to the model data.
    """
    assert exp_data_url.startswith("https://drive.google.com/file/d/")
    gdrive_file_id = exp_data_url.split("/")[-1]
    output_data_path = os.path.join(exp_data_destination, gdrive_file_id)
    downloaded_zip_path = os.path.join(exp_data_destination, f"{gdrive_file_id}.zip")
    if os.path.isfile(downloaded_zip_path) and os.path.isdir(output_data_path):
        return output_data_path
    os.makedirs(output_data_path, exist_ok=True)
    zip_path = download_file_from_google_drive(gdrive_file_id, downloaded_zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    return output_data_path


def extract_tracker_data(tracker, conf):
    """
    Get a dictionnary collecting interesting fields of the tracker and experimental settings

    Args:
        tracker (covid19sim.track.Tracker): Tracker toring simulation data
        conf (dict): Experimental Configuration

    returns:
        dict: the extracted data
    """
    timenow = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    data = dict()
    data['intervention_day'] = conf.get('INTERVENTION_DAY')
    data['intervention'] = conf.get('INTERVENTION')
    data['risk_model'] = conf.get('RISK_MODEL')
    data['adoption_rate'] = getattr(tracker, 'adoption_rate', 1.0)
    data['expected_mobility'] = tracker.expected_mobility
    data['serial_interval'] = tracker.get_serial_interval()
    data['all_serial_intervals'] = tracker.serial_intervals
    data['generation_times'] = tracker.get_generation_time()
    data['mobility'] = tracker.mobility
    data['n_init_infected'] = tracker.n_infected_init
    data['contacts'] = dict(tracker.contacts)
    data['cases_per_day'] = tracker.cases_per_day
    data['ei_per_day'] = tracker.ei_per_day
    data['r_0'] = tracker.r_0
    data['R'] = tracker.r
    data['n_humans'] = tracker.n_humans
    data['s'] = tracker.s_per_day
    data['e'] = tracker.e_per_day
    data['i'] = tracker.i_per_day
    data['r'] = tracker.r_per_day
    data['avg_infectiousness_per_day'] = tracker.avg_infectiousness_per_day
    data['risk_precision_global'] = tracker.compute_risk_precision(False)
    data['risk_precision'] = tracker.risk_precision_daily
    data['human_monitor'] = tracker.human_monitor
    data['infection_monitor'] = tracker.infection_monitor
    data['infector_infectee_update_messages'] = tracker.infector_infectee_update_messages
    data['risk_attributes'] = tracker.risk_attributes
    data['feelings'] = tracker.feelings
    data['rec_feelings'] = tracker.rec_feelings
    data['outside_daily_contacts'] = tracker.outside_daily_contacts
    data['test_monitor'] = tracker.test_monitor
    data['encounter_distances'] = tracker.encounter_distances
    data['effective_contacts_since_intervention'] = tracker.compute_effective_contacts(since_intervention=True)
    data['effective_contacts_all_days'] = tracker.compute_effective_contacts(since_intervention=False)
    data['humans_state'] = tracker.humans_state
    data['humans_rec_level'] = tracker.humans_rec_level
    data['humans_intervention_level'] = tracker.humans_intervention_level
    data['humans_has_app'] = dict((human.name, human.has_app) for human in tracker.city.humans)
    data['day_encounters'] = dict(tracker.day_encounters)
    data['daily_age_group_encounters'] = dict(tracker.daily_age_group_encounters)
    data['tracked_humans'] = dict({human.name:human.my_history for human in tracker.city.humans})
    data['age_histogram'] = tracker.city.age_histogram
    data['p_transmission'] = tracker.compute_probability_of_transmission()
    data['covid_properties'] = tracker.covid_properties
    data['human_has_app'] = tracker.human_has_app
    # data['dist_encounters'] = dict(tracker.dist_encounters)
    # data['time_encounters'] = dict(tracker.time_encounters)
    # data['day_encounters'] = dict(tracker.day_encounters)
    # data['hour_encounters'] = dict(tracker.hour_encounters)
    # data['daily_age_group_encounters'] = dict(tracker.daily_age_group_encounters)
    # data['age_distribution'] = tracker.age_distribution
    # data['sex_distribution'] = tracker.sex_distribution
    # data['house_size'] = tracker.house_size
    # data['house_age'] = tracker.house_age
    # data['symptoms'] = dict(tracker.symptoms)
    # data['transition_probability'] = dict(tracker.transition_probability)
    return data


def dump_tracker_data(data, outdir, name):
    """
    Writes the tracker's extracted data to outdir/name using dill.

    /!\ there are know incompatibility issues between python 3.7 and 3.8 regarding the dump/loading of data with dill/pickle

    Creates the outputdir if need be, including potential missing parents.

    Args:
        data (dict): tracker's extracted data
        outdir (str): directory where to dump the file
        name (str): the dump file's name
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / name, 'wb') as f:
        dill.dump(data, f)

def parse_search_configuration(conf):
    """
    Parses the OmegaConf to native types

    Args:
        conf (OmegaConf): Hydra configuration

    Returns:
        dict: parsed conf
    """
    return OmegaConf.to_container(conf, resolve=True)


def parse_configuration(conf):
    """
    Transforms an Omegaconf object to native python dict, parsing specific fields like:
    "1-15" age bin in YAML file becomes (1, 15) tuple, and datetime is parsed from string.

    ANY key-specific parsing should have its inverse in covid19sim.utils.dump_conf()

    Args:
        conf (omegaconf.OmegaConf): Hydra-loaded configuration

    Returns:
        dict: parsed configuration to use in experiment
    """
    if isinstance(conf, (OmegaConf, DictConfig)):
        conf = OmegaConf.to_container(conf, resolve=True)
    elif not isinstance(conf, dict):
        raise ValueError("Unknown configuration type {}".format(type(conf)))

    if "AGE_GROUP_CONTACT_AVG" in conf:
        conf['AGE_GROUP_CONTACT_AVG']['age_groups'] = [
            eval(age_group) for age_group in conf['AGE_GROUP_CONTACT_AVG']['age_groups']
        ]
        conf['AGE_GROUP_CONTACT_AVG']['contact_avg'] = np.array(conf['AGE_GROUP_CONTACT_AVG']['contact_avg'])

    if "SMARTPHONE_OWNER_FRACTION_BY_AGE" in conf:
        conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"] = {
            tuple(int(i) for i in k.split("-")): v
            for k, v in conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"].items()
        }

    if "NORMALIZED_SUSCEPTIBILITY_BY_AGE" in conf:
        conf["NORMALIZED_SUSCEPTIBILITY_BY_AGE"] = {
            tuple(int(i) for i in k.split("-")): v
            for k, v in conf["NORMALIZED_SUSCEPTIBILITY_BY_AGE"].items()
        }

    if "HUMAN_DISTRIBUTION" in conf:
        conf["HUMAN_DISTRIBUTION"] = {
            tuple(int(i) for i in k.split("-")): v
            for k, v in conf["HUMAN_DISTRIBUTION"].items()
        }

    if "MEAN_DAILY_INTERACTION_FOR_AGE_GROUP" in conf:
        conf["MEAN_DAILY_INTERACTION_FOR_AGE_GROUP"] = {
            tuple(int(i) for i in k.split("-")): v
            for k, v in conf["MEAN_DAILY_INTERACTION_FOR_AGE_GROUP"].items()
        }

    if "start_time" in conf:
        conf["start_time"] = datetime.datetime.strptime(
            conf["start_time"], "%Y-%m-%d %H:%M:%S"
        )

    assert "RISK_MODEL" in conf and conf["RISK_MODEL"] is not None

    try:
        conf["GIT_COMMIT_HASH"] = get_git_revision_hash()
    except subprocess.CalledProcessError as e:
        print(">> Contained git error:")
        print(e)
        print(">> Ignoring git hash")
        conf["GIT_COMMIT_HASH"] = "NO_GIT"
    return conf


def dump_conf(
        conf: dict,
        path: typing.Union[str, Path],
):
    """
    Perform a deep copy of the configuration dictionary, preprocess the elements into strings
    to reverse the preprocessing performed by `parse_configuration` and then, dumps the content into a `.yaml` file.

    Args:
        conf (dict): configuration dictionary to be written in a file
        path (str | Path): `.yaml` file where the configuration is written
    """

    copy_conf = deepcopy(conf)

    if "AGE_GROUP_CONTACT_AVG" in copy_conf:
        copy_conf['AGE_GROUP_CONTACT_AVG']['age_groups'] = \
            ["(" + ", ".join([str(i) for i in age_group]) + ")"
             for age_group in copy_conf["AGE_GROUP_CONTACT_AVG"]['age_groups']]
        copy_conf['AGE_GROUP_CONTACT_AVG']['contact_avg'] = copy_conf['AGE_GROUP_CONTACT_AVG']['contact_avg'].tolist()

    if "SMARTPHONE_OWNER_FRACTION_BY_AGE" in copy_conf:
        copy_conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"] = {
            "-".join([str(i) for i in k]): v
            for k, v in copy_conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"].items()
        }

    if "HUMAN_DISTRIBUTION" in copy_conf:
        copy_conf["HUMAN_DISTRIBUTION"] = {
            "-".join([str(i) for i in k]): v
            for k, v in copy_conf["HUMAN_DISTRIBUTION"].items()
        }

    if "NORMALIZED_SUSCEPTIBILITY_BY_AGE" in copy_conf:
        copy_conf["NORMALIZED_SUSCEPTIBILITY_BY_AGE"] = {
                "-".join([str(i) for i in k]): v
                for k, v in copy_conf["NORMALIZED_SUSCEPTIBILITY_BY_AGE"].items()
            }

    if "MEAN_DAILY_INTERACTION_FOR_AGE_GROUP" in copy_conf:
        copy_conf["MEAN_DAILY_INTERACTION_FOR_AGE_GROUP"] = {
                "-".join([str(i) for i in k]): v
                for k, v in copy_conf["MEAN_DAILY_INTERACTION_FOR_AGE_GROUP"].items()
            }

    if "start_time" in copy_conf:
        copy_conf["start_time"] = copy_conf["start_time"].strftime("%Y-%m-%d %H:%M:%S")

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print("WARNING configuration already exists in {}. Overwriting.".format(
            str(path.parent)
        ))

    with path.open("w") as f:
        yaml.safe_dump(copy_conf, f)


def relativefreq2absolutefreq(
        bins_fractions: dict,
        n_elements: int,
        rng
) -> dict:
    """
    Convert relative frequencies to absolute frequencies such that the number of elements sum to n_entity.
    First, we assign `math.floor(fraction*n_entity)` to each bin and then, we assign the remaining elements randomly
    until we have `n_entity`.
    Args:
        bins_fractions (dict): each key is the bin description and each value is the relative frequency.
        n_elements (int): the total number of elements to assign.
        rng: a random generator for randomly assigning the remaining elements
    Returns:
        histogram (dict): each key is the bin description and each value is the absolute frequency.
    """
    histogram = {}
    for my_bin, fraction in bins_fractions.items():
        histogram[my_bin] = math.floor(fraction * n_elements)
    while np.sum(list(histogram.values())) < n_elements:
        bins = list(histogram.keys())
        random_bin = rng.choice(len(bins))
        histogram[bins[random_bin]] += 1

    assert np.sum(list(histogram.values())) == n_elements

    return histogram


def get_git_revision_hash():
    """Get current git hash the code is run from

    Returns:
        str: git hash
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

def get_test_false_negative_rate(test_type, days_since_exposure, conf, interpolate="step"):
    rates = conf['TEST_TYPES'][test_type]['P_FALSE_NEGATIVE']['rate']
    days = conf['TEST_TYPES'][test_type]['P_FALSE_NEGATIVE']['days_since_exposure']
    if interpolate == "step":
        for x, y in zip(days, rates):
            if days_since_exposure <= x:
                return y
        return y
    else:
        raise

def get_p_infection(infector, infectors_infectiousness, infectee, social_contact_factor, contagion_knob, mask_efficacy_factor, hygiene_efficacy_factor, self, h):
    # probability of transmission
    # It is similar to Oxford COVID-19 model described in Section 4.
    rate_of_infection = infectee.normalized_susceptibility * social_contact_factor * 1 / infectee.mean_daily_interaction_age_group
    rate_of_infection *= infectors_infectiousness * infector.infection_ratio
    rate_of_infection *= contagion_knob
    p_infection = 1 - np.exp(-rate_of_infection)

    # factors that can reduce probability of transmission.
    # (no-source) How to reduce the transmission probability mathematically?
    mask_efficacy = (self.mask_efficacy + h.mask_efficacy)
    # mask_efficacy = p_infection - infector.mask_efficacy * p_infection - infectee.mask_efficacy * p_infection
    hygiene_efficacy = self.hygiene + h.hygiene
    reduction_factor = mask_efficacy * mask_efficacy_factor + hygiene_efficacy * hygiene_efficacy_factor
    p_infection *= np.exp(-reduction_factor)
    return p_infection


def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()

def zip_outdir(outdir):
    path = Path(outdir).resolve()
    assert path.exists()
    print(f"Zipping {outdir}...")
    start_time = time.time()
    command = "cd {}; zip -r -0 {}.zip {}".format(
        str(path.parent), path.name, path.name
    )
    subprocess_cmd(command)


def lp_solve_wasserstein(dist_0, dist_1):
    """Solve the optimal transport problem between two distributions as a
    Linear Program by minimizing the (squared) Wasserstein distance between the
    two distributions.

    The problem to be solved is [1, Equation 2.5]

        min_T   sum_{ij} T_{ij} * |i - j|^{2}
        st.     sum_{i} T_{ij} = dist1_{j}
                sum_{j} T_{ij} = dist0_{i}
                T >= 0

    Here we use the following heuristic first: we keep as much mass as possible
    fixed (i.e. priority is to stay in the same recommendation level). Using
    this heuristic, we then have a LP formulation with 12 variables (if dist0
    and dist1 take each 4 values, e.g. 4 recommendation levels) and 7 constraints.
    The variable T is encoded as a vector with the upper triangular values of the
    transport plan in the first half and the lower triangular values in the second half.

        T = [T_{01}, T_{02}, T_{03}, T_{12}, T_{13}, T_{23},
             T_{10}, T_{20}, T_{21}, T_{30}, T_{31}, T_{32}]

    The (equality) constraints are

        T_{01} + T_{02} + T_{03} = dist0_{0} - min(dist0_{0}, dist1_{0})
        T_{10} + T_{12} + T_{13} = dist0_{1} - min(dist0_{1}, dist1_{1})
        T_{20} + T_{21} + T_{23} = dist0_{2} - min(dist0_{2}, dist1_{2})
        T_{10} + T_{20} + T_{30} = dist1_{0} - min(dist0_{0}, dist1_{0})
        T_{01} + T_{21} + T_{31} = dist1_{1} - min(dist0_{1}, dist1_{1})
        T_{02} + T_{12} + T_{32} = dist1_{2} - min(dist0_{2}, dist1_{2})
                 sum_{ij} T_{ij} = 1 - sum_{i} min(dist0_{i}, dist1_{i})

    Note:
        [1] Justin Solomon, Optimal Transport on Discrete Domains
            (https://arxiv.org/abs/1801.07745)

    Args:
        dist_0 (np.ndarray): The distribution to move from. This array should
            have non-negative values, be normalized (i.e. sum to 1), and have
            the same shape as dist_1.
        dist_1 (np.ndarray): The distribution to move to. This array should
            have non-negative values, be normalized (i.e. sum to 1), and have
            the same shape as dist_0.

    Returns:
        np.ndarray: Array containing the solution of the Linear Program.
            This array contains the off-diagonal values of the optimal
            transport plan.
    """
    min_dist = np.minimum(dist_0, dist_1)

    # LP formulation
    c = np.array([1, 2, 3, 1, 2, 1, 1, 2, 1, 3, 2, 1], dtype=np.float_)
    A_eq = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float_)
    b_eq = np.hstack([dist_0[:-1] - min_dist[:-1],
                      dist_1[:-1] - min_dist[:-1],
                      1 - min_dist.sum()])

    # Solve LP
    result = linprog(c ** 2, A_eq=A_eq, b_eq=b_eq)

    return result.x

def lp_solution_to_transport_plan(dist_0, dist_1, solution):
    """Converts the solution of the LP given by lp_solve_wasserstein
    (off-diagonal values) into the full transport plan.

    Args:
        dist_0 (np.ndarray): The distribution to move from. This array should
            have non-negative values, be normalized (i.e. sum to 1), and have
            the same shape as dist_1.
        dist_1 (np.ndarray): The distribution to move to. This array should
            have non-negative values, be normalized (i.e. sum to 1), and have
            the same shape as dist_0.
        solution (np.ndarray): Array containing the solution of the Linear
            Program. This array contains the off-diagonal values of the optimal
            transport plan (upper triangular values in the first half, lower
            triangular values in the second half).

    Returns:
        np.ndarray: An array containing the full optimal transport plan. The
            transition matrix is a normalized version of the optimal transport plan.
    """
    # The diagonal of the transition matrix contains the minimum values
    # of both distributions, i.e. as much mass as possible is kept fixed.
    min_dist = np.minimum(dist_0, dist_1)
    transition = np.diag(min_dist)

    # Zero out small values of the solution
    solution[np.isclose(solution, 0.)] = 0.
    if not np.isclose(min_dist.sum(), 1):
        solution /= solution.sum() / (1 - min_dist.sum())

    # The solution contains the upper triangular values in the first half
    # of the solution, and the lower triangular values in the second half.
    upper, lower = solution[:solution.size // 2], solution[solution.size // 2:]
    transition[np.triu_indices_from(transition, k=1)] = upper
    transition[np.tril_indices_from(transition, k=-1)] = lower

    return transition

def get_rec_level_transition_matrix(source, target):
    """Compute the transition matrix to go from one distribution of
    recommendation levels (e.g. given by Digital Binary Tracing) to another
    distribution of recommendation levels (e.g. given by a Transformer).

    Args:
        source (np.ndarray): The source distribution. This distribution does
            not need to be normalized (i.e. array of counts).
        target (np.ndarray): The target distribution. This distribution does
        not need to be normalized (i.e. array of counts).

    Returns:
        np.ndarray: Transition matrix containing, where the value {i, j}
            corresponds to
            P(target recommendation level = j | source recommendation level = i)
    """
    # Normalize the distributions (in case we got counts)
    if np.isclose(source.sum(), 0) or np.isclose(target.sum(), 0):
        raise ValueError('The function `get_rec_level_transition_matrix` expects '
                         'two distributions, but got an array full of zeros. '
                         'source={0}, target={1}.'.format(source, target))
    dist_0 = source / source.sum()
    dist_1 = target / target.sum()

    solution = lp_solve_wasserstein(dist_0, dist_1)
    transport_plan = lp_solution_to_transport_plan(dist_0, dist_1, solution)

    # Leave the bins with no mass untouched (this ensures the transition matrix
    # is well defined everywhere, i.e. rows all sum to 1)
    diagonal = np.diag(transport_plan)
    np.fill_diagonal(transport_plan, np.where(diagonal == 0., 1., diagonal))

    return transport_plan / np.sum(transport_plan, axis=1, keepdims=True)
