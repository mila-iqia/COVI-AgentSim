"""
[summary]
"""
from collections import OrderedDict, namedtuple
from scipy.stats import norm, truncnorm, gamma
from functools import lru_cache
import datetime
import math

from covid19sim.configs.config import *
from covid19sim.interventions import *


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
        A probability of `None` is assigned when the symptom it can be skipped
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

SYMPTOMS_CONTEXTS = {'covid': {0: 'covid_pre_plateau', 1: 'covid_plateau_1', 2: 'covid_plateau_2',
                               3: 'covid_post_plateau_1', 4: 'covid_post_plateau_2'},
                     'cold': {0: 'cold', 1: 'cold_last_day'},
                     'flu': {0: 'flu_first_day', 1: 'flu', 2: 'flu_last_day'}}

SYMPTOMS = OrderedDict([
    # Sickness severity
    # A lot of symptoms are dependent on the sickness severity so severity
    # level needs to be first
    (
        'mild',
        SymptomProbability('mild', 1, {'covid_pre_plateau': -1,
                                       'covid_plateau_1': -1,
                                       'covid_plateau_2': -1,
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
        SymptomProbability('moderate', 0, {'covid_pre_plateau': -1,
                                           'covid_plateau_1': -1,
                                           'covid_plateau_2': -1,
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
        SymptomProbability('severe', 2, {'covid_pre_plateau': 0.0,
                                         'covid_plateau_1': -1,
                                         'covid_plateau_2': -1,
                                         'covid_post_plateau_1': -1,
                                         'covid_post_plateau_2': 0.0})
    ),
    (
        'extremely-severe',
        SymptomProbability('extremely-severe', 3, {'covid_pre_plateau': 0.0,
                                                   'covid_plateau_1': -1,
                                                   'covid_plateau_2': -1,
                                                   'covid_post_plateau_1': 0.0,
                                                   'covid_post_plateau_2': 0.0})
    ),

    # Symptoms

    (
        'fever',
        SymptomProbability('fever', 4, {'covid_pre_plateau': 0.2,
                                        'covid_plateau_1': 0.3,
                                        'covid_plateau_2': 0.8,
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
        SymptomProbability('chills', 5, {'covid_pre_plateau': 0.8,
                                         'covid_plateau_1': 0.5,
                                         'covid_plateau_2': 0.5,
                                         'covid_post_plateau_1': 0.0,
                                         'covid_post_plateau_2': 0.0})
    ),

    (
        'gastro',
        SymptomProbability('gastro', 6, {'covid_pre_plateau': -1,
                                          'covid_plateau_1': -1,
                                          'covid_plateau_2': -1,
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
        SymptomProbability('diarrhea', 7, {'covid_pre_plateau': 0.9,
                                           'covid_plateau_1': 0.9,
                                           'covid_plateau_2': 0.9,
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
        SymptomProbability('nausea_vomiting', 8, {'covid_pre_plateau': 0.7,
                                                  'covid_plateau_1': 0.7,
                                                  'covid_plateau_2': 0.7,
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
        SymptomProbability('fatigue', 9, {'covid_pre_plateau': -1,
                                          'covid_plateau_1': -1,
                                          'covid_plateau_2': -1,
                                          'covid_post_plateau_1': -1,
                                          'covid_post_plateau_2': -1,
                                          'cold': 0.8,
                                          'cold_last_day': 0.8,
                                          'flu_first_day': 0.4,
                                          'flu': 0.8,
                                          'flu_last_day': 0.8})
    ),
    (
        'unusual',
        SymptomProbability('unusual', 10, {'covid_pre_plateau': 0.2,
                                           'covid_plateau_1': 0.3,
                                           'covid_plateau_2': 0.5,
                                           'covid_post_plateau_1': 0.5,
                                           'covid_post_plateau_2': 0.5})
    ),
    (
        'hard_time_waking_up',
        SymptomProbability('hard_time_waking_up', 11, {'covid_pre_plateau': 0.6,
                                                       'covid_plateau_1': 0.6,
                                                       'covid_plateau_2': 0.6,
                                                       'covid_post_plateau_1': 0.6,
                                                       'covid_post_plateau_2': 0.6,
                                                       'flu_first_day': 0.3,
                                                       'flu': 0.5,
                                                       'flu_last_day': 0.4})
    ),
    (
        'headache',
        SymptomProbability('headache', 12, {'covid_pre_plateau': 0.5,
                                            'covid_plateau_1': 0.5,
                                            'covid_plateau_2': 0.5,
                                            'covid_post_plateau_1': 0.5,
                                            'covid_post_plateau_2': 0.5})
    ),
    (
        'confused',
        SymptomProbability('confused', 13, {'covid_pre_plateau': 0.1,
                                            'covid_plateau_1': 0.1,
                                            'covid_plateau_2': 0.1,
                                            'covid_post_plateau_1': 0.1,
                                            'covid_post_plateau_2': 0.1})
    ),
    (
        'lost_consciousness',
        SymptomProbability('lost_consciousness', 14, {'covid_pre_plateau': 0.1,
                                                      'covid_plateau_1': 0.1,
                                                      'covid_plateau_2': 0.1,
                                                      'covid_post_plateau_1': 0.1,
                                                      'covid_post_plateau_2': 0.1})
    ),

    # Respiratory symptoms
    # 'trouble_breathing' is a dependency of all this category so it should be
    # inserted before them
    (
        'trouble_breathing',
        SymptomProbability('trouble_breathing', 15, {'covid_pre_plateau': -1,
                                                     'covid_plateau_1': -1,
                                                     'covid_plateau_2': -1,
                                                     'covid_post_plateau_1': -1,
                                                     'covid_post_plateau_2': -1,
                                                     'cold': 0.1,
                                                     'cold_last_day': 0.0})
    ),
    (
        'sneezing',
        SymptomProbability('sneezing', 16, {'covid_pre_plateau': 0.2,
                                            'covid_plateau_1': 0.3,
                                            'covid_plateau_2': 0.3,
                                            'covid_post_plateau_1': 0.3,
                                            'covid_post_plateau_2': 0.3,
                                            'cold': 0.4,
                                            'cold_last_day': 0.0})
    ),
    (
        'cough',
        SymptomProbability('cough', 17, {'covid_pre_plateau': 0.6,
                                         'covid_plateau_1': 0.9,
                                         'covid_plateau_2': 0.9,
                                         'covid_post_plateau_1': 0.9,
                                         'covid_post_plateau_2': 0.9,
                                         'cold': 0.8,
                                         'cold_last_day': 0.8})
    ),
    (
        'runny_nose',
        SymptomProbability('runny_nose', 18, {'covid_pre_plateau': 0.1,
                                              'covid_plateau_1': 0.2,
                                              'covid_plateau_2': 0.2,
                                              'covid_post_plateau_1': 0.2,
                                              'covid_post_plateau_2': 0.2,
                                              'cold': 0.8,
                                              'cold_last_day': 0.8})
    ),
    (
        'sore_throat',
        SymptomProbability('sore_throat', 20, {'covid_pre_plateau': 0.5,
                                               'covid_plateau_1': 0.8,
                                               'covid_plateau_2': 0.8,
                                               'covid_post_plateau_1': 0.8,
                                               'covid_post_plateau_2': 0.8,
                                               'cold': 0.0,
                                               'cold_last_day': 0.6})
    ),
    (
        'severe_chest_pain',
        SymptomProbability('severe_chest_pain', 21, {'covid_pre_plateau': 0.4,
                                                     'covid_plateau_1': 0.5,
                                                     'covid_plateau_2': 0.5,
                                                     'covid_post_plateau_1': 0.15,
                                                     'covid_post_plateau_2': 0.15})
    ),

    # 'trouble_breathing' is a dependency of any '*_trouble_breathing' so it
    # needs to be inserted before this position
    (
        'light_trouble_breathing',
        SymptomProbability('light_trouble_breathing', 24, {'covid_pre_plateau': -1,
                                                          'covid_plateau_1': -1,
                                                          'covid_plateau_2': -1,
                                                          'covid_post_plateau_1': -1,
                                                          'covid_post_plateau_2': -1})
    ),
    (
        'mild_trouble_breathing',
        SymptomProbability('mild_trouble_breathing', 23, {})
    ),
    (
        'moderate_trouble_breathing',
        SymptomProbability('moderate_trouble_breathing', 25, {'covid_pre_plateau': -1,
                                                              'covid_plateau_1': -1,
                                                              'covid_plateau_2': -1,
                                                              'covid_post_plateau_1': -1,
                                                              'covid_post_plateau_2': -1})
    ),
    (
        'heavy_trouble_breathing',
        SymptomProbability('heavy_trouble_breathing', 26, {'covid_pre_plateau': 0,
                                                           'covid_plateau_1': -1,
                                                           'covid_plateau_2': -1,
                                                           'covid_post_plateau_1': -1,
                                                           'covid_post_plateau_2': -1})
    ),

    (
        'loss_of_taste',
        SymptomProbability('loss_of_taste', 22, {'covid_pre_plateau': 0.25,
                                                 'covid_plateau_1': 0.3,
                                                 'covid_plateau_2': 0.35,
                                                 'covid_post_plateau_1': 0.0,
                                                 'covid_post_plateau_2': 0.0,
                                                 'cold': 0.2,
                                                 'cold_last_day': 0.0})
    ),

    (
        'aches',
        SymptomProbability('aches', 19, {'flu_first_day': 0.3,
                                         'flu': 0.5,
                                         'flu_last_day': 0.8})
    )
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


def _sample_viral_load_piecewise(rng, initial_viral_load=0, age=40):
    """
    This function samples a piece-wise linear viral load model which increases, plateaus, and drops

    Args:
        rng ([type]): [description]
        initial_viral_load (int, optional): [description]. Defaults to 0.
        age (int, optional): [description]. Defaults to 40.
    """
    # https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
	# https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30196-1/fulltext
    plateau_start = truncnorm((PLATEAU_START_CLIP_LOW - PLATEAU_START_MEAN)/PLATEAU_START_STD, (PLATEAU_START_CLIP_HIGH - PLATEAU_START_MEAN) / PLATEAU_START_STD, loc=PLATEAU_START_MEAN, scale=PLATEAU_START_STD).rvs(1, random_state=rng)
    plateau_end = plateau_start + truncnorm((PLATEAU_DURATION_CLIP_LOW - PLATEAU_DURATION_MEAN)/PLEATEAU_DURATION_STD,
                                            (PLATEAU_DURATION_CLIP_HIGH - PLATEAU_DURATION_MEAN) / PLEATEAU_DURATION_STD,
                                            loc=PLATEAU_DURATION_MEAN, scale=PLEATEAU_DURATION_STD).rvs(1, random_state=rng)
    recovered = plateau_end + ((age/10)-1) # age is a determining factor for the recovery time
    recovered = recovered + initial_viral_load * VIRAL_LOAD_RECOVERY_FACTOR \
                          + truncnorm((RECOVERY_CLIP_LOW - RECOVERY_MEAN) / RECOVERY_STD,
                                        (RECOVERY_CLIP_HIGH - RECOVERY_MEAN) / RECOVERY_STD,
                                        loc=RECOVERY_MEAN, scale=RECOVERY_STD).rvs(1, random_state=rng)

    base = age/200 # peak viral load varies linearly with age
    # plateau_mean =  initial_viral_load - (base + MIN_VIRAL_LOAD) / (base + MIN_VIRAL_LOAD, base + MAX_VIRAL_LOAD) # transform initial viral load into a range
    # plateau_height = rng.normal(plateau_mean, 1)
    plateau_height = rng.uniform(base + MIN_VIRAL_LOAD, base + MAX_VIRAL_LOAD)
    return plateau_height, plateau_start.item(), plateau_end.item(), recovered.item()


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
    [summary]

    Args:
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    p = rng.rand()
    if p < .4:
        return 'female'
    elif p < .8:
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
                           viral_load_recovered, age, incubation_days, really_sick, extremely_sick,
                           rng, preexisting_conditions, carefulness):
    """
    [summary]

    Args:
        initial_viral_load ([type]): [description]
        viral_load_plateau_start ([type]): [description]
        viral_load_plateau_end ([type]): [description]
        viral_load_recovered ([type]): [description]
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

    # Before onset of symptoms (incubation)
    # ====================================================
    for day in range(math.ceil(incubation_days)):
        progression.append([])

    # Before the symptom's plateau
    # ====================================================
    phase_i = 0
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


    # During the plateau of symptoms Part 1
    # ====================================================
    phase_i = 1
    phase = symptoms_contexts[phase_i]

    if extremely_sick:
        symptoms_per_phase[phase_i].append('extremely-severe')
    elif really_sick or len(preexisting_conditions) >2 or 'moderate' in symptoms_per_phase[phase_i-1] or \
            initial_viral_load > 0.6:
        symptoms_per_phase[phase_i].append('severe')
    elif rng.rand() < p_gastro:
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    if 'fever' in symptoms_per_phase[phase_i-1] or initial_viral_load > 0.8 or \
            rng.rand() < SYMPTOMS['fever'].probabilities[phase]:
        symptoms_per_phase[phase_i].append('fever')
        if rng.rand() < SYMPTOMS['chills'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('chills')


    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later
    if 'gastro' in symptoms_per_phase[phase_i-1] or rng.rand() < p_gastro *.5:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    if rng.rand() < p_lethargy + (p_gastro/2): #if you had gastro symptoms before more likely to be lethargic now
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
    p_respiratory = initial_viral_load - (carefulness * 0.25) # e.g. 0.5 - 0.7*0.25 = 0.5-0.17
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

    if 'loss_of_taste' in symptoms_per_phase[phase_i-1] or \
            rng.rand() < SYMPTOMS['loss_of_taste'].probabilities[phase]:
        symptoms_per_phase[phase_i].append('loss_of_taste')

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')
    if ('severe' in symptoms_per_phase[phase_i] or 'extremely-severe' in symptoms_per_phase[phase_i]) and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('heavy_trouble_breathing')


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

    symptom_onset_delay = round(viral_load_plateau_start - incubation_days)
    plateau_duration = round(viral_load_plateau_end - viral_load_plateau_start)

    # same delay in symptom plateau as there was in symptom onset
    pre_plateau_duration = symptom_onset_delay
    plateau_1_duration = plateau_duration // 2
    plateau_2_duration = plateau_duration - plateau_1_duration
    post_plateau_1_duration = symptom_onset_delay
    post_plateau_2_duration = round(viral_load_recovered - viral_load_plateau_end - 2)

    for duration, symptoms in zip((pre_plateau_duration, plateau_1_duration, plateau_2_duration,
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
    symptoms = ['sneezing']
    if rng.rand() < P_SEVERE_ALLERGIES:
        symptoms.append('mild_trouble_breathing')
        # commented out because these are not used elsewhere for now
        # if rng.rand() < 0.4:
        #     symptoms.append ('hives')
        # if rng.rand() < 0.3:
        #     symptoms.append('swelling')
    if rng.rand() < 0.3:
        symptoms.append ('sore_throat')
    if rng.rand() < 0.2:
        symptoms.append('fatigue')
    if rng.rand() < 0.3:
        symptoms.append('hard_time_waking_up')
    if rng.rand() < 0.6:
        symptoms.append('headache')
    progression = [symptoms]
    return progression

def _get_flu_progression(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick):
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

    progression = [[] for day in range(FLU_INCUBATION)]

    symptoms_per_phase = [[] for _ in range(len(symptoms_contexts))]

    progression = []

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
        mean = AVG_FLU_DURATION + 2 -2*carefulness
    else:
        mean = AVG_FLU_DURATION - 2*carefulness

    len_flu = rng.normal(mean,3)

    if len_flu < 2:
        len_flu = 3
    else:
        len_flu = round(len_flu)

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

    progression = [[]]
    symptoms_per_phase = [[] for _ in range(len(symptoms_contexts))]

    # Day 2-4ish if it's a longer cold, if 2 days long this doesn't get added
    phase_i = 0
    phase = symptoms_contexts[phase_i]

    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    for symptom in ('runny_nose', 'cough', 'trouble_breathing', 'loss_of_taste', 'fatigue', 'sneezing'):
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
	area = rng.dirichlet(np.ones(math.ceil(num/2)))*(total_area/2)
	area = np.append(area,rng.dirichlet(np.ones(math.floor(num/2)))*(total_area/2))
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

def get_intervention(key, RISK_MODEL=None, TRACING_ORDER=None, TRACE_SYMPTOMS=None, TRACE_RISK_UPDATE=None, SHOULD_MODIFY_BEHAVIOR=True):
	"""
    [summary]

    Args:
        key ([type]): [description]
        RISK_MODEL ([type], optional): [description]. Defaults to None.
        TRACING_ORDER ([type], optional): [description]. Defaults to None.
        TRACE_SYMPTOMS ([type], optional): [description]. Defaults to None.
        TRACE_RISK_UPDATE ([type], optional): [description]. Defaults to None.
        SHOULD_MODIFY_BEHAVIOR (bool, optional): [description]. Defaults to True.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
	if key == "Lockdown":
		return Lockdown()
	elif key == "WearMask":
		return WearMask(MASKS_SUPPLY)
	elif key == "SocialDistancing":
		return SocialDistancing()
	elif key == "Quarantine":
		return Quarantine()
	elif key == "Tracing":
		return Tracing(RISK_MODEL, TRACING_ORDER, TRACE_SYMPTOMS, TRACE_RISK_UPDATE, SHOULD_MODIFY_BEHAVIOR)
	elif key == "WashHands":
		return WashHands()
	elif key == "Stand2M":
		return Stand2M()
	elif key == "StayHome":
		return StayHome()
	elif key == "GetTested":
		raise NotImplementedError
	else:
		raise

def get_recommendations(risk_level):
    """
    [summary]

    Args:
        risk_level ([type]): [description]

    Returns:
        [type]: [description]
    """
    if risk_level == 0:
        return ['stay_home', 'wash_hands', 'stand_2m']
    if risk_level == 1:
        return ['stay_home', 'wash_hands', 'stand_2m', 'limit_contact']
    if risk_level == 2:
        return ['stay_home', 'wash_hands', 'stand_2m', 'limit_contact', 'wear_mask', 'monitor_symptoms']
    else:
        return ['stay_home', 'wash_hands', 'stand_2m', 'limit_contact', 'wear_mask', 'monitor_symptoms', 'get_tested', 'quarantine']

def calculate_average_infectiousness(human):
    cur_infectiousness = human.get_infectiousness_for_day(human.env.timestamp, human.is_infectious)
    is_infectious_tomorrow = True if human.infection_timestamp and human.env.timestamp - human.infection_timestamp + datetime.timedelta(days=1) >= datetime.timedelta(days=human.infectiousness_onset_days) else False
    tomorrows_infectiousness = human.get_infectiousness_for_day(human.env.timestamp + datetime.timedelta(days=1),
                                                                is_infectious_tomorrow)
    return (cur_infectiousness + tomorrows_infectiousness) / 2