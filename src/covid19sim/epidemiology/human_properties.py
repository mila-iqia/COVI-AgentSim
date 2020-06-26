from collections import OrderedDict, namedtuple
from covid19sim.utils.utils import normal_pdf

# NOTE: THE PREEXISTING CONDITION NAMES/IDs BELOW MUST MATCH THOSE IN frozen/helper.py
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
        ConditionProbability('heart_disease', 2, 1000, 'f', 0.311),
        ConditionProbability('heart_disease', 2, 1000, 'm', 0.44),
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
        ConditionProbability('COPD', 3, 65, 'a', 0.037),
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
def _get_random_sex(rng, conf):
    """
    This function returns the sex at birth of the person.
    Other is associated with 'prefer not to answer' for the CanStats census.

    Args:
        rng (): A random number generator

    Returns:
        [str]: Possible choices of sex {female, male, other}
    """
    p = rng.rand()
    if p < conf.get('P_FEMALE'):
        return 'female'
    elif p < conf.get('P_FEMALE') + conf.get('P_MALE'):
        return 'male'
    else:
        return 'other'


def may_develop_severe_illness(age, sex, rng):
    """
    Liklihood of getting really sick (i.e., requiring hospitalization) from Covid-19
    Args:
        age ([int]): [description]
        sex ([int]): [description]
        rng ([RandState]): [description]

    Returns:
        Boolean: returns True if this person would likely require hospitalization given that they contracted Covid-19
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
        p_pregnant_at_age = normal_pdf(age, 27, 5)
        if rng.rand() < p_pregnant_at_age:
            conditions.append('pregnant')

    return conditions

def _get_inflammatory_disease_level(rng, preexisting_conditions, inflammatory_conditions):
    cond_count = 0
    for cond in inflammatory_conditions:
        if cond in preexisting_conditions:
          cond_count += 1
    if cond_count > 3:
        cond_count = 3
    return cond_count


def get_carefulness(age, rng, conf):
    # &carefulness
    if rng.rand() < conf.get("P_CAREFUL_PERSON"):
        carefulness = min((max(round(rng.normal(55, 10)), 0) + age / 2) / 100, 1)
    else:
        carefulness = min((max(round(rng.normal(25, 10)), 0) + age / 2) / 100, 1)
    return carefulness

def get_age_bin(age, conf):
    # normalized susceptibility and mean daily interaction for this age group
    # required for Oxford COVID-19 infection model
    age_bins = conf['NORMALIZED_SUSCEPTIBILITY_BY_AGE'].keys()
    for l, u in age_bins:
        # NOTE  & FIXME: lower limit is exclusive
        if l < age <= u:
            bin = (l, u)
            break
    return bin
