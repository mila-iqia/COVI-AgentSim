from collections import OrderedDict, namedtuple
from covid19sim.utils.constants import AGE_BIN_WIDTH_5, AGE_BIN_WIDTH_10

AGE_BIN_ID = namedtuple("AGE_BIN_ID", ['index', 'bin'])
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


HEART_DISEASE_IF_SMOKER_OR_DIABETES_MODIFIER = 4.
CANCER_OR_COPD_IF_SMOKER_MODIFIER = 1.3 / 0.95  # 1.3684210526
IMMUNO_SUPPRESSED_IF_CANCER_MODIFIER = 1.2 / 0.98  # 1.2244897959


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
        ConditionProbability('heart_disease', 2, 20, 'a', 0.0005),
        ConditionProbability('heart_disease', 2, 35, 'a', 0.0025),
        ConditionProbability('heart_disease', 2, 50, 'f', 0.0065),
        ConditionProbability('heart_disease', 2, 50, 'm', 0.0105),
        ConditionProbability('heart_disease', 2, 50, 'a', 0.0085),
        ConditionProbability('heart_disease', 2, 75, 'f', 0.065),
        ConditionProbability('heart_disease', 2, 75, 'm', 0.089),
        ConditionProbability('heart_disease', 2, 75, 'a', 0.075),
        ConditionProbability('heart_disease', 2, 1000, 'f', 0.1555),
        ConditionProbability('heart_disease', 2, 1000, 'm', 0.22),
        ConditionProbability('heart_disease', 2, 1000, 'a', 0.1875)
    ]),
    # 'smoker' is a dependency of 'cancer' so it needs to be inserted
    # before this position
    ('cancer', [
        ConditionProbability('cancer', 6, 30, 'a', 0.0002755),
        ConditionProbability('cancer', 6, 60, 'a', 0.002755),
        ConditionProbability('cancer', 6, 90, 'a', 0.02755),
        ConditionProbability('cancer', 6, 1000, 'a', 0.0475)
    ]),
    # 'smoker' is a dependency of 'COPD' so it needs to be inserted
    # before this position
    ('COPD', [
        ConditionProbability('COPD', 3, 35, 'a', 0.0),
        ConditionProbability('COPD', 3, 50, 'a', 0.01425),
        ConditionProbability('COPD', 3, 65, 'a', 0.03515),
        ConditionProbability('COPD', 3, 1000, 'a', 0.07125)
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
        ConditionProbability('immuno-suppressed', 0, 40, 'a', 0.0049),
        ConditionProbability('immuno-suppressed', 0, 65, 'a', 0.03528),
        ConditionProbability('immuno-suppressed', 0, 85, 'a', 0.0441),
        ConditionProbability('immuno-suppressed', 0, 1000, 'a', 0.196)
    ]),
    ('lung_disease', [
        ConditionProbability('lung_disease', 8, -1, 'a', -1)
    ]),
    ('pregnant', [
        ConditionProbability('pregnant', 9, -1, 'f', -1)
    ]),
    ('allergies', [
        ConditionProbability('allergies', 10, 1000, 'a', 0.2) # proportion of population; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6121311/
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
    Likelihood of getting really sick (i.e., requiring hospitalization) from Covid-19
    Args:
        age ([int]): [description]
        sex ([int]): [description]
        rng ([RandState]): [description]

    Returns:
        Boolean: returns True if this person would likely require hospitalization given that they contracted Covid-19
    """
    # age     < 10  < 20  < 30  < 40   < 50  < 60  < 70  < 80  < 90  default
    female = [0.02, 0.002, 0.05, 0.05, 0.13, 0.18, 0.16, 0.24, 0.17, 0.03]
    male = [0.002, 0.02, 0.03, 0.07, 0.13, 0.17, 0.22, 0.22, 0.15, 0.03]
    other = [0.02, 0.02, 0.04, 0.07, 0.13, 0.18, 0.24, 0.24, 0.18, 0.03]

    hospitalization_likelihood = other
    if sex.lower().startswith('f'):
        hospitalization_likelihood = female
    elif sex.lower().startswith('m'):
        hospitalization_likelihood = male

    if age > 90:
        age = 90
    
    index_of_age_category = (age - (age % 10))//10 # round down to nearest 10, then floor divide by 10
    return rng.rand() < hospitalization_likelihood[index_of_age_category]


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
        # 'diabetes' and 'smoker' are dependencies of 'heart_disease' so their
        # attribution to the list of pre-existing conditions need to be already
        # computed at this point
        if c_name == 'heart_disease':
            if 'diabetes' in conditions or 'smoker' in conditions:
                modifier = HEART_DISEASE_IF_SMOKER_OR_DIABETES_MODIFIER
        # 'smoker' is a dependencies of 'cancer' and 'COPD' so its attribution
        # to the list of pre-existing conditions need to be already computed at
        # this point
        if c_name in ('cancer', 'COPD'):
            if 'smoker' in conditions:
                modifier = CANCER_OR_COPD_IF_SMOKER_MODIFIER
        # TODO: 'immuno-suppressed' condiction is currently excluded when
        #  setting the 'stroke' modifier value. Is that wanted?
        if c_name == 'stroke':
            modifier = len(conditions)
        if c_name == 'immuno-suppressed':
            if 'cancer' in conditions:
                modifier = IMMUNO_SUPPRESSED_IF_CANCER_MODIFIER
        # Randomly append the pre-existing condition to the list
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
        if cond in preexisting_conditions and cond_count < 3:
          cond_count += 1
    return cond_count

def get_carefulness(age, rng, conf):
    if not conf["AGE_AFFECTS_CAREFULNESS"]:
        age = conf["MEDIAN_AGE_REGION"]
    # &carefulness
    loc = 55 if rng.rand() < conf.get("P_CAREFUL_PERSON") else 25
    carefulness = (max(round(rng.normal(loc, 10)), 0) + age / 2) / 100
    carefulness = min(carefulness + 0.01, 1)
    return carefulness

def get_age_bin(age, width=10):
    """
    Various data sources like demographics and epidemiological parameters are available per age group.
    The range of age groups vary from one source to another.
    This function returns an appropriate group for a particular age.

    Args:
        age (int): age of `human`
        width (int): number of ages included in each age group. For example,
            age bins of the form 0-9 have a width of 10 (both limits inclusive)
            age bins of the form 0-4 have a width of 5 (both limits inclusive)

    Returns:
        age_bin (AGE_BIN_ID): a namedtuple with the following keys:
            bin: idenitfier for the age group that can be used to look up values in various data sources
                ranges from 0-9 if width = 10
                ranges from 0-16 if width = 5
            index (int): ordering of this bin in the list

    """
    if width == 10:
        age_bins = AGE_BIN_WIDTH_10
    elif width == 5:
        age_bins = AGE_BIN_WIDTH_5
    else:
        raise

    for i, (l, u) in enumerate(age_bins):
        if l <= age <= u:
            bin = (l, u)
            break

    return AGE_BIN_ID(index=i, bin=bin)
