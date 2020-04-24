import numpy as np
from scipy.stats import norm, truncnorm, gamma
import datetime
import math
import json
from bitarray import bitarray
from config import *
from functools import lru_cache

def log(str, logfile=None, timestamp=False):
	if timestamp:
		str = f"[{datetime.datetime.now()}] {str}"

	print(str)
	if logfile is not None:
		with open(logfile, mode='a') as f:
			print(str, file=f)

def _sample_viral_load_gamma(rng, shape_mean=4.5, shape_std=.15, scale_mean=1., scale_std=.15):
    """ This function samples the shape and scale of a gamma distribution, then returns it"""
    shape = rng.normal(shape_mean, shape_std)
    scale = rng.normal(scale_mean, scale_std)
    return gamma(shape, scale=scale)


def _sample_viral_load_piecewise(rng, age=40):
    """ This function samples a piece-wise linear viral load model which increases, plateaus, and drops """
    # https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
	# https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30196-1/fulltext
    plateau_start = truncnorm((PLATEAU_START_CLIP_LOW - PLATEAU_START_MEAN)/PLATEAU_START_STD, (PLATEAU_START_CLIP_HIGH - PLATEAU_START_MEAN) / PLATEAU_START_STD, loc=PLATEAU_START_MEAN, scale=PLATEAU_START_STD).rvs(1, random_state=rng)
    plateau_end = plateau_start + truncnorm((PLATEAU_DURATION_CLIP_LOW - PLATEAU_DURATION_MEAN)/PLEATEAU_DURATION_STD,
                                            (PLATEAU_DURATION_CLIP_HIGH - PLATEAU_DURATION_MEAN) / PLEATEAU_DURATION_STD,
                                            loc=PLATEAU_DURATION_MEAN, scale=PLEATEAU_DURATION_STD).rvs(1, random_state=rng)
    recovered = plateau_end + ((age/10)-1) # age is a determining factor for the recovery time
    recovered = recovered + truncnorm((RECOVERY_CLIP_LOW - RECOVERY_MEAN) / RECOVERY_STD,
                                        (RECOVERY_CLIP_HIGH - RECOVERY_MEAN) / RECOVERY_STD,
                                        loc=RECOVERY_MEAN, scale=RECOVERY_STD).rvs(1, random_state=rng)
    base = age/200 # peak viral load varies linearly with age
    plateau_height = rng.uniform(base + MIN_VIRAL_LOAD, base + MAX_VIRAL_LOAD)
    return plateau_height, plateau_start.item(), plateau_end.item(), recovered.item()

def _normalize_scores(scores):
    return np.array(scores)/np.sum(scores)

# &canadian-demgraphics
def _get_random_age(rng):
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
    p = rng.rand()
    if p < .4:
        return 'female'
    elif p < .8:
        return 'male'
    else:
        return 'other'


def _get_mask_wearing(carefulness, simulation_days, rng):
    return [rng.rand() < carefulness*BASELINE_P_MASK for day in range(simulation_days)]


# 2D Array of symptoms; first axis is days after exposure (infection), second is an array of symptoms
def _get_covid_symptoms(viral_load_plateau_start, viral_load_plateau_end,
                            viral_load_recovered, age, incubation_days, really_sick, extremely_sick,
                            rng, preexisting_conditions):
    progression = []
    # Before the plateau
    symptoms1 = []
    if really_sick or extremely_sick or len(preexisting_conditions) > 2:
        symptoms1.append('moderate')
    else :
        symptoms1.append('mild')

    if rng.rand() < 0.9:
        symptoms1.append('fever')
    if rng.rand() < 0.7:
        symptoms1.append('cough')
    if rng.rand() < 0.1:
        symptoms1.append('runny_nose')
    if rng.rand() < 0.5:
        symptoms1.append('fatigue')
    if rng.rand() < 0.3:
        symptoms1.append('trouble_breathing')
    if rng.rand() < 0.4:
        symptoms1.append('gastro')
    if rng.rand() < 0.2*(age/3):
        symptoms1.append('unusual')

	#TODO CHECK THESE! PUT IN QUICKLY WITHOUT VERIFYING
    if rng.rand() < 0.5:
        symptoms1.append('sneezing')
    if rng.rand() < 0.3:
        symptoms1.append('diarrhea')
    if rng.rand() < 0.2:
        symptoms1.append('nausea_vomiting')
    if rng.rand() < 0.5:
        symptoms1.append('headache')
    if rng.rand() < 0.2:
        symptoms1.append('hard_time_waking_up')
    if rng.rand() < 0.6:
        symptoms1.append('sore_throat')
    if rng.rand() < 0.3:
        symptoms1.append('chills')
    if rng.rand() < 0.05:
        symptoms1.append('severe_chest_pain')
    if rng.rand() < 0.1:
        symptoms1.append('confused')

    if really_sick or extremely_sick or len(preexisting_conditions)>2:
        if rng.rand() < 0.1:
            symptoms1.append('lost_consciousness')

    if 'mild' and 'trouble_breathing' in symptoms1:
        symptoms1.append('light_trouble_breathing')

    if 'moderate' and 'trouble_breathing' in symptoms1:
        symptoms1.append('moderate_trouble_breathing')

    for day in range(math.ceil(viral_load_plateau_start)):
        progression.append(symptoms1)

    # During the plateau
    symptoms2 = []
    if really_sick or len(preexisting_conditions) >2 or 'moderate' in symptoms1:
        symptoms2.append('severe')
    elif extremely_sick:
        symptoms2.append('extremely-severe')
    elif rng.rand() < 0.1:
        symptoms2.append('moderate')
    else:
        symptoms2.append('mild')

    if 'fever' in symptoms1 or rng.rand() < 0.9:
        symptoms2.append('fever')
    if rng.rand() < 0.85:
        symptoms2.append('cough')
    if rng.rand() < 0.8:
        symptoms2.append('fatigue')
    if rng.rand() < 0.7:
        symptoms2.append('trouble_breathing')
    if rng.rand() < 0.1:
        symptoms2.append('runny_nose')
    if rng.rand() < 0.4:
        symptoms2.append('loss_of_taste')
    if rng.rand() < 0.1:
        symptoms2.append('gastro')
    if rng.rand() < 0.2*(age/3):
        symptoms2.append('unusual')

	#TODO CHECK THESE! PUT IN QUICKLY WITHOUT VERIFYING
    if rng.rand() < 0.5:
        symptoms2.append('sneezing')
    if rng.rand() < 0.3:
        symptoms2.append('diarrhea')
    if rng.rand() < 0.2:
        symptoms2.append('nausea_vomiting')
    if rng.rand() < 0.5:
        symptoms2.append('headache')
    if rng.rand() < 0.2:
        symptoms2.append('hard_time_waking_up')
    if rng.rand() < 0.6:
        symptoms2.append('sore_throat')
    if rng.rand() < 0.3:
        symptoms2.append('chills')
    if rng.rand() < 0.1:
        symptoms2.append('severe_chest_pain')
    if rng.rand() < 0.1:
        symptoms2.append('confused')
    if really_sick or extremely_sick or len(preexisting_conditions)>2:
        if rng.rand() < 0.6:
            symptoms2.append('lost_consciousness')
    if 'mild' in symptoms2 and 'trouble_breathing' in symptoms2:
        symptoms2.append('light_trouble_breathing')
    if 'moderate'in symptoms2 and 'trouble_breathing' in symptoms2:
        symptoms2.append('moderate_trouble_breathing')
    if ('severe' in symptoms2 or 'extremely-severe' in symptoms2) and 'trouble_breathing' in symptoms2:
        symptoms2.append('heavy_trouble_breathing')

    for day in range(math.ceil(viral_load_plateau_end - viral_load_plateau_start)):
        progression.append(symptoms2)

    # After the plateau
    symptoms3 = []
    if really_sick or extremely_sick:
        symptoms3.append('moderate')
    else:
        symptoms3.append('mild')
    if rng.rand() < 0.3:
        symptoms3.append('cough')
    if rng.rand() < 0.8:
        symptoms3.append('fatigue')
    if rng.rand() < 0.5:
        symptoms3.append('aches')
    if rng.rand() < 0.3:
        symptoms3.append('trouble_breathing')
    if rng.rand() < 0.2:
        symptoms3.append('gastro')

	#TODO CHECK THESE! PUT IN QUICKLY WITHOUT VERIFYING
    if rng.rand() < 0.5:
        symptoms3.append('sneezing')
    if rng.rand() < 0.3:
        symptoms3.append('diarrhea')
    if rng.rand() < 0.2:
        symptoms3.append('nausea_vomiting')
    if rng.rand() < 0.5:
        symptoms3.append('headache')
    if rng.rand() < 0.2:
        symptoms3.append('hard_time_waking_up')
    if rng.rand() < 0.6:
        symptoms3.append('sore_throat')
    if rng.rand() < 0.3:
        symptoms3.append('chills')
    if rng.rand() < 0.1:
        symptoms3.append('severe_chest_pain')
    if rng.rand() < 0.1:
        symptoms3.append('confused')
    if really_sick or extremely_sick or len(preexisting_conditions)>2:
        if rng.rand() < 0.6:
            symptoms3.append('lost_consciousness')

    if 'mild' in symptoms3 and 'trouble_breathing' in symptoms3:
        symptoms3.append('light_trouble_breathing')
    if 'moderate' in symptoms3 and 'trouble_breathing' in symptoms3:
        symptoms3.append('moderate_trouble_breathing')
    if ('severe' in symptoms3 or 'extremely-severe' in symptoms3) and 'trouble_breathing' in symptoms3:
        symptoms3.append('heavy_trouble_breathing')

    for day in range(math.ceil(viral_load_recovered - viral_load_plateau_end)):
        progression.append(symptoms3)

    return progression

def _get_flu_symptoms(age, rng, sim_days, carefulness, preexisting_conditions, really_sick, extremely_sick):
    symptoms_array = [[] for i in range(sim_days)]

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = 5 - round(carefulness)
    else:
        mean = 3 - round(carefulness)

    len_flu = rng.normal(mean,3)
    if len_flu < 1:
        len_flu = 1
    else:
        len_flu = round(len_flu)

    len_flu = min(len_flu, sim_days-1)
    symptoms = []
    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms.append('moderate')
    else:
        symptoms.append('mild')
    if rng.rand() < 0.8:
        symptoms.append('fever')
    if rng.rand() < 0.6:
        symptoms.append('gastro')
    if rng.rand() < 0.6:
        symptoms.append('aches')
    if rng.rand() < 0.3:
        symptoms.append('fatigue')

    progression = []
    for day in range(len_flu):
        progression.append(symptoms)

    start_day = None
    if rng.rand() < P_FLU: #gets a cold
        start_day = rng.choice(range(sim_days-len_flu))
        for day in range(len_flu):
            symptoms_array[start_day+day] = symptoms

    return progression, start_day, symptoms_array

def _get_flu_symptoms_v2(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick):
    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = 4 - round(carefulness)
    else:
        mean = 3 - round(carefulness)

    len_cold = rng.normal(mean,3)
    if len_cold < 1:
        len_cold = 1
    else:
        len_cold = round(len_cold)

    symptoms = []
    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms.append('moderate')
    else:
        symptoms.append('mild')
    if rng.rand() < 0.8:
        symptoms.append('fever')
    if rng.rand() < 0.6:
        symptoms.append('gastro')
    if rng.rand() < 0.6:
        symptoms.append('aches')
    if rng.rand() < 0.3:
        symptoms.append('fatigue')

    return symptoms

def _get_cold_symptoms(age, rng, sim_days, carefulness, preexisting_conditions, really_sick, extremely_sick):

    symptoms_array = [[] for i in range(sim_days)]

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = 4 - round(carefulness)
    else:
        mean = 3 - round(carefulness)

    len_cold = rng.normal(mean,3)
    if len_cold < 1:
        len_cold = 1
    else:
        len_cold = round(len_cold)

    len_cold = min(len_cold, sim_days-1)
    symptoms = []

    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms.append('moderate')
    else:
        symptoms.append('mild')
    if rng.rand() < 0.8:
        symptoms.append('runny_nose')
    if rng.rand() < 0.8:
        symptoms.append('cough')
    if rng.rand() < 0.1:
        symptoms.append('trouble_breathing')
    if rng.rand() < 0.2:
        symptoms.append('loss_of_taste')
    if rng.rand() < 0.2:
        symptoms.append('fatigue')
    if rng.rand() < 0.6:
        symptoms.append('sneezing')

    progression = []
    for day in range(len_cold):
        progression.append(symptoms)

    start_day = None
    if rng.rand() < P_COLD: #gets a cold
        start_day = rng.choice(range(sim_days-len_cold))
        for day in range(len_cold):
            symptoms_array[start_day+day] = symptoms


    return progression, start_day, symptoms_array

def _get_cold_symptoms_v2(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick):
    symptoms = []

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = 4 - round(carefulness)
    else:
        mean = 3 - round(carefulness)

    len_cold = rng.normal(mean,3)
    if len_cold < 1:
        len_cold = 1
    else:
        len_cold = round(len_cold)

    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms.append('moderate')
    else:
        symptoms.append('mild')

    if rng.rand() < 0.8:
        symptoms.append('runny_nose')
    if rng.rand() < 0.8:
        symptoms.append('cough')
    if rng.rand() < 0.1:
        symptoms.append('trouble_breathing')
    if rng.rand() < 0.2:
        symptoms.append('loss_of_taste')
    if rng.rand() < 0.2:
        symptoms.append('fatigue')
    if rng.rand() < 0.6:
        symptoms.append('sneezing')

    return symptoms

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
    #if rng.rand() < 0.6 + age/200:
    #    conditions = None
    #else:
    conditions = []

    # &smoking
    if age < 12:
        pass
    elif age < 18:
        if rng.rand() < 0.03:
            conditions.append('smoker')
    elif age < 65:
        if rng.rand() < 0.185:
            conditions.append('smoker')
    else:
        if rng.rand() < 0.09:
            conditions.append('smoker')

    # &diabetes
    if age < 18:
        if rng.rand() < .005:
            conditions.append('diabetes')
    elif age < 35:
        if rng.rand() < .009:
            conditions.append('diabetes')
    elif age < 50:
        if rng.rand() < .039:
            conditions.append('diabetes')
    elif age < 75:
        if rng.rand() < .13:
            conditions.append('diabetes')
    else:
        if rng.rand() < .179:
            conditions.append('diabetes')

    # &heart disease
    if 'diabetes' or 'smoker' in conditions:
        modifier = 2
    else:
        modifier = 0.5
    if age < 20:
        if rng.rand() < modifier *.001:
            conditions.append('heart_disease')
    elif age < 35:
        if rng.rand() < modifier * .005:
            conditions.append('heart_disease')
    elif age < 50:
        if sex.lower().startswith('f'):
            if rng.rand() < modifier * .013:
                conditions.append('heart_disease')
        elif sex.lower().startswith('m'):
            if rng.rand() < modifier * .021:
                conditions.append('heart_disease')
        else:
            if rng.rand() < modifier * .017:
                conditions.append('heart_disease')
    elif age < 75:
        if sex.lower().startswith('f'):
            if rng.rand() < modifier * .13:
                conditions.append('heart_disease')
        elif sex.lower().startswith('m'):
            if rng.rand() < modifier * .178:
                conditions.append('heart_disease')
        else:
            if rng.rand() < modifier * .15:
                conditions.append('heart_disease')
    else:
        if sex.lower().startswith('f'):
            if rng.rand() < modifier * .311:
                conditions.append('heart_disease')
        elif sex.lower().startswith('m'):
            if rng.rand() < modifier * .44:
                conditions.append('heart_disease')
        else:
            if rng.rand() < modifier * .375:
                conditions.append('heart_disease')

    # &cancer
    modifier = 1.3 if 'smoker' in conditions else 0.95
    if age < 30:
        if rng.rand() < modifier * 0.00029:
            conditions.append('cancer')
    elif age < 60:
        if rng.rand() < modifier * 0.0029:
            conditions.append('cancer')
    elif age < 90:
        if rng.rand() < modifier * 0.029:
            conditions.append('cancer')
    else:
        if rng.rand() < modifier * 0.05:
            conditions.append('cancer')


    # &COPD
    if age < 35:
        pass
    elif age < 50:
        if rng.rand() < modifier * .015:
            conditions.append('COPD')
    elif age < 65:
        if rng.rand() < modifier * .037:
            conditions.append('COPD')
    else:
        if rng.rand() < modifier * .075:
            conditions.append('COPD')

    # &asthma
    if age < 10:
        if sex.lower().startswith('f'):
            if rng.rand() < .07:
                conditions.append('asthma')
        elif sex.lower().startswith('m'):
            if rng.rand() < .12:
                conditions.append('asthma')
        else:
            if rng.rand() < .09:
                conditions.append('asthma')
    elif age < 25:
        if sex.lower().startswith('f'):
            if rng.rand() < .15:
                conditions.append('asthma')
        elif sex.lower().startswith('m'):
            if rng.rand() < .19:
                conditions.append('asthma')
        else:
            if rng.rand() < .17:
                conditions.append('asthma')
    elif age < 75:
        if sex.lower().startswith('f'):
            if rng.rand() < .11:
                conditions.append('asthma')
        elif sex.lower().startswith('m'):
            if rng.rand() < .06:
                conditions.append('asthma')
        else:
            if rng.rand() < .08:
                conditions.append('asthma')
    else:
        if sex.lower().startswith('f'):
            if rng.rand() < .12:
                conditions.append('asthma')
        elif sex.lower().startswith('m'):
            if rng.rand() < .08:
                conditions.append('asthma')
        else:
            if rng.rand() < .1:
                conditions.append('asthma')


    # &stroke
    modifier = len(conditions)
    if age < 20:
        pass
    elif age < 40:
        if rng.rand() < modifier * 0.01:
            conditions.append('stroke')
    elif age < 60:
        if rng.rand() < modifier * 0.03:
            conditions.append('stroke')
    elif age < 80:
        if rng.rand() < modifier * 0.04:
            conditions.append('stroke')
    else:
        if rng.rand() < modifier * 0.07:
            conditions.append('stroke')


    # &immuno-suppressed (3.6% on average)
    modifier = 1.2 if 'cancer' in conditions else 0.98
    if age < 40:
        if rng.rand() < modifier * 0.005:
            conditions.append('immuno-suppressed')
    elif age < 65:
        if rng.rand() < modifier * 0.036:
            conditions.append('immuno-suppressed')
    elif age < 85:
        if rng.rand() < modifier * 0.045:
            conditions.append('immuno-suppressed')
    else:
        if rng.rand() < modifier * 0.20:
            conditions.append('immuno-suppressed')

    #TODO PUT IN QUICKLY WITHOUT VERIFICATION OF NUMBERS
    if 'asthma' in conditions or 'COPD' in conditions:
        conditions.append('lung_disease')

    if sex.lower().startswith('f') and age > 18 and age < 50:
        p_pregnant = rng.normal(27,5)
        if rng.rand() < p_pregnant:
            conditions.append('pregnant')

    return conditions

# &canadian-demgraphics
def _get_random_age_multinomial(AGE_DISTRIBUTION, rng):
    x = list(zip(*AGE_DISTRIBUTION.items()))
    idx = rng.choice(range(len(x[0])), p=x[1])
    age_group = x[0][idx]
    return rng.uniform(age_group[0], age_group[1])

def _get_random_area(num, total_area, rng):
	''' Using Dirichlet distribution since it generates a "distribution of probabilities"
	which will ensure that the total area allotted to a location type remains conserved
	while also maintaining a uniform distribution'''

	# Keeping max at area/2 to ensure no location is allocated more than half of the total area allocated to its location type
	area = rng.dirichlet(np.ones(math.ceil(num/2)))*(total_area/2)
	area = np.append(area,rng.dirichlet(np.ones(math.floor(num/2)))*(total_area/2))
	return area

def _draw_random_discreet_gaussian(avg, scale, rng):
    # https://stackoverflow.com/a/37411711/3413239
    irange, normal_pdf = _get_integer_pdf(avg, scale, 2)
    return int(rng.choice(irange, size=1, p=normal_pdf))

def _json_serialize(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

def compute_distance(loc1, loc2):
    return np.sqrt((loc1.lat - loc2.lat) ** 2 + (loc1.lon - loc2.lon) ** 2)

def _encode_message(message):
    # encode a contact message as a string
    # TODO: clean up the bitarray => string transformation
    return str(np.array(message[0].tolist()).astype(int).tolist()) + "_" + str(np.array(message[1].tolist()).astype(int).tolist()) + "_" + str(message[2]) + "_" + str(message[3])

def _decode_message(message):
    # decode a string-encoded message into a tuple
    # TODO: make this a namedtuple
    m_i = message.split("_")
    obs_uid = bitarray(json.loads(m_i[0]))
    risk = bitarray(json.loads(m_i[1]))
    date_sent = datetime.datetime.strptime(m_i[2], '%Y-%m-%d %H:%M:%S')
    unobs_uid = int(m_i[3])
    return obs_uid, risk, date_sent, unobs_uid

@lru_cache(500)
def _get_integer_pdf(avg, scale, num_sigmas=2):
    irange = np.arange(avg - num_sigmas * scale, avg + num_sigmas * scale + 1)
    normal_pdf = norm.pdf(irange - avg)
    normal_pdf /= normal_pdf.sum()
    return irange, normal_pdf

# https://stackoverflow.com/questions/51843297/convert-real-numbers-to-binary-and-vice-versa-in-python
def float_to_binary(x, m, n):
    """Convert the float value `x` to a binary string of length `m + n`
    where the first `m` binary digits are the integer part and the last
    'n' binary digits are the fractional part of `x`.
    """
    x_scaled = round(x * 2 ** n)
    return '{:0{}b}'.format(x_scaled, m + n)

def binary_to_float(bstr, m, n):
    """Convert a binary string in the format '00101010100' to its float value."""
    return int(bstr, 2) / 2 ** n

def probas_to_risk_mapping(probas,
                           num_bins,
                           lower_cutoff=None,
                           upper_cutoff=None):
    """Create a mapping from probabilities returned by the model to discrete
    risk levels, with a number of predictions in each bins being approximately
    equivalent.

    Parameters
    ----------
    probas : `np.ndarray` instance
        The array of probabilities returned by the model.

    num_bins : int
        The number of bins. For example, `num_bins=16` for risk messages on
        4 bits.

    lower_cutoff : float, optional
        Ignore values smaller than `lower_cutoff` in the creation of the bins.
        This avoids any bias towards values which are too close to 0. If `None`,
        then do not cut off the small probabilities.

    upper_cutoff : float, optional
        Ignore values larger than `upper_cutoff` in the creation of the bins.
        This avoids any bias towards values which are too close to 1. If `None`,
        then do not cut off the large probabilities.

    Returns
    -------
    mapping : `np.ndarray` instance
        The mapping from probabilities to discrete risk levels. This mapping has
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
    """Create a callable, based on a mapping, that takes probabilities (in
    [0, 1]) and returns a discrete risk level (in [0, num_bins - 1]).

    Parameters
    ----------
    mapping : `np.ndarray` instance
        The mapping from probabilities to discrete risk levels. See
        `probas_to_risk_mapping`.

    Returns
    proba_to_risk : callable
        Function taking probabilities and returning discrete risk levels.
    """
    def _proba_to_risk(probas):
        return np.maximum(np.searchsorted(mapping, probas, side='left') - 1, 0)

    return _proba_to_risk
