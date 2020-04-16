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

def _get_mask_wearing(carefulness, simulation_days, rng):
	return [rng.rand() < carefulness*BASELINE_P_MASK for day in range(simulation_days)]


def _sample_viral_load_gamma(rng, shape_mean=4.5, shape_std=.15, scale_mean=1., scale_std=.15):
	""" This function samples the shape and scale of a gamma distribution, then returns it"""
	shape = rng.normal(shape_mean, shape_std)
	scale = rng.normal(scale_mean, scale_std)
	return gamma(shape, scale=scale)


def _sample_viral_load_piecewise(rng, age=40):
	""" This function samples a piece-wise linear viral load model which increases, plateaus, and drops """
	# https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
	plateau_start = truncnorm((PLATEAU_START_CLIP_LOW - PLATEAU_START_MEAN)/PLATEAU_START_STD, (PLATEAU_START_CLIP_HIGH - PLATEAU_START_MEAN) / PLATEAU_START_STD, loc=PLATEAU_START_MEAN, scale=PLATEAU_START_STD).rvs(1, random_state=rng)
	plateau_end = plateau_start + truncnorm((PLATEAU_DURATION_CLIP_LOW - PLATEAU_DURATION_MEAN)/PLEATEAU_DURATION_STD,
											(PLATEAU_DURATION_CLIP_HIGH - PLATEAU_DURATION_MEAN) / PLEATEAU_DURATION_STD,
											loc=PLATEAU_DURATION_MEAN, scale=PLEATEAU_DURATION_STD).rvs(1, random_state=rng)
	recovered = plateau_end + ((age/10)-1) # age is a determining factor for the recovery time
	recovered = recovered + truncnorm((plateau_end - RECOVERY_MEAN) / RECOVERY_STD,
										(RECOVERY_CLIP_HIGH - RECOVERY_MEAN) / RECOVERY_STD,
										loc=RECOVERY_MEAN, scale=RECOVERY_STD).rvs(1, random_state=rng)
	plateau_height = rng.uniform(MIN_VIRAL_LOAD, MAX_VIRAL_LOAD)
	return plateau_height, plateau_start, plateau_end, recovered

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

# 2D Array of symptoms; first axis is days after exposure (infection), second is an array of symptoms
def _get_all_symptoms(viral_load_plateau_start, viral_load_plateau_end,
	                        viral_load_recovered, age, incubation_days, really_sick, extremely_sick,
							rng, preexisting_conditions):
        # Before showing symptoms
        symptoms_array = [[] for i in range(incubation_days)]
        # Before the plateau
        for day in range(round(viral_load_plateau_start)-1):
            symptoms = []
            if really_sick or extremely_sick or any(preexisting_conditions):
                symptoms.append('moderate')
            else :
                symptoms.append('mild')
            if rng.rand() < 0.9:
                symptoms.append('fever')
            if rng.rand() < 0.7:
                symptoms.append('cough')
            if rng.rand() < 0.5:
                symptoms.append('fatigue')
            if rng.rand() < 0.3:
                symptoms.append('trouble_breathing')
            if rng.rand() < 0.4:
                symptoms.append('gastro')
            symptoms_array.append(symptoms)

        # During the plateau
        for day in range(round(viral_load_plateau_end - viral_load_plateau_start)):
            symptoms = []
            if really_sick or any(preexisting_conditions):
                symptoms.append('severe')
            elif extremely_sick:
                symptoms.append('extremely-severe')
            elif rng.rand() < 0.4:
                symptoms.append('moderate')
            else:
                symptoms.append('mild')
            if rng.rand() < 0.9:
                symptoms.append('fever')
            if rng.rand() < 0.85:
                symptoms.append('cough')
            if rng.rand() < 0.8:
                symptoms.append('fatigue')
            if rng.rand() < 0.7:
                symptoms.append('trouble_breathing')
            if rng.rand() < 0.1:
                symptoms.append('runny_nose')
            if rng.rand() < 0.4:
                symptoms.append('loss_of_taste')
            if rng.rand() < 0.1:
                symptoms.append('gastro')
            symptoms_array.append(symptoms)

        # After the plateau
        for day in range(round(viral_load_recovered - viral_load_plateau_end)):
            symptoms = []
            if really_sick or extremely_sick:
                symptoms.append('moderate')
            else:
                symptoms.append('mild')
            if rng.rand() < 0.3:
                symptoms.append('cough')
            if rng.rand() < 0.8:
                symptoms.append('fatigue')
            if rng.rand() < 0.5:
                symptoms.append('aches')
            if rng.rand() < 0.3:
                symptoms.append('trouble_breathing')
            if rng.rand() < 0.2:
                symptoms.append('gastro')
            symptoms_array.append(symptoms)
        return symptoms_array

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
	#	conditions = None
	#else:
	conditions = []

	# &immuno-suppressed (3.6% on average)
	if age < 40:
		if rng.rand() < 0.005:
			conditions.append('immuno-suppressed')
	elif age < 65:
		if rng.rand() < 0.036:
			conditions.append('immuno-suppressed')
	elif age < 85:
		if rng.rand() < 0.045:
			conditions.append('immuno-suppressed')
	else:
		if rng.rand() < 0.20:
			conditions.append('immuno-suppressed')

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
	if age < 20:
		if rng.rand() < .001:
			conditions.append('heart_disease')
	elif age < 35:
		if rng.rand() < .005:
			conditions.append('heart_disease')
	elif age < 50:
		if sex.lower().startswith('f'):
			if rng.rand() < .013:
				conditions.append('heart_disease')
		elif sex.lower().startswith('m'):
			if rng.rand() < .021:
				conditions.append('heart_disease')
		else:
			if rng.rand() < .017:
				conditions.append('heart_disease')
	elif age < 75:
		if sex.lower().startswith('f'):
			if rng.rand() < .13:
				conditions.append('heart_disease')
		elif sex.lower().startswith('m'):
			if rng.rand() < .178:
				conditions.append('heart_disease')
		else:
			if rng.rand() < .15:
				conditions.append('heart_disease')
	else:
		if sex.lower().startswith('f'):
			if rng.rand() < .311:
				conditions.append('heart_disease')
		elif sex.lower().startswith('m'):
			if rng.rand() < .44:
				conditions.append('heart_disease')
		else:
			if rng.rand() < .375:
				conditions.append('heart_disease')

	# &COPD
	if age < 35:
		pass
	elif age < 50:
		if rng.rand() < .015:
			conditions.append('COPD')
	elif age < 65:
		if rng.rand() < .037:
			conditions.append('COPD')
	else:
		if rng.rand() < .075:
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

	return conditions


# &canadian-demgraphics
def _get_random_age_multinomial(AGE_DISTRIBUTION, rng):
    x = list(zip(*AGE_DISTRIBUTION.items()))
    idx = rng.choice(range(len(x[0])), p=x[1])
    age_group = x[0][idx]
    return rng.uniform(age_group[0], age_group[1])


def _get_random_area(location_type, num, total_area, rng):
	''' Using Dirichlet distribution since it generates a "distribution of probabilities"
	which will ensure that the total area allotted to a location type remains conserved
	while also maintaining a uniform distribution'''
	perc_dist = {"store":0.15, "misc":0.15, "workplace":0.2, "household":0.3, "park":0.05, 'hospital': 0.6, "school":0.05, "senior_residency":0.05}

	# Keeping max at area/2 to ensure no location is allocated more than half of the total area allocated to its location type
	area = rng.dirichlet(np.ones(math.ceil(num/2)))*(perc_dist[location_type]*total_area/2)
	area = np.append(area,rng.dirichlet(np.ones(math.floor(num/2)))*(perc_dist[location_type]*total_area/2))

	return area

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
