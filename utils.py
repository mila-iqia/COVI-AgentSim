import numpy as np
from scipy.stats import truncnorm
import datetime

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


def _draw_random_discreet_gaussian(avg, scale, rng):
    # https://stackoverflow.com/a/37411711/3413239
    return int(truncnorm(a=-1, b=1, loc=avg, scale=scale).rvs(1, random_state=rng).round().astype(int)[0])

def _json_serialize(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

def compute_distance(loc1, loc2):
    return np.sqrt((loc1.lat - loc2.lat) ** 2 + (loc1.lon - loc2.lon) ** 2)
