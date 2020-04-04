import numpy as np
from scipy.stats import truncnorm
import datetime

def _normalize_scores(scores):
    return np.array(scores)/np.sum(scores)

# &canadian-demgraphics
def _get_random_age():
	# random normal centered on 50 with stdev 25
	draw = np.random.normal(50, 25, 1)
	if draw < 0:
		# if below 0, shift to a bump centred around 30
		age = round(30 + np.random.normal(0, 4))
	else:
		age = round(float(draw))
	return age

# &sex
def _get_random_sex():
	p = np.random.rand()
	if p < .4:
		return 'female'
	elif p < .8:
		return 'male'
	else:
		return 'other'

# &preexisting-conditions
def _get_preexisting_conditions(age, sex):
	#if np.random.rand() < 0.6 + age/200: 
	#	conditions = None
	#else:
	conditions = []

	# &diabetes
	if age < 18:
		if np.random.rand() < .005:
			conditions.append('diabetes')
	elif age < 35:
		if np.random.rand() < .009:
			conditions.append('diabetes')
	elif age < 50:
		if np.random.rand() < .039:
			conditions.append('diabetes')
	elif age < 75:
		if np.random.rand() < .13:
			conditions.append('diabetes')
	else:
		if np.random.rand() < .179:
			conditions.append('diabetes')
	
	# &heart disease
	if age < 20:
		if np.random.rand() < .001:
			conditions.append('heart_disease')
	elif age < 35:
		if np.random.rand() < .005:
			conditions.append('heart_disease')
	elif age < 50:
		if sex.lower().startswith('f'):
			if np.random.rand() < .013:
				conditions.append('heart_disease')
		elif sex.lower().startswith('m'):
			if np.random.rand() < .021:
				conditions.append('heart_disease')
		else:
			if np.random.rand() < .017:
				conditions.append('heart_disease')
	elif age < 75:
		if sex.lower().startswith('f'):
			if np.random.rand() < .13:
				conditions.append('heart_disease')
		elif sex.lower().startswith('m'):
			if np.random.rand() < .178:
				conditions.append('heart_disease')
		else:
			if np.random.rand() < .15:
				conditions.append('heart_disease')
	else:
		if sex.lower().startswith('f'):
			if np.random.rand() < .311:
				conditions.append('heart_disease')
		elif sex.lower().startswith('m'):
			if np.random.rand() < .44:
				conditions.append('heart_disease')
		else:
			if np.random.rand() < .375:
				conditions.append('heart_disease')

	# &COPD
	if age < 35:
		pass
	elif age < 50:
		if np.random.rand() < .015:
			conditions.append('COPD')
	elif age < 65:
		if np.random.rand() < .037:
			conditions.append('COPD')
	else:
		if np.random.rand() < .075:
			conditions.append('COPD')

	# &asthma 
	if age < 10:
		if sex.lower().startswith('f'):
			if np.random.rand() < .07:
				conditions.append('asthma')
		elif sex.lower().startswith('m'):
			if np.random.rand() < .12:
				conditions.append('asthma')
		else:
			if np.random.rand() < .09:
				conditions.append('asthma')
	elif age < 25:
		if sex.lower().startswith('f'):
			if np.random.rand() < .15:
				conditions.append('asthma')
		elif sex.lower().startswith('m'):
			if np.random.rand() < .19:
				conditions.append('asthma')
		else:
			if np.random.rand() < .17:
				conditions.append('asthma')
	elif age < 75:
		if sex.lower().startswith('f'):
			if np.random.rand() < .11:
				conditions.append('asthma')
		elif sex.lower().startswith('m'):
			if np.random.rand() < .06:
				conditions.append('asthma')
		else:
			if np.random.rand() < .08:
				conditions.append('asthma')
	else:
		if sex.lower().startswith('f'):
			if np.random.rand() < .12:
				conditions.append('asthma')
		elif sex.lower().startswith('m'):
			if np.random.rand() < .08:
				conditions.append('asthma')
		else:
			if np.random.rand() < .1:
				conditions.append('asthma')

	return conditions

def _draw_random_discreet_gaussian(avg, scale):
    # https://stackoverflow.com/a/37411711/3413239
    return int(truncnorm(a=-1, b=1, loc=avg, scale=scale).rvs(1).round().astype(int)[0])

def _json_serialize(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()
