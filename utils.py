import numpy as np
from scipy.stats import truncnorm
import datetime
import math

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

def _get_random_area(location_type, num, total_area, rng):
	''' Using Dirichlet distribution since it generates a "distribution of probabilities" 
	which will ensure that the total area allotted to a location type remains conserved 
	while also maintaining a uniform distribution'''
	perc_dist = {"store":0.15, "misc":0.15, "workplace":0.2, "household":0.3, "park":0.5}
	
	# Keeping max at area/2 to ensure no location is allocated more than half of the total area allocated to its location type 
	area = rng.dirichlet(np.ones(math.ceil(num/2)))*(perc_dist[location_type]*total_area/2)
	area = np.append(area,rng.dirichlet(np.ones(math.floor(num/2)))*(perc_dist[location_type]*total_area/2))
	
	return area

def _draw_random_discreet_gaussian(avg, scale, rng):
    # https://stackoverflow.com/a/37411711/3413239
    return int(truncnorm(a=-1, b=1, loc=avg, scale=scale).rvs(1, random_state=rng).round().astype(int)[0])

def _json_serialize(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

def compute_distance(loc1, loc2):
    return np.sqrt((loc1.lat - loc2.lat) ** 2 + (loc1.lon - loc2.lon) ** 2)
