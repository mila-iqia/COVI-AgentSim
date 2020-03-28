import numpy as np
from scipy.stats import truncnorm

def _normalize_scores(scores):
    return np.array(scores)/np.sum(scores)

def _draw_random_discreet_gaussian(avg, scale):
    # https://stackoverflow.com/a/37411711/3413239
    return int(truncnorm(a=-1, b=1, loc=avg, scale=scale).rvs(1).round().astype(int)[0])
