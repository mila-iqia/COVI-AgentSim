import numpy as np
import scipy.stats

def ci(samples, level=0.95, axis=0):
    # https://en.wikipedia.org/wiki/Confidence_interval#Basic_steps
    # https://stackoverflow.com/a/15034143/6826667
    num_samples = samples.shape[axis]
    mean = np.mean(samples, axis=axis)
    
    # Standard error of the mean
    se = scipy.stats.sem(samples, axis=0)
    
    # Critical value from standard Student t table
    t = scipy.stats.t.ppf(0.5 * (1 + level), num_samples - 1)
    
    # Confidence interval t_{alpha, n-1} * sigma / sqrt(n)
    interval = t * se

    return (mean - interval, mean + interval)

def std(samples, axis=0, **kwargs):
    mean = np.mean(samples, axis=axis)
    std = np.std(samples, axis=axis)

    return (mean - std, mean + std)

def se(samples, level=0.95, axis=0):
    mean = np.mean(samples, axis=axis)
    se = scipy.stats.sem(samples, axis=0)

    return (mean - se, mean + se)