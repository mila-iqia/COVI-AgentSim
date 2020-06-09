import numpy as np

def generate_synthetic_cases_per_day(t, Rt, cases_per_day0, serial_interval_dist):
    """
    """

    cases_per_day = np.zeros_like(t)
    cases_per_day[0] = cases_per_day0
    for i in range(1,len(t)):
        cases_per_day[i] = np.random.poisson(Rt[i]*np.sum(cases_per_day[max(0,i-len(serial_interval_dist)):i][::-1] * \
            serial_interval_dist[:i]))

    return cases_per_day