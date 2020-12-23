"""
Extracts metrics from tracker
"""
import numpy as np
import datetime

from covid19sim.utils.constants import POSITIVE_TEST_RESULT, NEGATIVE_TEST_RESULT

def SEIR_Map(state):
    """
    Encodes the literal SEIR state to an integer.

    Args:
        (str): State of the human i.e. S, E, I, or R
    """
    if state == "S":
        return 0
    if state == "E":
        return 1
    if state == "I":
        return 2
    if state == "R":
        return 3

def get_quarantined_states(data):
    """
    Extracts relevant keys and values from tracker data to return a compact numpy array representing SEIR states of humans

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        humans_quarantined_state (np.array): 2D binary array of (n_people, n_days) where each element is 1 if human was quarantining at that time
    """
    all_humans = sorted(data['humans_state'].keys(), key = lambda x: int(x.split(":")[-1]))
    assert len(all_humans) == data['n_humans']

    n_obs = len(data['humans_quarantined_state'][all_humans[0]])
    humans_quarantined_state = np.zeros((data['n_humans'], n_obs))

    for i, human in enumerate(all_humans):
        humans_quarantined_state[i] = data['humans_quarantined_state'][human]

    return humans_quarantined_state

def get_SEIR_states(data):
    """
    Extracts relevant keys and values from tracker data to return a compact numpy array representing SEIR states of humans

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        humans_state (np.array): 2D array of (n_people, n_days) where each element is integer encoded SEIR state of humans
    """
    all_humans = sorted(data['humans_state'].keys(), key = lambda x: int(x.split(":")[-1]))
    assert len(all_humans) == data['n_humans']

    n_obs = len(data['humans_quarantined_state'][all_humans[0]])
    humans_state = np.zeros((data['n_humans'], n_obs))

    for i, human in enumerate(all_humans):
        humans_state[i] = list(map(SEIR_Map, data['humans_state'][human]))

    return humans_state

def get_SEIR_quarantined_states(data):
    """
    Extracts relevant keys and values from tracker data to return a compact numpy array representing SEIR and Quarantined states of humans

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        humans_state (np.array): 2D array of (n_people, n_days) where each element is integer encoded SEIR state of humans
        humans_quarantined_state (np.array): 2D binary array of (n_people, n_days) where each element is 1 if human was quarantining at that time
    """
    return get_SEIR_states(data), get_quarantined_states(data)


##################################################################
##############               DAILY SERIES           ##############
##################################################################

def _daily_fraction_of_population_infected(data):
    """
    Returns a time series of number of daily infections as a fraction of population

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is fraction representing the proportion of population infected on that simulation day
    """
    n_people = data['n_humans']
    cases_per_day = data['cases_per_day']
    cases_per_day[0] = 0 # on day 0 there are no infections but this array contains initially infected people
    return np.array(cases_per_day) / n_people

def _daily_fraction_quarantine(data):
    """
    Returns a time series of total number of people that quarantined on a simulation day

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is an integer representing the number of people quarantining on that simulation day
    """
    n_people = data['n_humans']
    states, quarantined_states = get_SEIR_quarantined_states(data)
    daily_quarantine = quarantined_states.sum(axis=0)
    return daily_quarantine / n_people

def _daily_fraction_ill(data):
    """
    Returns a time series of total number of people that quarantined on a simulation day

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is an integer representing the number of people quarantining on that simulation day
    """
    n_people = data['n_humans']
    states, quarantined_states = get_SEIR_quarantined_states(data)
    daily_quarantine = quarantined_states.sum(axis=0)
    return daily_quarantine / n_people

def _daily_false_quarantine(data):
    """
    Returns a time series of fraction of population quarantining on a simulation day

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is a fraction of population that false quarantined on that simulation day
    """
    intervention_day = data['intervention_day']
    n_people = data['n_humans']

    states, quarantined_states = get_SEIR_quarantined_states(data)
    states = states[:, intervention_day:]
    quarantined_states = quarantined_states[:, intervention_day:]

    #
    false_quarantine = ((quarantined_states == 1) & ( (states == 0) | (states == 3) )).sum(axis=0)
    daily_false_quarantine = false_quarantine / n_people
    return daily_false_quarantine

def _daily_false_susceptible_recovered(data):
    """
    Returns a time series of fraction of population that was falsely identified as non-risky i.e. susceptible or recovered

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is a fraction of population that was falsely classified as susceptible or recovered
    """
    intervention_day = data['intervention_day']
    n_people = data['n_humans']

    states, quarantined_states = get_SEIR_quarantined_states(data)
    states = states[:, intervention_day:]
    quarantined_states = quarantined_states[:, intervention_day:]

    daily_false_not_quarantine = ((quarantined_states == 0) & ((states == 1) | (states == 2))).sum(axis=0)
    daily_false_not_quarantine = daily_false_not_quarantine / n_people

    return daily_false_not_quarantine

def _daily_susceptible_recovered(data):
    """
    Returns a time series of fraction of population that is either S or R

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is a fraction for that simulation day
    """
    intervention_day = data['intervention_day']
    n_people = data['n_humans']

    states, _ = get_SEIR_quarantined_states(data)
    return ((states == 0) | (states == 1)).sum(axis=0) / n_people

def _daily_infected(data):
    """
    Returns a time series of fraction of population that is either E or I

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is a fraction for that simulation day
    """
    intervention_day = data['intervention_day']
    n_people = data['n_humans']

    states, _ = get_SEIR_quarantined_states(data)
    return ((states == 1) | (states == 2)).sum(axis=0) / n_people

def _daily_fraction_risky_classified_as_non_risky(data):
    """
    Returns a time series of ** fraction of infected people ** that are not in quarantine

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is a fraction for each simulation day
    """
    risky = _daily_infected(data)
    classified_non_risky = _daily_false_susceptible_recovered(data)

    m = np.zeros_like(risky)
    np.divide(classified_non_risky, risky, where=risky!=0, out=m)
    return m

def _daily_fraction_non_risky_classified_as_risky(data):
    """
    Returns a time series of ** fraction of not-infected people (S or R) ** that are in quarantine

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each element is a fraction for each simulation day
    """
    non_risky = _daily_susceptible_recovered(data)
    classified_risky = _daily_false_quarantine(data)

    m = np.zeros_like(non_risky)
    np.divide(classified_risky, non_risky, where=non_risky!=0, out=m)
    return m

def _daily_number_of_tests(data):
    """
    Returns a time series of number of tests per day

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is the number of tests on that simulation day
    """
    min_date = datetime.datetime.strptime(data['SIMULATION_START_TIME'], "%Y-%m-%d %H:%M:%S").date()
    # n_days = data['simulation_days']
    max_date = max(x['test_time'].date() for x in data['test_monitor'])
    n_days = (max_date - min_date).days + 1

    n_tests_per_day = np.zeros(n_days)
    for test in data['test_monitor']:
        day_idx = (test['test_time'].date() - min_date).days
        n_tests_per_day[day_idx] +=  1

    return n_tests_per_day

def _daily_positive_test_results(data):
    """
    Returns a time series of number of positive tests per simulation day

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is the number of positive tests on that simulation day
    """
    min_date = datetime.datetime.strptime(data['SIMULATION_START_TIME'], "%Y-%m-%d %H:%M:%S").date()
    # n_days = data['simulation_days']
    max_date = max(x['test_time'].date() for x in data['test_monitor'])
    n_days = (max_date - min_date).days + 1

    n_positive_tests_per_day = np.zeros(n_days)
    for test in data['test_monitor']:
        if (
            test['result_time'].date() <= max_date
            and test['test_result'] == POSITIVE_TEST_RESULT
        ):
            day_idx = (test['result_time'].date() - min_date).days
            n_positive_tests_per_day[day_idx] +=  1

    return n_positive_tests_per_day

def X_daily_fraction_ill_not_working(data):
    """
    Returns a time series of fraction of population that is ill and not working

    Args:
       (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is the fraction of population that cancelled work due to illness
    """
    raise NotImplementedError()

def _daily_fraction_cumulative_cases(data):
    """
    Returns a series where each value is a true fraction of population that is infected upto some simulation day

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is the above described fraction
    """
    x = data['cases_per_day']
    return np.cumsum(x) / data['n_humans']

def _daily_incidence(data):
    """
    Returns a series where each value is disease incidence i.e. infected / susceptible per 1000 people

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is the above described fraction
    """
    daily_n_susceptible = data['s']
    daily_cases = data['cases_per_day']
    incidence = []
    for s, n in zip(daily_n_susceptible, daily_cases[1:]):
        incidence.append(n / s)

    return np.array(incidence) * 1000

def _daily_prevalence(data):
    """
    Returns a series where each value is a true fraction of currently infected population.

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is the above described fraction
    """
    n_infected_per_day = data['ei_per_day']
    n_people = data['n_humans']
    prevalence = np.array(n_infected_per_day) / n_people
    return prevalence


##################################################################
##############               SCALARS                ##############
##################################################################

def _mean_effective_contacts(data):
    """
    Returns mean effective contacts across the population i.e. actual interactions that took place.

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (float): Average interactions that took place across the population on a typical day
    """
    return data['effective_contacts_since_intervention']

def _mean_healthy_effective_contacts(data):
    """
    Returns mean effective contacts across the population between non-risky individuals only

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (float): Average 0 risk interactions that took place across the population on a typical day
    """
    return data['healthy_effective_contacts_since_intervention']

def _percentage_total_infected(data):
    """
    Returns the fraction of population infected by the end of the simulation

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (float): Fraction of population infected by the end of the simulation
    """
    return 100 * sum(data['cases_per_day'])/data['n_humans']

def _positivity_rate(data):
    """
    Returns a ** fraction of positive test results ** of all the tests

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (float): positivty rate
    """
    test_monitor = data['test_monitor']
    return sum(x['test_result'] == POSITIVE_TEST_RESULT for x in test_monitor)/len(test_monitor)


##################################################################
##############           OTHER SERIES               ##############
##################################################################

def _cumulative_infected_by_recovered_people(data):
    """
    Returns a series of a total number of infected people by recovered people.
    Value at `idx` position in the array is cumulative infected up until `idx` number of infected people have recovered.

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is as described above.
    """
    n_infected_per_recovered = [x[1] for x in data['recovered_stats']['timestamps']]
    return np.cumsum(n_infected_per_recovered)

def _fraction_cumulative_infected_by_recovered_people(data):
    """
    Returns a series of a fraction of infected population by recovered people.
    Value at `idx` position in the array is the fraction of cumulative infected up until `idx` number of infected people have recovered.

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is as described above.
    """
    return _cumulative_infected_by_recovered_people(data) / data['n_humans']

def _proxy_R_estimated_by_recovered_people(data):
    """
    Returns a series where each value is estimated R, where estimation is by average outdegree of recovered infectors.
    A value at `idx` represents estimated R when `idx` number of infectors have recovered.

    Args:
        (dict): tracker data loaded from pkl file.

    Returns:
        (np.array): 1D array where each value is as described above.
    """
    cumulative_infected = _cumulative_infected_by_recovered_people(data)
    n_recovered = np.arange(1, len(cumulative_infected)+1)
    return cumulative_infected / n_recovered
