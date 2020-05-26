import datetime
import math
import warnings
from pathlib import PosixPath

import numpy as np
import pandas as pd
import pytest

from covid19sim import utils
from covid19sim.base import Env, City
from covid19sim.simulator import Human
from tests.utils import get_test_conf


@pytest.mark.parametrize('test_conf_name', ['test_covid_testing.yaml'])
def test_age_distribution(
        test_conf_name: str,
):
    """
        Test for the expected demographic statistics related to the number of people in the population
            - age distribution

    Args:
        test_conf_name (str): the filename of the configuration file used for testing
    """
    warnings.filterwarnings('ignore')

    conf = get_test_conf(test_conf_name)

    seed = 0
    n_people = 1000
    init_percent_sick = 0.01
    rng = np.random.RandomState(seed=seed)
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    city = City(
        env=env,
        n_people=n_people,
        init_percent_sick=init_percent_sick,
        rng=rng,
        x_range=city_x_range,
        y_range=city_y_range,
        Human=Human,
        conf=conf,
    )

    # Check the actual population size is the same than specified
    assert len(city.humans) == n_people

    # Demographics
    population = []
    for human in city.humans:
        population.append([
            human.age,
            human.sex,
        ])

    df = pd.DataFrame.from_records(
        data=population,
        columns=['age', 'sex']
    )

    # Check the age distribution
    age_histogram = {}
    for key, item in conf.get('HUMAN_DISTRIBUTION').items():
        age_histogram[key] = item['p']
    intervals = pd.IntervalIndex.from_tuples(age_histogram.keys(), closed='left')
    age_grouped = df.groupby(pd.cut(df['age'], intervals))
    age_grouped = age_grouped.agg({'age': 'count'})
    assert age_grouped.age.sum() == n_people
    stats = age_grouped.age.apply(lambda x: x / n_people)
    assert np.allclose(stats.to_numpy(), np.array(list(age_histogram.values())), atol=0.01)

    # TODO: Check the sex distribution

    # TODO: Check the profession distribution


@pytest.mark.parametrize('test_conf_name', ['test_covid_testing.yaml'])
def test_household_distribution(
        test_conf_name: str,
        avg_household_error_tol: float = 0.01
):
    """
        Test for the expected demographic statistics related to the households
            - compare the average number of people per household to Canadian statistics
            - distribution of the number of people per household

    Args:
        test_conf_name (str): the filename of the configuration file used for testing
    """
    warnings.filterwarnings('ignore')

    conf = get_test_conf(test_conf_name)

    seed = 0
    n_people = 1000
    init_percent_sick = 0.01
    rng = np.random.RandomState(seed=seed)
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    city = City(
        env=env,
        n_people=n_people,
        init_percent_sick=init_percent_sick,
        rng=rng,
        x_range=city_x_range,
        y_range=city_y_range,
        Human=Human,
        conf=conf,
    )

    # Value from Canada Statistics - Census profile 2006 (ref: https://tinyurl.com/qsf2q8d)
    avg_household_size = 2.4
    n_people_in_households = 0
    for household in city.households:
        n_people_in_households += len(household.residents)
    emp_avg_household_size = n_people_in_households/len(city.households)

    # TODO: this test is broken for now (our avg is 3.10 while the Canadian avg is 2.4)
    # assert math.fabs(emp_avg_household_size - avg_household_size) < avg_household_error_tol, \
    #    f'The empirical average household size is {emp_avg_household_size:.2f}'  \
    #    f' while the statistics for Canada is {avg_household_size:.2f}'

    # Verify that each human is associated to a household
    for human in city.humans:
        assert human.household


@pytest.mark.parametrize('test_conf_name', ['test_covid_testing.yaml'])
@pytest.mark.parametrize('app_uptake', [None, 0.25, 0.5, 1.0])
def test_app_distribution(
        test_conf_name: str,
        app_uptake: float
):
    """
        Test for the expected demographic statistics related to the app users
            - age distribution of the app users

    Args:
        test_conf_name (str): the filename of the configuration file used for testing
        app_uptake (float): probability that an individual with a smartphone has the app

    """
    warnings.filterwarnings('ignore')

    conf = get_test_conf(test_conf_name)

    if app_uptake:
        conf['APP_UPTAKE'] = app_uptake

    n_people = 1000
    init_percent_sick = 0.01
    start_time = datetime.datetime(2020, 2, 28, 0, 0)

    seed = 0
    rng = np.random.RandomState(seed=seed)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    city = City(
        env=env,
        n_people=n_people,
        init_percent_sick=init_percent_sick,
        rng=rng,
        x_range=city_x_range,
        y_range=city_y_range,
        Human=Human,
        conf=conf,
    )

    # Demographics
    population = []
    for human in city.humans:
        population.append([
            human.age,
            human.sex,
            human.has_app,
        ])

    df = pd.DataFrame.from_records(
        data=population,
        columns=['age', 'sex', 'has_app']
    )

    # Check the age distribution of the app users
    if conf.get('APP_UPTAKE') < 0:
        age_app_histogram = conf.get('SMARTPHONE_OWNER_FRACTION_BY_AGE')
        age_app_groups = [(low, up) for low, up in age_app_histogram]  # make the age groups contiguous
        intervals = pd.IntervalIndex.from_tuples(age_app_groups, closed='left')
        age_grouped = df.groupby(pd.cut(df['age'], intervals))
        age_grouped = age_grouped.agg({'age': 'count', 'has_app': 'sum'})
        assert age_grouped.age.sum() == n_people
        age_stats = age_grouped.age.apply(lambda x: x / n_people)
        app_stats = age_grouped.has_app.apply(lambda x: x / n_people)
        assert np.allclose(age_stats.to_numpy(), app_stats.to_numpy())

    else:

        abs_age_histogram = utils.relativefreq2absolutefreq(
            bins_fractions={age_bin: specs['p'] for age_bin, specs in conf.get('HUMAN_DISTRIBUTION').items()},
            n_elements=n_people,
            rng=city.rng
        )
        n_apps_per_age = {
            k: math.ceil(abs_age_histogram[k]*v*conf.get('APP_UPTAKE'))
            for k, v in conf.get("SMARTPHONE_OWNER_FRACTION_BY_AGE").items()
        }
        n_apps = np.sum(list(n_apps_per_age.values()))

        intervals = pd.IntervalIndex.from_tuples(n_apps_per_age.keys(), closed='left')
        age_grouped = df.groupby(pd.cut(df['age'], intervals))
        age_grouped = age_grouped.agg({'age': 'count', 'has_app': 'sum'})
        assert age_grouped.age.sum() == n_people
        assert age_grouped.has_app.sum() == n_apps
        stats = age_grouped.has_app.apply(lambda x: x / n_apps)

        assert np.allclose(stats.to_numpy(), np.array(list(n_apps_per_age.values()))/n_apps)
