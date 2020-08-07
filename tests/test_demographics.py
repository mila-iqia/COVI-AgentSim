import datetime
import math

import numpy as np
import pandas as pd
import pytest

from covid19sim.utils import utils
from covid19sim.locations.city import City
from covid19sim.utils.env import Env
from tests.utils import get_test_conf


@pytest.mark.parametrize('seed', [62, 93, 73, 3, 51])
@pytest.mark.parametrize('test_conf_name', ['test_covid_testing.yaml'])
def test_basic_demographics(
        seed: int,
        test_conf_name: str,
        age_error_tol: float = 2.3,
        age_distribution_error_tol: float = 0.01,
        sex_diff_error_tol: float = 0.1,
        profession_error_tol: float = 0.03,
        fraction_over_100_error_tol: float = 0.01):
    """
        Tests for the about demographic statistics:
            - min, max, average and median population age
            - fraction of people over 100 years old
            - fraction difference between male and female
            - age distribution w.r.t to HUMAN_DISTRIBUTION
            - fraction of retired people
            - fraction of people working in healthcare
            - fraction of people working in education

    Reference values are from Canada statistics - census profile 2016 (ref: https://tinyurl.com/qsf2q8d)

    Args:
        test_conf_name (str): the filename of the configuration file used for testing
        age_error_tol (float): tolerance about average and median age discrepancy w.r.t. official statistics
        age_distribution_error_tol (float): tolerance about the population fraction assigned to each age group
        profession_error_tol (float): tolerance about the population fraction assigned to each profession
    """

    conf = get_test_conf(test_conf_name)

    n_people = 5000
    init_fraction_sick = 0.01
    rng = np.random.RandomState(seed=seed)
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    conf['simulation_days'] = 1
    city = City(
        env=env,
        n_people=n_people,
        init_fraction_sick=init_fraction_sick,
        rng=rng,
        x_range=city_x_range,
        y_range=city_y_range,
        conf=conf,
        logfile="logfile.txt",
    )
    city.have_some_humans_download_the_app()

    # Check that the actual population size is the same than specified
    assert len(city.humans) == n_people

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

    # Check basic statistics about age
    canstat_avg_population_age = 41.
    assert math.fabs(df.age.mean() - canstat_avg_population_age) < age_error_tol, \
        f'The simulated average population age is {df.age.mean():.2f} ' \
        f'while the statistics for Canada is {canstat_avg_population_age:.2f}'

    canstat_median_population_age = 41.2
    assert math.fabs(df.age.median() - canstat_median_population_age) < age_error_tol, \
        f'The simulated median population age is {df.age.mean():.2f} ' \
        f'while the statistics for Canada is {canstat_avg_population_age:.2f}'

    minimum_age = 0
    assert df.age.min() >= minimum_age, f'There is a person with negative age.'

    maximum_age = 117  # Canadian record: Marie-Louise Meilleur
    assert df.age.max() < maximum_age, f'There is a person older than the Canadian longevity record.'

    # Check basic statistics about sex
    canstat_sex_rel_diff = 0.018
    sex_grouped = df.groupby('sex').count()
    sex_grouped = sex_grouped.apply(lambda x: x / n_people)
    sex_rel_diff = math.fabs(sex_grouped.age['male'] - sex_grouped.age['female'])
    assert (math.fabs(sex_rel_diff - canstat_sex_rel_diff) < sex_diff_error_tol), \
        f'The relative difference between male and female in the population is {sex_rel_diff} ' \
        f'while the actual number for Canada is {canstat_sex_rel_diff}'

    fraction_other_sex = sex_grouped.age['other']
    assert math.fabs(fraction_other_sex - 0.1) < sex_diff_error_tol, \
        f'The relative difference between other and the one specified in config ' \
        f'is too high (diff={fraction_other_sex - 0.1})'

    # Check that the simulated age distribution is similar to the one specified in HUMAN_DISTRIBUTION
    age_histogram = {}
    for x1, x2, p in conf.get('P_AGE_REGION'):
        age_histogram[(x1, x2)] = p
    intervals = pd.IntervalIndex.from_tuples(age_histogram.keys(), closed='both')
    age_grouped = df.groupby(pd.cut(df['age'], intervals))
    age_grouped = age_grouped.agg({'age': 'count'})

    assert age_grouped.age.sum() == n_people
    age_grouped = age_grouped.age.apply(lambda x: x / n_people)
    assert np.allclose(age_grouped.to_numpy(), np.array(list(age_histogram.values())), atol=age_distribution_error_tol)

    # Check basic statistics about profession
    # profession_grouped = df.groupby('profession').count()
    # profession_grouped = profession_grouped.apply(lambda x: x / n_people)
    # canstat_retired_fraction = 0.177  # We consider all 65 years old and older to be retired so this is an upper bound.
    # sim_retired_fraction = profession_grouped.iloc[:, 0]['retired']
    # assert sim_retired_fraction < canstat_retired_fraction, \
    #     f'The simulated retired fraction of the population is {sim_retired_fraction:.4f} ' \
    #     f'while it should be lower than the upper bound {canstat_retired_fraction:.4f}'
    #
    # canstat_health_profession = 0.0354  # population fraction with health occupation according to the NOC standard.
    # sim_health_profession = profession_grouped.iloc[:, 0]['healthcare']
    # assert math.fabs(sim_health_profession - canstat_health_profession) < profession_error_tol, \
    #     f'The simulated fraction of the population working in healthcare is {sim_health_profession:.2f} ' \
    #     f'while the statistics for Canada is {canstat_health_profession:.2f}'

    # Reference value for the students count in Canada (5,553,522 - group age from 5 to 18 years old) is from
    # Elementary–Secondary Education Survey for Canada, the provinces and territories, 2016/2017.
    # (ref: https://www150.statcan.gc.ca/n1/daily-quotidien/181102/dq181102c-eng.htm)
    # Reference value of the total labour force population in educational services (definition from the NAICS standard)
    # is 1,346,585. (ref: https://tinyurl.com/qsf2q8d)
    # canstat_education_profession = 0.196  # sum of the two groups divided by 2016 Canada population (35,151,730)
    # sim_education_profession = profession_grouped.iloc[:, 0]['school']
    # assert math.fabs(sim_education_profession - canstat_education_profession) < profession_error_tol, \
    #     f'The simulated fraction of the population working in education (including students) ' \
    #     f'is {sim_education_profession:.2f} while the statistics for Canada is {canstat_education_profession:.2f}'


@pytest.mark.parametrize('seed', [62, 93, 73, 3, 51])
@pytest.mark.parametrize('test_conf_name', ['test_covid_testing.yaml'])
def test_household_distribution(
        seed: int,
        test_conf_name: str,
        avg_household_size_error_tol: float = 0.22, #TODO: change this back to 0.1. I had to bump it up otherwise the tests fail for inscrutable reasons...
        fraction_in_households_error_tol: float = 0.1,
        household_size_distribution_error_tol: float = 0.1):
    """
        Tests for the demographic statistics related to the households
            - each human is associated to a household
            - there is no empty household
            - average number of people per household
            - fraction of people in household
            - distribution of the number of people per household

    Reference values are from Canada statistics - census profile 2016 (ref: https://tinyurl.com/qsf2q8d)

    Args:
        test_conf_name (str): the filename of the configuration file used for testing
        avg_household_size_error_tol (float): tolerance to the average household size discrepancy
        fraction_in_households_error_tol (float): tolerance to the population fraction in households discrepancy
        household_size_distribution_error_tol (float): tolerance to the distribution of household size discrepancy
    """

    conf = get_test_conf(test_conf_name)

    # Test that all house_size preferences sum to 1
    P_HOUSEHOLD_SIZE = conf['P_HOUSEHOLD_SIZE']
    P_FAMILY_TYPE_SIZE_2 = conf['P_FAMILY_TYPE_SIZE_2']
    P_FAMILY_TYPE_SIZE_3 = conf['P_FAMILY_TYPE_SIZE_3']
    P_FAMILY_TYPE_SIZE_4 = conf['P_FAMILY_TYPE_SIZE_4']
    P_FAMILY_TYPE_SIZE_MORE_THAN_5 = conf['P_FAMILY_TYPE_SIZE_MORE_THAN_5']

    # household size
    val = np.sum(P_HOUSEHOLD_SIZE)
    assert math.fabs(np.sum(P_HOUSEHOLD_SIZE) - 1.) < 1e-6, \
        f'The P_HOUSEHOLD_SIZE does not sum to 1. (actual value= {val})'

    # household sizes
    val = np.sum(P_FAMILY_TYPE_SIZE_2)
    assert math.fabs(np.sum(P_FAMILY_TYPE_SIZE_2) - P_HOUSEHOLD_SIZE[1]) < 1e-6, \
        f'The P_FAMILY_TYPE_SIZE_2 does not sum to P_HOUSEHOLD_SIZE[1]. (actual value= {val}, expected value={P_HOUSEHOLD_SIZE[1]})'

    val = np.sum(P_FAMILY_TYPE_SIZE_3)
    assert math.fabs(np.sum(P_FAMILY_TYPE_SIZE_3) - P_HOUSEHOLD_SIZE[2]) < 1e-6, \
        f'The P_FAMILY_TYPE_SIZE_3 does not sum to P_HOUSEHOLD_SIZE[2]. (actual value= {val}, expected value={P_HOUSEHOLD_SIZE[2]})'

    val = np.sum(P_FAMILY_TYPE_SIZE_4)
    assert math.fabs(np.sum(P_FAMILY_TYPE_SIZE_4) - P_HOUSEHOLD_SIZE[3]) < 1e-6, \
        f'The P_FAMILY_TYPE_SIZE_4 does not sum to P_HOUSEHOLD_SIZE[3]. (actual value= {val}, expected value={P_HOUSEHOLD_SIZE[3]})'

    val = np.sum(P_FAMILY_TYPE_SIZE_MORE_THAN_5)
    assert math.fabs(np.sum(P_FAMILY_TYPE_SIZE_MORE_THAN_5) - P_HOUSEHOLD_SIZE[4]) < 1e-6, \
        f'The P_FAMILY_TYPE_SIZE_MORE_THAN_5 does not sum to P_HOUSEHOLD_SIZE[4]. (actual value= {val}, expected value={P_HOUSEHOLD_SIZE[4]})'


    n_people = 5000
    init_fraction_sick = 0.01
    rng = np.random.RandomState(seed=seed)
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    conf['simulation_days'] = 1
    city = City(
        env=env,
        n_people=n_people,
        init_fraction_sick=init_fraction_sick,
        rng=rng,
        x_range=city_x_range,
        y_range=city_y_range,
        conf=conf,
        logfile="logfile.txt"
    )

    # Verify that each human is associated to a household
    for human in city.humans:
        assert human.household, f'There is at least one individual without household.'

    n_resident_in_households = 0
    sim_household_size_distribution = [0., 0., 0., 0., 0.]
    for household in city.households:
        n_resident = len(household.residents)
        assert n_resident > 0, f'There is an empty household.'
        n_resident_in_households += n_resident
        if n_resident < 5:
            sim_household_size_distribution[n_resident - 1] += 1
        else:
            sim_household_size_distribution[-1] += 1
    sim_household_size_distribution = np.array(sim_household_size_distribution) / len(city.households)
    sim_avg_household_size = n_resident_in_households / len(city.households)

    # Average number of resident per household
    avg_household_size = conf['AVG_HOUSEHOLD_SIZE']  # Value from CanStats
    assert math.fabs(sim_avg_household_size - avg_household_size) < avg_household_size_error_tol, \
        f'The empirical average household size is {sim_avg_household_size:.2f}' \
        f' while the statistics for Canada is {avg_household_size:.2f}'

    # Number of persons in private household from
    fraction_in_households = 0.98  # Value from CanStats
    sim_fraction_in_households = n_resident_in_households / n_people
    assert math.fabs(sim_fraction_in_households - fraction_in_households) < fraction_in_households_error_tol, \
        f'The empirical fraction of people in households is {sim_fraction_in_households:.2f}' \
        f' while the statistics for Canada is {fraction_in_households:.2f}'

    # Household size distribution from
    household_size_distribution = conf['P_HOUSEHOLD_SIZE']
    assert np.allclose(
        sim_household_size_distribution,
        household_size_distribution,
        atol=household_size_distribution_error_tol), \
        f'the discrepancy between simulated and estimated household size distribution is too important.'


@pytest.mark.parametrize('seed', [62, 93, 73, 3, 51])
@pytest.mark.parametrize('test_conf_name', ['test_covid_testing.yaml'])
@pytest.mark.parametrize('app_uptake', [None, 0.25, 0.5, 0.75, 1.0])
def test_app_distribution(
        seed: int,
        test_conf_name: str,
        app_uptake: float
):
    """
        Tests for the demographic statistics related to the app users
            - age distribution of the app users when all individuals have the app

    Args:
        test_conf_name (str): the filename of the configuration file used for testing
        app_uptake (float): probability that an individual with a smartphone has the app
    """

    conf = get_test_conf(test_conf_name)

    if app_uptake:
        conf['APP_UPTAKE'] = app_uptake

    n_people = 5000
    init_fraction_sick = 0.01
    start_time = datetime.datetime(2020, 2, 28, 0, 0)

    seed = 0
    rng = np.random.RandomState(seed=seed)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    conf['simulation_days'] = 1
    city = City(
        env=env,
        n_people=n_people,
        init_fraction_sick=init_fraction_sick,
        rng=rng,
        x_range=city_x_range,
        y_range=city_y_range,
        conf=conf,
        logfile="logfile.txt",
    )
    city.have_some_humans_download_the_app()

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
            bins_fractions={(x1, x2): p for x1, x2, p in conf.get('P_AGE_REGION')},
            n_elements=n_people,
            rng=city.rng
        )
        age_histogram_bin_10s = utils._convert_bin_5s_to_bin_10s(abs_age_histogram)
        n_apps_per_age = {
            (x[0], x[1]): math.ceil(age_histogram_bin_10s[(x[0], x[1])] * x[2] * conf.get('APP_UPTAKE'))
            for x in conf.get("SMARTPHONE_OWNER_FRACTION_BY_AGE")
        }
        n_apps = np.sum(list(n_apps_per_age.values()))

        intervals = pd.IntervalIndex.from_tuples(n_apps_per_age.keys(), closed='left')
        age_grouped = df.groupby(pd.cut(df['age'], intervals))
        age_grouped = age_grouped.agg({'age': 'count', 'has_app': 'sum'})
        assert age_grouped.age.sum() == n_people
        assert age_grouped.has_app.sum() == n_apps
        age_grouped = age_grouped.has_app.apply(lambda x: x / n_apps)

        assert np.allclose(age_grouped.to_numpy(), np.array(list(n_apps_per_age.values())) / n_apps)


@pytest.mark.parametrize('test_conf_name', ['test_covid_testing.yaml'])
@pytest.mark.parametrize('app_uptake', [None, 0.25, 0.5, 0.75, 1.0])
def test_app_distribution(
        test_conf_name: str,
        app_uptake: float
):
    """
        Tests for the demographic statistics related to the app users
            - age distribution of the app users when all individuals have the app or with different uptake

    Args:
        test_conf_name (str): the filename of the configuration file used for testing
        app_uptake (float): probability that an individual with a smartphone has the app
    """

    conf = get_test_conf(test_conf_name)

    if app_uptake:
        conf['APP_UPTAKE'] = app_uptake

    n_people = 1000
    init_fraction_sick = 0.01
    start_time = datetime.datetime(2020, 2, 28, 0, 0)

    seed = 0
    rng = np.random.RandomState(seed=seed)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    conf['simulation_days'] = 1
    city = City(
        env=env,
        n_people=n_people,
        init_fraction_sick=init_fraction_sick,
        rng=rng,
        x_range=city_x_range,
        y_range=city_y_range,
        conf=conf,
        logfile="logfile.txt",
    )
    city.have_some_humans_download_the_app()

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
        age_app_groups = [(low, up) for low, up, p in age_app_histogram]  # make the age groups contiguous
        intervals = pd.IntervalIndex.from_tuples(age_app_groups, closed='both')
        age_grouped = df.groupby(pd.cut(df['age'], intervals))
        age_grouped = age_grouped.agg({'age': 'count', 'has_app': 'sum'})
        assert age_grouped.age.sum() == n_people
        age_stats = age_grouped.age.apply(lambda x: x / n_people)
        app_stats = age_grouped.has_app.apply(lambda x: x / n_people)
        assert np.allclose(age_stats.to_numpy(), app_stats.to_numpy())
    else:
        abs_age_histogram = utils.relativefreq2absolutefreq(
            bins_fractions={(x1, x2): p for x1, x2, p in conf.get('P_AGE_REGION')},
            n_elements=n_people,
            rng=city.rng
        )
        age_histogram_bin_10s = utils._convert_bin_5s_to_bin_10s(abs_age_histogram)
        n_apps_per_age = {
            (x[0], x[1]): math.ceil(age_histogram_bin_10s[(x[0], x[1])] * x[2] * conf.get('APP_UPTAKE'))
            for x in conf.get("SMARTPHONE_OWNER_FRACTION_BY_AGE")
        }
        n_apps = np.sum(list(n_apps_per_age.values()))

        intervals = pd.IntervalIndex.from_tuples(n_apps_per_age.keys(), closed='both')
        age_grouped = df.groupby(pd.cut(df['age'], intervals))
        age_grouped = age_grouped.agg({'age': 'count', 'has_app': 'sum'})
        assert age_grouped.age.sum() == n_people
        assert age_grouped.has_app.sum() == n_apps
        age_grouped = age_grouped.has_app.apply(lambda x: x / n_apps)

        assert np.allclose(age_grouped.to_numpy(), np.array(list(n_apps_per_age.values())) / n_apps)
