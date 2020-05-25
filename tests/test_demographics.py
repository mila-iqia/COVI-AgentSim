import datetime
import math
from pathlib import PosixPath

import numpy as np
import pandas as pd
import os
import pytest

from covid19sim import utils
from tests.utils import get_test_conf

from covid19sim.run import simulate
import warnings


@pytest.mark.parametrize('test_conf_name', ['test_covid_testing.yaml'])
@pytest.mark.parametrize('p_has_app', [1.0])
def test_age_distribution(
        tmp_path: PosixPath,
        test_conf_name: str,
        p_has_app: float
):
    """
        Run one simulation and test for the expected demographic statistics that includes:
            - number of people in the population
            - age distribution
            - age distribution of the app users (TODO: the stats `SMARTPHONE_OWNER_FRACTION_BY_AGE` is not consistent with
            `HUMAN_DISTRIBUTION` with `HAS_APP` is fixed to 1.)

    Args:
        tmp_path (PosixPath): a temporary directory provided by the Pytest fixture
        p_has_app (float): probability that an individual has the app.
        If not None, this value overwrites the one provided in
    """
    warnings.filterwarnings('ignore')

    conf = get_test_conf(test_conf_name)

    if p_has_app:
        conf['APP_UPTAKE'] = p_has_app

    outfile = os.path.join(tmp_path, "data")
    n_people = 1000
    city, monitors, _ = simulate(
        n_people=n_people,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=2,
        outfile=outfile,
        init_percent_sick=0.1,
        out_chunk_size=500,
        return_city=True,
        conf=conf
    )
    monitors[0].dump()
    monitors[0].join_iothread()

    # Check the actual population size is the same than specified
    assert len(city.humans) == n_people

    # Demographics
    population = []
    for human in city.humans:
        population.append([
            human.age,
            human.sex,
            human.has_app,
            human.profession,
            human.workplace
        ])

    df = pd.DataFrame.from_records(
        data=population,
        columns=['age', 'sex', 'has_app', 'profession', 'workplace']
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
    print(stats.to_numpy())
    print(np.array(list(age_histogram.values())))
    assert np.allclose(stats.to_numpy(), np.array(list(age_histogram.values())), atol=0.01)

    # Check the age distribution of the app users
    if conf.get('APP_UPTAKE') < 0:
        age_app_histogram = conf.get('SMARTPHONE_OWNER_FRACTION_BY_AGE')
        age_app_groups = [(low, up + 1) for low, up in age_app_histogram]  # make the age groups contiguous
        intervals = pd.IntervalIndex.from_tuples(age_app_groups, closed='left')
        age_grouped = df.groupby(pd.cut(df['age'], intervals))
        age_grouped = age_grouped.agg({'age': 'count', 'has_app': 'sum'})
        assert age_grouped.age.sum() == n_people
        age_stats = age_grouped.age.apply(lambda x: x / n_people)
        app_stats = age_grouped.has_app.apply(lambda x: x / n_people)
        assert np.allclose(age_stats.to_numpy(), app_stats.to_numpy())

    # TODO: there is a bug with the app users creation.
    else:

        abs_age_histogram = utils.relativefreq2absolutefreq(
            bins_fractions={age_bin: specs['p'] for age_bin, specs in conf.get('HUMAN_DISTRIBUTION').items()},
            n_elements=n_people,
            rng=city.rng
        )
        # app users
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


