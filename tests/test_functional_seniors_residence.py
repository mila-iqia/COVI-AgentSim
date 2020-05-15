import pytest
import tempfile
import os
import numpy as np

from covid19sim.simulator import Human
from covid19sim.base import City, Env, EmptyCity
from covid19sim.monitors import EventMonitor, TimeMonitor, SEIRMonitor
from covid19sim.configs.exp_config import ExpConfig
from covid19sim.configs import config
from covid19sim.frozen.helper import SYMPTOMS_META, SYMPTOMS_META_IDMAP
from covid19sim.configs.constants import TICK_MINUTE
import datetime
from pathlib import Path

def test_functional_seniors_residence():
    """ Run a simulation of 1 infection in a seniors residence, and perform some sanity checks """

    with tempfile.TemporaryDirectory() as output_dir:

        rng = np.random.RandomState(42)

        # Config
        start_time=datetime.datetime(2020, 2, 28, 0, 0)
        simulation_days=40
        city_x_range = (0,1000)
        city_y_range = (0,1000)

        # Find the test_configs directory, and load the required config yaml
        path = Path(__file__).parent
        ExpConfig.load_config(path/"test_configs"/"naive_local.yml")

        env = Env(start_time)
        city = EmptyCity(env, rng, city_x_range, city_y_range, start_time)

        sr = city.create_location(config.LOCATION_DISTRIBUTION['senior_residency'], 'senior_residency', 0, area=1000)
        city.senior_residencys.append(sr)

        N = 10

        # Create humans
        ages = city.rng.randint(*(65,100), size=N)

        infection = [None]*N
        # One initial infection
        infection[0] = city.start_time

        humans = [Human(env=city.env, city=city,
                                rng=rng,
                                name=i,
                                age=ages[i],
                                household=sr,
                                workplace=sr,
                                profession='retired',
                                rho=config.RHO,
                                gamma=config.GAMMA,
                                has_app=True,
                                infection_timestamp=infection[i]
                                ) for i in range(N)]


        city.humans = humans
        city.initWorld()

        outfile=os.path.join(output_dir, 'test1')

        monitors = [EventMonitor(f=1800, dest=outfile, chunk_size=None), SEIRMonitor(f=1440), TimeMonitor(1440)]

        monitors[0].dump()
        monitors[0].join_iothread()

        env.process(city.run(1440/24, outfile, start_time, SYMPTOMS_META_IDMAP, port=668, n_jobs=1))

        for human in city.humans:
                env.process(human.run(city=city))

        for m in monitors:
                env.process(m.run(env, city=city))

        env.run(until=simulation_days * 24 * 60 / TICK_MINUTE)

        # Check dead humans are removed from the residence
        assert sum([h.is_dead for h in city.humans])==N-len(sr.humans)

        # Check there are no humans that are infectious
        assert not any([h.is_infectious for h in city.humans])

        # Check there are some dead
        assert sum([h.is_dead for h in city.humans])>0

        # Check some stats on number dead
        #len([h for h in city.humans if h.dead])/len(city.humans)

        # Curve fit to outbreak propagation 
        # TODO

