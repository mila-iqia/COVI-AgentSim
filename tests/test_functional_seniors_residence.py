import datetime
import os
import tempfile
import unittest.mock

import numpy as np
from tests.utils import get_test_conf

from covid19sim.locations.city import City
from covid19sim.utils.env import Env
from covid19sim.log.monitors import EventMonitor
from covid19sim.human import Human
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from covid19sim.utils.demographics import assign_households_to_humans, create_locations_and_assign_workplace_to_humans


def fake_run_app(*args, **kwargs):
    return (), ()


def test_functional_seniors_residence():
    """ Run a simulation of 1 infection in a seniors residence, and perform some sanity checks """

    with tempfile.TemporaryDirectory() as output_dir:

        rng = np.random.RandomState(42)

        # Config
        start_time = datetime.datetime(2020, 2, 28, 0, 0)
        simulation_days = 40
        city_x_range = (0, 1000)
        city_y_range = (0, 1000)

        # Find the test_configs directory, and load the required config yaml
        conf = get_test_conf("naive_local.yaml")

        env = Env(start_time)
        city = City(env, 1000, 0.01, rng, city_x_range, city_y_range, conf)

        N = 10

        # Create humans
        ages = city.rng.randint(*(65, 100), size=N)

        infection = [None] * N
        # One initial infection
        infection[0] = city.start_time

        humans = [
            Human(
                env=city.env,
                city=city,
                name=i,
                age=ages[i],
                rng=rng,
                infection_timestamp=infection[i],
                conf=conf,
            )
            for i in range(N)
        ]
        # pick one human randomly and make sure it cannot recover (for later checks)
        humans[np.random.randint(N)].never_recovers = True

        city.initialize_humans_and_locations()
        # TODO: Add way to just add senior residencies -- below won't work anymore. Also maybe switch this back to empty city?

        # sr = city.create_location(
        #     conf.get("LOCATION_DISTRIBUTION")["senior_residency"],
        #     "senior_residency",
        #     0,
        #     area=1000,
        # )
        # city.senior_residencys.append(sr)

        city.humans = humans
        city.hd = {h.name: h for h in humans}
        city.initWorld()

        outfile = os.path.join(output_dir, "test1")

        monitors = [
            EventMonitor(f=SECONDS_PER_HOUR*30, dest=outfile, chunk_size=None),
        ]

        monitors[0].dump()
        monitors[0].join_iothread()

        env.process(city.run(SECONDS_PER_HOUR, outfile))

        for human in city.humans:
            env.process(human.run(city=city))

        for m in monitors:
            env.process(m.run(env, city=city))

        with unittest.mock.patch.object(
                City, "run_app",
                new=fake_run_app) as mock:
            env.run(until=env.ts_initial+simulation_days*SECONDS_PER_DAY)

        # Check dead humans are removed from the residence
        assert sum([h.is_dead for h in city.humans]) == N - len(sr.humans)

        # Check there are no humans that are infectious
        assert not any([h.is_infectious for h in city.humans])

        # Check there are some dead
        assert sum([h.is_dead for h in city.humans]) > 0

        # Check some stats on number dead
        # len([h for h in city.humans if h.dead])/len(city.humans)

        # Curve fit to outbreak propagation
        # TODO


if __name__ == "__main__":
    test_functional_seniors_residence()
