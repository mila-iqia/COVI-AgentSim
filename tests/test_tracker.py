import datetime

import numpy as np

from covid19sim.env import Env
from covid19sim.human import Human
from covid19sim.log.track import Tracker
from tests.utils import get_test_conf
from covid19sim.locations.city import EmptyCity
import tempfile

def test_track_serial_interval():
    """
    Test the various cases of serial interval tracking    
    """

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
        city = EmptyCity(env, rng, city_x_range, city_y_range, conf)

        sr = city.create_location(
            conf.get("LOCATION_DISTRIBUTION")["senior_residency"],
            "senior_residency",
            0,
            area=1000,
        )
        city.senior_residencys.append(sr)

        N = 10

        # Create humans
        ages = city.rng.randint(*(65, 100), size=N)

        infection = [None] * N
        
        humans = [
            Human(
                env=city.env,
                city=city,
                name=i,
                age=ages[i],
                rng=rng,
                has_app=False,
                infection_timestamp=infection[i],
                household=sr,
                workplace=sr,
                profession="retired",
                rho=conf.get("RHO"),
                gamma=conf.get("GAMMA"),
                conf=conf,
            )
            for i in range(N)
        ]

        city.humans = humans
        city.initWorld()

        t = Tracker(env, city)

        # Create some infections
        infections = [(0,1), (0,2), (1,3), (2,5)]
        for infector, infectee in infections:
            to_human = humans[infectee]
            from_human = humans[infector]
            t.serial_interval_book_to[to_human.name][from_human.name] = (to_human, from_human)
            t.serial_interval_book_from[from_human.name][to_human.name] = (to_human, from_human)

        # check no interval is registered for only to_human symptoms 
        # or only from_human symptoms
        humans[1].covid_symptom_start_time = datetime.datetime(2020, 2, 28, 0, 0)
        t.track_serial_interval(humans[1].name)
        assert len(t.serial_intervals)==0

        humans[5].covid_symptom_start_time = datetime.datetime(2020, 2, 28, 0, 0)+datetime.timedelta(days=4)
        t.track_serial_interval(humans[5].name)
        assert len(t.serial_intervals)==0

        # check a negative interval is registered for subsequent infector symptoms
        humans[0].covid_symptom_start_time = datetime.datetime(2020, 2, 28, 0, 0)+datetime.timedelta(days=1)
        t.track_serial_interval(humans[0].name)
        assert len(t.serial_intervals)==1
        assert t.serial_intervals[0]==-1.0

        # check infector and infectee intervals are registered
        humans[2].covid_symptom_start_time = datetime.datetime(2020, 2, 28, 0, 0)+datetime.timedelta(days=2)
        t.track_serial_interval(humans[2].name)
        assert len(t.serial_intervals)==3
        # Intervals (2,5) and (0,2) should be registered
        assert sorted(t.serial_intervals[-2:])==[1,2]

        # assert calling twice has no effect
        t.track_serial_interval(humans[2].name)
        assert len(t.serial_intervals)==3
        # Intervals (2,5) and (0,2) should be registered
        assert sorted(t.serial_intervals[-2:])==[1,2]

        # check what's left in the serial_interval_book_to, serial_interval_book_from
        assert humans[1].name in t.serial_interval_book_to[humans[3].name]
        assert len(t.serial_interval_book_to[humans[3].name])==1

        assert humans[3].name in t.serial_interval_book_from[humans[1].name]
        assert len(t.serial_interval_book_from[humans[1].name])==1

        #check all the others are empty
        for i in [5,0,2]:
            assert len(t.serial_interval_book_from[humans[i].name])==0
            assert len(t.serial_interval_book_to[humans[i].name])==0



        

