import unittest
import datetime
import numpy as np
from covid19sim.interventions.tracing import Heuristic
from tests.utils import get_test_conf
from covid19sim.human import Human
from covid19sim.utils.env import Env
from covid19sim.locations.city import EmptyCity
from covid19sim.inference.message_utils import EncounterMessage, GenericMessageType, UpdateMessage

class DummyHuman:
    pass

class HeuristicTest(unittest.TestCase):
    def setUp(self):
        self.conf = get_test_conf("naive_local.yaml")
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 40
        self.city_x_range = (0, 1000)
        self.city_y_range = (0, 1000)
        self.rng = np.random.RandomState(42)
        self.heuristic = Heuristic(version=1, conf=self.conf)

    def test_1(self):
        env = Env(self.start_time)
        city = EmptyCity(env, self.rng, self.city_x_range, self.city_y_range, self.conf)
        sr = city.create_location(
            self.conf.get("LOCATION_DISTRIBUTION")["senior_residency"],
            "senior_residency",
            0,
            area=1000,
        )

        human1 = Human(env=city.env, city=city, name=1, age=42, rng=self.rng, has_app=True, infection_timestamp=self.start_time,
            household=sr, workplace=sr, profession="retired", rho=self.conf.get("RHO"), gamma=self.conf.get("GAMMA"),
            conf=self.conf)
        setattr(human1, "_heuristic_rec_level", 0)
        human2 = Human(env=city.env, city=city, name=2, age=6*9, rng=self.rng, has_app=True, infection_timestamp=self.start_time,
            household=sr, workplace=sr, profession="retired", rho=self.conf.get("RHO"), gamma=self.conf.get("GAMMA"),
            conf=self.conf)
        setattr(human2, "_heuristic_rec_level", 0)
        for human in [human1, human2]:
            human.set_intervention(self.heuristic)
        hd = {h.name: h for h in [human1, human2]}
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=10,
            old_risk_level=0,
            update_time=env.timestamp.date(),
            encounter_time=datetime.datetime.combine(env.timestamp.date(), datetime.datetime.min.time()),
            _sender_uid=human2.name,
            _receiver_uid=human1.name,
            _real_encounter_time=env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}
        risk_history = self.heuristic.compute_risk(human1, mailbox, hd)
        print(risk_history)
        print(human1._heuristic_rec_level)
