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
        self.env = Env(self.start_time)
        self.city = EmptyCity(self.env, self.rng, self.city_x_range, self.city_y_range, self.conf)
        self.sr = self.city.create_location(
            self.conf.get("LOCATION_DISTRIBUTION")["senior_residency"],
            "senior_residency",
            0,
            area=1000,
        )

        self.human1 = Human(env=self.city.env, city=self.city, name=1, age=42, rng=self.rng, has_app=True, infection_timestamp=self.start_time,
            household=self.sr, workplace=self.sr, profession="retired", rho=self.conf.get("RHO"), gamma=self.conf.get("GAMMA"),
            conf=self.conf)
        self.human1.set_intervention(self.heuristic)
        setattr(self.human1, "_heuristic_rec_level", 0)

        self.human2 = Human(env=self.city.env, city=self.city, name=2, age=6*9, rng=self.rng, has_app=True, infection_timestamp=self.start_time,
            household=self.sr, workplace=self.sr, profession="retired", rho=self.conf.get("RHO"), gamma=self.conf.get("GAMMA"),
            conf=self.conf)
        self.human2.set_intervention(self.heuristic)
        setattr(self.human2, "_heuristic_rec_level", 0)

        self.humans = [self.human1, self.human2]
        self.hd = {h.name: h for h in self.humans}

    def test_handle_risk_messages(self):

        # Risk Level 5
        # m1 = UpdateMessage(
        #     uid=None,
        #     new_risk_level=5,
        #     old_risk_level=0,
        #     update_time=self.env.timestamp.date(),
        #     encounter_time=datetime.datetime.combine(self.env.timestamp.date(), datetime.datetime.min.time()),
        #     _sender_uid=self.human2.name,
        #     _receiver_uid=self.human1.name,
        #     _real_encounter_time=self.env.timestamp,
        #     _exposition_event=None,  # we don't decide this here, it will be done in the caller
        # )
        #
        # mailbox = {None: [m1]}
        # risk_history, rec_level = self.heuristic.handle_risk_messages(self.human1, mailbox)
        # assert rec_level == 0
        # assert risk_history == []
        #
        # # Risk Level 6 today
        # m1 = UpdateMessage(
        #     uid=None,
        #     new_risk_level=6,
        #     old_risk_level=0,
        #     update_time=self.env.timestamp,
        #     encounter_time=self.env.timestamp,
        #     _sender_uid=self.human2.name,
        #     _receiver_uid=self.human1.name,
        #     _real_encounter_time=self.env.timestamp,
        #     _exposition_event=None,  # we don't decide this here, it will be done in the caller
        # )
        #
        # mailbox = {None: [m1]}
        # risk_history, rec_level = self.heuristic.handle_risk_messages(self.human1, mailbox)
        # assert rec_level == 0
        # assert risk_history == []

        # Risk level 6, but 2 days have elapsed
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=6,
            old_risk_level=0,
            update_time=self.env.timestamp.date(),
            encounter_time=self.env.timestamp - datetime.timedelta(days=2),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}
        risk_history, rec_level = self.heuristic.handle_risk_messages(self.human1, mailbox)
        assert rec_level == 2
        assert risk_history == [0.20009698]

        # Risk Level 8
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=8,
            old_risk_level=0,
            update_time=self.env.timestamp.date(),
            encounter_time=datetime.datetime.combine(self.env.timestamp.date(), datetime.datetime.min.time()),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}
        risk_history, rec_level = self.heuristic.handle_risk_messages(self.human1, mailbox)
        assert rec_level == 2
        assert risk_history == [0.42782824,0.42782824, 0.42782824, 0.42782824, 0.42782824, 0.42782824]

    def test_symptoms(self):
        assert True

    def test_recovered(self):
        assert True

    def test_multi(self):
        assert True