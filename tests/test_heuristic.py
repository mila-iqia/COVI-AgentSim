import unittest
import datetime
import numpy as np
from covid19sim.interventions.tracing import Heuristic
from tests.utils import get_test_conf
from covid19sim.human import Human
from covid19sim.utils.env import Env
from covid19sim.locations.city import EmptyCity
from covid19sim.inference.message_utils import UpdateMessage
from covid19sim.epidemiology.symptoms import MODERATE, SEVERE, EXTREMELY_SEVERE

class TrackerMock:
    def track_tested_results(self, *args, **kwargs):
        pass

    def track_covid_properties(self, *args, **kwargs):
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
        try:
            if self.city.tracker is None:
                self.city.tracker = TrackerMock()
        except AttributeError:
            self.city.tracker = TrackerMock()

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

##################################
#############TESTS################
##################################

    def test_handle_tests_positive(self):
        self.human1.set_test_info("lab", "positive")
        self.env = Env(self.start_time + datetime.timedelta(days=3))
        self.human1.env = self.env
        risk_history, rec_level = self.heuristic.handle_tests(self.human1)
        assert rec_level == 3
        assert risk_history == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def test_handle_tests_positive_lt_2_days(self):
        self.human1.set_test_info("lab", "positive")
        self.env = Env(self.start_time + datetime.timedelta(days=1))
        self.human1.env = self.env
        risk_history, rec_level= self.heuristic.handle_tests(self.human1)
        assert rec_level == 0
        assert risk_history == []

    def test_handle_tests_negative_3_days(self):
        self.human1.set_test_info("lab", "negative")
        self.env = Env(self.start_time + datetime.timedelta(days=3))
        self.human1.env = self.env
        risk_history, rec_level = self.heuristic.handle_tests(self.human1)
        assert rec_level == 0
        assert risk_history == [0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698]


##################################
###########RISK MESSAGES##########
##################################

    def test_handle_risk_messages_lev_5(self):

        # Risk Level 5
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=5,
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
        assert rec_level == 0
        assert risk_history == []

    def test_handle_risk_messages_lev_6(self):
        # Risk Level 6 today
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=6,
            old_risk_level=0,
            update_time=self.env.timestamp,
            encounter_time=self.env.timestamp,
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}
        risk_history, rec_level = self.heuristic.handle_risk_messages(self.human1, mailbox)
        assert rec_level == 0
        assert risk_history == []

    def test_handle_risk_messages_lev_6_2_days_elapsed(self):

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

    def test_handle_risk_messages_lev_8_2_days_elapsed(self):

        # Risk Level 8
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=8,
            old_risk_level=0,
            update_time=self.env.timestamp,
            encounter_time=self.env.timestamp - datetime.timedelta(days=2),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}
        risk_history, rec_level = self.heuristic.handle_risk_messages(self.human1, mailbox)
        assert rec_level == 2
        assert risk_history == [0.42782824]

    def test_handle_risk_messages_lev_8_5_days_elapsed(self):

        # Risk Level 8
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=8,
            old_risk_level=0,
            update_time=self.env.timestamp,
            encounter_time=self.env.timestamp - datetime.timedelta(days=5),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}
        risk_history, rec_level = self.heuristic.handle_risk_messages(self.human1, mailbox)
        assert rec_level == 2
        assert risk_history == [0.42782824, 0.42782824, 0.42782824, 0.42782824]

    def test_handle_risk_messages_lev_12_5_days_elapsed(self):

        # Risk Level 12
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=12,
            old_risk_level=0,
            update_time=self.env.timestamp,
            encounter_time=self.env.timestamp - datetime.timedelta(days=5),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}
        risk_history, rec_level = self.heuristic.handle_risk_messages(self.human1, mailbox)
        assert rec_level == 3
        assert risk_history == [0.79687407, 0.79687407, 0.79687407, 0.79687407]

##################################
#############SYMPTOMS#############
##################################

    def test_symptoms_empty(self):
        reported_symptoms = []
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)
        risk_history, rec_level = self.heuristic.handle_symptoms(self.human1)
        assert rec_level == 0
        assert risk_history == []

    def test_symptoms_mild(self):
        reported_symptoms = ["mild"]
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)
        risk_history, rec_level = self.heuristic.handle_symptoms(self.human1)
        assert rec_level == 2
        assert risk_history == [0.79687407, 0.79687407, 0.79687407, 0.79687407, 0.79687407, 0.79687407, 0.79687407]

    def test_symptoms_moderate(self):
        reported_symptoms = [MODERATE]
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)
        risk_history, rec_level = self.heuristic.handle_symptoms(self.human1)
        assert rec_level == 3
        assert risk_history == [0.90514533, 0.90514533, 0.90514533, 0.90514533, 0.90514533, 0.90514533, 0.90514533]

    def test_symptoms_severe(self):
        reported_symptoms = [SEVERE]
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)
        risk_history, rec_level = self.heuristic.handle_symptoms(self.human1)
        assert rec_level == 3
        assert risk_history == [0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601]


    def test_symptoms_exteremely_severe(self):
        reported_symptoms = [EXTREMELY_SEVERE]
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)
        risk_history, rec_level = self.heuristic.handle_symptoms(self.human1)
        assert rec_level == 3
        assert risk_history == [0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601]

    def test_negative_test_and_symptoms(self):
        # Should hit "recovered" and just default to lab results
        self.human1.set_test_info("lab", "negative")
        self.env = Env(self.start_time + datetime.timedelta(days=3))
        self.human1.env = self.env
        reported_symptoms = [SEVERE]
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)
        mailbox = {}

        risk_history = self.heuristic.compute_risk(self.human1, mailbox, self.hd)
        assert self.human1._heuristic_rec_level == 0
        assert risk_history == [0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698]


    def test_low_risk_message_and_severe_symptoms(self):
        self.env = Env(self.start_time + datetime.timedelta(days=3))
        self.human1.env = self.env
        reported_symptoms = [SEVERE]
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)

        # Risk Level 1
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=1,
            old_risk_level=0,
            update_time=self.env.timestamp,
            encounter_time=self.env.timestamp - datetime.timedelta(days=5),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}

        risk_history = self.heuristic.compute_risk(self.human1, mailbox, self.hd)
        assert self.human1._heuristic_rec_level == 3
        assert risk_history == [0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601, 0.94996601]


    def test_high_risk_message_and_mild_symptoms_diff_days(self):
        self.env = Env(self.start_time + datetime.timedelta(days=3))
        self.human1.env = self.env
        reported_symptoms = ["mild"]
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)

        # Risk Level 1
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=13,
            old_risk_level=0,
            update_time=self.env.timestamp,
            encounter_time=self.env.timestamp - datetime.timedelta(days=2),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}

        risk_history = self.heuristic.compute_risk(self.human1, mailbox, self.hd)
        assert self.human1._heuristic_rec_level == 3
        # We get a hetereogenous array here because of the mixed signals between symptoms and risk messages
        assert risk_history == [0.8408014, 0.79687407, 0.79687407, 0.79687407, 0.79687407, 0.79687407, 0.79687407]

    def test_recovered(self):
        # they were at a rec level of 1, then they recovered
        mailbox = {None: []}
        self.env = Env(self.start_time + datetime.timedelta(days=3))
        self.human1.env = self.env
        self.human1._rec_level = 1
        self.human1._heuristic_rec_level = 1

        risk_history = self.heuristic.compute_risk(self.human1, mailbox, self.hd)
        assert self.human1._heuristic_rec_level == 0
        assert risk_history == [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]


    def test_high_risk_history_no_new_signal(self):
        # they were at a rec level of 1, then they recovered
        mailbox = {None: []}
        self.env = Env(self.start_time + datetime.timedelta(days=3))
        self.human1.env = self.env
        self.human1._rec_level = 3
        self.human1._heuristic_rec_level = 3
        self.human1.risk_history_map = {1: 0.9, 2: 0.9, 3: 0.9}

        risk_history = self.heuristic.compute_risk(self.human1, mailbox, self.hd)
        assert self.human1._heuristic_rec_level == 0
        assert risk_history == [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    def test_high_risk_history_mild_symptom(self):
        # they were at a rec level of 1, then they recovered
        mailbox = {None: []}
        self.env = Env(self.start_time + datetime.timedelta(days=3))
        self.human1.env = self.env
        self.human1._rec_level = 3
        self.human1._heuristic_rec_level = 3
        self.human1.risk_history_map = {1: 0.9, 2: 0.9, 3: 0.9}
        reported_symptoms = ["mild"]
        self.human1.rolling_all_reported_symptoms.appendleft(reported_symptoms)

        risk_history = self.heuristic.compute_risk(self.human1, mailbox, self.hd)
        assert self.human1._heuristic_rec_level == 3
        # This basically says for the last three days you maintain the high risk signal, but then you update older signals.
        # not sure this would ever happen in the real algorithm since the only time we are really writing signals higher
        # than 0.79 is for positive test result that writes for 14 days.
        assert risk_history == [0.9, 0.9, 0.9, 0.79687407, 0.79687407,0.79687407,0.79687407]


    def test_handle_tests_negative_8_days(self):
        # The scenario is you get a negative lab test 8 days ago, but you got a moderate risk message two days ago.
        self.human1.set_test_info("lab", "negative")
        self.env = Env(self.start_time + datetime.timedelta(days=8))
        self.human1.env = self.env
        m1 = UpdateMessage(
            uid=None,
            new_risk_level=8,
            old_risk_level=0,
            update_time=self.env.timestamp.date(),
            encounter_time=self.env.timestamp - datetime.timedelta(days=2),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}
        risk_history = self.heuristic.compute_risk(self.human1, mailbox, self.hd)

        assert self.human1._heuristic_rec_level == 2
        assert risk_history == [0.42782824, 0.01, 0.01, 0.01, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.20009698]


    def test_handle_tests_negative_8_days_but_high_risk_before(self):
        # The scenario is you get a negative lab test 8 days ago, but you got a moderate risk message 12 days ago.
        # This triggers the "recovery" mode so you get 7 baseline risks instead of
        # [0.9, 0.9, 0.9, 0.9, 0.20009698, 0.20009698, 0.20009698, 0.20009698, 0.9]
        self.human1.set_test_info("lab", "negative")
        self.env = Env(self.start_time + datetime.timedelta(days=8))
        self.human1.env = self.env
        self.human1.risk_history_map = {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.9, 6: 0.9, 7: 0.9, 8: 0.9, 9: 0.9}

        m1 = UpdateMessage(
            uid=None,
            new_risk_level=8,
            old_risk_level=0,
            update_time=self.env.timestamp.date(),
            encounter_time=self.env.timestamp - datetime.timedelta(days=12),
            _sender_uid=self.human2.name,
            _receiver_uid=self.human1.name,
            _real_encounter_time=self.env.timestamp,
            _exposition_event=None,  # we don't decide this here, it will be done in the caller
        )

        mailbox = {None: [m1]}

        risk_history = self.heuristic.compute_risk(self.human1, mailbox, self.hd)

        assert self.human1._heuristic_rec_level == 0
        assert risk_history == [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

