import unittest
import numpy as np
from covid19sim.interventions import WashHands, WearMask
from covid19sim.epidemiology.p_infection import get_p_infection
from tests.utils import get_test_conf

class DummyHuman:
    pass

class BehaviorTest(unittest.TestCase):
    def setUp(self):
        self.conf = get_test_conf("naive_local.yaml")

        self.human1 = DummyHuman()
        self.human1.rng = np.random.RandomState(1)
        self.human1.carefulness = 0
        self.human1.risk_level = 0
        self.human1.hygiene = 0
        self.human1.age = 20
        self.human1.normalized_susceptibility = self.conf['NORMALIZED_SUSCEPTIBILITY_BY_AGE'][(20, 30)]
        self.human1.mean_daily_interaction_age_group = self.conf['MEAN_DAILY_INTERACTION_FOR_AGE_GROUP'][(20, 30)]
        self.human1.mask_efficacy = 0
        self.human1.infectiousness = 0.5
        self.human1.infection_ratio = self.conf['MILD_INFECTION_RATIO']

        self.human2 = DummyHuman()
        self.human2.rng = np.random.RandomState(1)
        self.human2.carefulness = 0
        self.human2.risk_level = 0
        self.human2.hygiene = 0
        self.human2.age = 20
        self.human2.normalized_susceptibility = self.conf['NORMALIZED_SUSCEPTIBILITY_BY_AGE'][(20, 30)]
        self.human2.mean_daily_interaction_age_group = self.conf['MEAN_DAILY_INTERACTION_FOR_AGE_GROUP'][(20, 30)]
        self.human2.mask_efficacy = 0
        self.human2.infection_ratio = self.conf['MILD_INFECTION_RATIO']

        self.location_social_contact_factor = 0.5
        self.t_near = 10

    def test_default_p_infection(self):
        infector = self.human1
        infectee = self.human2
        p_infection = get_p_infection(infector, infector.infectiousness, infectee, self.location_social_contact_factor, self.conf["CONTAGION_KNOB"], self.conf['MASK_EFFICACY_FACTOR'], self.conf['HYGIENE_EFFICACY_FACTOR'], self.human1, self.human2)
        print(f"p_infection default: {p_infection}")

    def test_wash_hands_p_infection(self):
        WashHands().modify_behavior(self.human1)

        assert self.human1.hygiene != 0
        infector = self.human1
        infectee = self.human2
        p_infection = get_p_infection(infector, infector.infectiousness, infectee, self.location_social_contact_factor, self.conf["CONTAGION_KNOB"], self.conf['MASK_EFFICACY_FACTOR'], self.conf['HYGIENE_EFFICACY_FACTOR'], self.human1, self.human2)
        print(f"p_infection one washes hands: {p_infection}")

    def test_wash_hands_p_infection_both(self):
        WashHands().modify_behavior(self.human1)
        WashHands().modify_behavior(self.human2)

        assert self.human1.hygiene != 0
        infector = self.human1
        infectee = self.human2
        p_infection = get_p_infection(infector, infector.infectiousness, infectee, self.location_social_contact_factor, self.conf["CONTAGION_KNOB"], self.conf['MASK_EFFICACY_FACTOR'], self.conf['HYGIENE_EFFICACY_FACTOR'], self.human1, self.human2)
        print(f"p_infection both wash hands: {p_infection}")

    def test_mask_p_infection_one(self):
        WearMask().modify_behavior(self.human1)
        self.human1.mask_efficacy = self.conf.get("MASK_EFFICACY_NORMIE")

        infector = self.human1
        infectee = self.human2
        p_infection = get_p_infection(infector, infector.infectiousness, infectee, self.location_social_contact_factor, self.conf["CONTAGION_KNOB"], self.conf['MASK_EFFICACY_FACTOR'], self.conf['HYGIENE_EFFICACY_FACTOR'], self.human1, self.human2)
        print(f"p_infection one wears mask: {p_infection}")


    def test_mask_p_infection_both(self):
        WearMask().modify_behavior(self.human1)
        self.human1.mask_efficacy = self.conf.get("MASK_EFFICACY_NORMIE")

        WearMask().modify_behavior(self.human2)
        self.human2.mask_efficacy = self.conf.get("MASK_EFFICACY_NORMIE")

        infector = self.human1
        infectee = self.human2
        p_infection = get_p_infection(infector, infector.infectiousness, infectee, self.location_social_contact_factor, self.conf["CONTAGION_KNOB"], self.conf['MASK_EFFICACY_FACTOR'], self.conf['HYGIENE_EFFICACY_FACTOR'], self.human1, self.human2)
        print(f"p_infection both wear mask: {p_infection}")


