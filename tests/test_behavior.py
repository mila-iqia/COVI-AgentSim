import unittest
from covid19sim.interventions import WashHands
import numpy as np

class DummyHuman:
    pass

class BehaviorTest(unittest.TestCase):
    def test_behavior_modification(self):
        human = DummyHuman()
        human.rng = np.random.RandomState(1)
        human.carefulness = 0
        human.risk_level = 0
        human.hygiene = 0
        WashHands().modify_behavior(human)
        assert human.hygiene != 0
        import pdb; pdb.set_trace()
    def test_behavior_revert(self):
        assert True