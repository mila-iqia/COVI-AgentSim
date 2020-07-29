import unittest
import numpy as np
from collections import Counter
from covid19sim.epidemiology.symptoms import _get_covid_symptoms, _get_covid_fever_probability


######################## COVID ################################

class CovidSymptomTests(unittest.TestCase):
    def test_incubation(self):
        """
        check returns false if not far enough from intervention day
        """
        assert True
        rng = np.random.RandomState(1)

        symptoms = _get_covid_symptoms(symptoms_progression=[[], []], phase_id=10, rng=rng, really_sick=False,
                        extremely_sick=False, age=20, initial_viral_load=0.5,
                        carefulness=0.5, preexisting_conditions=[])
        assert type(symptoms) == list
        assert len(symptoms) == 0


    # def test_onset(self):
    #     """
    #     check returns false if not far enough from intervention day
    #     """
    #     assert True
    #     rng = np.random.RandomState()
    #     sampled_symptoms = []
    #     iters = 100
    #     for i in range(iters):
    #         symptoms = _get_covid_symptoms(symptoms_progression=[[], []], phase_id=11, rng=rng, really_sick=False,
    #                         extremely_sick=False, age=20, initial_viral_load=0.5,
    #                         carefulness=0.5, preexisting_conditions=[])
    #         sampled_symptoms.extend(symptoms)
    #     print(sampled_symptoms)
    #     count = Counter(sampled_symptoms)
    #     print(count.most_common(10))
    #     assert type(symptoms) == list

    # def test_onset_fever(self):
    #     """
    #     check returns false if not far enough from intervention day
    #     """
    #     rng = np.random.RandomState()
    #     sampled_symptoms = []
    #     iters = 100
    #     for i in range(iters):
    #         symptoms = _get_covid_symptoms(symptoms_progression=[[], []], phase_id=11, rng=rng, really_sick=False,
    #                         extremely_sick=False, age=20, initial_viral_load=0.5,
    #                         carefulness=0.5, preexisting_conditions=[])
    #         sampled_symptoms.extend(symptoms)
    #     count = Counter(sampled_symptoms)
    #     assert type(symptoms) == list



    def test_onset_fever_proba(self):
        # Test each phase
        proba = _get_covid_fever_probability(phase_id=11, really_sick=False, extremely_sick=False,
            preexisting_conditions=[], initial_viral_load=0.5)
        assert proba == 0.2
        # proba = _get_covid_fever_probability(phase_id=12, really_sick=False, extremely_sick=False,
        #     preexisting_conditions=[], initial_viral_load=0.5)
        # assert proba == 0.8
        # proba = _get_covid_fever_probability(phase_id=13, really_sick=False, extremely_sick=False,
        #     preexisting_conditions=[], initial_viral_load=0.5)
        # assert proba == 0.0
        # proba = _get_covid_fever_probability(phase_id=14, really_sick=False, extremely_sick=False,
        #     preexisting_conditions=[], initial_viral_load=0.5)
        # assert proba == 0.0
        #
        # # Cond 1
        # proba = _get_covid_fever_probability(phase_id=11, really_sick=True, extremely_sick=False,
        #     preexisting_conditions=[], initial_viral_load=0.5)
        # assert proba == 0.4
        # proba = _get_covid_fever_probability(phase_id=11, really_sick=False, extremely_sick=False,
        #     preexisting_conditions=[], initial_viral_load=0.61)
        # assert proba == 0.4
        # proba = _get_covid_fever_probability(phase_id=11, really_sick=False, extremely_sick=False,
        #     preexisting_conditions=["a", "b", "c"], initial_viral_load=0.5)
        # assert proba == 0.4
        #
        # proba = _get_covid_fever_probability(phase_id=12, really_sick=False, extremely_sick=False,
        #     preexisting_conditions=[], initial_viral_load=0.61)
        # assert proba == 1.
