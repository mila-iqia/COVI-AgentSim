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

    def test_onset_fever(self):
        """
        check returns false if not far enough from intervention day
        """
        rng = np.random.RandomState()
        sampled_symptoms = []
        iters = 100
        for i in range(iters):
            symptoms = _get_covid_symptoms(symptoms_progression=[[], []], phase_id=11, rng=rng, really_sick=False,
                            extremely_sick=False, age=20, initial_viral_load=0.5,
                            carefulness=0.5, preexisting_conditions=[])
            sampled_symptoms.extend(symptoms)
        print(sampled_symptoms)
        count = Counter(sampled_symptoms)
        print(count.most_common(10))
        assert type(symptoms) == list



    def test_onset_fever_proba(self):

        # SymptomProbability('fever', 4, {COVID_INCUBATION: 0.0,
        #                                 COVID_ONSET: 0.2,
        #                                 COVID_PLATEAU: 0.8,
        #                                 COVID_POST_PLATEAU_1: 0.0,
        #                                 COVID_POST_PLATEAU_2: 0.0,
        #                                 FLU_FIRST_DAY: 0.7,
        #                                 FLU_MAIN: 0.7,
        #                                 FLU_LAST_DAY: 0.3})

        proba = _get_covid_fever_probability(phase_id=11, really_sick=False, extremely_sick=False,
            preexisting_conditions=[], initial_viral_load=0.5)
        assert proba == 0.2

        proba = _get_covid_fever_probability(phase_id=12, really_sick=False, extremely_sick=False,
            preexisting_conditions=[], initial_viral_load=0.5)
        assert proba == 0.8