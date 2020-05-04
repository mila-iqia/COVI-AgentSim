import unittest

import numpy as np

from covid19sim.frozen.helper import PREEXISTING_CONDITIONS_META, SYMPTOMS_META
from covid19sim.utils import _get_covid_progression, _get_cold_progression, _get_flu_progression, \
    _get_preexisting_conditions, PREEXISTING_CONDITIONS, SYMPTOMS, SYMPTOMS_CONTEXTS


class Symptoms(unittest.TestCase):
    def test_symptoms_structure(self):
        s_ids = set()

        # Legacy all_possible_symptoms
        all_possible_symptoms = ['moderate', 'mild', 'severe', 'extremely-severe', 'fever',
                                 'chills', 'gastro', 'diarrhea', 'nausea_vomiting', 'fatigue',
                                 'unusual', 'hard_time_waking_up', 'headache', 'confused',
                                 'lost_consciousness', 'trouble_breathing', 'sneezing',
                                 'cough', 'runny_nose', 'aches', 'sore_throat', 'severe_chest_pain',
                                 'loss_of_taste', 'mild_trouble_breathing', 'light_trouble_breathing',
                                 'moderate_trouble_breathing',
                                 'heavy_trouble_breathing']

        for s_name, s_prob in SYMPTOMS.items():
            self.assertEqual(s_name, s_prob.name)
            self.assertEqual(s_prob.id, SYMPTOMS_META[s_name])
            if s_prob.id < len(all_possible_symptoms):
                self.assertEqual(s_name, all_possible_symptoms[s_prob.id])

            self.assertNotIn(s_prob.id, s_ids)
            s_prob_contexts = set(s_prob.probabilities.keys())
            if s_prob_contexts:
                self.assertTrue([1 for _, symptoms_contexts in SYMPTOMS_CONTEXTS.items()
                                 if set(symptoms_contexts.values()).issubset(s_prob_contexts)])
            s_ids.add(s_prob.id)


class CovidProgression(unittest.TestCase):
    symptoms_contexts = SYMPTOMS_CONTEXTS['covid']

    n_people = 1000  # Test too slow to use 10000 ppl
    initial_viral_load_options = (0.10, 0.25, 0.50, 0.75)
    viral_load_plateau_start = 2
    viral_load_plateau_end = 4
    viral_load_recovered = 7
    ages_options = (25, 50, 75)
    incubation_days = 1
    really_sick_options = (True, False)
    extremely_sick_options = (True, False)
    preexisting_conditions_options = (tuple(), ('pre1', 'pre2'), ('pre1', 'pre2', 'pre3'))
    carefulness_options = (0.10, 0.25, 0.50, 0.75)

    def setUp(self):
        self.out_of_context_symptoms = set()
        for _, prob in SYMPTOMS.items():
            for context in prob.probabilities:
                if context in self.symptoms_contexts.values():
                    break
            else:
                self.out_of_context_symptoms.add(prob.id)

    @staticmethod
    def _get_id(symptom):
        return SYMPTOMS[symptom].id

    @staticmethod
    def _get_probability(symptom_probs, symptom_context):
        if isinstance(symptom_probs, str):
            symptom_probs = SYMPTOMS[symptom_probs]
        if isinstance(symptom_context, int):
            symptom_context = CovidProgression.symptoms_contexts[symptom_context]
        return symptom_probs.probabilities[symptom_context]

    def test_covid_progression(self):
        """
            Test the distribution of the covid symptoms
        """
        rng = np.random.RandomState(1234)

        for initial_viral_load in self.initial_viral_load_options:
            for age in self.ages_options:
                for really_sick in self.really_sick_options:
                    for extremely_sick in self.extremely_sick_options if really_sick else (False,):
                        for preexisting_conditions in self.preexisting_conditions_options:
                            for carefulness in self.carefulness_options:
                                self._test_covid_progression(
                                    rng, initial_viral_load, age, really_sick,
                                    extremely_sick, preexisting_conditions,
                                    carefulness)

    def _test_covid_progression(self, rng, initial_viral_load, age, really_sick,
                                extremely_sick, preexisting_conditions,
                                carefulness):
        _get_id = self._get_id
        _get_probability = self._get_probability
        symptoms_contexts = self.symptoms_contexts

        # List of list of set of symptoms (str). Each set contains the list of symptoms in a day
        computed_dist = [[set(day_symptoms) for day_symptoms in
                          _get_covid_progression(initial_viral_load, self.viral_load_plateau_start,
                                                 self.viral_load_plateau_end, self.viral_load_recovered,
                                                 age, self.incubation_days, really_sick,
                                                 extremely_sick,
                                                 rng, preexisting_conditions, carefulness)]
                         for _ in range(self.n_people)]

        probs = [[0] * len(SYMPTOMS) for _ in symptoms_contexts]

        for human_symptoms in computed_dist:
            # To simplify the tests, we expect each stage to last 1 day
            self.assertEqual(len(human_symptoms),
                             len(symptoms_contexts) + self.incubation_days)

            for day_symptoms in human_symptoms[:self.incubation_days]:
                self.assertEqual(len(day_symptoms), 0)

            for i, day_symptoms in enumerate(human_symptoms[self.incubation_days:]):
                self.assertEqual(len([s for s in ('mild', 'moderate', 'severe', 'extremely-severe')
                                      if s in day_symptoms]), 1)
                self.assertIn(
                    len([s for s in ('light_trouble_breathing', 'moderate_trouble_breathing',
                                     'heavy_trouble_breathing')
                         if s in day_symptoms]),
                    (0, 1) if 'trouble_breathing' in day_symptoms else (0,))

                for s_name, s_prob in SYMPTOMS.items():
                    probs[i][s_prob.id] += int(s_name in day_symptoms)

        for symptoms_probs in probs:
            for i in range(len(symptoms_probs)):
                symptoms_probs[i] /= self.n_people

        for s_name, s_prob in SYMPTOMS.items():
            s_id = s_prob.id

            if s_id in {_get_id('gastro'), _get_id('diarrhea'), _get_id('nausea_vomiting'),

                        _get_id('fatigue'), _get_id('unusual'), _get_id('hard_time_waking_up'),
                        _get_id('headache'), _get_id('confused'), _get_id('lost_consciousness'),

                        _get_id('trouble_breathing'), _get_id('sneezing'), _get_id('cough'),
                        _get_id('runny_nose'), _get_id('sore_throat'), _get_id('severe_chest_pain'),
                        _get_id('light_trouble_breathing'), _get_id('moderate_trouble_breathing'),
                        _get_id('heavy_trouble_breathing')}:
                # Skip this test as maintaining of the tests would be
                # as complex as maintaining the code
                continue

            for i, (context, expected_prob) in enumerate((c, p) for c, p in s_prob.probabilities.items()
                                                         if c in symptoms_contexts):
                prob = probs[i][s_id]

                if s_id == _get_id('lost_consciousness') and \
                        not (really_sick or extremely_sick or len(preexisting_conditions) > 2):
                    expected_prob = 0

                if i == 0:
                    if really_sick or extremely_sick or len(preexisting_conditions) > 2 \
                            or initial_viral_load > 0.6:
                        if s_id == _get_id('moderate'):
                            expected_prob = 1
                        elif s_id == _get_id('mild'):
                            expected_prob = 0
                    elif s_id == _get_id('mild'):
                        expected_prob = 1
                    elif s_id == _get_id('moderate'):
                        expected_prob = 0

                    if s_id == _get_id('chills') and \
                            not extremely_sick:
                        expected_prob = 0

                    elif s_id == _get_id('lost_consciousness') and \
                            not (really_sick or extremely_sick or len(preexisting_conditions) > 2):
                        expected_prob = 0

                    elif s_id == _get_id('severe_chest_pain') and \
                            not extremely_sick:
                        expected_prob = 0

                    elif s_id in {_get_id('fever'), _get_id('chills')}:
                        # Skip this test as maintaining of the tests would be
                        # as complex as maintaining the code
                        continue

                elif i == 1:
                    if extremely_sick:
                        if s_id == _get_id('extremely-severe'):
                            expected_prob = 1
                        elif s_id in {_get_id('mild'), _get_id('moderate'), _get_id('severe')}:
                            expected_prob = 0
                    elif really_sick or len(preexisting_conditions) > 2 or initial_viral_load > 0.6:
                        if s_id == _get_id('severe'):
                            expected_prob = 1
                        elif s_id in {_get_id('mild'), _get_id('moderate'), _get_id('extremely-severe')}:
                            expected_prob = 0
                    elif s_id in {_get_id('severe'), _get_id('extremely-severe')}:
                        expected_prob = 0
                    elif s_id in {_get_id('mild'), _get_id('moderate')}:
                        # Skip this test as maintaining of the tests would be
                        # as complex as maintaining the code
                        continue

                    if s_id == _get_id('lost_consciousness') and \
                            not (really_sick or extremely_sick or len(preexisting_conditions) > 2):
                        expected_prob = 0

                    elif s_id == _get_id('severe_chest_pain') and \
                            not extremely_sick:
                        expected_prob = 0

                    elif s_id == _get_id('loss_of_taste'):
                        p = _get_probability('loss_of_taste', 0)
                        expected_prob = p + (1 - p) * expected_prob

                    elif s_id in {_get_id('fever'), _get_id('chills')}:
                        # Skip this test as maintaining of the tests would be
                        # as complex as maintaining the code
                        continue

                elif i == 2:
                    if extremely_sick:
                        if s_id == _get_id('extremely-severe'):
                            expected_prob = 1
                        elif s_id in {_get_id('mild'), _get_id('moderate'), _get_id('severe')}:
                            expected_prob = 0
                    elif really_sick or len(preexisting_conditions) > 2 or initial_viral_load > 0.6:
                        if s_id == _get_id('severe'):
                            expected_prob = 1
                        elif s_id in {_get_id('mild'), _get_id('moderate'), _get_id('extremely-severe')}:
                            expected_prob = 0
                    elif s_id in {_get_id('severe'), _get_id('extremely-severe')}:
                        expected_prob = 0
                    elif s_id in {_get_id('mild'), _get_id('moderate')}:
                        # Skip this test as maintaining of the tests would be
                        # as complex as maintaining the code
                        continue

                    if s_id == _get_id('lost_consciousness') and \
                            not (really_sick or extremely_sick or len(preexisting_conditions) > 2):
                        expected_prob = 0

                    elif s_id == _get_id('severe_chest_pain') and \
                            not extremely_sick:
                        expected_prob = 0

                    elif s_id == _get_id('loss_of_taste'):
                        p0 = _get_probability('loss_of_taste', 0)
                        p1 = p0 + (1 - p0) * _get_probability('loss_of_taste', 1)
                        expected_prob = p1 + (1 - p1) * expected_prob

                    elif s_id in {_get_id('fever'), _get_id('chills')}:
                        # Skip this test as maintaining of the tests would be
                        # as complex as maintaining the code
                        continue

                elif i == 3:
                    if extremely_sick:
                        if s_id == _get_id('severe'):
                            expected_prob = 1
                        elif s_id in {_get_id('mild'), _get_id('moderate')}:
                            expected_prob = 0
                    elif really_sick:
                        if s_id == _get_id('moderate'):
                            expected_prob = 1
                        elif s_id in {_get_id('mild'), _get_id('severe')}:
                            expected_prob = 0
                    elif s_id == _get_id('mild'):
                        expected_prob = 1
                    elif s_id in {_get_id('moderate'), _get_id('severe')}:
                        expected_prob = 0

                    if s_id == _get_id('lost_consciousness') and \
                            not (really_sick or extremely_sick or len(preexisting_conditions) > 2):
                        expected_prob = 0

                    elif s_id == _get_id('severe_chest_pain') and \
                            not extremely_sick:
                        expected_prob = 0

                elif i == 4:
                    if extremely_sick:
                        if s_id == _get_id('moderate'):
                            expected_prob = 1
                        elif s_id == _get_id('mild'):
                            expected_prob = 0
                    elif s_id == _get_id('mild'):
                        expected_prob = 1
                    elif s_id == _get_id('moderate'):
                        expected_prob = 0

                    if s_id == _get_id('lost_consciousness') and \
                            not (really_sick or extremely_sick or len(preexisting_conditions) > 2):
                        expected_prob = 0

                    elif s_id == _get_id('severe_chest_pain') and \
                            not extremely_sick:
                        expected_prob = 0

                self.assertAlmostEqual(
                    prob, expected_prob,
                    delta=0 if expected_prob in (0, 1) else max(0.015, expected_prob * 0.25),
                    msg=f"Computation of the symptom [{s_name}] yielded an unexpected "
                    f"probability for initial_viral_load {initial_viral_load}, "
                    f"age {age}, really_sick {really_sick}, extremely_sick {extremely_sick}, "
                    f"preexisting_conditions {len(preexisting_conditions)} and "
                    f"carefulness {carefulness} in context {i}:{context}")

            if s_id in self.out_of_context_symptoms:
                prob = sum(probs[i][s_id] for i in range(len(probs))) / len(probs)

                self.assertEqual(prob, 0.0,
                                 msg=f"Symptom [{s_name}] should not be present is the "
                                 f"list of symptoms. initial_viral_load {initial_viral_load}, "
                                 f"age {age}, really_sick {really_sick}, extremely_sick {extremely_sick}, "
                                 f"preexisting_conditions {len(preexisting_conditions)} "
                                 f"and carefulness {carefulness}")


class ColdSymptoms(unittest.TestCase):
    def test_cold_v2_symptoms(self):
        """
            Test the distribution of the cold symptoms
        """
        symptoms_contexts = SYMPTOMS_CONTEXTS['cold']

        def _get_id(symptom):
            return SYMPTOMS[symptom].id

        out_of_context_symptoms = set()
        for _, prob in SYMPTOMS.items():
            for context in prob.probabilities:
                if context in symptoms_contexts.values():
                    break
            else:
                out_of_context_symptoms.add(prob.id)

        n_people = 10000

        rng = np.random.RandomState(1234)

        age = 25
        carefulness = 0.50
        really_sick_options = (True, False)
        extremely_sick_options = (True, False)
        preexisting_conditions_options = (tuple(), ('pre1', 'pre2'))

        s_ids = set()

        for s_name, s_prob in SYMPTOMS.items():
            self.assertNotIn(s_prob.id, s_ids)
            s_ids.add(s_prob.id)

        for really_sick in really_sick_options:
            for extremely_sick in extremely_sick_options if really_sick else (False,):
                for preexisting_conditions in preexisting_conditions_options:
                    # To simplify the tests, we expect each stage to last 1 day
                    computed_dist = [[set(day_symptoms) for day_symptoms in
                                      _get_cold_progression(age, rng, carefulness, preexisting_conditions,
                                                            really_sick, extremely_sick)]
                                     for _ in range(n_people)]

                    probs = [[0] * len(SYMPTOMS) for _ in symptoms_contexts]

                    for human_symptoms in computed_dist:
                        for day_symptoms in human_symptoms[:1]:
                            self.assertEqual(len(day_symptoms), 0)

                        for i, day_symptoms in enumerate((human_symptoms[1], human_symptoms[-1])):
                            self.assertEqual(len([s for s in ('mild', 'moderate') if s in day_symptoms]), 1)

                            for s_name, s_prob in SYMPTOMS.items():
                                probs[i][s_prob.id] += int(s_name in day_symptoms)

                    for symptoms_probs in probs:
                        for i in range(len(symptoms_probs)):
                            symptoms_probs[i] /= n_people

                    for s_name, s_prob in SYMPTOMS.items():
                        s_id = s_prob.id

                        for i, (context, expected_prob) in enumerate((c, p) for c, p in s_prob.probabilities.items()
                                                                     if c in symptoms_contexts):
                            prob = probs[i][s_id]

                            if i == 0:
                                if really_sick or extremely_sick or any(preexisting_conditions):
                                    if s_id == _get_id('moderate'):
                                        expected_prob = 1
                                    elif s_id == _get_id('mild'):
                                        expected_prob = 0
                                elif s_id == _get_id('mild'):
                                    expected_prob = 1
                                elif s_id == _get_id('moderate'):
                                    expected_prob = 0

                            elif i == 1:
                                if s_id == _get_id('mild'):
                                    expected_prob = 1
                                elif s_id == _get_id('moderate'):
                                    expected_prob = 0

                            self.assertAlmostEqual(prob, expected_prob,
                                                   delta=0 if expected_prob in (0, 1)
                                                   else max(0.015, expected_prob * 0.05),
                                                   msg=f"Computation of the symptom [{s_name}] yielded an "
                                                   f"unexpected probability for age {age}, really_sick {really_sick}, "
                                                   f"extremely_sick {extremely_sick}, "
                                                   f"preexisting_conditions {len(preexisting_conditions)} "
                                                   f"and carefulness {carefulness} in context {context}")

                        if s_id in out_of_context_symptoms:
                            prob = sum(probs[i][s_id] for i in range(len(probs))) / len(probs)

                            self.assertEqual(prob, 0.0,
                                             msg=f"Symptom [{s_name}] should not be present is the "
                                             f"list of symptoms. age {age}, really_sick {really_sick}, "
                                             f"extremely_sick {extremely_sick}, "
                                             f"preexisting_conditions {len(preexisting_conditions)} "
                                             f"and carefulness {carefulness}")


class FluSymptoms(unittest.TestCase):
    def test_flu_progression(self):
        """
            Test the distribution of the flu symptoms
        """
        symptoms_contexts = SYMPTOMS_CONTEXTS['flu']

        def _get_id(symptom):
            return SYMPTOMS[symptom].id

        out_of_context_symptoms = set()
        for _, prob in SYMPTOMS.items():
            for context in prob.probabilities:
                if context in symptoms_contexts.values():
                    break
            else:
                out_of_context_symptoms.add(prob.id)

        n_people = 10000

        rng = np.random.RandomState(1234)

        age = 25
        carefulness = 0.50
        really_sick_options = (True, False)
        extremely_sick_options = (True, False)
        preexisting_conditions_options = (tuple(), ('pre1', 'pre2'))

        s_ids = set()

        for s_name, s_prob in SYMPTOMS.items():
            self.assertNotIn(s_prob.id, s_ids)
            s_ids.add(s_prob.id)

        for really_sick in really_sick_options:
            for extremely_sick in extremely_sick_options if really_sick else (False,):
                for preexisting_conditions in preexisting_conditions_options:
                    # To simplify the tests, we expect each stage to last 1 day
                    computed_dist = [[set(day_symptoms) for day_symptoms in
                                      _get_flu_progression(age, rng, carefulness, preexisting_conditions,
                                                           really_sick, extremely_sick)]
                                     for _ in range(n_people)]

                    probs = [[0] * len(SYMPTOMS) for _ in symptoms_contexts]

                    for human_symptoms in computed_dist:
                        for i, day_symptoms in enumerate((*human_symptoms[0:2], human_symptoms[-1])):
                            self.assertEqual(len([s for s in ('mild', 'moderate') if s in day_symptoms]), 1)

                            for s_name, s_prob in SYMPTOMS.items():
                                probs[i][s_prob.id] += int(s_name in day_symptoms)

                    for symptoms_probs in probs:
                        for i in range(len(symptoms_probs)):
                            symptoms_probs[i] /= n_people

                    for s_name, s_prob in SYMPTOMS.items():
                        s_id = s_prob.id

                        for i, (context, expected_prob) in enumerate((c, p) for c, p in s_prob.probabilities.items()
                                                                     if c in symptoms_contexts):
                            prob = probs[i][s_id]

                            if i == 0:
                                if really_sick or extremely_sick or any(preexisting_conditions):
                                    if s_id == _get_id('moderate'):
                                        expected_prob = 1
                                    elif s_id == _get_id('mild'):
                                        expected_prob = 0
                                elif s_id == _get_id('mild'):
                                    expected_prob = 1
                                elif s_id == _get_id('moderate'):
                                    expected_prob = 0

                            elif i == 1:
                                if s_id == _get_id('mild'):
                                    expected_prob = 1
                                elif s_id == _get_id('moderate'):
                                    expected_prob = 0

                            self.assertAlmostEqual(prob, expected_prob,
                                                   delta=0 if expected_prob in (0, 1)
                                                   else max(0.015, expected_prob * 0.05),
                                                   msg=f"Computation of the symptom [{s_name}] yielded an "
                                                   f"unexpected probability for age {age}, really_sick {really_sick}, "
                                                   f"extremely_sick {extremely_sick}, "
                                                   f"preexisting_conditions {len(preexisting_conditions)} "
                                                   f"and carefulness {carefulness} in context {context}")

                        if s_id in out_of_context_symptoms:
                            prob = sum(probs[i][s_id] for i in range(len(probs))) / len(probs)

                            self.assertEqual(prob, 0.0,
                                             msg=f"Symptom [{s_name}] should not be present is the "
                                             f"list of symptoms. age {age}, really_sick {really_sick}, "
                                             f"extremely_sick {extremely_sick}, "
                                             f"preexisting_conditions {len(preexisting_conditions)} "
                                             f"and carefulness {carefulness}")


class PreexistingConditions(unittest.TestCase):
    def test_preexisting_conditions_struct(self):
        c_ids = set()

        for c_name, c_probs in PREEXISTING_CONDITIONS.items():
            c_id = c_probs[0].id

            self.assertEqual(c_id, PREEXISTING_CONDITIONS_META[c_name])

            self.assertNotIn(c_id, c_ids)
            c_ids.add(c_id)

            for c_prob in c_probs:
                self.assertEqual(c_name, c_prob.name)
                self.assertEqual(c_prob.id, c_id)

    def test_preexisting_conditions(self):
        """
            Test the distribution of the preexisting conditions
        """
        def _get_id(cond):
            return PREEXISTING_CONDITIONS[cond][0].id

        def _get_probability(cond_probs, age, sex):
            if isinstance(cond_probs, str):
                cond_probs = PREEXISTING_CONDITIONS[cond_probs]
            return next((c_prob.probability for c_prob in cond_probs
                         if age < c_prob.age and (c_prob.sex in ('a', sex))))

        n_people = 10000

        rng = np.random.RandomState(1234)

        for c_name, c_probs in PREEXISTING_CONDITIONS.items():
            c_id = c_probs[0].id

            # TODO: Implement test for stroke and pregnant
            if c_id in (_get_id('stroke'), _get_id('pregnant')):
                continue

            for c_prob in c_probs:
                age, sex = c_prob.age - 1, c_prob.sex
                expected_prob = c_prob.probability

                if c_id in (_get_id('cancer'), _get_id('COPD')):
                    modifer_prob = _get_probability('smoker', age, sex)
                    expected_prob = expected_prob * (1.3 * modifer_prob + 0.95 * (1 - modifer_prob))

                if c_id == _get_id('heart_disease'):
                    modifer_prob = _get_probability('diabetes', age, sex) + \
                                   _get_probability('smoker', age, sex)
                    expected_prob = expected_prob * (2 * modifer_prob + 0.5 * (1 - modifer_prob))

                if c_id == _get_id('immuno-suppressed'):
                    modifer_prob = _get_probability('cancer', age, sex)
                    expected_prob = expected_prob * (1.2 * modifer_prob + 0.98 * (1 - modifer_prob))

                if c_id == _get_id('lung_disease'):
                    expected_prob = _get_probability('asthma', age, sex) + \
                                    _get_probability('COPD', age, sex)

                computed_dist = [_get_preexisting_conditions(age, sex, rng) for _ in range(n_people)]

                prob = len([1 for conditions in computed_dist if c_name in conditions]) / n_people

                self.assertAlmostEqual(prob, expected_prob,
                                       delta=0 if not expected_prob else max(0.015, expected_prob * 0.05),
                                       msg=f"Computation of the preexisting conditions [{c_name}] yielded an "
                                       f"unexpected probability for age {age} and sex {sex}")
