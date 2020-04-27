import unittest

import numpy as np

from utils import _get_preexisting_conditions, PREEXISTING_CONDITIONS


class PreexistingConditions(unittest.TestCase):
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

        c_ids = set()

        for c_name, c_probs in PREEXISTING_CONDITIONS.items():
            c_id = c_probs[0].id

            self.assertNotIn(c_id, c_ids)
            c_ids.add(c_id)

            # TODO: Implement test for stroke and pregnant
            if c_id in (_get_id('stroke'), _get_id('pregnant')):
                continue

            for c_prob in c_probs:
                self.assertEqual(c_prob.id, c_probs[0].id)

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
