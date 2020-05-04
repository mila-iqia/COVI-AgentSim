import numpy as np
import unittest

from covid19sim.utils import PREEXISTING_CONDITIONS

from covid19sim.frozen.helper import conditions_to_np, encode_age, encode_sex, PREEXISTING_CONDITIONS_META


class ModelsHelperTest(unittest.TestCase):

    def test_frozen_names_and_ids(self):
        self.assertEqual(len(PREEXISTING_CONDITIONS_META), len(PREEXISTING_CONDITIONS))
        self.assertTrue(np.array_equal(np.unique(np.asarray(list(PREEXISTING_CONDITIONS_META.values()))),
                                       np.arange(len(PREEXISTING_CONDITIONS_META))))
        for cond_meta, cond in zip(PREEXISTING_CONDITIONS_META.items(), PREEXISTING_CONDITIONS.items()):
            self.assertEqual(cond_meta[0], cond[0])
            self.assertEqual(cond_meta[1], cond[1][0].id)

    def test_conditions_to_np(self):
        conditions = [(k, v[0].id) for k, v in PREEXISTING_CONDITIONS.items()]

        np_conditions = conditions_to_np([c[0] for c in conditions])

        self.assertEqual(len(np_conditions), len(conditions))
        self.assertEqual(np_conditions.sum(), len(conditions))
        self.assertEqual(np_conditions.max(), 1)
        self.assertEqual(np_conditions.min(), 1)

        np_conditions = conditions_to_np([])

        self.assertEqual(len(np_conditions), len(conditions))
        self.assertEqual(np_conditions.sum(), 0)
        self.assertEqual(np_conditions.max(), 0)
        self.assertEqual(np_conditions.min(), 0)

        for condition, id in conditions:
            np_conditions = conditions_to_np([condition])

            self.assertEqual(np_conditions[id], 1)
            self.assertEqual(len(np_conditions), len(conditions))
            self.assertEqual(np_conditions.sum(), 1)
            self.assertEqual(np_conditions.max(), 1)
            self.assertEqual(np_conditions.min(), 0)

    def test_encode_age(self):
        self.assertEqual(encode_age(0), 0)
        self.assertEqual(encode_age(10), 10)
        self.assertEqual(encode_age(-1), -1)
        self.assertEqual(encode_age(None), -1)

    def test_encode_sex(self):
        self.assertEqual(encode_sex('F'), 1)
        self.assertEqual(encode_sex('f'), 1)
        self.assertEqual(encode_sex('FEMALE'), 1)
        self.assertEqual(encode_sex('female'), 1)
        self.assertEqual(encode_sex('M'), 2)
        self.assertEqual(encode_sex('m'), 2)
        self.assertEqual(encode_sex('MALE'), 2)
        self.assertEqual(encode_sex('male'), 2)
        self.assertEqual(encode_sex('O'), 0)
        self.assertEqual(encode_sex('o'), 0)
        self.assertEqual(encode_sex('OTHERS'), 0)
        self.assertEqual(encode_sex('others'), 0)
        self.assertEqual(encode_sex(''), -1)
        self.assertEqual(encode_sex(None), -1)
