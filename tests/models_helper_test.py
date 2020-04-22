import unittest

from utils import PREEXISTING_CONDITIONS

from models.helper import conditions_to_np


class ModelsHelperTest(unittest.TestCase):
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
