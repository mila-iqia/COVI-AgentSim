import numpy as np
import unittest
import warnings

from covid19sim.utils import PREEXISTING_CONDITIONS, SYMPTOMS

from covid19sim.frozen.helper import conditions_to_np, symptoms_to_np, \
    encode_age, encode_sex, PREEXISTING_CONDITIONS_META, \
    SYMPTOMS_META


class ModelsHelperTest(unittest.TestCase):

    def test_frozen_symptoms_names_and_ids(self):
        self.assertEqual(len(SYMPTOMS_META), len(SYMPTOMS))
        self.assertTrue(np.array_equal(np.unique(np.asarray(list(SYMPTOMS_META.values()))),
                                       np.arange(len(SYMPTOMS_META))))
        for symp_meta, symp in zip(SYMPTOMS_META.items(), SYMPTOMS.items()):
            self.assertEqual(symp_meta[0], symp[0])
            self.assertEqual(symp_meta[1], symp[1].id)

        # Legacy all_possible_symptoms
        all_possible_symptoms = ['moderate', 'mild', 'severe', 'extremely-severe', 'fever',
                                 'chills', 'gastro', 'diarrhea', 'nausea_vomiting', 'fatigue',
                                 'unusual', 'hard_time_waking_up', 'headache', 'confused',
                                 'lost_consciousness', 'trouble_breathing', 'sneezing',
                                 'cough', 'runny_nose', 'aches', 'sore_throat', 'severe_chest_pain',
                                 'loss_of_taste', 'mild_trouble_breathing', 'light_trouble_breathing',
                                 'moderate_trouble_breathing',
                                 'heavy_trouble_breathing']

        for s_name, s_id in SYMPTOMS_META.items():
            if s_id < len(all_possible_symptoms):
                self.assertEqual(s_name, all_possible_symptoms[s_id])

        for s_name in all_possible_symptoms:
            self.assertIn(s_name, SYMPTOMS_META)

    def test_frozen_conditions_names_and_ids(self):
        self.assertEqual(len(PREEXISTING_CONDITIONS_META), len(PREEXISTING_CONDITIONS))
        self.assertTrue(np.array_equal(np.unique(np.asarray(list(PREEXISTING_CONDITIONS_META.values()))),
                                       np.arange(len(PREEXISTING_CONDITIONS_META))))
        for cond_meta, cond in zip(PREEXISTING_CONDITIONS_META.items(), PREEXISTING_CONDITIONS.items()):
            self.assertEqual(cond_meta[0], cond[0])
            self.assertEqual(cond_meta[1], cond[1][0].id)

    def test_conditions_to_np(self):
        np_conditions = conditions_to_np([c for c in PREEXISTING_CONDITIONS_META])

        self.assertEqual(len(np_conditions), len(PREEXISTING_CONDITIONS_META))
        self.assertEqual(np_conditions.sum(), len(PREEXISTING_CONDITIONS_META))
        self.assertEqual(np_conditions.max(), 1)
        self.assertEqual(np_conditions.min(), 1)

        np_conditions = conditions_to_np([])

        self.assertEqual(len(np_conditions), len(PREEXISTING_CONDITIONS_META))
        self.assertEqual(np_conditions.sum(), 0)
        self.assertEqual(np_conditions.max(), 0)
        self.assertEqual(np_conditions.min(), 0)

        for condition, c_id in PREEXISTING_CONDITIONS_META.items():
            np_conditions = conditions_to_np([condition])

            self.assertEqual(np_conditions[c_id], 1)
            self.assertEqual(len(np_conditions), len(PREEXISTING_CONDITIONS_META))
            self.assertEqual(np_conditions.sum(), 1)
            self.assertEqual(np_conditions.max(), 1)
            self.assertEqual(np_conditions.min(), 0)

    def test_symptoms_to_np(self):
        np_symptoms = symptoms_to_np([[s for s in SYMPTOMS_META]] * 14, [])

        warnings.warn("Lenght of the output of covid19sim.frozen.helper.symptoms_to_np() is "
                      "len(SYMPTOMS_META) + 1. This doesn't look right.")
        self.assertEqual(np_symptoms.shape, (14, len(SYMPTOMS_META) + 1))
        self.assertEqual(np_symptoms.sum(), len(SYMPTOMS_META) * 14)
        self.assertEqual(np_symptoms.max(), 1)
        self.assertEqual(np_symptoms.min(), 0)

        np_symptoms = symptoms_to_np([[]] * 14, [])

        self.assertEqual(np_symptoms.shape, (14, len(SYMPTOMS_META) + 1))
        self.assertEqual(np_symptoms.sum(), 0)
        self.assertEqual(np_symptoms.max(), 0)
        self.assertEqual(np_symptoms.min(), 0)

        for symptom, s_id in SYMPTOMS_META.items():
            np_symptoms = symptoms_to_np([[symptom]] * 14, [])

            self.assertEqual(np_symptoms.shape, (14, len(SYMPTOMS_META) + 1))

            for i in range(np_symptoms.shape[0]):
                self.assertTrue((np_symptoms[i] == np_symptoms[0]).all())

            self.assertEqual(np_symptoms[0][s_id], 1)
            self.assertEqual(np_symptoms[0].sum(), 1)
            self.assertEqual(np_symptoms[0].max(), 1)
            self.assertEqual(np_symptoms[0].min(), 0)

        raw_symptoms = [[] for _ in range(14)]

        for i, symptom in zip(range(len(raw_symptoms)), SYMPTOMS_META):
            raw_symptoms[i].append(symptom)

        np_symptoms = symptoms_to_np(raw_symptoms, [])

        self.assertEqual(np_symptoms.shape, (14, len(SYMPTOMS_META) + 1))
        self.assertEqual(np_symptoms.sum(), 14)
        self.assertEqual(np_symptoms.max(), 1)
        self.assertEqual(np_symptoms.min(), 0)

        for i, symptom_id in zip(range(len(raw_symptoms)), SYMPTOMS_META.values()):
            self.assertEqual(np_symptoms[i][symptom_id], 1)

        self.assertEqual(np_symptoms.sum(), 14)
        self.assertEqual(np_symptoms.max(), 1)
        self.assertEqual(np_symptoms.min(), 0)

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
