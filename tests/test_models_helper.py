import numpy as np
import unittest
import warnings

from covid19sim.epidemiology.human_properties import PREEXISTING_CONDITIONS
from covid19sim.epidemiology.symptoms import SYMPTOMS, STR_TO_SYMPTOMS

from covid19sim.inference.helper import conditions_to_np, symptoms_to_np, \
    encode_age, encode_sex, encode_test_result, PREEXISTING_CONDITIONS_META


class ModelsHelperTest(unittest.TestCase):

    def test_legacy_all_possible_symptoms(self):
        # Legacy all_possible_symptoms
        all_possible_symptoms = ['moderate', 'mild', 'severe', 'extremely-severe', 'fever',
                                 'chills', 'gastro', 'diarrhea', 'nausea_vomiting', 'fatigue',
                                 'unusual', 'hard_time_waking_up', 'headache', 'confused',
                                 'lost_consciousness', 'trouble_breathing', 'sneezing',
                                 'cough', 'runny_nose', 'aches', 'sore_throat', 'severe_chest_pain',
                                 'loss_of_taste', 'mild_trouble_breathing', 'light_trouble_breathing',
                                 'moderate_trouble_breathing',
                                 'heavy_trouble_breathing']

        for symptom in SYMPTOMS:
            if symptom.id < len(all_possible_symptoms):
                self.assertEqual(str(symptom), all_possible_symptoms[symptom.id])

        for s_name in all_possible_symptoms:
            self.assertIn(s_name, STR_TO_SYMPTOMS)

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
        np_symptoms = symptoms_to_np([[s for s in SYMPTOMS]] * 14, {"TRACING_N_DAYS_HISTORY": 14})

        self.assertEqual(np_symptoms.shape, (14, len(SYMPTOMS)))
        self.assertEqual(np_symptoms.sum(), len(SYMPTOMS) * 14)
        self.assertEqual(np_symptoms.max(), 1)
        self.assertEqual(np_symptoms.min(), 1)

        np_symptoms = symptoms_to_np([[]] * 14, {"TRACING_N_DAYS_HISTORY": 14})

        self.assertEqual(np_symptoms.shape, (14, len(SYMPTOMS)))
        self.assertEqual(np_symptoms.sum(), 0)
        self.assertEqual(np_symptoms.max(), 0)
        self.assertEqual(np_symptoms.min(), 0)

        for symptom in SYMPTOMS:
            np_symptoms = symptoms_to_np([[symptom]] * 14, {"TRACING_N_DAYS_HISTORY": 14})

            self.assertEqual(np_symptoms.shape, (14, len(SYMPTOMS)))

            for i in range(np_symptoms.shape[0]):
                self.assertTrue((np_symptoms[i] == np_symptoms[0]).all())

            self.assertEqual(np_symptoms[0][symptom.id], 1)
            self.assertEqual(np_symptoms[0].sum(), 1)
            self.assertEqual(np_symptoms[0].max(), 1)
            self.assertEqual(np_symptoms[0].min(), 0)

        raw_symptoms = [[] for _ in range(14)]

        for i, symptom in zip(range(len(raw_symptoms)), SYMPTOMS):
            raw_symptoms[i].append(symptom)

        np_symptoms = symptoms_to_np(raw_symptoms, {"TRACING_N_DAYS_HISTORY": 14})

        self.assertEqual(np_symptoms.shape, (14, len(SYMPTOMS)))
        self.assertEqual(np_symptoms.sum(), 14)
        self.assertEqual(np_symptoms.max(), 1)
        self.assertEqual(np_symptoms.min(), 0)

        for i, symptom in zip(range(len(raw_symptoms)), SYMPTOMS):
            self.assertEqual(np_symptoms[i][symptom.id], 1)

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

    def test_encode_test_result(self):
        self.assertEqual(encode_test_result('positive'), 1)
        self.assertEqual(encode_test_result('POSITIVE'), 1)
        self.assertEqual(encode_test_result('negative'), -1)
        self.assertEqual(encode_test_result('NEGATIVE'), -1)
        self.assertEqual(encode_test_result(None), 0)
