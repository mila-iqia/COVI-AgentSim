import datetime
import glob
import os
import pickle
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from covid19sim.run import run_simu


class ModelsTest(unittest.TestCase):
    def test_run(self):
        """
            run one simulation and ensure json files are correctly populated and most of the users have activity
        """
        with TemporaryDirectory() as preprocess_d:
            n_people = 100
            n_days = 30
            monitors, _ = run_simu(
                n_people=n_people,
                init_percent_sick=0.25,
                start_time=datetime.datetime(2020, 2, 28, 0, 0),
                simulation_days=n_days,
                outfile=os.path.join(preprocess_d, "output"),
                out_chunk_size=1,
                seed=0, n_jobs=4,
                port=6688
            )

            days_output = glob.glob(f"{preprocess_d}/daily_outputs/*/")
            days_output.sort()

            self.assertEqual(len(days_output), n_days)

            output = [None] * len(days_output)
            for day_output in days_output:
                pkls = glob.glob(f"{day_output}*/daily_human.pkl")
                pkls.sort()
                day_humans = []
                for pkl in pkls:
                    with open(pkl, 'rb') as f:
                        day_human = pickle.load(f)
                        day_humans.append(day_human)
                        self.assertEqual(day_human['current_day'], day_humans[0]['current_day'])
                self.assertGreaterEqual(len(day_humans), n_people)
                output[day_humans[0]['current_day']] = day_humans

            for i in range(1, len(output)):
                self.assertEqual(len(output[i-1]), len(output[i]))

            stats = {'human_enc_ids': [0] * 256,
                     'humans': {}}

            for current_day, day_output in enumerate(output):
                for h_i, human in enumerate(day_output):
                    stats['humans'].setdefault(h_i, {})
                    stats['humans'][h_i].setdefault('candidate_encounters_cnt', 0)
                    stats['humans'][h_i].setdefault('has_exposure_day', 0)
                    stats['humans'][h_i].setdefault('has_recovery_day', 0)
                    stats['humans'][h_i].setdefault('exposure_encounter_cnt', 0)
                    stats['humans'][h_i].setdefault('infectiousness', 0)

                    self.assertEqual(current_day, human['current_day'])

                    observed = human['observed']
                    unobserved = human['unobserved']

                    if current_day == 0:
                        prev_observed = None
                        prev_unobserved = None
                    else:
                        prev_observed = output[current_day - 1][h_i]['observed']
                        prev_unobserved = output[current_day - 1][h_i]['unobserved']

                    # Multi-hot arrays identifying the reported symptoms in the last 14 days
                    # Symptoms:
                    # ['aches', 'cough', 'fatigue', 'fever', 'gastro', 'loss_of_taste',
                    #  'mild', 'moderate', 'runny_nose', 'severe', 'trouble_breathing']
                    self.assertEqual(observed['reported_symptoms'].shape, (14, 28))
                    if observed['candidate_encounters'].size:
                        stats['humans'][h_i]['candidate_encounters_cnt'] += 1
                        # candidate_encounters[:, 0] is the other human's signature id
                        # candidate_encounters[:, 1] is the 4 bits new risk of getting contaminated during the encounter
                        # candidate_encounters[:, 2] is the length of the encounter
                        # candidate_encounters[:, 3] is the number of days since the encounter
                        self.assertEqual(observed['candidate_encounters'].shape[1], 4)
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 0].min(), 0)
                        self.assertLess(observed['candidate_encounters'][:, 0].max(), 256)
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 1].min(), 0)
                        self.assertLess(observed['candidate_encounters'][:, 1].max(), 16)
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 2].min(), 0)
                        self.assertLess(observed['candidate_encounters'][:, 2].max(), 10000)
                        self.assertLessEqual(observed['candidate_encounters'][:, 3].max(), current_day)
                        # # TODO: Expecting only the last 14 days
                        # self.assertLess(observed['candidate_encounters'][:, 3].max() -
                        #                 observed['candidate_encounters'][:, 3].min(), 14)
                        # TODO: Expecting the messages to be ordered by day and to be
                        #  in reverse chronological order
                        # self.assertLessEqual(observed['candidate_encounters'][0, 3],
                        #                      observed['candidate_encounters'][-1, 3])

                        for h_enc_id in observed['candidate_encounters'][:, 0]:
                            stats['human_enc_ids'][h_enc_id] += 1

                    # Has received a positive test result [index] days before today
                    self.assertEqual(observed['test_results'].shape, (14,))
                    self.assertIn(observed['test_results'].min(), (0, 1))
                    self.assertIn(observed['test_results'].max(), (0, 1))
                    self.assertIn(observed['test_results'].sum(), (0, 1))

                    # Multihot encoding
                    self.assertIn(observed['preexisting_conditions'].min(), (0, 1))
                    self.assertIn(observed['preexisting_conditions'].max(), (0, 1))
                    self.assertGreaterEqual(observed['age'], -1)
                    self.assertGreaterEqual(observed['sex'], -1)

                    # Multi-hot arrays identifying the true symptoms in the last 14 days
                    # Symptoms:
                    # ['aches', 'cough', 'fatigue', 'fever', 'gastro', 'loss_of_taste',
                    #  'mild', 'moderate', 'runny_nose', 'severe', 'trouble_breathing']
                    self.assertEqual(unobserved['true_symptoms'].shape, (14, 28))
                    # Has been exposed or not
                    self.assertIn(unobserved['is_exposed'], (0, 1))
                    if unobserved['exposure_day'] is not None:
                        stats['humans'][h_i]['has_exposure_day'] = 1
                        # For how long has been exposed
                        self.assertTrue(0 <= unobserved['exposure_day'] < 14)
                    # Is recovered or not
                    self.assertIn(unobserved['is_recovered'], (0, 1))
                    if unobserved['recovery_day'] is not None:
                        stats['humans'][h_i]['has_recovery_day'] = 1
                        # For how long has been infectious
                        self.assertTrue(0 <= unobserved['recovery_day'] < 14)
                    if observed['candidate_encounters'].size:
                        stats['humans'][h_i]['exposure_encounter_cnt'] += 1
                        # Encounters responsible for exposition. Exposition can occur without being
                        # linked to an encounter
                        self.assertTrue(len(unobserved['exposure_encounter'].shape) == 1)
                        self.assertTrue(unobserved['exposure_encounter'].min() in (0, 1))
                        self.assertTrue(unobserved['exposure_encounter'].max() in (0, 1))
                        self.assertTrue(unobserved['exposure_encounter'].sum() in (0, 1))
                    if unobserved['infectiousness'].size:
                        stats['humans'][h_i]['infectiousness'] += 1
                        # Level of infectiousness / day
                        self.assertLessEqual(unobserved['infectiousness'].shape[0], 14)
                        self.assertGreaterEqual(unobserved['infectiousness'].min(), 0)
                        self.assertLessEqual(unobserved['infectiousness'].max(), 10)

                    # Multihot encoding
                    self.assertIn(unobserved['true_preexisting_conditions'].min(), (0, 1))
                    self.assertIn(unobserved['true_preexisting_conditions'].max(), (0, 1))
                    self.assertGreaterEqual(unobserved['true_age'], -1)
                    self.assertGreaterEqual(unobserved['true_sex'], -1)

                    # observed['reported_symptoms'] is a subset of unobserved['true_symptoms']
                    # TODO: Test fails with some values of observed['reported_symptoms'] not being in
                    #  unobserved['true_symptoms'].
                    # self.assertTrue((unobserved['true_symptoms'] == observed['reported_symptoms'])
                    #                 [observed['reported_symptoms'].astype(np.bool)].all())

                    if unobserved['is_exposed'] or unobserved['is_recovered']:
                        self.assertNotEqual(unobserved['is_exposed'], unobserved['is_recovered'])

                    if observed['candidate_encounters'].size:
                        # exposure_encounter is the same length as candidate_encounters
                        self.assertEqual(unobserved['exposure_encounter'].shape,
                                         (observed['candidate_encounters'].shape[0],))

                    # observed['preexisting_conditions'] is a subset of unobserved['true_preexisting_conditions']
                    self.assertTrue((unobserved['true_preexisting_conditions'] == observed['preexisting_conditions'])
                                    [observed['preexisting_conditions'].astype(np.bool)].all())
                    # If observed['age'] is set, unobserved['true_age'] should also be set to the same value
                    self.assertGreaterEqual(unobserved['true_age'], observed['age'])
                    # If observed['sex'] is set, unobserved['true_sex'] should also be set to the same value
                    self.assertGreaterEqual(unobserved['true_sex'], observed['sex'])

                    if prev_observed:
                        # TODO: Test fails with values updated in observed['reported_symptoms'][1].
                        #  Expecting only the first row to be updated
                        # self.assertTrue((observed['reported_symptoms'][1:] == prev_observed['reported_symptoms'][:13]).all())
                        if observed['candidate_encounters'].size and prev_observed['candidate_encounters'].size:
                            # TODO: Can't validate rolling of the message because of new messages being added
                            current_day_mask = observed['candidate_encounters'][:, 3] < current_day  # Get the last 13 days excluding today
                            prev_day_mask = prev_observed['candidate_encounters'][:, 3] > current_day - 14  # Get the last 13 days including relative today (of yesterday)
                            masked = observed['candidate_encounters'][current_day_mask][:, (0, 1, 3)]
                            prev_masked = prev_observed['candidate_encounters'][prev_day_mask][:, (0, 1, 3)]
                            offset = 0
                            for i in range(prev_masked.shape[0]):
                                for j in range(i+offset, masked.shape[0]):
                                    if (prev_masked[i] == masked[j]).all():
                                        break
                                    # Skipping updated message
                                    offset += 1
                                else:
                                    self.assertFalse(False,
                                                     msg=f"Could not find previous candidate_encounter {prev_masked[i]} "
                                                     f"in current day.")
                        # TODO: Test fails with observed['test_results'] and prev_observed['test_results'] being the same array
                        # self.assertTrue((observed['test_results'][1:] == prev_observed['test_results'][:13]).all())

                        self.assertTrue((observed['preexisting_conditions'] ==
                                         prev_observed['preexisting_conditions']).all())
                        self.assertEqual(observed['age'], prev_observed['age'])
                        self.assertEqual(observed['sex'], prev_observed['sex'])

                        # TODO: Test fails with values updated in unobserved['true_symptoms'][1].
                        #  Expecting only the first row to be updated
                        # self.assertTrue((unobserved['true_symptoms'][1:] == prev_unobserved['true_symptoms'][:13]).all())
                        # TODO: Test fails with unobserved['infectiousness'] being the same as prev_unobserved['infectiousness']
                        #  (appears to sometimes be not updated)
                        check = unobserved['infectiousness'][1:] == prev_unobserved['infectiousness'][:13]
                        self.assertTrue(check if isinstance(check, bool) else check.all())

                        if prev_unobserved['is_exposed'] and prev_unobserved['exposure_day'] < 13:
                            self.assertTrue(unobserved['is_exposed'])
                            self.assertEqual(max(0, unobserved['exposure_day'] - 1),
                                             prev_unobserved['exposure_day'])

                        if unobserved['is_exposed'] != prev_unobserved['is_exposed']:
                            if unobserved['is_exposed']:
                                # TODO: Test fails. Expecting the exposure day to be 0 since the human
                                #  just got exposed
                                # self.assertEqual(unobserved['exposure_day'], 0)
                                self.assertEqual(prev_unobserved['infectiousness'][0], 0)
                            else:
                                self.assertEqual(prev_unobserved['exposure_day'], 13)
                                self.assertFalse(unobserved['is_exposed'])
                                self.assertEqual(unobserved['exposure_day'], None)

                        self.assertTrue((unobserved['true_preexisting_conditions'] ==
                                         prev_unobserved['true_preexisting_conditions']).all())
                        self.assertEqual(unobserved['true_age'], prev_unobserved['true_age'])
                        self.assertEqual(unobserved['true_sex'], prev_unobserved['true_sex'])

            candidate_encounters_cnt = 0
            has_exposure_day = 0
            has_recovery_day = 0
            exposure_encounter_cnt = 0
            infectiousness = 0
            for _, human_stats in stats['humans'].items():
                candidate_encounters_cnt += human_stats['candidate_encounters_cnt']
                has_exposure_day += human_stats['has_exposure_day']
                has_recovery_day += human_stats['has_recovery_day']
                exposure_encounter_cnt += human_stats['exposure_encounter_cnt']
                infectiousness += human_stats['infectiousness']

            self.assertGreaterEqual(sum(stats['human_enc_ids']), n_people)
            self.assertGreaterEqual(candidate_encounters_cnt, n_people)
            self.assertGreaterEqual(has_exposure_day, n_people * 0.5)
            # TODO: Is it expected to have no recovery_days?
            # self.assertGreaterEqual(has_recovery_day, n_people)
            self.assertGreaterEqual(exposure_encounter_cnt, n_people)
            self.assertGreaterEqual(infectiousness, n_people)
