import datetime
import glob
import os
import pickle
from collections import defaultdict
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from covid19sim.frozen.helper import conditions_to_np, encode_age, encode_sex
from covid19sim.configs.exp_config import ExpConfig


class MakeHumanAsMessageProxy:
    def __init__(self, test_case: unittest.TestCase):
        self.humans_logs = defaultdict(dict)
        self.test_case = test_case

    def make_human_as_message(self, human):
        tc = self.test_case

        message = ModelsTest.make_human_as_message(human)

        now = human.env.timestamp
        today = now.date()

        with tc.subTest(name=human.name, now=now):
            tc.assertEqual(human.last_date['symptoms_updated'], today)
            tc.assertIsInstance(human.name, str)
            tc.assertIsInstance(human.age, (int, np.integer))
            tc.assertGreaterEqual(human.age, 0)
            tc.assertIsInstance(human.sex, str)
            tc.assertIn(human.sex.lower()[0], {'f', 'm', 'o'})

            if human.obs_age is not None:
                tc.assertIsInstance(human.obs_age, (int, np.integer))
                tc.assertGreaterEqual(human.obs_age, 0)
            if human.obs_sex is not None:
                tc.assertIsInstance(human.obs_sex, str)
                tc.assertIn(human.obs_sex.lower()[0], {'f', 'm', 'o'})

            validate_human_message(tc, message, human)

        return message


def validate_human_message(test_case, human_as_message, human):
    test_case.assertEqual(human_as_message.name, human.name)
    test_case.assertEqual(human_as_message.age, encode_age(human.age))
    test_case.assertEqual(human_as_message.sex, encode_sex(human.sex))
    test_case.assertEqual(human_as_message.obs_age, encode_age(human.obs_age))
    test_case.assertEqual(human_as_message.obs_sex, encode_sex(human.obs_sex))

    test_case.assertEqual(human_as_message.infectiousnesses, human.infectiousnesses)
    test_case.assertEqual(human_as_message.infection_timestamp, human.infection_timestamp)
    test_case.assertEqual(human_as_message.recovered_timestamp, human.recovered_timestamp)

    test_case.assertEqual(human_as_message.clusters, human.clusters)
    test_case.assertEqual(human_as_message.exposure_message, human.exposure_message)
    test_case.assertEqual(human_as_message.carefulness, human.carefulness)
    test_case.assertEqual(human_as_message.has_app, human.has_app)

    test_case.assertEqual(human_as_message.preexisting_conditions.sum(), len(human.preexisting_conditions))
    test_case.assertTrue((human_as_message.preexisting_conditions ==
                          conditions_to_np(human.preexisting_conditions)).all())
    test_case.assertEqual(human_as_message.obs_preexisting_conditions.sum(),
                          len(human.obs_preexisting_conditions))
    test_case.assertTrue((human_as_message.obs_preexisting_conditions ==
                          conditions_to_np(human.obs_preexisting_conditions)).all())

    test_case.assertEqual(human_as_message.rolling_all_symptoms.shape[0], len(human.rolling_all_symptoms))
    for m_rolling_all_symptoms, h_rolling_all_symptoms in \
            zip(human_as_message.rolling_all_symptoms, human.rolling_all_symptoms):
        test_case.assertEqual(m_rolling_all_symptoms.sum(), len(h_rolling_all_symptoms))
    test_case.assertEqual(human_as_message.rolling_all_reported_symptoms.shape[0],
                          len(human.rolling_all_reported_symptoms))
    for m_rolling_all_reported_symptoms, h_rolling_all_reported_symptomsin in \
            zip(human_as_message.rolling_all_reported_symptoms, human.rolling_all_reported_symptoms):
        test_case.assertEqual(m_rolling_all_reported_symptoms.sum(), len(h_rolling_all_reported_symptomsin))

    test_case.assertEqual(len(human_as_message.test_results), len(human.test_results))

    test_case.assertEqual(len(human_as_message.messages), len([m for m in human.contact_book.messages
                                                               if m.day == human.contact_book.messages[-1].day]))
    test_case.assertEqual(len(human_as_message.update_messages),
                          len([u_m for u_m in human.contact_book.update_messages
                               if u_m.day == human.contact_book.update_messages[-1].day]))


class ModelsTest(unittest.TestCase):
    from covid19sim.models.run import make_human_as_message
    make_human_as_message_proxy = None

    def setUp(self):
        from covid19sim.models import run

        proxy = MakeHumanAsMessageProxy(self)
        ModelsTest.make_human_as_message_proxy = proxy
        run.make_human_as_message = proxy.make_human_as_message

    def tearDown(self):
        from covid19sim.models import run

        run.make_human_as_message = self.make_human_as_message

    def test_run(self):
        """
            run one simulation and ensure json files are correctly populated and most of the users have activity
        """
        from covid19sim.run import run_simu

        with TemporaryDirectory() as preprocess_d:
            n_people = 100
            n_days = 30
            monitors, _ = run_simu(
                n_people=n_people,
                start_time=datetime.datetime(2020, 2, 28, 0, 0),
                simulation_days=n_days,
                init_percent_sick=0.1,
                outfile=os.path.join(preprocess_d, "output"),
                out_chunk_size=1,
                seed=0, n_jobs=4,
                port=6688
            )
            days_output = glob.glob(f"{preprocess_d}/daily_outputs/*/")
            days_output.sort(key=lambda p: int(p.split(os.path.sep)[-2]))
            self.assertEqual(len(days_output), n_days - ExpConfig.get('INTERVENTION_DAY'))
            output = [[]] * len(days_output)

            for day_output in days_output:
                pkls = glob.glob(f"{day_output}*/daily_human-*.pkl")
                pkls.sort(key=lambda p: (int(p.split(os.path.sep)[-3]), int(p.split(os.path.sep)[-2])))
                day_humans = []
                for pkl in pkls:
                    with open(pkl, 'rb') as f:
                        day_human = pickle.load(f)
                        day_humans.append(day_human)
                        self.assertEqual(day_human['current_day'], day_humans[0]['current_day'])
                self.assertGreaterEqual(len(day_humans), n_people)
                output[day_humans[0]['current_day'] - ExpConfig.get('INTERVENTION_DAY')] = day_humans

            for i in range(1, len(output)):
                self.assertEqual(len(output[i-1]), len(output[i]))

            stats = {'human_enc_ids': [0] * 256,
                     'humans': {}}

            for current_day, day_output in zip(range(ExpConfig.get('INTERVENTION_DAY'), n_days),
                                               output):
                output_day_index = current_day - ExpConfig.get('INTERVENTION_DAY')
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

                    if current_day == ExpConfig.get('INTERVENTION_DAY'):
                        prev_observed = None
                        prev_unobserved = None
                    else:
                        prev_observed = output[output_day_index - 1][h_i]['observed']
                        prev_unobserved = output[output_day_index - 1][h_i]['unobserved']

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
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 0].min(), 0)  # cluster id
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 1].min(), 0)  # risk level
                        self.assertLess(observed['candidate_encounters'][:, 1].max(), 16)  # risk level
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 2].min(), 0)  # encounters
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 3].min(), 0)  # day idx

                    # Has received a positive test result [index] days before today
                    self.assertEqual(observed['test_results'].shape, (14,))
                    self.assertIn(observed['test_results'].min(), (-1, 0, 1))
                    self.assertIn(observed['test_results'].max(), (-1, 0, 1))
                    self.assertIn(observed['test_results'][observed['test_results'] == 1].sum(), (0, 1))

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
                    self.assertTrue((unobserved['true_symptoms'] == observed['reported_symptoms'])
                                    [observed['reported_symptoms'].astype(np.bool)].all())

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
                        self.assertTrue((observed['reported_symptoms'][1:] == prev_observed['reported_symptoms'][:13]).all())
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
                        self.assertTrue((observed['test_results'][1:] == prev_observed['test_results'][:13]).all())
                        self.assertTrue((observed['preexisting_conditions'] ==
                                         prev_observed['preexisting_conditions']).all())
                        self.assertEqual(observed['age'], prev_observed['age'])
                        self.assertEqual(observed['sex'], prev_observed['sex'])

                        self.assertTrue((unobserved['true_symptoms'][1:] == prev_unobserved['true_symptoms'][:13]).all())
                        check = unobserved['infectiousness'][1:] == prev_unobserved['infectiousness'][:13]
                        self.assertTrue(check if isinstance(check, bool) else check.all())

                        if prev_unobserved['is_exposed'] and prev_unobserved['exposure_day'] < 13:
                            self.assertTrue(unobserved['is_exposed'])
                            self.assertEqual(max(0, unobserved['exposure_day'] - 1),
                                             prev_unobserved['exposure_day'])

                        if unobserved['is_exposed'] != prev_unobserved['is_exposed']:
                            if unobserved['is_exposed']:
                                self.assertEqual(unobserved['exposure_day'], 0)
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

            # TODO: Validate the values to check against
            self.assertGreaterEqual(candidate_encounters_cnt, n_people)
            self.assertGreaterEqual(has_exposure_day, n_people * 0.5)
            self.assertGreaterEqual(has_recovery_day, n_people * 0.2)
            self.assertGreaterEqual(exposure_encounter_cnt, n_people)
            self.assertGreaterEqual(infectiousness, n_people)


class HumanAsMessageTest(unittest.TestCase):
    class EnvMock:
        def __init__(self, timestamp):
            self.timestamp = timestamp
    
    def test_human_as_message(self):
        from covid19sim.simulator import Human
        from covid19sim.models.run import make_human_as_message

        rng = np.random.RandomState(1234)

        today = datetime.datetime.today()

        env = self.EnvMock(today)

        human = Human(env=env, city={'city': 'city'}, name=1, age=25, rng=rng, has_app=True,
                      infection_timestamp=today, household={'household': 'household'},
                      workplace={'workplace': 'workplace'}, profession="profession", rho=0.3,
                      gamma=0.21, symptoms=[], test_results=None)

        message = make_human_as_message(human)

        for k in message.__dict__.keys():
            if k in ('messages', 'update_messages'):
                self.assertIn(k, human.contact_book.__dict__)
            else:
                self.assertIn(k, human.__dict__)

        validate_human_message(self, message, human)


if __name__ == "__main__":
    # Load the experimental configuration
    ExpConfig.load_config(os.path.join(os.path.dirname(__file__), "../src/covid19sim/configs/test_config.yml"))
    unittest.main()
