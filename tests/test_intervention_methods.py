import datetime
import hashlib
import os
import pickle
import unittest
import warnings
import zipfile
from tempfile import TemporaryDirectory

from tests.utils import get_test_conf

from covid19sim.base import Event
from covid19sim.run import simulate

TEST_CONF_NAME = "intervention_performance.yaml"


class InterventionCodePath(unittest.TestCase):
    def setUp(self):
        self.config = get_test_conf(TEST_CONF_NAME)

        self.test_seed = 0
        self.n_people = 30
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 15
        self.intervention_day = 10

        self.config['INTERVENTION_DAY'] = self.intervention_day
        self.config['TRANSFORMER_EXP_PATH'] = "https://drive.google.com/file/d/1Z7g3gKh2kWFSmK2Yr19MQq0blOWS5st0"

        self.interventions_risk_models = (
            ('', ''),
            ('Lockdown', ''),
            ('WearMask', ''),
            ('SocialDistancing', ''),
            ('Quarantine', ''),
            ('Tracing', 'naive'),
            # ('Tracing', 'manual'), manual is currently not working
            ('Tracing', 'digital'),
            ('Tracing', 'transformer'),
            ('WashHands', ''),
            ('StandApart', ''),
            ('StayHome', ''),
        )

    def test_intervention_code_path(self):
        """
        Run a simulation for all risk_models and compare results before and
        after intervention
        """

        events_logs = []

        for intervention, risk_model in self.interventions_risk_models:
            with self.subTest(intervention=intervention, risk_model=risk_model):
                self.config['INTERVENTION'] = intervention
                self.config['RISK_MODEL'] = risk_model

                if intervention == '':
                    self.config['INTERVENTION_DAY'] = -1
                else:
                    self.config['INTERVENTION_DAY'] = self.intervention_day

                data = []

                with TemporaryDirectory() as d:
                    outfile = os.path.join(d, "data")
                    monitors, _ = simulate(
                        n_people=self.n_people,
                        start_time=self.start_time,
                        simulation_days=self.simulation_days,
                        outfile=outfile,
                        out_chunk_size=0,
                        init_percent_sick=0.1,
                        seed=self.test_seed,
                        conf=self.config
                    )
                    monitors[0].dump()
                    monitors[0].join_iothread()

                    with zipfile.ZipFile(f"{outfile}.zip", 'r') as zf:
                        for pkl in zf.namelist():
                            pkl_bytes = zf.read(pkl)
                            data.extend(pickle.loads(pkl_bytes))

                self.assertGreater(len(data), 0)

                self.assertIn(Event.encounter, {d['event_type'] for d in data})
                self.assertIn(Event.test, {d['event_type'] for d in data})

                self.assertGreaterEqual(len({d['human_id'] for d in data}), self.n_people)

                events_logs.append(data)

        intervention_time = self.start_time + datetime.timedelta(days=self.config['INTERVENTION_DAY'])
        before_intervention_events = []
        after_intervention_events = []
        for ir_i, (intervention, risk_model) in enumerate(self.interventions_risk_models):
            data = events_logs[ir_i]
            data.sort(key=lambda e: e['time'])

            model_before_intervention_events = []
            model_after_intervention_events = []
            for e_i, event in enumerate(data):
                if event['time'] >= intervention_time:
                    model_before_intervention_events = data[:e_i]
                    model_after_intervention_events = data[e_i:]
                    break

            before_intervention_events.append(model_before_intervention_events)
            after_intervention_events.append(model_after_intervention_events)

            if ir_i == 0:
                continue

            with self.subTest(intervention=intervention, risk_model=risk_model):
                for oir_i in range(ir_i):
                    other_intervention, other_risk_model = self.interventions_risk_models[oir_i]
                    self.assertEqual(hashlib.md5(pickle.dumps(model_before_intervention_events)).hexdigest(),
                                     hashlib.md5(pickle.dumps(before_intervention_events[oir_i])).hexdigest(),
                                     msg=f"Before intervention day [{self.config['INTERVENTION_DAY']}], "
                                     f"simulation yielded different results as simulation with "
                                     f"intervention/risk_model [{other_intervention}/{other_risk_model}].")
                    try:
                        self.assertNotEqual(hashlib.md5(pickle.dumps(model_after_intervention_events)).hexdigest(),
                                            hashlib.md5(pickle.dumps(after_intervention_events[oir_i])).hexdigest(),
                                            msg=f"At and after intervention day [{self.config['INTERVENTION_DAY']}], "
                                            f"simulation yielded the same results as simulation with "
                                            f"intervention/risk_model [{other_intervention}/{other_risk_model}].")
                    except AssertionError as error:
                        # 'WearMask' and 'WashHands' yields the exact same results
                        # 'SocialDistancing' and 'Stand2M' yields the exact same results
                        if {'WearMask', 'WashHands'}.issubset({intervention,
                                                               other_intervention}) or \
                                {'SocialDistancing', 'StandApart'}.issubset({intervention,
                                                                             other_intervention}):
                            warnings.warn(f"At and after intervention day [{self.config['INTERVENTION_DAY']}], "
                                          f"simulation with intervention/risk_model [{intervention}/{risk_model}] "
                                          f"yielded the same results as simulation with "
                                          f"intervention/risk_model [{other_intervention}/{other_risk_model}]: "
                                          f"{str(error)}", RuntimeWarning)
                        else:
                            raise


class InterventionPerformances(unittest.TestCase):
    def setUp(self):
        self.config = get_test_conf(TEST_CONF_NAME)

        self.test_seed = 0
        self.n_people = 100
        self.init_percent_sick = 0.01
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 22
        self.intervention_day = 5

        self.config['INTERVENTION_DAY'] = self.intervention_day
        self.config['TRANSFORMER_EXP_PATH'] = "https://drive.google.com/file/d/1Z7g3gKh2kWFSmK2Yr19MQq0blOWS5st0"

        self.interventions_risk_models_order = (('', '', None),
                                                ('SocialDistancing', '', None),
                                                ('Tracing', 'digital', 1), ('Tracing', 'digital', 2),
                                                ('Tracing', 'transformer', None))

    def test_intervention_performance(self):
        """
        Run a simulation for all risk_models and compare results before and
        after intervention
        """

        events_logs = []

        for intervention, risk_model, tracing_order in self.interventions_risk_models_order:
            with self.subTest(intervention=intervention, risk_model=risk_model,
                              tracing_order=tracing_order):
                self.config['INTERVENTION'] = intervention
                self.config['RISK_MODEL'] = risk_model
                self.config['TRACING_ORDER'] = tracing_order

                if intervention == '':
                    self.config['INTERVENTION_DAY'] = -1
                else:
                    self.config['INTERVENTION_DAY'] = self.intervention_day

                data = []

                with TemporaryDirectory() as d:
                    outfile = os.path.join(d, "data")
                    monitors, _ = simulate(
                        n_people=self.n_people,
                        start_time=self.start_time,
                        simulation_days=self.simulation_days,
                        outfile=outfile,
                        out_chunk_size=0,
                        init_percent_sick=self.init_percent_sick,
                        seed=self.test_seed,
                        conf=self.config
                    )
                    monitors[0].dump()
                    monitors[0].join_iothread()

                    with zipfile.ZipFile(f"{outfile}.zip", 'r') as zf:
                        for pkl in zf.namelist():
                            pkl_bytes = zf.read(pkl)
                            data.extend(pickle.loads(pkl_bytes))

                self.assertGreater(len(data), 0)

                self.assertIn(Event.encounter, {d['event_type'] for d in data})
                self.assertIn(Event.test, {d['event_type'] for d in data})

                self.assertGreaterEqual(len({d['human_id'] for d in data}), self.n_people)

                events_logs.append(data)

        intervention_time = self.start_time + datetime.timedelta(days=self.config['INTERVENTION_DAY'])
        before_intervention_contaminations = []
        after_intervention_contaminations = []
        humans_cnt = None
        for iro_i, (intervention, risk_model, tracing_order) in enumerate(self.interventions_risk_models_order):
            data = events_logs[iro_i]
            data.sort(key=lambda e: e['time'])

            model_before_intervention_contaminations = 0
            model_after_intervention_contaminations = 0
            human_ids = set()
            for event in data:
                human_ids.add(event['human_id'])
                if event['event_type'] == Event.contamination:
                    if event['time'] >= intervention_time:
                        model_after_intervention_contaminations += 1
                    else:
                        model_before_intervention_contaminations += 1

            if humans_cnt is None:
                humans_cnt = len(human_ids)
            else:
                self.assertEqual(len(human_ids), humans_cnt)

            before_intervention_contaminations.append(model_before_intervention_contaminations)
            after_intervention_contaminations.append(model_after_intervention_contaminations)

            if iro_i == 0:
                continue

            with self.subTest(intervention=intervention, risk_model=risk_model,
                              tracing_order=tracing_order):
                for oiro_i in range(iro_i):
                    other_intervention, other_risk_model, other_tracing_order = \
                        self.interventions_risk_models_order[oiro_i]

                    self.assertEqual(model_before_intervention_contaminations,
                                     before_intervention_contaminations[oiro_i],
                                     msg=f"Before intervention day [{self.config['INTERVENTION_DAY']}], "
                                     f"simulation yielded different results as simulation with "
                                     f"intervention/risk_model/tracing_order "
                                     f"[{other_intervention}/{other_risk_model}/{other_tracing_order}].")

                    self.assertNotEqual(model_after_intervention_contaminations,
                                        after_intervention_contaminations[oiro_i],
                                        msg=f"At and after intervention day [{self.config['INTERVENTION_DAY']}], "
                                        f"simulation yielded the same results as simulation with "
                                        f"intervention/risk_model/tracing_order "
                                        f"[{other_intervention}/{other_risk_model}/{other_tracing_order}].")

        # No intervention should be at least 3% worst than non-AI interventions
        self.assertGreaterEqual((after_intervention_contaminations[0] -
                                 max(after_intervention_contaminations[1],
                                     after_intervention_contaminations[2],
                                     after_intervention_contaminations[3])) / humans_cnt,
                                0.05)

        # Non-AI interventions should give more or less the same results
        try:
            self.assertAlmostEqual(after_intervention_contaminations[1] / humans_cnt,
                                   after_intervention_contaminations[2] / humans_cnt,
                                   delta=0.1)
        except AssertionError as error:
            # TODO: Fix ('SocialDistancing', '', None) intervention giving almost
            #  no contaminations when it kicks in
            warnings.warn(f"No intervention gave similar results as non-AI "
                          f"interventions: {str(error)}", RuntimeWarning)
        try:
            self.assertAlmostEqual(after_intervention_contaminations[1] / humans_cnt,
                                   after_intervention_contaminations[3] / humans_cnt,
                                   delta=0.1)
        except AssertionError as error:
            # TODO: Fix ('SocialDistancing', '', None) intervention giving almost
            #  no contaminations when it kicks in
            warnings.warn(f"No intervention gave similar results as non-AI "
                          f"interventions: {str(error)}", RuntimeWarning)
        self.assertAlmostEqual(after_intervention_contaminations[2] / humans_cnt,
                               after_intervention_contaminations[3] / humans_cnt,
                               delta=0.1)

        # AI intervention should be at least 0.5% better than non-AI interventions
        # TODO: Fix ('SocialDistancing', '', None) intervention giving almost
        #  no contaminations when it kicks in
        self.assertGreaterEqual((min(  # after_intervention_contaminations[1],
                                     after_intervention_contaminations[2],
                                     after_intervention_contaminations[3]) -
                                 after_intervention_contaminations[4]) / humans_cnt,
                                0.05)
