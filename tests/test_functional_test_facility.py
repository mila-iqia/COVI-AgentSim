import datetime
import hashlib
import os
import pickle
import unittest
import warnings
import zipfile
from collections import namedtuple
from tempfile import TemporaryDirectory

from tests.utils import get_test_conf

from covid19sim.base import Event
from covid19sim.run import simulate

TEST_CONF_NAME = "intervention_performance.yaml"


InterventionProps = namedtuple('InterventionProps',
                               ['name', 'risk_model', 'tracing_order'])


def _run_simulation(test_case, intervention_properties, lab_capacity):
    intervention, risk_model, tracing_order = intervention_properties
    test_case.config['INTERVENTION'] = intervention
    test_case.config['RISK_MODEL'] = risk_model
    test_case.config['TRACING_ORDER'] = tracing_order
    test_case.config['TEST_TYPES']['lab']['capacity'] = lab_capacity

    if intervention == '':
        test_case.config['INTERVENTION_DAY'] = -1
    else:
        test_case.config['INTERVENTION_DAY'] = test_case.intervention_day

    data = []

    with TemporaryDirectory() as d:
        outfile = os.path.join(d, "data")
        monitors, _ = simulate(
            n_people=test_case.n_people,
            start_time=test_case.start_time,
            simulation_days=test_case.simulation_days,
            outfile=outfile,
            out_chunk_size=0,
            init_percent_sick=test_case.init_percent_sick,
            seed=test_case.test_seed,
            conf=test_case.config
        )
        monitors[0].dump()
        monitors[0].join_iothread()

        with zipfile.ZipFile(f"{outfile}.zip", 'r') as zf:
            for pkl in zf.namelist():
                pkl_bytes = zf.read(pkl)
                data.extend(pickle.loads(pkl_bytes))

    test_case.assertGreater(len(data), 0)

    test_case.assertIn(Event.encounter, {d['event_type'] for d in data})
    test_case.assertIn(Event.test, {d['event_type'] for d in data})

    test_case.assertGreaterEqual(len({d['human_id'] for d in data}), test_case.n_people)

    return data


def _split_events_on_time_and_count_by_type(events, time, event_type):
    events.sort(key=lambda e: e['time'])

    cnt_1 = 0
    cnt_2 = 0
    for event in events:
        if event['event_type'] == event_type:
            if event['time'] >= time:
                cnt_2 += 1
            else:
                cnt_1 += 1

    return cnt_1, cnt_2


class TestFacilityImpact(unittest.TestCase):
    def setUp(self):
        self.config = get_test_conf(TEST_CONF_NAME)

        self.test_seed = 0
        self.n_people = 30
        self.init_percent_sick = 0.1
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 20
        self.intervention_day = 2
        self.lab_capacities = [0.05, 0.1, 0.15]

        self.config['INTERVENTION_DAY'] = self.intervention_day
        self.config['TRANSFORMER_EXP_PATH'] = "https://drive.google.com/file/d/1Z7g3gKh2kWFSmK2Yr19MQq0blOWS5st0"

        self.non_ai_interventions = (
            InterventionProps('', '', None),  # No intervention
            InterventionProps('SocialDistancing', '', None),
            InterventionProps('Tracing', 'digital', 1),
            InterventionProps('Tracing', 'digital', 2)
        )

        self.ai_intervention = InterventionProps('Tracing', 'transformer', None)

    def test_lab_capacity_performance(self):
        """
        Test the impact of lab capacity on a few days of simulation
        """

        intervention_time = self.start_time + datetime.timedelta(days=self.intervention_day)
        contaminations_cache = {}

        # Run test non-ai interventions first as transformer takes significantly more
        # time to execute
        for i, intervention in enumerate(self.non_ai_interventions):
            contaminations_cache[intervention] = []
            for capacity in self.lab_capacities:
                data = _run_simulation(self, intervention, capacity)

                before_intervention_contaminations, after_intervention_contaminations = \
                    _split_events_on_time_and_count_by_type(data, intervention_time, Event.contamination)

                contaminations_cache[intervention].append(after_intervention_contaminations)

            contaminations = contaminations_cache[intervention]
            with self.subTest(intervention=intervention, capacities=self.lab_capacities,
                              contaminations=contaminations):
                if intervention == InterventionProps('', '', None):
                    self.assertEqual(contaminations[0], contaminations[1])
                    self.assertEqual(contaminations[0], contaminations[2])
                    self.assertEqual(contaminations[0], contaminations[3])
                else:
                    self.assertGreater(contaminations[0], contaminations[1])
                    self.assertGreater(contaminations[1], contaminations[2])
                    self.assertGreater(contaminations[2], contaminations[3])

        # Test ai intervention
        contaminations_cache[self.ai_intervention] = []
        for capacity in self.lab_capacities:
            data = _run_simulation(self, self.ai_intervention, capacity)

            before_intervention_contaminations, contaminations = \
                _split_events_on_time_and_count_by_type(data, intervention_time, Event.contamination)

            self.assertEqual(len(before_intervention_contaminations), 0)

            contaminations_cache[self.ai_intervention].append(contaminations)

        # Bigger lab capacities should improve performance of ('Tracing', 'transformer', None)
        contaminations = contaminations_cache[self.ai_intervention]
        with self.subTest(intervention=self.ai_intervention, capacities=self.lab_capacities,
                          contaminations=contaminations):
            self.assertGreater(contaminations[0], contaminations[1])
            self.assertGreater(contaminations[1], contaminations[2])
            self.assertGreater(contaminations[2], contaminations[3])
