import datetime
import hashlib
import os
import pickle
import unittest
import zipfile
from tempfile import TemporaryDirectory

from tests.utils import get_test_conf

from covid19sim.base import Event
from covid19sim.run import simulate

TEST_CONF_NAME = "test_models.yaml"


class ClusteringCodePaths(unittest.TestCase):
    def setUp(self):
        self.config = get_test_conf(TEST_CONF_NAME)

        self.test_seed = 0
        self.n_people = 30
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 10

        self.config['INTERVENTION_DAY'] = 5
        self.config['TRANSFORMER_EXP_PATH'] = "https://drive.google.com/file/d/1Z7g3gKh2kWFSmK2Yr19MQq0blOWS5st0"
        self.config['RISK_MODEL'] = 'transformer'

        self.cluster_algo_types = ('old', 'blind', 'naive', 'perfect', 'simple')

    def test_clustering_code_paths(self):
        """
        Run a simulation for all risk_models and compare results before and
        after intervention
        """

        events_logs = []

        for cluster_algo_type in self.cluster_algo_types:
            with self.subTest(cluster_algo_type=cluster_algo_type):
                self.config['CLUSTER_ALGO_TYPE'] = cluster_algo_type

                data = []

                with TemporaryDirectory() as d:
                    outfile = os.path.join(d, "data")
                    monitors, _ = simulate(
                        n_people=self.n_people,
                        start_time=self.start_time,
                        simulation_days=self.simulation_days,
                        outfile=outfile,
                        out_chunk_size=0,
                        init_percent_sick=0.25,
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
        for a_i, cluster_algo_type in enumerate(self.cluster_algo_types):
            data = events_logs[a_i]
            data.sort(key=lambda e: e['time'])

            model_before_intervention_events = []
            model_after_intervention_events = []
            for e_i, event in enumerate(data):
                if event['time'] >= intervention_time:
                    model_before_intervention_events = data[0:e_i]
                    model_after_intervention_events = data[e_i:]
                    break

            before_intervention_events.append(model_before_intervention_events)
            after_intervention_events.append(model_after_intervention_events)

            if a_i == 0:
                continue

            with self.subTest(cluster_algo_type=cluster_algo_type):
                for oa_i in range(a_i):
                    self.assertEqual(hashlib.md5(pickle.dumps(model_before_intervention_events)).hexdigest(),
                                     hashlib.md5(pickle.dumps(before_intervention_events[oa_i])).hexdigest(),
                                     msg=f"Before intervention day {self.config['INTERVENTION_DAY']}, "
                                     f"simulation with cluster_algo_type {cluster_algo_type} yielded different results "
                                     f"as simulaion with cluster_algo_type {self.cluster_algo_types[oa_i]}.")
                    self.assertNotEqual(hashlib.md5(pickle.dumps(model_after_intervention_events)).hexdigest(),
                                        hashlib.md5(pickle.dumps(after_intervention_events[oa_i])).hexdigest(),
                                        msg=f"At and after intervention day {self.config['INTERVENTION_DAY']}, "
                                        f"simulation with cluster_algo_type {cluster_algo_type} yielded the same results "
                                        f"as simulaion with cluster_algo_type {self.cluster_algo_types[oa_i]}.")
