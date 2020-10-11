# TODO: Fix this test. For some reason we hit an error in the data collection worker,
# specifically it hits The worker was at day {current_day}, but got a " f"message from day {day_idx}. Bonk!"
# import datetime
# import hashlib
# import os
# import pickle
# import unittest
# import zipfile
# from tempfile import TemporaryDirectory
#
# from tests.utils import get_test_conf
#
# from covid19sim.inference.server_utils import DataCollectionServer
# from covid19sim.run import simulate
#
# TEST_CONF_NAME = "test_models.yaml"
#
#
# class ClusteringCodePaths(unittest.TestCase):
#     def setUp(self):
#         self.config = get_test_conf(TEST_CONF_NAME)
#         self.test_seed = 0
#         self.n_people = 30
#         self.location_start_time = datetime.datetime(2020, 2, 28, 0, 0)
#         self.simulation_days = 10
#         self.config['COLLECT_LOGS'] = True
#         # self.config['INTERVENTION_DAY'] = 5 # TODO: uncomment this line and make it work
#         self.cluster_algo_types = ('gaen', 'blind')
#
#     def test_clustering_code_paths(self):
#         """
#         Run a simulation for all risk_models and compare results before and
#         after intervention
#         """
#
#         events_logs = []
#         for cluster_algo_type in self.cluster_algo_types:
#             with self.subTest(cluster_algo_type=cluster_algo_type):
#                 self.config['CLUSTER_ALGO_TYPE'] = cluster_algo_type
#
#                 data = []
#                 with TemporaryDirectory() as d:
#                     outfile = os.path.join(d, "data")
#                     os.makedirs(outfile, exist_ok=True)
#                     hdf5file = os.path.join(outfile, "train.hdf5")
#                     collection_server = DataCollectionServer(
#                         data_output_path=hdf5file,
#                         config_backup=self.config,
#                         human_count=self.n_people,
#                         simulation_days=self.simulation_days,
#                     )
#                     collection_server.start()
#                     _ = simulate(
#                         n_people=self.n_people,
#                         start_time=self.location_start_time,
#                         simulation_days=self.simulation_days,
#                         outfile=outfile,
#                         out_chunk_size=0,
#                         init_fraction_sick=0.25,
#                         seed=self.test_seed,
#                         conf=self.config
#                     )
#                     collection_server.stop_gracefully()
#                     collection_server.join()
#                     assert os.path.exists(hdf5file)
#
#                     with zipfile.ZipFile(f"{outfile}.zip", 'r') as zf:
#                         for pkl in zf.namelist():
#                             pkl_bytes = zf.read(pkl)
#                             data.extend(pickle.loads(pkl_bytes))
#
#                 self.assertGreater(len(data), 0)
#
#                 self.assertGreaterEqual(len({d['human_id'] for d in data}), self.n_people)
#
#                 events_logs.append(data)
#
#         intervention_time = self.location_start_time + datetime.timedelta(days=self.config['INTERVENTION_DAY'])
#         before_intervention_events = []
#         after_intervention_events = []
#         for a_i, cluster_algo_type in enumerate(self.cluster_algo_types):
#             data = events_logs[a_i]
#             data.sort(key=lambda e: e['time'])
#
#             model_before_intervention_events = []
#             model_after_intervention_events = []
#             for e_i, event in enumerate(data):
#                 if event['time'] >= intervention_time:
#                     model_before_intervention_events = data[0:e_i]
#                     model_after_intervention_events = data[e_i:]
#                     break
#
#             before_intervention_events.append(model_before_intervention_events)
#             after_intervention_events.append(model_after_intervention_events)
#
#             if a_i == 0:
#                 continue
#
#             with self.subTest(cluster_algo_type=cluster_algo_type):
#                 for oa_i in range(a_i):
#                     self.assertEqual(hashlib.md5(pickle.dumps(model_before_intervention_events)).hexdigest(),
#                                      hashlib.md5(pickle.dumps(before_intervention_events[oa_i])).hexdigest(),
#                                      msg=f"Before intervention day {self.config['INTERVENTION_DAY']}, "
#                                      f"simulation with cluster_algo_type {cluster_algo_type} yielded different results "
#                                      f"as simulaion with cluster_algo_type {self.cluster_algo_types[oa_i]}.")
#                     self.assertNotEqual(hashlib.md5(pickle.dumps(model_after_intervention_events)).hexdigest(),
#                                         hashlib.md5(pickle.dumps(after_intervention_events[oa_i])).hexdigest(),
#                                         msg=f"At and after intervention day {self.config['INTERVENTION_DAY']}, "
#                                         f"simulation with cluster_algo_type {cluster_algo_type} yielded the same results "
#                                         f"as simulaion with cluster_algo_type {self.cluster_algo_types[oa_i]}.")
