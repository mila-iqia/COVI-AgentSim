import numpy as np
import unittest

import covid19sim.frozen.clustering.perfect as clu
import covid19sim.frozen.message_utils as mu
from tests.utils import FakeHuman, generate_received_messages, Visit

never = 9999  # dirty macro to indicate a human will never get infected


class PerfectClusteringTests(unittest.TestCase):
    # here, we will run random encounter scenarios and verify that cluster are always homogenous

    def test_random_large_scale(self):
        n_trials = 5
        n_humans = 200
        n_visits = 3000
        n_expositions = 50
        max_timestamp = 30
        for _ in range(n_trials):
            # start by sampling a bunch of non-exposition visits...
            visits = []
            for _ in range(n_visits):
                visitor_real_uid = np.random.randint(n_humans)
                visited_real_uid = visitor_real_uid
                while visitor_real_uid == visited_real_uid:
                    visited_real_uid = np.random.randint(n_humans)
                visits.append(Visit(
                    visitor_real_uid=mu.RealUserIDType(visitor_real_uid),
                    visited_real_uid=mu.RealUserIDType(visited_real_uid),
                    exposition=False,
                    timestamp=mu.TimestampType(np.random.randint(max_timestamp + 1)),
                ))
            # ...then, had a handful of exposition visits to increase risk levels
            for _ in range(n_expositions):
                exposer_real_uid = np.random.randint(n_humans)
                exposed_real_uid = exposer_real_uid
                while exposer_real_uid == exposed_real_uid:
                    exposed_real_uid = np.random.randint(n_humans)
                visits.append(Visit(
                    visitor_real_uid=mu.RealUserIDType(exposer_real_uid),
                    visited_real_uid=mu.RealUserIDType(exposed_real_uid),
                    exposition=True,
                    timestamp=mu.TimestampType(np.random.randint(max_timestamp + 1)),
                ))
            # now, generate all humans with the spurious thingy tag so we dont have to set expo flags
            humans = [
                FakeHuman(
                    real_uid=idx,
                    exposition_timestamp=never,
                    visits_to_adopt=visits,
                    allow_spurious_exposition=True,
                ) for idx in range(n_humans)
            ]
            messages = generate_received_messages(humans)  # hopefully this is not too slow
            h0_messages = [msg for msgs in messages[0]["received_messages"].values() for msg in msgs]
            cluster_manager = clu.PerfectClusterManager(max_history_ticks_offset=never)
            cluster_manager.add_messages(h0_messages)
            for cluster in cluster_manager.clusters:
                self.assertTrue(cluster._is_homogenous())


if __name__ == "__main__":
    unittest.main()
