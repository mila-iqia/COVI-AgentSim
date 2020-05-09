import unittest

import covid19sim.frozen.clustering.perfect as clu
from tests.utils import generate_random_messages

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
            h0_messages, visits = generate_random_messages(
                n_humans=n_humans,
                n_visits=n_visits,
                n_expositions=n_expositions,
                max_timestamp=max_timestamp,
            )
            cluster_manager = clu.PerfectClusterManager(max_history_ticks_offset=never)
            cluster_manager.add_messages(h0_messages)
            for cluster in cluster_manager.clusters:
                self.assertTrue(cluster._is_homogenous())


if __name__ == "__main__":
    unittest.main()
