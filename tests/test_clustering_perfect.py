import unittest
import numpy as np
import covid19sim.inference.clustering.perfect as clu
from tests.utils import MessageContext


class PerfectClusteringTests(unittest.TestCase):
    # here, we will run random encounter scenarios and verify that cluster
    # are always homogenous

    def setUp(self):
        self.max_tick = 30
        self.message_context = MessageContext(max_tick=self.max_tick)

    def test_linear_saturation_history(self):
        n_trial = 5
        n_human = 200
        n_encounter = 40
        n_exposure = 50
        assert n_exposure <= n_human

        for _ in range(n_trial):
            for _ in range(n_exposure):
                self.message_context.insert_linear_saturation_risk_messages(
                    n_encounter=n_encounter,
                    init_risk_level=0,
                    final_risk_level=15,
                    exposure_tick=np.random.randint(
                        self.message_context.max_tick),
                )
            for _ in range(n_human - n_exposure):
                self.message_context.insert_linear_saturation_risk_messages(
                    n_encounter=n_encounter,
                    init_risk_level=0,
                    final_risk_level=15
                )
            cluster_manager = clu.PerfectClusterManager(
                max_history_offset=self.message_context.max_history_offset)
            cluster_manager.add_messages(self.message_context.contact_messages)
            for cluster in cluster_manager.clusters:
                self.assertTrue(cluster._is_homogenous())

    def test_random_history(self):
        n_trial = 5
        n_human = 200
        n_encounter = 30
        n_update = 10
        n_exposure = 50
        assert n_exposure <= n_human

        for _ in range(n_trial):
            for _ in range(n_exposure):
                self.message_context.insert_random_messages(
                    n_encounter=n_encounter,
                    n_update=n_update,
                    exposure_tick=np.random.randint(
                        self.message_context.max_tick),
                )
            for _ in range(n_human-n_exposure):
                self.message_context.insert_random_messages(
                    n_encounter=n_encounter,
                    n_update=n_update,
                )
            cluster_manager = clu.PerfectClusterManager(
                max_history_offset=self.message_context.max_history_offset)
            cluster_manager.add_messages(self.message_context.contact_messages)
            for cluster in cluster_manager.clusters:
                self.assertTrue(cluster._is_homogenous())


if __name__ == "__main__":
    unittest.main()
