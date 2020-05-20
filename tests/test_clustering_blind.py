import numpy as np
import unittest

import covid19sim.frozen.clustering.blind as clu
import covid19sim.frozen.message_utils as mu
from tests.utils import FakeHuman, generate_received_messages, generate_random_messages, Visit

never = 9999  # dirty macro to indicate a human will never get infected


class BlindClusteringTests(unittest.TestCase):
    # note: we only ever build & test clusters for a single human, assuming it would also work for others

    def test_same_day_visit_clusters(self):
        n_trials = 100
        for _ in range(n_trials):
            # first scenario: single day visits, same risk for both visitors, should give 1 cluster
            visits = [
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=2),
                Visit(visitor_real_uid=2, visited_real_uid=0, exposition=False, timestamp=2),
            ]
            humans = [FakeHuman(real_uid=idx, exposition_timestamp=never, visits_to_adopt=visits)
                      for idx in range(3)]
            messages = generate_received_messages(humans)
            h0_messages = [msg for msgs in messages[0]["received_messages"].values() for msg in msgs]
            cluster_manager = clu.BlindClusterManager(max_history_ticks_offset=never)
            cluster_manager.add_messages(h0_messages)
            self.assertEqual(len(cluster_manager.clusters), 1)
            self.assertEqual(cluster_manager.latest_refresh_timestamp, 2)
            embeddings = cluster_manager.get_embeddings_array()
            self.assertEqual(len(embeddings), 1)
            self.assertTrue((embeddings[:, 1] == 0).all())  # risk level
            self.assertTrue((embeddings[:, 2] == 2).all())  # message count
            self.assertTrue((embeddings[:, 3] == 0).all())  # timestamp offset
            # 2nd scenario: single day visits, diff risk for both visitors, should give 2 cluster
            humans = [FakeHuman(real_uid=idx, exposition_timestamp=never,
                                visits_to_adopt=visits, force_init_risk=mu.RiskLevelType(idx))
                      for idx in range(3)]
            messages = generate_received_messages(humans, minimum_risk_level_for_updates=1)
            h0_messages = [msg for msgs in messages[0]["received_messages"].values() for msg in msgs]
            cluster_manager = clu.BlindClusterManager(max_history_ticks_offset=never)
            cluster_manager.add_messages(h0_messages)
            self.assertEqual(len(cluster_manager.clusters), 2)
            self.assertEqual(cluster_manager.latest_refresh_timestamp, 2)
            expositions = cluster_manager._get_expositions_array()
            self.assertTrue(len(expositions) == 2 and sum(expositions) == 0)
            embeddings = cluster_manager.get_embeddings_array()
            self.assertEqual(len(embeddings), 2)
            self.assertTrue((embeddings[:, 1] > 0).all())  # risk level
            self.assertTrue((embeddings[:, 2] == 1).all())  # message count
            self.assertTrue((embeddings[:, 3] == 0).all())  # timestamp offset

    def test_cluster_risk_update(self):
        # scenario: single encounter that gets updated (should remain at 1 cluster)
        visits = [
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=0),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=never, visits_to_adopt=visits,
                      force_init_risk=np.uint8(0)),
            FakeHuman(real_uid=1, exposition_timestamp=never, visits_to_adopt=visits,
                      force_init_risk=np.uint8(7)),
        ]
        messages = generate_received_messages(humans)
        h0_messages = [msg for msgs in messages[0]["received_messages"].values() for msg in msgs]
        cluster_manager = clu.BlindClusterManager(max_history_ticks_offset=never)
        cluster_manager.add_messages(h0_messages)
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(7))
        cluster_manager.add_messages([
            mu.create_update_message(h0_messages[0], np.uint8(9), np.uint64(1))
        ])
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(9))
        # add a new encounter: it should not match the existing cluster due to diff risk
        cluster_manager.add_messages([
            mu.EncounterMessage(mu.create_new_uid(), risk_level=np.uint8(1), encounter_time=0)
        ])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[1].risk_level, np.uint8(1))
        # add a new encounter: it should match the existing cluster due to same risk
        new_encounter = \
            mu.EncounterMessage(mu.create_new_uid(), risk_level=np.uint8(9), encounter_time=0)
        cluster_manager.add_messages([new_encounter])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(9))
        self.assertEqual(len(cluster_manager.clusters[0].messages), 2)
        # update one of the two encounters in the first cluster; it should get split
        cluster_manager.add_messages([
            mu.create_update_message(new_encounter, np.uint8(13), np.uint64(1))
        ])
        self.assertEqual(len(cluster_manager.clusters), 3)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(9))
        self.assertEqual(cluster_manager.clusters[1].risk_level, np.uint8(13))
        self.assertEqual(cluster_manager.clusters[2].risk_level, np.uint8(1))
        self.assertTrue(all([len(c.messages) == 1 for c in cluster_manager.clusters]))

    def test_cleanup_outdated_cluster(self):
        # scenario: a new encounter is added that is waaay outdated; it should not create a cluster
        visits = [
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=2),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=5),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=8),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=never, visits_to_adopt=visits),
            FakeHuman(real_uid=1, exposition_timestamp=never, visits_to_adopt=visits),
        ]
        messages = generate_received_messages(humans)
        h0_messages = [msg for msgs in messages[0]["received_messages"].values() for msg in msgs]
        cluster_manager = clu.BlindClusterManager(max_history_ticks_offset=5)
        cluster_manager.add_messages(h0_messages)
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[0].first_update_time, np.uint8(5))
        self.assertEqual(cluster_manager.clusters[1].first_update_time, np.uint8(8))
        # new manually added encounters that are outdated should also be ignored
        cluster_manager.add_messages([
            mu.EncounterMessage(mu.create_new_uid(), risk_level=np.uint8(1), encounter_time=0)
        ])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[0].first_update_time, np.uint8(5))
        self.assertEqual(cluster_manager.clusters[1].first_update_time, np.uint8(8))

    def test_random_large_scale(self):
        n_trials = 25
        n_humans = 50
        n_visits = 2000
        n_expositions = 15
        max_timestamp = 10
        for _ in range(n_trials):
            h0_messages, visits = generate_random_messages(
                n_humans=n_humans,
                n_visits=n_visits,
                n_expositions=n_expositions,
                max_timestamp=max_timestamp,
            )
            cluster_manager = clu.BlindClusterManager(max_history_ticks_offset=never)
            cluster_manager.add_messages(h0_messages)
            self.assertLessEqual(
                len(cluster_manager.clusters),
                (mu.message_uid_mask + 1) * (mu.risk_level_mask + 1) * (max_timestamp + 1)
            )
            homogeneity_scores = cluster_manager._get_homogeneity_scores()
            for id in homogeneity_scores:
                self.assertLessEqual(homogeneity_scores[id], 1.0)
                min_homogeneity = 1 / sum([v.visited_real_uid == 0 for v in visits])
                self.assertLessEqual(min_homogeneity, homogeneity_scores[id])


if __name__ == "__main__":
    unittest.main()
