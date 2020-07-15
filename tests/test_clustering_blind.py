import numpy as np
import unittest

import covid19sim.inference.clustering.blind as clu
import covid19sim.inference.message_utils as mu

from tests.utils import MessageContextManager, ObservedRisk, \
                        RiskLevelBoundsCheckedType


class BlindClusteringTests(unittest.TestCase):
    """
    testcase for a blind clustering algorithm
    """
    def setUp(self):
        self.max_tick = 20
        self.message_context = MessageContextManager(max_tick=self.max_tick)

    def test_same_day_visit_same_risk_clusters(self):
        """
        first scenario: single day visits, same risk for n visitors,
        should give one cluster
        """
        n_human = 3
        encounter_tick = 2
        encounter_risk_level = 1
        assert encounter_tick <= self.max_tick
        for idx in range(n_human):
            self.message_context.insert_messages(
                ObservedRisk(
                    encounter_tick=encounter_tick,
                    encounter_risk_level=encounter_risk_level
                ),
                tick_to_uid_map={0: idx},
            )
        cluster_manager = clu.BlindClusterManager(
            max_history_offset=self.message_context.max_history_offset,
            generate_backw_compat_embeddings=True,
        )
        cluster_manager.add_messages(self.message_context.contact_messages)
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(
            cluster_manager.latest_refresh_timestamp,
            ObservedRisk.toff(encounter_tick))
        embeddings = cluster_manager.get_embeddings_array()
        self.assertEqual(len(embeddings), 1)
        self.assertTrue(
            (embeddings[:, 1] == encounter_risk_level).all())  # risk level
        self.assertTrue((embeddings[:, 2] == n_human).all())  # message count
        self.assertTrue((embeddings[:, 3] == 0).all())  # timestamp offset

    def test_same_day_visit_diff_risk_clusters(self):
        """
        2nd scenario: single day visits, diff risk for n visitors,
        should give n cluster
        """
        n_human = 3
        encounter_tick = 0
        assert encounter_tick <= self.max_tick
        assert n_human <= mu.risk_level_mask
        for idx in range(n_human):
            self.message_context.insert_messages(
                ObservedRisk(
                    encounter_tick=encounter_tick,
                    encounter_risk_level=idx
                ),
                tick_to_uid_map={0: idx},
            )
        cluster_manager = clu.BlindClusterManager(
            max_history_offset=self.message_context.max_history_offset,
            generate_backw_compat_embeddings=True,
        )
        cluster_manager.add_messages(self.message_context.contact_messages)
        self.assertEqual(len(cluster_manager.clusters), n_human)
        self.assertEqual(
            cluster_manager.latest_refresh_timestamp,
            ObservedRisk.toff(encounter_tick)
        )
        expositions = cluster_manager._get_expositions_array()
        self.assertTrue(len(expositions) == n_human and sum(expositions) == 0)
        embeddings = cluster_manager.get_embeddings_array()
        self.assertEqual(len(embeddings), n_human)
        self.assertEqual(
            set(embeddings[:, 1]), set(range(n_human))
        )  # risk level
        # message count
        self.assertTrue((sum(embeddings[:, 2]) == n_human).all())
        self.assertTrue((embeddings[:, 3] == 0).all())  # timestamp offset

    def test_cluster_single_encounter_functional(self):
        """
        functional test for BlindClusterManager, single encounter
        """
        encounter_tick = 0
        encounter_risk_level = 7
        assert encounter_tick <= self.max_tick
        o = ObservedRisk(
            encounter_tick=encounter_tick,
            encounter_risk_level=encounter_risk_level
        )
        self.message_context.insert_messages(o)
        cluster_manager = clu.BlindClusterManager(
            max_history_offset=self.message_context.max_history_offset
        )
        cluster_manager.add_messages(self.message_context.contact_messages)
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(
            cluster_manager.clusters[0].risk_level,
            encounter_risk_level
        )

    def test_cluster_single_encounter_and_update_functional(self):
        """
        functional test for BlindClusterManager, single encounter and update
        """
        encounter_tick = 0
        encounter_risk_level = 7
        update_tick1 = 1
        update_tick2 = 1
        update_risk_level1 = 9
        update_risk_level2 = 10
        assert encounter_tick <= update_tick1 <= update_tick2 <= self.max_tick
        o = ObservedRisk(
            encounter_tick=encounter_tick,
            encounter_risk_level=encounter_risk_level
        ).update(
            update_tick=update_tick1,
            update_risk_level=update_risk_level1
        ).update(
            update_tick=update_tick2,
            update_risk_level=update_risk_level2
        )
        self.message_context.insert_messages(o)
        cluster_manager = clu.BlindClusterManager(
            max_history_offset=self.message_context.max_history_offset
        )
        cluster_manager.add_messages(self.message_context.contact_messages)
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(
            cluster_manager.clusters[0].risk_level,
            RiskLevelBoundsCheckedType(update_risk_level2)
        )

    def test_cluster_eventual_same_risk_functional(self):
        """
        scenario: single encounter that gets updated (should remain at
        one cluster), add a new encounter: it should not match the
        existing cluster due to diff risk
        """
        encounter_tick_a = encounter_tick_b = 0
        encounter_risk_level_a = update_risk_level_b = 7
        encounter_risk_level_b = 1
        update_tick_b = 1
        assert encounter_tick_a <= self.max_tick
        assert encounter_tick_b <= update_tick_b <= self.max_tick
        o_a = ObservedRisk(
            encounter_tick=encounter_tick_a,
            encounter_risk_level=encounter_risk_level_a
        )
        o_b = ObservedRisk(
            encounter_tick=encounter_tick_b,
            encounter_risk_level=encounter_risk_level_b
        ).update(
            update_tick=update_tick_b,
            update_risk_level=update_risk_level_b
        )
        self.message_context.insert_messages(observed_risks=o_a)
        self.message_context.insert_messages(observed_risks=o_b)
        cluster_manager = clu.BlindClusterManager(
            max_history_offset=self.message_context.max_history_offset
        )
        cluster_manager.add_messages(self.message_context.contact_messages)
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(
            cluster_manager.clusters[0].risk_level,
            encounter_risk_level_a
        )

    def test_cluster_same_risk_repeated_contact_functional(self):
        """
        scenario: a same user with unchanging risk making a couple of contacts
        over days. With the message cluster expiring messages with older ticks
        than max_tick_day_offset, the remaining messages each constitute a
        distinct cluster
        """
        encounter_ticks = [2, 5, 8]
        max_tick_day_offset = 5
        encounter_risk_level = 0
        assert all(t <= self.max_tick for t in encounter_ticks)
        o = [
            ObservedRisk(
                encounter_tick=tick,
                encounter_risk_level=encounter_risk_level
            )
            for tick in encounter_ticks
        ]
        self.message_context.insert_messages(observed_risks=o)
        cluster_manager = clu.BlindClusterManager(
            max_history_offset=ObservedRisk.time_offset * max_tick_day_offset
        )
        cluster_manager.add_messages(self.message_context.contact_messages)
        cutoff_day = max(encounter_ticks) - max_tick_day_offset
        expected_ticks = {ObservedRisk.toff(t) for t in encounter_ticks
                          if t >= cutoff_day}
        self.assertEqual(len(cluster_manager.clusters), len(expected_ticks))
        clustered_first_update_time = \
            {clu.first_update_time for clu in cluster_manager.clusters}
        self.assertEqual(clustered_first_update_time, expected_ticks)

    def test_cluster_cleanup_outdated_functional(self):
        """
        scenario: a new encounter is added that is waaay outdated; it should
        not create a cluster new manually added encounters that are outdated
        should also be ignored
        """
        encounter_tick_a = 0
        encounter_risk_level_a = 7
        update_tick_a = 1
        update_risk_level_a = 9
        encounter_tick_b = 13
        max_tick_day_offset = 3
        encounter_risk_level_b = 15
        assert encounter_tick_a <= update_tick_a < \
            encounter_tick_b - max_tick_day_offset <= self.max_tick
        o_a = ObservedRisk(
            encounter_tick=encounter_tick_a,
            encounter_risk_level=encounter_risk_level_a
        ).update(
            update_tick=update_tick_a,
            update_risk_level=update_risk_level_a
        )
        o_b = ObservedRisk(
            encounter_tick=encounter_tick_b,
            encounter_risk_level=encounter_risk_level_b
        )
        self.message_context.insert_messages(observed_risks=o_a)
        self.message_context.insert_messages(observed_risks=o_b)
        cluster_manager = clu.BlindClusterManager(
            max_history_offset=ObservedRisk.time_offset * max_tick_day_offset
        )
        cluster_manager.add_messages(self.message_context.contact_messages)
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(
            cluster_manager.clusters[0].risk_level, encounter_risk_level_b)

    def test_random_large_scale(self):
        """
        test if random large tests cover all the event space for clusters
        and its corresponding homogeneity
        """
        n_trial = 5
        n_human = 50
        n_encounter = 30
        n_update = 10
        n_exposure = 15
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
                    exposure_tick=np.random.randint(
                        self.message_context.max_tick),
                )
            cluster_manager = clu.BlindClusterManager(
                max_history_offset=self.message_context.max_history_offset
            )
            cluster_manager.add_messages(self.message_context.contact_messages)
            self.assertLessEqual(
                len(cluster_manager.clusters),
                (mu.message_uid_mask + 1) *
                (mu.risk_level_mask + 1) * (self.message_context.max_tick + 1)
            )
            homogeneity_scores = cluster_manager._get_homogeneity_scores()
            for id in homogeneity_scores:
                self.assertLessEqual(homogeneity_scores[id], 1.0)
                min_homogeneity = 1 / len(self.message_context.contact_messages)
                self.assertLessEqual(min_homogeneity, homogeneity_scores[id])


if __name__ == "__main__":
    unittest.main()
