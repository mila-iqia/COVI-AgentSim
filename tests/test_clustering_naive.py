import numpy as np
import unittest

import covid19sim.frozen.clustering.blind as blind
import covid19sim.frozen.clustering.naive as naive
import covid19sim.frozen.clustering.perfect as perfect
import covid19sim.frozen.clustering.simple as simple
import covid19sim.frozen.message_utils as mu
import covid19sim.frozen.utils
from tests.utils import FakeHuman, generate_received_messages, Visit, generate_random_messages

never = 9999  # dirty macro to indicate a human will never get infected
ui = np.uint64  # dirty macro to cast integers to np.uint64


class NaiveClusteringTests(unittest.TestCase):
    # note: we only ever build & test clusters for a single human, assuming it would also work for others

    def test_same_day_visit_clusters(self):
        n_trials = 100
        for _ in range(n_trials):
            # scenario: single day visits, 1 cluster per visit, not enough visits to overlap the clusters
            visits = [
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=2),
                Visit(visitor_real_uid=2, visited_real_uid=0, exposition=False, timestamp=2),
            ]
            # we will cheat and hard-code some initial 4-bit uids to make sure there is no overlap
            humans = [
                FakeHuman(
                    real_uid=idx,
                    exposition_timestamp=never,
                    visits_to_adopt=visits,
                    # with only two visitors, there should never be cluster uid overlap at day 2
                    force_init_uid=np.uint8(idx),
                ) for idx in range(3)
            ]
            day2_uids = [h.rolling_uids[2] for h in humans]
            self.assertTrue(len(np.unique(day2_uids)) == 3)
            messages = generate_received_messages(humans)
            # now we need to get all messages sent to human 0 to do our actual clustering analysis
            h0_messages = messages[0]["received_messages"]
            self.assertEqual(len(h0_messages), 3)  # three timesteps in book
            self.assertEqual(sum([len(msgs) for msgs in h0_messages.values()]), 2)
            self.assertEqual(len(h0_messages[2]), 2)
            h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
            cluster_manager = naive.NaiveClusterManager(
                ticks_per_uid_roll=1,
                max_history_ticks_offset=never,
            )
            cluster_manager.add_messages(h0_messages)
            self.assertEqual(len(cluster_manager.clusters), 2)
            self.assertEqual(cluster_manager.latest_refresh_timestamp, 2)
            expositions = cluster_manager._get_expositions_array()
            self.assertEqual(len(expositions), 2)
            self.assertEqual(sum(expositions), 0)
            embeddings = cluster_manager.get_embeddings_array()
            self.assertEqual(len(np.unique(embeddings[:, 0])), 2)  # unique cluster ids
            self.assertTrue((embeddings[:, 1] == 0).all())  # risk level
            self.assertTrue((embeddings[:, 2] == 1).all())  # message count
            self.assertTrue((embeddings[:, 3] == 0).all())  # timestamp offset

    def test_same_day_visit_clusters_overlap(self):
        # scenario: single day visits, and some visits will share the same cluster
        visits = [
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=0),
            Visit(visitor_real_uid=2, visited_real_uid=0, exposition=False, timestamp=0),
            Visit(visitor_real_uid=3, visited_real_uid=0, exposition=False, timestamp=0),
            Visit(visitor_real_uid=4, visited_real_uid=0, exposition=False, timestamp=0),
            Visit(visitor_real_uid=5, visited_real_uid=0, exposition=False, timestamp=0),
        ]
        # we will cheat and hard-code some initial 4-bit uids to make sure there is overlap
        humans = [
            FakeHuman(
                real_uid=idx,
                exposition_timestamp=never,
                visits_to_adopt=visits,
                force_init_uid=np.uint8(max(idx - 1, 0) % 3),
                force_init_risk=np.uint8(7),
            ) for idx in range(6)
        ]
        day0_visitor_uids = [h.rolling_uids[0] for h in humans[1:]]
        self.assertTrue(len(np.unique(day0_visitor_uids)) == 3)  # two visits will be overlapped
        messages = generate_received_messages(humans)
        h0_messages = messages[0]["received_messages"]
        self.assertEqual(len(h0_messages), 1)  # single timestep in book
        self.assertEqual(len(h0_messages[0]), 5)  # all 5 encounter messages in day 0
        h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
        cluster_manager = naive.NaiveClusterManager(
            ticks_per_uid_roll=1,
            max_history_ticks_offset=never,
        )
        cluster_manager.add_messages(h0_messages)
        self.assertEqual(len(cluster_manager.clusters), 3)
        self.assertEqual(cluster_manager.latest_refresh_timestamp, 0)
        expositions = cluster_manager._get_expositions_array()
        self.assertTrue(len(expositions) == 3 and sum(expositions) == 0)
        embeddings = cluster_manager.get_embeddings_array()
        self.assertEqual(len(np.unique(embeddings[:, 0])), 3)  # unique cluster ids
        self.assertTrue((embeddings[:, 1] == 7).all())  # risk level
        self.assertTrue(np.logical_and(embeddings[:, 2] > 0, embeddings[:, 2] < 3).all())
        self.assertTrue((embeddings[:, 3] == 0).all())  # timestamp

    def test_multi_day_visit_clusters_overlap(self):
        # scenario: multi day visits, and some visits will share the same cluster
        # we will cheat and hard-code some uids to make sure there is meaninful overlap
        init_uids = [[np.uint8(max(human_idx - 1, 0) % 2)] for human_idx in range(9)]
        for timestamp_idx in range(1, 3):
            for human_idx in range(9):
                init_uids[human_idx].append(
                    (init_uids[human_idx][-1] << 1) + np.uint8(max(human_idx - 1, 0) >> max(3 - timestamp_idx, 0))
                )
        visits = [  # first sub-scenario: shrinking number of visits (should end w/ 3 clusters)
            *[Visit(visitor_real_uid=id, visited_real_uid=0, exposition=False, timestamp=0)
              for id in range(1, 9)],
            *[Visit(visitor_real_uid=id, visited_real_uid=0, exposition=False, timestamp=1)
              for id in range(1, 5)],
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=2),
            Visit(visitor_real_uid=3, visited_real_uid=0, exposition=False, timestamp=2)
        ]
        humans = [
            FakeHuman(
                real_uid=idx,
                exposition_timestamp=never,
                visits_to_adopt=visits,
                force_init_uid=init_uids[idx] if idx < 5 else init_uids[idx][:2],
                force_init_risk=np.uint8(7),
            ) for idx in range(9)
        ]
        messages = generate_received_messages(humans)
        h0_messages = [msg for msgs in messages[0]["received_messages"].values() for msg in msgs]
        cluster_manager = naive.NaiveClusterManager(
            ticks_per_uid_roll=1,
            max_history_ticks_offset=never,
        )
        cluster_manager.add_messages(h0_messages)
        self.assertEqual(len(cluster_manager.clusters), 3)
        self.assertTrue(
            (cluster_manager.clusters[0].get_encounter_count() == 7) and
            (cluster_manager.clusters[1].get_encounter_count() == 6) and
            (cluster_manager.clusters[2].get_encounter_count() == 1)
        )
        # now try again with the same uid assignments, but more visits --- we should have more clusters
        visits = []
        for timestamp_idx in range(3):
            visits.extend(
                [Visit(visitor_real_uid=id, visited_real_uid=0, exposition=False, timestamp=timestamp_idx)
                 for id in (range(1, 9) if timestamp_idx < 1 else range(1, 5))]
            )
        humans = [
            FakeHuman(
                real_uid=idx,
                exposition_timestamp=never,
                visits_to_adopt=visits,
                force_init_uid=init_uids[idx] if idx < 5 else init_uids[idx][:2],
                force_init_risk=np.uint8(7),
            ) for idx in range(9)
        ]
        messages = generate_received_messages(humans)
        cluster_manager = naive.NaiveClusterManager(
            ticks_per_uid_roll=1,
            max_history_ticks_offset=never,
        )
        cluster_manager.add_messages([msg for msgs in messages[0]["received_messages"].values() for msg in msgs])
        self.assertEqual(len(cluster_manager.clusters), 4)
        self.assertTrue(
            (cluster_manager.clusters[0].get_encounter_count() == 7) and
            (cluster_manager.clusters[1].get_encounter_count() == 7) and
            (cluster_manager.clusters[2].get_encounter_count() == 1) and
            (cluster_manager.clusters[3].get_encounter_count() == 1)
        )

    def test_cluster_risk_update(self):
        # scenario: single day visits, and some visits will share the same cluster
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
        h0_messages = messages[0]["received_messages"]
        self.assertEqual(len(h0_messages), 1)  # single timestep in book
        self.assertEqual(len(h0_messages[0]), 1)  # single encounter message in day 0
        h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
        cluster_manager = naive.NaiveClusterManager(
            ticks_per_uid_roll=1,
            max_history_ticks_offset=never,
        )
        cluster_manager.add_messages(h0_messages)
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(7))
        # we will add a manual update for one of the visits
        cluster_manager.add_messages([
            mu.create_update_message(h0_messages[0], np.uint8(9), np.uint64(1))
        ])
        self.assertEqual(len(cluster_manager.clusters), 1)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(9))
        # add a new encounter: it should not match the existing cluster due to diff risk
        cluster_manager.add_messages([
            mu.EncounterMessage(humans[1].rolling_uids[0], risk_level=np.uint8(1), encounter_time=0)
        ])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[1].risk_level, np.uint8(1))
        # add a new encounter: it should match the existing cluster due to same risk
        new_encounter = \
            mu.EncounterMessage(humans[1].rolling_uids[0], risk_level=np.uint8(9), encounter_time=0)
        cluster_manager.add_messages([new_encounter])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertEqual(cluster_manager.clusters[0].risk_level, np.uint8(9))
        self.assertEqual(cluster_manager.clusters[0].get_encounter_count(), 2)
        # update one of the two encounters in the first cluster; it should be split into two parts
        cluster_manager.add_messages([
            mu.create_update_message(new_encounter, np.uint8(13), np.uint64(1))
        ])
        self.assertEqual(len(cluster_manager.clusters), 3)
        self.assertTrue(all([c.risk_level in [9, 1, 13] for c in cluster_manager.clusters]))
        self.assertTrue(all([c.get_encounter_count() == 1 for c in cluster_manager.clusters]))
        # update the second of the two encounters in the first cluster; we should now be back at two
        cluster_manager.add_messages([
            mu.create_update_message(new_encounter, np.uint8(13), np.uint64(1))
        ])
        self.assertEqual(len(cluster_manager.clusters), 2)
        self.assertTrue(all([c.risk_level in [1, 13] for c in cluster_manager.clusters]))
        self.assertTrue(sum([c.get_encounter_count() for c in cluster_manager.clusters]) == 3)

    def test_cleanup_outdated_cluster(self):
        n_trials = 100
        for _ in range(n_trials):
            # scenario: a new encounter is added that is waaay outdated; it should not create a cluster
            visits = [
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=2),
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=8),
            ]
            humans = [
                FakeHuman(real_uid=0, exposition_timestamp=never, visits_to_adopt=visits),
                FakeHuman(real_uid=1, exposition_timestamp=never, visits_to_adopt=visits),
            ]
            messages = generate_received_messages(humans)
            h0_messages = messages[0]["received_messages"]
            h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
            cluster_manager = naive.NaiveClusterManager(
                ticks_per_uid_roll=1,
                max_history_ticks_offset=5,
            )
            cluster_manager.add_messages(h0_messages)
            self.assertEqual(len(cluster_manager.clusters), 1)
            self.assertEqual(cluster_manager.clusters[0].first_update_time, np.uint8(8))
            # new manually added encounters that are outdated should also be ignored
            cluster_manager.add_messages([
                mu.EncounterMessage(humans[1].rolling_uids[0], risk_level=np.uint8(1), encounter_time=0)
            ])
            self.assertEqual(len(cluster_manager.clusters), 1)
            self.assertEqual(cluster_manager.clusters[0].first_update_time, np.uint8(8))

    def test_cleanup_outdated_but_updated_cluster(self):
        n_trials = 100
        for _ in range(n_trials):
            # scenario: the early encounters are beyond the max history offset from the later encounters,
            # but these should all be clustered and kept due to the partial uid matching strategy
            visits = [
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=2),
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=4),
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=7),
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=8),
                Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=10),
            ]
            humans = [
                FakeHuman(real_uid=0, exposition_timestamp=never, visits_to_adopt=visits),
                FakeHuman(real_uid=1, exposition_timestamp=never, visits_to_adopt=visits),
            ]
            messages = generate_received_messages(humans)
            h0_messages = messages[0]["received_messages"]
            h0_messages = [msg for msgs in h0_messages.values() for msg in msgs]
            cluster_manager = naive.NaiveClusterManager(
                ticks_per_uid_roll=1,
                max_history_ticks_offset=5,
            )
            cluster_manager.add_messages(h0_messages)
            self.assertEqual(len(cluster_manager.clusters), 1)
            self.assertEqual(cluster_manager.clusters[0].first_update_time, np.uint8(2))
            # new manually added encounters that are outdated should be ignored
            cluster_manager.add_messages([
                mu.EncounterMessage(humans[1].rolling_uids[0], risk_level=np.uint8(1), encounter_time=0)
            ])
            self.assertEqual(len(cluster_manager.clusters), 1)
            self.assertEqual(cluster_manager.clusters[0].first_update_time, np.uint8(2))

    def test_random_large_scale(self):
        n_trials = 5
        for iter_idx in range(n_trials):
            h0_messages, visits = generate_random_messages(
                n_humans=100,
                n_visits=5000,
                n_expositions=25,
                max_timestamp=14,
            )
            blind_cluster_manager = blind.BlindClusterManager(max_history_ticks_offset=never)
            blind_cluster_manager.add_messages(h0_messages)
            naive_cluster_manager = naive.NaiveClusterManager(ticks_per_uid_roll=1,
                                                              max_history_ticks_offset=never)
            naive_cluster_manager.add_messages(h0_messages)
            perfect_cluster_manager = perfect.PerfectClusterManager(max_history_ticks_offset=never)
            perfect_cluster_manager.add_messages(h0_messages)
            simple_cluster_manager = simple.SimplisticClusterManager(max_history_ticks_offset=never)
            simple_cluster_manager.add_messages(h0_messages)
            self.assertGreaterEqual(
                len(simple_cluster_manager.clusters),
                len(naive_cluster_manager.clusters),
            )
            blind_homogeneity_scores = blind_cluster_manager._get_homogeneity_scores()
            naive_homogeneity_scores = naive_cluster_manager._get_homogeneity_scores()
            perfect_homogeneity_scores = perfect_cluster_manager._get_homogeneity_scores()
            simple_homogeneity_scores = simple_cluster_manager._get_homogeneity_scores()
            for id in naive_homogeneity_scores:
                self.assertLessEqual(naive_homogeneity_scores[id], 1.0)
                min_homogeneity = 1 / sum([v.visited_real_uid == 0 for v in visits])
                self.assertLessEqual(min_homogeneity, naive_homogeneity_scores[id])
                if id in perfect_homogeneity_scores:
                    self.assertEqual(perfect_homogeneity_scores[id], 1.0)

            print(f"---- iter#{iter_idx + 1} ----")
            print(f"\tsimplistic clustr average homogeneity = {np.mean(list(simple_homogeneity_scores.values()))}")
            print(f"\tsimplistic clustr count error = {simple_cluster_manager._get_cluster_count_error()}")
            print(f"\tsimplistic clustr count = {len(simple_cluster_manager.clusters)}")
            print(f"\tperfect clustr average homogeneity = {np.mean(list(perfect_homogeneity_scores.values()))}")
            print(f"\tperfect clustr count error = {perfect_cluster_manager._get_cluster_count_error()}")
            print(f"\tperfect clustr count = {len(perfect_cluster_manager.clusters)}")
            print(f"\tnaive clustr average homogeneity = {np.mean(list(naive_homogeneity_scores.values()))}")
            print(f"\tnaive clustr count error = {naive_cluster_manager._get_cluster_count_error()}")
            print(f"\tnaive clustr count = {len(naive_cluster_manager.clusters)}")
            print(f"\tblind clustr average homogeneity = {np.mean(list(blind_homogeneity_scores.values()))}")
            print(f"\tblind clustr count error = {blind_cluster_manager._get_cluster_count_error()}")
            print(f"\tblind clustr count = {len(blind_cluster_manager.clusters)}")

    def test_random_large_scale_batch(self):
        n_trials = 25
        np.random.seed(0)
        for _ in range(n_trials):
            h0_messages, visits = generate_random_messages(
                n_humans=10,
                n_visits=1000,
                n_expositions=5,
                max_timestamp=14,
            )
            encounter_messages = [m for m in h0_messages if isinstance(m, mu.EncounterMessage)]
            update_messages = [m for m in h0_messages if isinstance(m, mu.UpdateMessage)]
            batched_encounter_messages = \
                covid19sim.frozen.utils.convert_messages_to_batched_new_format(encounter_messages)
            batched_update_messages = \
                covid19sim.frozen.utils.convert_messages_to_batched_new_format(update_messages)
            batched_naive_cluster_manager = naive.NaiveClusterManager(
                ticks_per_uid_roll=1,
                max_history_ticks_offset=never,
            )
            naive_cluster_manager = naive.NaiveClusterManager(
                ticks_per_uid_roll=1,
                max_history_ticks_offset=never,
            )
            batched_naive_cluster_manager.add_messages(batched_encounter_messages)
            naive_cluster_manager.add_messages(encounter_messages)
            self.assertEqual(
                batched_naive_cluster_manager.clusters,
                naive_cluster_manager.clusters,
            )
            batched_naive_cluster_manager.add_messages(batched_update_messages)
            naive_cluster_manager.add_messages(update_messages)
            self.assertEqual(
                len(batched_naive_cluster_manager.clusters),
                len(naive_cluster_manager.clusters),
            )
            for c_idx, (c1, c2) in \
                    enumerate(zip(batched_naive_cluster_manager.clusters,
                                  naive_cluster_manager.clusters)):
                # note: the only thing we can't match is the cluster id
                self.assertEqual(c1.risk_level, c2.risk_level)
                self.assertEqual(c1.first_update_time, c2.first_update_time)
                self.assertEqual(c1.latest_update_time, c2.latest_update_time)
                self.assertEqual(c1.messages_by_timestamp.keys(),
                                 c2.messages_by_timestamp.keys())
                for t in c1.messages_by_timestamp:
                    c1_msgs = c1.messages_by_timestamp[t]
                    c2_msgs = c1.messages_by_timestamp[t]
                    self.assertEqual(len(c1_msgs), len(c2_msgs))
                    for m1, m2 in zip(c1_msgs, c2_msgs):
                        self.assertEqual(m1, m2)
            simple_cluster_manager = simple.SimplisticClusterManager(max_history_ticks_offset=never)
            simple_cluster_manager.add_messages(h0_messages)
            self.assertGreaterEqual(
                len(simple_cluster_manager.clusters),
                len(batched_naive_cluster_manager.clusters),
            )


if __name__ == "__main__":
    unittest.main()
