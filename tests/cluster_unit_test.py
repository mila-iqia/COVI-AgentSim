import unittest
import datetime
from bitarray import bitarray
from collections import defaultdict

from models.risk_models import RiskModelTristan
from models.utils import Message, UpdateMessage, encode_message
from models.clusters import Clusters

class ScoreMessagesUnitTest(unittest.TestCase):

    def test_score_bad_match_same_day_run(self):
        """
        Tests messages with mutually exclusive uids on the same day are scored lowly
        """
        # uid, risk, day, time_received, true sender id
        message1 = Message(bitarray("0000"), 0, 0, "human:0")
        message2 = Message(bitarray("0001"), 0, 0, "human:1")
        clusters = Clusters()
        clusters.add_message(message1)
        best_cluster, best_message, best_score = clusters.score_matches(message2, 0)
        self.assertEqual(best_score, -1)
        self.assertEqual(message1, best_message)

    def test_score_good_match_same_day_run(self):
        """
        Tests messages with the same uids on the same day are scored highly
        """
        # uid, risk, day, true sender id
        message1 = Message(bitarray("0000"), 0, 0, "human:1")
        m_enc = encode_message(message1)
        message2 = Message(bitarray("0000"), 0, 0, "human:1")
        clusters = Clusters()
        clusters.add_message(message1)
        best_cluster, best_message, best_score = clusters.score_matches(message2, 0)
        self.assertEqual(best_cluster, 0)
        self.assertEqual(best_message, message1)
        self.assertEqual(best_score, 3)

    def test_score_good_match_one_day_run(self):
        """
        Tests messages with similar uids on the different day are scored lowly
        """
        # uid, risk, day, true sender id
        message1 = Message(bitarray("0000"), 0, 0, "human:1")
        message2 = Message(bitarray("0001"), 0, 1, "human:1")
        clusters = Clusters()
        clusters.add_message(message1)
        best_cluster, best_message, best_score = clusters.score_matches(message2, 0)
        self.assertEqual(best_cluster, 0)
        self.assertEqual(best_message, message1)
        self.assertEqual(best_score, 2)

    def test_score_bad_match_one_day_run(self):
        """
        Tests messages with mutually exclusive uids seperated by a day are scored lowly
        """
        # uid, risk, day, true sender id
        message1 = Message(bitarray("0000"), 0, 0, "human:1")
        message2 = Message(bitarray("0100"), 0, 1, "human:1")
        clusters = Clusters()
        clusters.add_message(message1)
        best_cluster, best_message, best_score = clusters.score_matches(message2, 0)
        self.assertEqual(best_cluster, 0)
        self.assertEqual(best_message, message1)
        self.assertEqual(best_score, -1)


class RiskModelIntegrationTest(unittest.TestCase):

    def test_add_message_to_cluster_new_cluster_run(self):
        """
        Tests messages with mutually exclusive uids on the same day are scored lowly
        """
        # make new old message clusters
        message = Message(bitarray("0000"), 0, 0, "human:1")
        clusters = Clusters()
        clusters.add_message(message)

        # make new message
        new_message = Message(bitarray("0001"), 0, 0, "human:1")
        # add message to clusters

        clusters.add_message(new_message)
        num_clusters = len(clusters)
        self.assertEqual(num_clusters, 2)

    def test_add_message_to_cluster_same_cluster_run(self):
        """
        Tests that the add_message_to_cluster function adds messages with the same uid on the same day to the same cluster.
        """
        # make new old message clusters
        message = Message(bitarray("0000"), 0, 0, "human:1")
        clusters = Clusters()
        clusters.add_message(message)

        # make new message
        new_message = Message(bitarray("0000"), 0, 0, "human:1")
        # add message to clusters
        clusters.add_message(new_message)
        self.assertEqual(len(clusters), 1)

    # def test_cluster_add_two_identical_messages_one_update_run(self):
    #     """
    #     If you get two messages on the same day with the same id and have clustered them together,
    #     but get just one update message the same day with that user id, you now have evidence that
    #     those two messages should not be clustered together. Therefore, we should break up that cluster.
    #     """
    #     received_at = datetime.datetime(2020, 2, 29, 0, 0, 0)
    #
    #     message1 = Message(bitarray("0000"), 0, 0, 1)
    #     clusters = Clusters()
    #     clusters.add_message(message1)
    #
    #     message2 = Message(bitarray("0000"), 0, 0, 1)
    #     clusters.add_message(message2)
    #     self.assertEqual(len(clusters), 1)
    #
    #     update_messages = [UpdateMessage(bitarray("0000"), 0, 15, 0, received_at, 1)]
    #     clusters = clusters.update_records(update_messages)
    #     self.assertEqual(len(clusters), 2)


    def test_cluster_two_identical_messages_two_updates_run(self):
        """
        If you get two messages on the same day with the same id and have clustered them together,
        then get two update message for that day with that user id, you now have evidence that
        those two messages should be clustered together.
        """
        received_at = datetime.datetime(2020, 2, 29, 0, 0, 0)

        message1 = Message(bitarray("0000"), 0, 0, "human:0")
        message2 = Message(bitarray("0000"), 0, 0, "human:0")
        clusters = Clusters()

        clusters.add_message(message1)
        clusters.add_message(message2)
        self.assertEqual(len(clusters), 1)

        update_messages = [UpdateMessage(bitarray("0000"), 15, 0, 15, received_at, "human:1"), UpdateMessage(bitarray("0000"), 15, 0, 15, received_at, "human:1")]
        clusters.update_records(update_messages)
        self.assertEqual(len(clusters), 1)

    # def test_cluster_two_messages_two_updates_diff_days_run(self):
    #     """
    #     """
    #     received_at = datetime.datetime(2020, 2, 29, 0, 0, 0)
    #
    #     message1 = Message(bitarray("0000"), 0, 0, 0)
    #     message2 = Message(bitarray("0001"), 0, 1, 0)
    #     clusters = Clusters()
    #
    #     clusters.add_message(message1)
    #     clusters.add_message(message2)
    #
    #     self.assertEqual(len(clusters), 1)
    #
    #     update_messages = [UpdateMessage(bitarray("0001"), 15, 0, 0, received_at, 0), UpdateMessage(bitarray("0001"), 15, 1, 1, received_at, 0)]
    #     import pdb; pdb.set_trace()
    #
    #     clusters.update_records(update_messages)
    #
    #     self.assertEqual(len(clusters), 1)

    def test_purge(self):
        """ Tests the purge functionality"""
        message1 = Message(bitarray("0000"), 0, 0, "human:0")
        message2 = Message(bitarray("1111"), 0, 1, "human:0")
        clusters = Clusters()
        clusters.add_message(message1)
        clusters.add_message(message2)

        clusters.purge(13)
        self.assertEqual(len(clusters), 2)
        clusters.purge(14)
        self.assertEqual(len(clusters), 1)
        clusters.purge(15)
        self.assertEqual(len(clusters), 0)

    # def test_mutex_cluster_run(self):
    #     """
    #     If you get two messages on the same day with the same id and have clustered them together,
    #     but get just one update message the same day with that user id, you now have evidence that
    #     those two messages should not be clustered together. Therefore, we should break up that cluster.
    #     """
    #     # make new old message clusters
    #     old_message1 = Message(bitarray("0000"), 0, 0, 0)
    #     old_message2 = Message(bitarray("0000"), 0, 0, 1)
    #
    #     clusters = {encode_message(old_message1): 0, encode_message(old_message2): 0}
    #     clusters = defaultdict(list)
    #     clusters[0].append(encode_message(old_message1))
    #     clusters[0].append(encode_message(old_message2))
    #
    #     # make new message
    #     # uid, new_risk, old_risk, day, received_at, unobs_id
    #     update_message = UpdateMessage(bitarray("0000"), 15, 0, recieved_at, 1)
    #
    #     # add message to clusters
    #     clusters = RiskModelTristan.add_message_to_cluster(clusters, new_message)
    #     num_clusters = len(set(clusters.values()))
    #     self.assertEqual(num_clusters, 1)
    #
