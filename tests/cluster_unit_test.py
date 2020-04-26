import unittest
from frozen.utils import Message, encode_message
from frozen.clusters import Clusters

class ScoreMessagesUnitTest(unittest.TestCase):

    def test_score_bad_match_same_day_run(self):
        """
        Tests messages with mutually exclusive uids on the same day are scored lowly
        """
        # uid, risk, day, time_received, true sender id
        current_day = 0
        message1 = Message(0, 0, current_day, "human:0")
        message2 = Message(1, 0, current_day, "human:1")
        clusters = Clusters()
        clusters.add_messages([encode_message(message1)], current_day)
        best_cluster, best_message, best_score = clusters.score_matches(message2, current_day)
        self.assertEqual(best_score, -1)
        self.assertEqual(message1, best_message)

    def test_score_good_match_same_day_run(self):
        """
        Tests messages with the same uids on the same day are scored highly
        """
        # uid, risk, day, true sender id
        current_day = 0
        message1 = Message(0, 0, current_day, "human:1")
        message2 = Message(0, 0, current_day, "human:1")
        clusters = Clusters()
        clusters.add_messages([encode_message(message1)], current_day)
        best_cluster, best_message, best_score = clusters.score_matches(message2, current_day)
        self.assertEqual(best_cluster, 0)
        self.assertEqual(best_message, message1)
        self.assertEqual(best_score, 3)

    def test_score_good_match_one_day_run(self):
        """
        Tests messages with similar uids on the different day are scored mediumly
        """
        # uid, risk, day, true sender id
        current_day = 0
        clusters = Clusters()
        message1 = Message(0, 0, 0, "human:1")
        clusters.add_messages([encode_message(message1)], current_day)
        message2 = Message(1, 0, 1, "human:1")

        best_cluster, best_message, best_score = clusters.score_matches(message2, 1)
        self.assertEqual(best_cluster, 0)
        self.assertEqual(best_message, message1)
        self.assertEqual(best_score, 2)

    def test_score_bad_match_one_day_run(self):
        """
        Tests messages with mutually exclusive uids seperated by a day are scored lowly
        """
        # uid, risk, day, true sender id
        message1 = Message(0, 0, 0, "human:1")
        message2 = Message(6, 0, 1, "human:1")
        clusters = Clusters()
        clusters.add_messages([encode_message(message1)], 0)
        best_cluster, best_message, best_score = clusters.score_matches(message2, 1)
        self.assertEqual(best_cluster, 0)
        self.assertEqual(best_message, message1)
        self.assertEqual(best_score, -1)


class RiskModelIntegrationTest(unittest.TestCase):

    def test_add_message_to_cluster_new_cluster_run(self):
        """
        Tests messages with mutually exclusive uids on the same day are scored lowly
        """
        # make new old message clusters
        message = Message(0, 0, 0, "human:1")
        clusters = Clusters()
        clusters.add_messages([encode_message(message)], 0)

        # make new message
        new_message = Message(1, 0, 0, "human:1")
        # add message to clusters

        clusters.add_messages([encode_message(new_message)], 0)
        num_clusters = len(clusters)
        self.assertEqual(num_clusters, 2)

    def test_add_message_to_cluster_same_cluster_run(self):
        """
        Tests that the add_message_to_cluster function adds messages with the same uid on the same day to the same cluster.
        """
        # make new old message clusters
        message = Message(0, 0, 0, "human:1")
        clusters = Clusters()
        clusters.add_messages([encode_message(message)], 0)

        # make new message
        new_message = Message(0, 0, 0, "human:1")
        # add message to clusters
        clusters.add_messages([encode_message(new_message)], 0)
        self.assertEqual(len(clusters), 1)


    # TODO: fix this once we have a better understanding of the intended behaviour for update_records
    # def test_cluster_two_identical_messages_two_updates_run(self):
    #     """
    #     If you get two messages on the same day with the same id and have clustered them together,
    #     then get two update message for that day with that user id, you now have evidence that
    #     those two messages should be clustered together.
    #     """
    #     received_at = datetime.datetime(2020, 2, 29, 0, 0, 0)
    #     from models.dummy_human import DummyHuman
    #     from models.utils import encode_update_message
    #     human = DummyHuman()
    #     message1 = Message(0, 0, 0, "human:0")
    #     message2 = Message(0, 0, 0, "human:0")
    #     clusters = Clusters()
    #
    #     clusters.add_messages([encode_message(message1)], 0)
    #     clusters.add_messages([encode_message(message2)], 0)
    #     self.assertEqual(len(clusters), 1)
    #
    #     update_messages = [encode_update_message(UpdateMessage(0, 15, 0, 15, received_at, "human:1")),
    #                        encode_update_message(UpdateMessage(0, 15, 0, 15, received_at, "human:1"))]
    #     import pdb; pdb.set_trace()
    #     clusters.update_records(update_messages, human)
    #     self.assertEqual(len(clusters), 1)

    def test_purge(self):
        """ Tests the purge functionality"""
        message1 = Message(0, 0, 0, "human:0")
        message2 = Message(15, 0, 1, "human:0")
        clusters = Clusters()
        clusters.add_messages([encode_message(message1)], 0)
        clusters.add_messages([encode_message(message2)], 0)

        clusters.purge(13)
        self.assertEqual(len(clusters), 2)
        clusters.purge(14)
        self.assertEqual(len(clusters), 1)
        clusters.purge(15)
        self.assertEqual(len(clusters), 0)
