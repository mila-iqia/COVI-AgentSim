import unittest
from bitarray import bitarray
from models.risk_models import RiskModelTristan
from models.utils import Message, UpdateMessage, encode_message
from models.dummy_human import DummyHuman

class ScoreMessagesUnitTest(unittest.TestCase):

    def test_score_bad_match_same_day_run(self):
        """
        Tests messages with mutually exclusive uids on the same day are scored lowly
        """
        # uid, risk, day, true sender id
        message1 = Message(bitarray("0000"), 0, 0, 0)
        m_enc = encode_message(message1)
        message2 = Message(bitarray("0001"), 0, 0, 1)
        old_messages = [m_enc]
        scores = RiskModelTristan.score_matches(old_messages, message2)
        self.assertEqual(scores[m_enc], -1)

    def test_score_good_match_same_day_run(self):
        """
        Tests messages with the same uids on the same day are scored highly
        """
        # uid, risk, day, true sender id
        message1 = Message(bitarray("0000"), 0, 0, 0)
        m_enc = encode_message(message1)
        message2 = Message(bitarray("0000"), 0, 0, 0)
        old_messages = [m_enc]
        scores = RiskModelTristan.score_matches(old_messages, message2)
        self.assertEqual(scores[m_enc], 3)

    def test_score_good_match_one_day_run(self):
        """
        Tests messages with similar uids on the different day are scored lowly
        """
        # uid, risk, day, true sender id
        message1 = Message(bitarray("0000"), 0, 0, 0)
        m_enc = encode_message(message1)
        message2 = Message(bitarray("0001"), 0, 1, 0)
        old_messages = [m_enc]
        scores = RiskModelTristan.score_matches(old_messages, message2)
        self.assertEqual(scores[m_enc], 2)

    def test_score_bad_match_one_day_run(self):
        """
        Tests messages with mutually exclusive uids seperated by a day are scored lowly
        """
        # uid, risk, day, true sender id
        message1 = Message(bitarray("0000"), 0, 0, 0)
        m_enc = encode_message(message1)
        message2 = Message(bitarray("0100"), 0, 1, 0)
        old_messages = [m_enc]
        scores = RiskModelTristan.score_matches(old_messages, message2)
        self.assertEqual(scores[m_enc], -1)

class RiskModelIntegrationTest(unittest.TestCase):

    def test_add_message_to_cluster_new_cluster_run(self):
        """
        Tests messages with mutually exclusive uids on the same day are scored lowly
        """
        # make new old message clusters
        old_message = Message(bitarray("0000"), 0, 0, 0)
        m_enc = encode_message(old_message)
        human = DummyHuman()
        human.M = {m_enc: 0}
        clusters = human.M

        # make new message
        new_message = Message(bitarray("0001"), 0, 0, 1)
        # add message to clusters
        clusters = RiskModelTristan.add_message_to_cluster(clusters, new_message)
        num_clusters = len(set(clusters.values()))
        self.assertEqual(num_clusters, 2)

    def test_add_message_to_cluster_same_cluster_run(self):
        """
        Tests messages with mutually exclusive uids on the same day are scored lowly
        """
        # make new old message clusters
        old_message = Message(bitarray("0000"), 0, 0, 0)
        m_enc = encode_message(old_message)
        human = DummyHuman()
        human.M = {m_enc: 0}
        clusters = human.M

        # make new message
        new_message = Message(bitarray("0000"), 0, 0, 1)
        # add message to clusters
        clusters = RiskModelTristan.add_message_to_cluster(clusters, new_message)
        num_clusters = len(set(clusters.values()))
        self.assertEqual(num_clusters, 1)

    def test_mutex_cluster_run(self):
        """
        Tests messages with mutually exclusive uids on the same day are scored lowly
        """
        # make new old message clusters
        old_message1 = Message(bitarray("0000"), 0, 0, 0)
        old_message2 = Message(bitarray("0000"), 0, 0, 1)

        clusters = {encode_message(old_message1): 0, encode_message(old_message2): 0}

        # make new message
        update_message = UpdateMessage(bitarray("0000"), 0, 0, 15, 1)
        # add message to clusters
        clusters = RiskModelTristan.add_message_to_cluster(clusters, new_message)
        num_clusters = len(set(clusters.values()))
        self.assertEqual(num_clusters, 1)

