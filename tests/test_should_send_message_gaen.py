import datetime
import numpy as np
import unittest

from covid19sim.city import City


class DummyContactBook(object):
    pass


class DummyHuman(object):
    pass


class DummyCity(object):
    pass


class ShouldSendMessageGaenTests(unittest.TestCase):
    def test_intervention_day(self):
        """
        check returns false if not far enough from intervention day
        """
        cur_day = 10
        daily_update_message_budget_sent_gaen = 0
        current_timestamp = datetime.datetime.now()
        risk_change = 2

        city = DummyCity()
        city.conf = dict(
            BURN_IN_DAYS=2,
            DAYS_BETWEEN_MESSAGES=2,
            INTERVENTION_DAY=10,
            UPDATES_PER_DAY=4,
            MESSAGE_BUDGET_GAEN=1,
            n_people=1000,
        )
        city.rng = np.random.RandomState(0)
        city.risk_change_hist = {0: 12, 1: 1}
        city.risk_change_histogram_sum = sum(city.risk_change_hist.values())
        city.sent_messages_by_day = {cur_day: daily_update_message_budget_sent_gaen}
        human = DummyHuman()
        human.contact_book = DummyContactBook()
        human.contact_book.latest_update_time = current_timestamp - datetime.timedelta(days=cur_day)

        res = City._check_should_send_message_gaen(
            city,
            current_day_idx=cur_day,
            current_timestamp=current_timestamp,
            human=human,
            risk_change_score=risk_change,
        )
        self.assertFalse(res)

        city.conf["INTERVENTION_DAY"] = 9
        res = City._check_should_send_message_gaen(
            city,
            current_day_idx=cur_day,
            current_timestamp=current_timestamp,
            human=human,
            risk_change_score=risk_change,
        )
        self.assertFalse(res)

    def test_last_update(self):
        """
        check returns false if last update is too recent
        """
        cur_day = 10
        daily_update_message_budget_sent_gaen = 0
        current_timestamp = datetime.datetime.now()
        risk_change = 2

        city = DummyCity()
        city.conf = dict(
            BURN_IN_DAYS=2,
            DAYS_BETWEEN_MESSAGES=2,
            INTERVENTION_DAY=5,
            UPDATES_PER_DAY=4,
            MESSAGE_BUDGET_GAEN=1,
            n_people=1000,
        )
        city.rng = np.random.RandomState(0)
        city.risk_change_histogram = {0: 12, 1: 1}
        city.risk_change_histogram_sum = sum(city.risk_change_histogram.values())
        city.sent_messages_by_day = {cur_day: daily_update_message_budget_sent_gaen}
        human = DummyHuman()
        human.contact_book = DummyContactBook()
        human.contact_book.latest_update_time = current_timestamp

        res = City._check_should_send_message_gaen(
            city,
            current_day_idx=cur_day,
            current_timestamp=current_timestamp,
            human=human,
            risk_change_score=risk_change,
        )
        self.assertFalse(res)

    def test_should_send_risk_change_true_det(self):
        """
        check returns True if in last bucket, which is smaller than total message budget
        """
        cur_day = 10
        daily_update_message_budget_sent_gaen = 0
        current_timestamp = datetime.datetime.now()
        risk_change = 1

        city = DummyCity()
        city.conf = dict(
            BURN_IN_DAYS=2,
            DAYS_BETWEEN_MESSAGES=2,
            INTERVENTION_DAY=5,
            UPDATES_PER_DAY=4,
            MESSAGE_BUDGET_GAEN=1,
            n_people=1000,
        )
        city.rng = np.random.RandomState(0)
        city.risk_change_histogram = {0: 1000, 1: 1}
        city.risk_change_histogram_sum = sum(city.risk_change_histogram.values())
        city.sent_messages_by_day = {cur_day: daily_update_message_budget_sent_gaen}
        human = DummyHuman()
        human.contact_book = DummyContactBook()
        human.contact_book.latest_update_time = current_timestamp - datetime.timedelta(days=cur_day)

        res = City._check_should_send_message_gaen(
            city,
            current_day_idx=cur_day,
            current_timestamp=current_timestamp,
            human=human,
            risk_change_score=risk_change,
        )
        self.assertTrue(res)

    def test_last_bucket_prob(self):
        """
        check if you're in the last bucket but it's larger than message budget, total messages = budget for this update (=> /UPDATES_PER_DAY)
        """
        cur_day = 10
        daily_update_message_budget_sent_gaen = 0
        current_timestamp = datetime.datetime.now()
        risk_change = 1  # risk_change HAS to be in risk_change_histogram

        city = DummyCity()
        city.conf = dict(
            BURN_IN_DAYS=2,
            DAYS_BETWEEN_MESSAGES=1,
            INTERVENTION_DAY=5,
            UPDATES_PER_DAY=4,
            MESSAGE_BUDGET_GAEN=1,
            n_people=1000,
        )
        city.rng = np.random.RandomState(0)
        city.risk_change_histogram = {0: 60, 1: 40}
        city.risk_change_histogram_sum = sum(city.risk_change_histogram.values())
        city.sent_messages_by_day = {cur_day: daily_update_message_budget_sent_gaen}
        human = DummyHuman()
        human.contact_book = DummyContactBook()
        human.contact_book.latest_update_time = current_timestamp - datetime.timedelta(days=cur_day)

        results = []
        for i in range(1000):
            res = City._check_should_send_message_gaen(
                city,
                current_day_idx=cur_day,
                current_timestamp=current_timestamp,
                human=human,
                risk_change_score=risk_change,
            )
            results.append(res)
            if res:
                if cur_day not in city.sent_messages_by_day:
                    city.sent_messages_by_day[cur_day] = 0
                city.sent_messages_by_day[cur_day] += 1

        self.assertAlmostEqual(1 / 4, np.mean(results), 2)

    def test_middle_bucket_prob(self):
        """
        checks that if in previous to last bucket and last bucket is smaller than
        budget, then messages sent correspond to the number of remaining messages
        """
        cur_day = 10
        daily_update_message_budget_sent_gaen = 0
        current_timestamp = datetime.datetime.now()
        risk_change = 1  # risk_change HAS to be in risk_change_histogram

        city = DummyCity()
        city.conf = dict(
            BURN_IN_DAYS=2,
            DAYS_BETWEEN_MESSAGES=1,
            INTERVENTION_DAY=5,
            UPDATES_PER_DAY=4,
            MESSAGE_BUDGET_GAEN=1,
            n_people=1000,
        )
        city.rng = np.random.RandomState(0)
        city.risk_change_histogram = {0: 50, 1: 40, 2: 10}
        city.risk_change_histogram_sum = sum(city.risk_change_histogram.values())
        city.sent_messages_by_day = {cur_day: daily_update_message_budget_sent_gaen}
        human = DummyHuman()
        human.contact_book = DummyContactBook()
        human.contact_book.latest_update_time = current_timestamp - datetime.timedelta(days=cur_day)
        results = []
        for i in range(1000):
            res = City._check_should_send_message_gaen(
                city,
                current_day_idx=cur_day,
                current_timestamp=current_timestamp,
                human=human,
                risk_change_score=risk_change,
            )
            results.append(res)
            if res:
                if cur_day not in city.sent_messages_by_day:
                    city.sent_messages_by_day[cur_day] = 0
                city.sent_messages_by_day[cur_day] += 1
        # allowed messages: 100 / 4 = 25
        # already sent messages: 10
        # remaining to send for second bucket: 15
        self.assertAlmostEqual(1 / 4 - 10 / 100, np.mean(results), 2)

    def test_middle_bucket_last_is_full(self):
        """
        If the last bucket is larger than the budget then no message is sent when in the second largest bucket
        """
        cur_day = 10
        daily_update_message_budget_sent_gaen = 0
        current_timestamp = datetime.datetime.now()
        risk_change = 1  # risk_change HAS to be in risk_change_histogram

        city = DummyCity()
        city.conf = dict(
            BURN_IN_DAYS=2,
            DAYS_BETWEEN_MESSAGES=1,
            INTERVENTION_DAY=5,
            UPDATES_PER_DAY=4,
            MESSAGE_BUDGET_GAEN=1,
            n_people=1000,
        )
        city.rng = np.random.RandomState(0)
        city.risk_change_histogram = {0: 40, 1: 20, 2: 40}
        city.risk_change_histogram_sum = sum(city.risk_change_histogram.values())
        city.sent_messages_by_day = {cur_day: daily_update_message_budget_sent_gaen}
        human = DummyHuman()
        human.contact_book = DummyContactBook()
        human.contact_book.latest_update_time = current_timestamp - datetime.timedelta(days=cur_day)

        res = City._check_should_send_message_gaen(
            city,
            current_day_idx=cur_day,
            current_timestamp=current_timestamp,
            human=human,
            risk_change_score=risk_change,
        )
        self.assertFalse(res)

    def test_last_bucket_low_budget(self):
        """
        Everything works still with a very low budget
        """
        cur_day = 10
        daily_update_message_budget_sent_gaen = 0
        current_timestamp = datetime.datetime.now()
        risk_change = 2  # risk_change HAS to be in risk_change_histogram

        city = DummyCity()
        city.conf = dict(
            BURN_IN_DAYS=2,
            DAYS_BETWEEN_MESSAGES=2,
            INTERVENTION_DAY=5,
            UPDATES_PER_DAY=4,
            MESSAGE_BUDGET_GAEN=0.01,
            n_people=1000,
        )
        city.rng = np.random.RandomState(0)
        city.risk_change_histogram = {0: 40, 1: 20, 2: 40}
        city.risk_change_histogram_sum = sum(city.risk_change_histogram.values())
        city.sent_messages_by_day = {cur_day: daily_update_message_budget_sent_gaen}
        human = DummyHuman()
        human.contact_book = DummyContactBook()
        human.contact_book.latest_update_time = current_timestamp - datetime.timedelta(days=cur_day)

        results = []
        for i in range(1000):
            res = City._check_should_send_message_gaen(
                city,
                current_day_idx=cur_day,
                current_timestamp=current_timestamp,
                human=human,
                risk_change_score=risk_change,
            )
            results.append(res)
            if res:
                if cur_day not in city.sent_messages_by_day:
                    city.sent_messages_by_day[cur_day] = 0
                city.sent_messages_by_day[cur_day] += 1

        self.assertAlmostEqual(city.conf["MESSAGE_BUDGET_GAEN"] / 4, np.mean(results), 2)
