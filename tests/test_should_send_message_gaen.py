import numpy as np
import unittest
from tests.utils import FakeHuman
from covid19sim.utils import should_send_message_gaen
from pytest import approx


class ShouldSendMessageGaenTests(unittest.TestCase):
    def test_intervention_day(self):
        """
        check returns false if not far enough from intervention day
        """
        cur_day = 10
        BURN_IN_DAYS = 2
        DAYS_BETWEEN_MESSAGES = 2
        INTERVENTION_DAY = 10
        last_sent_update_gaen = 0
        rng = np.random.RandomState(0)
        daily_update_message_budget_sent_gaen = 0
        message_budget = 1
        UPDATES_PER_DAY = 4
        n_people = 1000

        risk_change_hist = {0: 12, 1: 1}
        risk_change_hist_sum = sum(risk_change_hist.values())
        risk_change = 2
        res = should_send_message_gaen(
            risk_change,
            cur_day,
            last_sent_update_gaen,
            risk_change_hist,
            risk_change_hist_sum,
            rng,
            daily_update_message_budget_sent_gaen,
            message_budget,
            INTERVENTION_DAY,
            UPDATES_PER_DAY,
            n_people,
            BURN_IN_DAYS,
            DAYS_BETWEEN_MESSAGES,
        )
        self.assertFalse(res)

        INTERVENTION_DAY = 9
        res = should_send_message_gaen(
            risk_change,
            cur_day,
            last_sent_update_gaen,
            risk_change_hist,
            risk_change_hist_sum,
            rng,
            daily_update_message_budget_sent_gaen,
            message_budget,
            INTERVENTION_DAY,
            UPDATES_PER_DAY,
            n_people,
            BURN_IN_DAYS,
            DAYS_BETWEEN_MESSAGES,
        )
        self.assertFalse(res)

    def test_last_update(self):
        """
        check returns false if last update is too recent
        """
        cur_day = 10
        BURN_IN_DAYS = 2
        DAYS_BETWEEN_MESSAGES = 2
        INTERVENTION_DAY = 5
        last_sent_update_gaen = 10
        rng = np.random.RandomState(0)
        daily_update_message_budget_sent_gaen = 0
        message_budget = 1
        UPDATES_PER_DAY = 4
        n_people = 1000

        risk_change_hist = {0: 12, 1: 1}
        risk_change_hist_sum = sum(risk_change_hist.values())
        risk_change = 2
        res = should_send_message_gaen(
            risk_change,
            cur_day,
            last_sent_update_gaen,
            risk_change_hist,
            risk_change_hist_sum,
            rng,
            daily_update_message_budget_sent_gaen,
            message_budget,
            INTERVENTION_DAY,
            UPDATES_PER_DAY,
            n_people,
            BURN_IN_DAYS,
            DAYS_BETWEEN_MESSAGES,
        )
        self.assertFalse(res)

    def test_should_send_risk_change_true_det(self):
        """
        check returns True if in last bucket, which is smaller than total message budget
        """
        cur_day = 10
        BURN_IN_DAYS = 2
        DAYS_BETWEEN_MESSAGES = 2
        INTERVENTION_DAY = 5
        last_sent_update_gaen = 0
        rng = np.random.RandomState(0)
        daily_update_message_budget_sent_gaen = 0
        message_budget = 1
        UPDATES_PER_DAY = 4
        n_people = 1000

        risk_change_hist = {0: 1000, 1: 1}
        risk_change_hist_sum = sum(risk_change_hist.values())
        risk_change = 1  # risk_change HAS to be in risk_change_hist
        res = should_send_message_gaen(
            risk_change,
            cur_day,
            last_sent_update_gaen,
            risk_change_hist,
            risk_change_hist_sum,
            rng,
            daily_update_message_budget_sent_gaen,
            message_budget,
            INTERVENTION_DAY,
            UPDATES_PER_DAY,
            n_people,
            BURN_IN_DAYS,
            DAYS_BETWEEN_MESSAGES,
        )
        self.assertTrue(res)

    def test_last_bucket_prob(self):
        """
        check if you're in the last bucket but it's larger than message budget, total messages = budget for this update (=> /UPDATES_PER_DAY)
        """
        cur_day = 10
        BURN_IN_DAYS = 2
        DAYS_BETWEEN_MESSAGES = 1
        INTERVENTION_DAY = 5
        last_sent_update_gaen = 0
        rng = np.random.RandomState(0)
        daily_update_message_budget_sent_gaen = 0
        message_budget = 1
        UPDATES_PER_DAY = 4
        n_people = 1000

        risk_change_hist = {0: 60, 1: 40}
        risk_change_hist_sum = sum(risk_change_hist.values())
        risk_change = 1  # risk_change HAS to be in risk_change_hist
        results = []
        for i in range(1000):
            res = should_send_message_gaen(
                risk_change,
                cur_day,
                last_sent_update_gaen,
                risk_change_hist,
                risk_change_hist_sum,
                rng,
                daily_update_message_budget_sent_gaen,
                message_budget,
                INTERVENTION_DAY,
                UPDATES_PER_DAY,
                n_people,
                BURN_IN_DAYS,
                DAYS_BETWEEN_MESSAGES,
            )
            results.append(res)

        self.assertAlmostEqual(1 / 4, np.mean(results), 2)

    def test_middle_bucket_prob(self):
        """
        checks that if in previous to last bucket and last bucket is smaller than
        budget, then messages sent correspond to the number of remaining messages
        """
        cur_day = 10
        BURN_IN_DAYS = 2
        DAYS_BETWEEN_MESSAGES = 1
        INTERVENTION_DAY = 5
        last_sent_update_gaen = 0
        rng = np.random.RandomState(0)
        daily_update_message_budget_sent_gaen = 0
        message_budget = 1
        UPDATES_PER_DAY = 4
        n_people = 1000

        risk_change_hist = {0: 50, 1: 40, 2: 10}
        risk_change_hist_sum = sum(risk_change_hist.values())
        risk_change = 1  # risk_change HAS to be in risk_change_hist
        results = []
        for i in range(1000):
            res = should_send_message_gaen(
                risk_change,
                cur_day,
                last_sent_update_gaen,
                risk_change_hist,
                risk_change_hist_sum,
                rng,
                daily_update_message_budget_sent_gaen,
                message_budget,
                INTERVENTION_DAY,
                UPDATES_PER_DAY,
                n_people,
                BURN_IN_DAYS,
                DAYS_BETWEEN_MESSAGES,
            )
            results.append(res)
        # allowed messages: 100 / 4 = 25
        # already sent messages: 10
        # remaining to send for second bucket: 15
        self.assertAlmostEqual(1 / 4 - 10 / 100, np.mean(results), 2)

    def test_middle_bucket_last_is_full(self):
        """
        If the last bucket is larger than the budget then no message is sent when in the second largest bucket
        """
        cur_day = 10
        BURN_IN_DAYS = 2
        DAYS_BETWEEN_MESSAGES = 1
        INTERVENTION_DAY = 5
        last_sent_update_gaen = 0
        rng = np.random.RandomState(0)
        daily_update_message_budget_sent_gaen = 0
        message_budget = 1
        UPDATES_PER_DAY = 4
        n_people = 1000

        risk_change_hist = {0: 40, 1: 20, 2: 40}
        risk_change_hist_sum = sum(risk_change_hist.values())
        risk_change = 1  # risk_change HAS to be in risk_change_hist
        res = should_send_message_gaen(
            risk_change,
            cur_day,
            last_sent_update_gaen,
            risk_change_hist,
            risk_change_hist_sum,
            rng,
            daily_update_message_budget_sent_gaen,
            message_budget,
            INTERVENTION_DAY,
            UPDATES_PER_DAY,
            n_people,
            BURN_IN_DAYS,
            DAYS_BETWEEN_MESSAGES,
        )
        self.assertFalse(res)

    def test_last_bucket_low_budget(self):
        """
        Everything works still with a very low budget
        """
        cur_day = 10
        BURN_IN_DAYS = 2
        DAYS_BETWEEN_MESSAGES = 2
        INTERVENTION_DAY = 5
        last_sent_update_gaen = 0
        rng = np.random.RandomState(0)
        daily_update_message_budget_sent_gaen = 0
        message_budget = 0.01  # 0.0025
        UPDATES_PER_DAY = 4
        n_people = 1000

        risk_change_hist = {0: 40, 1: 20, 2: 40}
        risk_change_hist_sum = sum(risk_change_hist.values())
        risk_change = 2  # risk_change HAS to be in risk_change_hist
        results = []
        for i in range(1000):
            res = should_send_message_gaen(
                risk_change,
                cur_day,
                last_sent_update_gaen,
                risk_change_hist,
                risk_change_hist_sum,
                rng,
                daily_update_message_budget_sent_gaen,
                message_budget,
                INTERVENTION_DAY,
                UPDATES_PER_DAY,
                n_people,
                BURN_IN_DAYS,
                DAYS_BETWEEN_MESSAGES,
            )
            results.append(res)

        self.assertAlmostEqual(message_budget / 4, np.mean(results), 2)
