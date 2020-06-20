import logging
from collections import defaultdict
from orderedset import OrderedSet
from covid19sim.utils import get_test_false_negative_rate
from covid19sim.locations.hospital import Hospital, ICU

class TestFacility(object):
    """
    Implements queue behavior for tests.
    It keeps a queue of `Human`s who need a test.
    Depending on the daily budget of testing, tests are administered to `Human` according to a scoring function.
    """

    def __init__(self, test_type_preference, max_capacity_per_test_type, env, conf):
        self.test_type_preference = test_type_preference
        self.max_capacity_per_test_type = max_capacity_per_test_type

        self.test_count_today = defaultdict(int)
        self.env = env
        self.conf = conf
        self.test_queue = OrderedSet()
        self.last_date_to_check_tests = self.env.timestamp.date()

    def reset_tests_capacity(self):
        """
        Resets the tests capactiy back to the allowed budget each day.
        """
        if self.last_date_to_check_tests != self.env.timestamp.date():
            self.last_date_to_check_tests = self.env.timestamp.date()
            for k in self.test_count_today.keys():
                self.test_count_today[k] = 0

            # clear queue
            # TODO : check more scenarios about when the person can be removed from a queue
            to_remove = []
            for human in self.test_queue:
                if not any(human.symptoms) and not human._test_recommended:
                    to_remove.append(human)

            _ = [self.test_queue.remove(human) for human in to_remove]

    def get_available_test(self):
        """
        Returns a first type that is available according to preference hierarchy

        See TEST_TYPES in core.yaml

        Returns:
            str: available test_type
        """
        for test_type in self.test_type_preference:
            if self.test_count_today[test_type] < self.max_capacity_per_test_type[test_type]:
                self.test_count_today[test_type] += 1
                return test_type

    def add_to_test_queue(self, human):
        """
        Adds `Human` to the test queue.

        Args:
            human (Human): `Human` object.
        """
        if human in self.test_queue:
            return
        self.test_queue.add(human)

    def clear_test_queue(self):
        """
        It is called at the same frequency as `while` in City.run.
        Triages `Human` in queue to administer tests.
        With probability P_FALSE_NEGATIVE the test will be negative, otherwise it will be positive

        See TEST_TYPES in core.yaml
        """
        # reset here. if reset at end, it results in carry-over of remaining test at the 0th hour.
        self.reset_tests_capacity()
        test_triage = sorted(list(self.test_queue), key=lambda human: -self.score_test_need(human))
        for human in test_triage:
            test_type = self.get_available_test()
            if test_type:
                if human.infection_timestamp is not None:
                    if human.rng.rand() < get_test_false_negative_rate(test_type, human.days_since_covid, human.conf):
                        unobserved_result = 'negative'
                    else:
                        unobserved_result = 'positive'
                else:
                    if human.rng.rand() < self.conf['TEST_TYPES'][test_type]["P_FALSE_POSITIVE"]:
                        unobserved_result = "positive"
                    else:
                        unobserved_result = "negative"

                human.set_test_info(test_type, unobserved_result)  # /!\ sets other attributes related to tests
                self.test_queue.remove(human)

            else:
                # no more tests available
                break

        logging.debug(f"Cleared the test queue for {len(test_triage)} humans. "
                      f"Out of those, {len(test_triage) - len(self.test_queue)} "
                      f"were tested")

    def score_test_need(self, human):
        """
        Score `Human`s according to some criterion. Highest score gets the test first.
        Note: this can be replaced by a better heuristic.

        Args:
            human (Human): `Human` object.

        Returns:
            float: score value indicating chances of `Human` getting a test.
        """
        score = 0

        if 'severe' in human.symptoms:
            score += self.conf['P_TEST_SEVERE']
        elif 'moderate' in human.symptoms:
            score += self.conf['P_TEST_MODERATE']
        elif 'mild' in human.symptoms:
            score += self.conf['P_TEST_MILD']

        if isinstance(human.location, (Hospital, ICU)):
            score += 1

        if human._test_recommended:
            score += 0.3  # @@@@@@ FIXME THIS IS ARBITRARY

        return score

