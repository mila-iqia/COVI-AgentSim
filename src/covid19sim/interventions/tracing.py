"""

These are logic engines for doing tracing.

"""
import typing
import copy
import numpy as np
from itertools import islice
from covid19sim.interventions.behaviors import Quarantine
from covid19sim.epidemiology.symptoms import MODERATE, SEVERE, EXTREMELY_SEVERE
from covid19sim.interventions.tracing_utils import _get_behaviors_for_level, create_behavior

if typing.TYPE_CHECKING:
    from covid19sim.human import Human
    from covid19sim.locations.city import PersonalMailboxType


class BaseMethod(object):
    """
    Implements contact tracing and assigns risk_levels to Humans.

    This object carries a bunch of flags & is responsible for determining the risk level of humans when
    the transformer is not used for risk level inference. To do so, it will use the targeted human's
    contact book to pull statistics about its recent contacts.

    If the transformer is used, this object becomes fairly useless, and will only be used to apply
    recommended behavior changes.

    The name of this class is probably not the best, feel free to suggest alternatives.

    Attributes:
            Default Behaviors: can be set in the config as a list of strings corresponding to keys in `create_behavior`

    """
    def __init__(self, conf: dict):
        """
        Initializes the tracing object.

        Args:
            conf (dict): configuration to parse settings from.
        """
        self.default_behaviors = [create_behavior(key, conf) for key in conf.get("DEFAULT_BEHAVIORS", [])]

    def get_behaviors(self, human: "Human"):
        # note: if there is overlap between default behaviors & recommended ones, the latter will override
        # the former due to the use of list extend calls below + the modify/revert logic in behaviors
        behaviors = copy.deepcopy(self.default_behaviors)

        if human.has_app:
            if human.city.daily_rec_level_mapping is None:
                intervention_level = human.rec_level
            else:
                # QKFIX: There are 4 recommendation levels, the value is hard-coded here
                probas = human.city.daily_rec_level_mapping[human.rec_level]
                intervention_level = human.rng.choice(4, p=probas)
            human._intervention_level = intervention_level
            behaviors.extend(_get_behaviors_for_level(intervention_level))

        if not any([isinstance(rec, Quarantine) for rec in behaviors]) and \
                any([h.test_result == "positive" for h in human.household.humans]):
            # in short, if we are not already quarantining and if there is a member
            # of the household that got a positive test (even if they didn't tell their
            # app, whether because they don't have one or don't care), the housemates should
            # know and get the same (max-level) recommendations, including quarantine
            behaviors.extend(_get_behaviors_for_level(level=3))
        return behaviors

    @staticmethod
    def get_recommendations_level(human, thresholds, max_risk_level, intervention_start=False):
        """
        Converts the risk level to recommendation level.

        Args:
            human (Human): the human for which we want to compute the recommendation.
            thresholds (list|tuple): exactly 3 values to convert risk_level to rec_level
            max_risk_level (int): maximum allowed risk_level for sanity check (assert risk_level <= max_risk_level)

        Returns:
            recommendation level (int): App recommendation level which takes on a range of 0-3.
        """
        if human.risk_level <= thresholds[0]:
            return 0
        elif thresholds[0] < human.risk_level <= thresholds[1]:
            return 1
        elif thresholds[1] < human.risk_level <= thresholds[2]:
            return 2
        elif thresholds[2] < human.risk_level <= max_risk_level:
            return 3
        else:
            raise

    def compute_risk(
            self,
            human: "Human",
            mailbox: "PersonalMailboxType",
            humans_map: typing.Dict[str, "Human"],
    ):
        """
        Computes the infection risk of a human based on the statistics of its past contacts.

        Args:
            human: the human for which to generate the contact tracing counts.
            mailbox: centralized mailbox with all recent update messages for the target human.
            humans_map: a human-name-to-human-object reference map to pass to the contact book functions.

        Returns:
            float: a scalar value.
        """
        raise NotImplementedError


class BinaryDigitalTracing(BaseMethod):
    """
        Attributes:
            max_depth (int, optional): The number of hops away from the source to consider while tracing.
            The term `order` is also used for this. Defaults to 1.
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.max_depth = conf.get("TRACING_ORDER")

    def compute_risk(
            self,
            human: "Human",
            mailbox: "PersonalMailboxType",
            humans_map: typing.Dict[str, "Human"],
    ):
        t = 0

        positive_test_counts = human.contact_book.get_positive_contacts_counts(
            humans_map=humans_map,
            max_order=self.max_depth,
            make_sure_15min_minimum_between_contacts=False,
        )
        for order, count in positive_test_counts.items():
            t += count * np.exp(-2 * (order - 1))

        if t > 0:
            risk = 1.0
        else:
            risk = 0.0

        return risk if isinstance(risk, list) else [risk]


class Heuristic(BaseMethod):

    def __init__(self, version, conf):
        super().__init__(conf)
        self.version = version
        if self.version == 1:
            self.high_risk_threshold, self.high_risk_rec_level = 12, 3
            self.moderate_risk_threshold, self.moderate_risk_rec_level = 10, 2
            self.mild_risk_threshold, self.mild_risk_rec_level = 6, 1

            self.severe_symptoms_risk_level, self.severe_symptoms_rec_level = 12, 3
            self.moderate_symptoms_risk_level, self.moderate_symptoms_rec_level = 10, 3
            self.mild_symptoms_risk_level, self.mild_symptoms_rec_level = 7, 2

        elif self.version == 2:
            self.high_risk_threshold, self.high_risk_rec_level = 10, 2
            self.moderate_risk_threshold, self.moderate_risk_rec_level = 8, 1
            self.mild_risk_threshold, self.mild_risk_rec_level = 4, 1

            self.severe_symptoms_risk_level, self.severe_symptoms_rec_level = 10, 2
            self.moderate_symptoms_risk_level, self.moderate_symptoms_rec_level = 8, 2
            self.mild_symptoms_risk_level, self.mild_symptoms_rec_level = 6, 1

        else:
            raise NotImplementedError()

        self.risk_mapping = conf.get("RISK_MAPPING")

    def get_recommendations_level(self, human, thresholds, max_risk_level, intervention_start=False):
        """
        /!\ Overwrites _heuristic_rec_level on the very first day of intervention; note that
        the `_heuristic_rec_level` attribute must be set in each human before calling this via
        the `intervention_start` kwarg.
        """
        # Most of the logic for recommendations level update is given in the
        # "BaseMEthod" class (with "heuristic" tracing method). The recommendations
        # level for the heuristic tracing algorithm are dependent on messages
        # received in the mailbox, which get_recommendations_level does not have
        # access to under the current API.

        if intervention_start:
            setattr(human, "_heuristic_rec_level", 0)
        else:
            assert hasattr(human, '_heuristic_rec_level'), f"heuristic recommendation level not set for {human}"

        # if human.age >= 70:
        #     rec_level = 1 if (self.version == 2) else 2
        #     rec_level = max(human.rec_level, rec_level)
        #     setattr(human, "_heuristic_rec_level", rec_level)

        return getattr(human, '_heuristic_rec_level')

    def risk_level_to_risk(self, risk_level):
        risk_level = min(risk_level, 15)
        return self.risk_mapping[risk_level + 1]

    def compute_risk(self, human, mailbox, humans_map: typing.Dict[str, "Human"]):
        """
        Computes risk according to heuristic.

        /!\ Note 0: for heuristic float risk values do not mean anything, therefore, self.risk_level_to_risk is used
        to convert desired risk_level to float risk value.

        /!\ Note 1: Side-effect - we set `_heuristic_rec_level` attribute in this function. This is required because
        heuristic doesn't have a concept of risk_level to rec_level mapping. The function `self.get_recommendations_level`
        will overwrite rec_level attribute of human via `update_recommendations_level`.
        """

        no_message_gt3_past_7_days = True
        no_positive_test_result_past_14_days = True
        latest_negative_test_result_num_days = None
        high_risk_message, high_risk_num_days = -1, -1
        moderate_risk_message, moderate_risk_num_days = -1, -1
        mild_risk_message, mild_risk_num_days = -1, -1

        # Check if the user received messages with specific risk level
        # TODO: mailbox only contains update messages, and not encounter messages
        # TODO: use the result of the clustering algorithm to find the number of
        #       encounters with another user with high risk level
        for _, update_messages in mailbox.items():
            for update_message in update_messages:
                encounter_day = (human.env.timestamp - update_message.encounter_time).days
                risk_level = update_message.new_risk_level

                if (encounter_day < 7) and (risk_level >= 3):
                    no_message_gt3_past_7_days = False

                # conservative approach - keep max risk above threshold along with max days in the past
                if (risk_level >= self.high_risk_threshold):
                    high_risk_message = max(high_risk_message, risk_level)
                    high_risk_num_days = max(encounter_day, high_risk_num_days)

                elif (risk_level >= self.moderate_risk_threshold):
                    moderate_risk_message = max(moderate_risk_message, risk_level)
                    moderate_risk_num_days = max(encounter_day, moderate_risk_message)

                elif (risk_level >= self.mild_risk_threshold):
                    mild_risk_message = max(mild_risk_message, risk_level)
                    mild_risk_num_days = max(encounter_day, moderate_risk_message)

        for test_result, test_time, _ in human.test_results:
            result_day = (human.env.timestamp - test_time).days
            if result_day >= 0 and result_day < human.conf.get("TRACING_N_DAYS_HISTORY"):
                no_positive_test_result_past_14_days &= (test_result != "positive")

                # keep the date of latest negative test result
                if (test_result == "negative" and ((latest_negative_test_result_num_days is None)
                        or (latest_negative_test_result_num_days > result_day))):
                    latest_negative_test_result_num_days = result_day

        no_symptoms_past_7_days = \
            not any(islice(human.rolling_all_reported_symptoms, (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)))
        assert human.rec_level == getattr(human, '_heuristic_rec_level'), "rec level mismatch"

        risk_histories = [[human.risk]]
        rec_levels = [0]
        if human.reported_test_result == "positive":
            risk_histories.append([self.risk_level_to_risk(15)] * human.conf.get("TRACING_N_DAYS_HISTORY"))
            rec_levels.append(3)

        if (no_positive_test_result_past_14_days
              and latest_negative_test_result_num_days is not None):
            risk_histories.append([self.risk_level_to_risk(1)] * max(latest_negative_test_result_num_days, 2))
            rec_levels.append(0)

        if human.all_reported_symptoms:
            # for some symptoms set R for now and all past 7 days
            if EXTREMELY_SEVERE in human.all_reported_symptoms:
                new_risk_level = self.severe_symptoms_risk_level
                new_rec_level = self.severe_symptoms_rec_level

            elif SEVERE in human.all_reported_symptoms:
                new_risk_level = self.severe_symptoms_risk_level
                new_rec_level = self.severe_symptoms_rec_level

            elif MODERATE in human.all_reported_symptoms:
                new_risk_level = self.moderate_symptoms_risk_level
                new_rec_level = self.moderate_symptoms_rec_level

            else:
                new_risk_level = self.mild_symptoms_risk_level
                new_rec_level = self.mild_symptoms_rec_level

            # if it's more conservative to go with the number of experienced symptoms, do that.
            if len(human.all_reported_symptoms) > 2 * new_rec_level:
                new_risk_level = len(human.all_reported_symptoms)
                new_rec_level = min(new_risk_level // 2, 3)

            risk_histories.append([self.risk_level_to_risk(new_risk_level)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2))
            rec_levels.append(new_rec_level)

        if high_risk_message > 0:
            # TODO: Decrease the risk level depending on the number of encounters (N > 5)
            updated_risk = self.risk_level_to_risk(max(human.risk_level, high_risk_message - 5))
            risk_histories.append([updated_risk] * max(high_risk_num_days - 2, 1)) # Update at least 1 day
            rec_levels.append(self.high_risk_rec_level)

        elif moderate_risk_message > 0:
            # Set the risk level to max(R' - 5, R) for all days after day D + 2
            # (with at least 1 update for the current day)
            updated_risk = self.risk_level_to_risk(max(human.risk_level, moderate_risk_message - 5))
            risk_histories.append([updated_risk] * max(moderate_risk_num_days - 2, 1))
            rec_levels.append(self.moderate_risk_rec_level)

        elif mild_risk_message > 0:
            # Set the risk level to max(R' - 5, R) for all days after day D + 2
            # (with at least 1 update for the current day)
            updated_risk = self.risk_level_to_risk(max(human.risk_level, mild_risk_message - 5))
            risk_histories.append([updated_risk] * max(mild_risk_num_days - 2, 1))
            rec_levels.append(self.mild_risk_rec_level)

        # If we don't actually appear risky, then set to low risk
        if (human.rec_level > 0 and no_positive_test_result_past_14_days
              and no_symptoms_past_7_days and no_message_gt3_past_7_days):
            # Set risk level R = 0 for now and all past 7 days
            risk_history = [self.risk_level_to_risk(0)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)
            setattr(human, '_heuristic_rec_level', 0)
        # if no rules applied, don't change rec level and return baseline risk

        # apply most conservative rec level and risk history
        else:
            setattr(human, '_heuristic_rec_level', max(rec_levels))
            most_conservative_risk_history = [0.]
            for risk_history in risk_histories:
                if sum(risk_history) > sum(most_conservative_risk_history):
                    most_conservative_risk_history = risk_history
            risk_history = most_conservative_risk_history
        return risk_history if type(risk_history) == list else [risk_history]
