"""
Encodes the logic getting recommendations. Probably should be merged with recommendation_manager.py
"""

import typing
from itertools import islice
from covid19sim.interventions.behaviors import Quarantine
from covid19sim.interventions.intervention_utils import _get_tracing_recommendations

if typing.TYPE_CHECKING:
    from covid19sim.human import Human


class RecommendationGetter(object):
    """
    A base class to modify behavior based on the type of intervention.
    """

    def get_recommendations(self, human: "Human"):
        recommendations = self._get_recommendations_impl(human)
        if not any([isinstance(rec, Quarantine) for rec in recommendations]) and \
                any([h.test_result == "positive" for h in human.household.humans]):
            # in short, if we are not already quarantining and if there is a member
            # of the household that got a positive test (even if they didn't tell their
            # app, whether because they don't have one or don't care, they should still
            # warn their housemates)
            recommendations.append(Quarantine())
        return recommendations

    def _get_recommendations_impl(self, human: "Human"):
        return []


class RiskBasedRecommendationGetter(RecommendationGetter):
    """
    Implements recommendation based behavior modifications.
    The risk level is mapped to a recommendation level. The thresholds to do so are fixed.
    These thresholds are decided using a heuristic, which is outside the scope of this class.
    Each recommendation level is a list of different `RecommendationGetter`.

    It uses `Human.recommendations_to_follow` to keep a record of various recommendations
    that `Human` is currently following.
    """

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

    def _get_recommendations_impl(self, human: "Human"):
        # If there is a mapping available for recommendation levels in the
        # configuration file, use the intervention level randomly picked from
        # this transition matrix, based on the recommendation level. The update
        # of the recommendation levels are not altered.
        if not human.has_app:
            return []

        if human.city.daily_rec_level_mapping is None:
            intervention_level = human.rec_level
        else:
            # QKFIX: There are 4 recommendation levels, the value is hard-coded here
            probas = human.city.daily_rec_level_mapping[human.rec_level]
            intervention_level = human.rng.choice(4, p=probas)
        human._intervention_level = intervention_level
        return _get_tracing_recommendations(intervention_level)

    def revert(self, human):
        # TODO: refactor this
        raise Exception("This is never called")


class HeuristicRecommendationGetter(RiskBasedRecommendationGetter):

    def __init__(self, version, conf):
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
        # "NonMLRiskComputer" class (with "heuristic" tracing method). The recommendations
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

    def compute_risk(self, human, mailbox):
        """
        Computes risk according to heuristic.

        /!\ Note 0: for heuristic float risk values do not mean anything, therefore, self.risk_level_to_risk is used
        to convert desired risk_level to float risk value.

        /!\ Note 1: Side-effect - we set `_heuristic_rec_level` attribute in this function. This is required because
        heuristic doesn't have a concept of risk_level to rec_level mapping. The function `self.get_recommendations_level`
        will overwrite rec_level attribute of human via `update_recommendations_level`.
        """
        risk = human.risk

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

        if human.reported_test_result == "positive":
            # Update risk for the past 14 days (2 weeks)
            risk = [self.risk_level_to_risk(15)] * human.conf.get("TRACING_N_DAYS_HISTORY")
            setattr(human, '_heuristic_rec_level', 3)

        elif (no_positive_test_result_past_14_days
              and latest_negative_test_result_num_days is not None):
            # Set risk level R = 1 for now and all past D days
            risk = [self.risk_level_to_risk(1)] * latest_negative_test_result_num_days
            setattr(human, '_heuristic_rec_level', 0)

        elif human.all_reported_symptoms:
            # for some symptoms set R for now and all past 7 days
            if "extremely-severe" in human.all_reported_symptoms:
                new_risk_level = self.severe_symptoms_risk_level
                new_rec_level = self.severe_symptoms_rec_level

            elif "severe" in human.all_reported_symptoms:
                new_risk_level = self.severe_symptoms_risk_level
                new_rec_level = self.severe_symptoms_rec_level

            elif "moderate" in human.all_reported_symptoms:
                new_risk_level = self.moderate_symptoms_risk_level
                new_rec_level = self.moderate_symptoms_rec_level

            else:
                new_risk_level = self.mild_symptoms_risk_level
                new_rec_level = self.mild_symptoms_rec_level

            risk = [self.risk_level_to_risk(new_risk_level)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)
            setattr(human, '_heuristic_rec_level', new_rec_level)

        elif (human.rec_level > 0 and no_positive_test_result_past_14_days
              and no_symptoms_past_7_days and no_message_gt3_past_7_days):
            # Set risk level R = 0 for now and all past 7 days
            risk = [self.risk_level_to_risk(0)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)
            setattr(human, '_heuristic_rec_level', 0)

        elif high_risk_message > 0:
            # TODO: Decrease the risk level depending on the number of encounters (N > 5)
            updated_risk = max(human.risk_level, self.risk_level_to_risk(high_risk_message - 5))
            risk = [updated_risk] * max(high_risk_num_days - 2, 1) # Update at least 1 day
            setattr(human, '_heuristic_rec_level', self.high_risk_rec_level)

        elif moderate_risk_message > 0:
            # Set the risk level to max(R' - 5, R) for all days after day D + 2
            # (with at least 1 update for the current day)
            updated_risk = max(human.risk_level, self.risk_level_to_risk(moderate_risk_message - 5))
            risk = [updated_risk] * max(moderate_risk_num_days - 2, 1)
            setattr(human, '_heuristic_rec_level', self.moderate_risk_rec_level)

        elif mild_risk_message > 0:
            # Set the risk level to max(R' - 5, R) for all days after day D + 2
            # (with at least 1 update for the current day)
            updated_risk = max(human.risk_level, self.risk_level_to_risk(mild_risk_message - 5))
            risk = [updated_risk] * max(mild_risk_num_days - 2, 1)
            setattr(human, '_heuristic_rec_level', self.mild_risk_rec_level)

        return risk


class BinaryTracing(RecommendationGetter):
    """
    Implements two recommendations for binary tracing.
    There are only two levels, i.e., 0 and 1.
    At the start of this intervention, everyone is initialized with recommendations
    in the level 0.
    """

    def _get_recommendations_impl(self, human: "Human"):
        # If there is a mapping available for recommendation levels in the
        # configuration file, use the intervention level randomly picked from
        # this transition matrix, based on the recommendation level. The update
        # of the recommendation levels are not altered.
        if not human.has_app:
            return []

        if human.city.daily_rec_level_mapping is None:
            intervention_level = human.rec_level
        else:
            # QKFIX: There are 4 recommendation levels, the value is hard-coded here
            probas = human.city.daily_rec_level_mapping[human.rec_level]
            intervention_level = human.rng.choice(4, p=probas)
        human._intervention_level = intervention_level
        return _get_tracing_recommendations(intervention_level)

    def revert(self, human):
        # TODO: refactor such that we can delete this function
        raise Exception("This is never called")
