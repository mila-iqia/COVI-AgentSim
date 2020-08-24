"""

These are logic engines for doing tracing.

"""
import typing
from itertools import islice
from covid19sim.epidemiology.symptoms import MODERATE, SEVERE, EXTREMELY_SEVERE
from covid19sim.inference.heavy_jobs import DummyMemManager

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
        pass

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
    #
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
            t += count

        risk = 1.0 * (t > 0)

        return risk if isinstance(risk, list) else [risk]


class Heuristic(BaseMethod):

    def __init__(self, version, conf):
        super().__init__(conf)
        self.version = version
        if self.version == 1:
            self.high_risk_threshold, self.high_risk_rec_level = 12, 3
            self.moderate_risk_threshold, self.moderate_risk_rec_level = 10, 2
            self.mild_risk_threshold, self.mild_risk_rec_level = 6, 2

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

        elif self.version == 3:
            self.high_risk_threshold, self.high_risk_rec_level = 15, 3
            self.moderate_risk_threshold, self.moderate_risk_rec_level = 13, 2
            self.mild_risk_threshold, self.mild_risk_rec_level = 10, 1

            self.severe_symptoms_risk_level, self.severe_symptoms_rec_level = 13, 3
            self.moderate_symptoms_risk_level, self.moderate_symptoms_rec_level = 10, 2
            self.mild_symptoms_risk_level, self.mild_symptoms_rec_level = 5, 1

        else:
            raise NotImplementedError()

        self.risk_mapping = conf.get("RISK_MAPPING")
        self.conf = conf

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

    def extract_clusters(self, human):
        try:
            cluster_mgr_map = DummyMemManager.get_cluster_mgr_map()
            prepend_str = list(cluster_mgr_map.keys())[0].split(":")[0]
            clusters = cluster_mgr_map[":".join([prepend_str, human.name])].clusters
            processed = []
            for c in clusters:
                encounter_day = (human.env.timestamp - c.first_update_time).days
                processed.append((encounter_day, c.risk_level, len(c._real_encounter_times)))
        except (IndexError, KeyError):
            return []
        return processed

    def compute_risk(self, human, clusters, humans_map: typing.Dict[str, "Human"]):
        """
         Computes risk according to heuristic.

         /!\ Note 0: for heuristic float risk values do not mean anything, therefore, self.risk_level_to_risk is used
         to convert desired risk_level to float risk value.

         /!\ Note 1: Side-effect - we set `_heuristic_rec_level` attribute in this function. This is required because
         heuristic doesn't have a concept of risk_level to rec_level mapping. The function `self.get_recommendations_level`
         will overwrite rec_level attribute of human via `update_recommendations_level`.
        """
        cur_risk_history = list(human.risk_history_map.values())
        test_protection_window = 8
        # if you have a positive test result, it over-rides everything else (we ignore symptoms and risk messages)
        test_risk_history, test_rec_level = self.handle_tests(human)
        if test_rec_level == 3:
            _heuristic_rec_level = test_rec_level
            setattr(human, '_heuristic_rec_level', _heuristic_rec_level)
            return test_risk_history

        message_risk_history, message_rec_level, risk_override = self.handle_risk_messages(human, clusters)

        symptoms_risk_history, symptoms_rec_level = self.handle_symptoms(human)

        recovery_risk_history, recovery_rec_level = self.handle_recovery(human, clusters)

        # if we recovered, ignore the other signals
        if len(recovery_risk_history) == 7:
            setattr(human, '_heuristic_rec_level', recovery_rec_level)
            return recovery_risk_history

        # compute max risk history and rec level
        risk_history = self.compute_max_risk_history([cur_risk_history, message_risk_history, symptoms_risk_history])
        rec_level = max(message_rec_level,  symptoms_rec_level, human.rec_level)

        # if you have a negative test result, we want to overwrite the above
        if len(test_risk_history) > 0 and not risk_override:
            rec_level, risk_history = self.apply_negative_test(rec_level, risk_history, test_risk_history, test_protection_window)
        setattr(human, '_heuristic_rec_level', rec_level)

        # self.debug_heuristic_plot(message_rec_level, symptoms_rec_level, human.rec_level, rec_level)
        # Sometimes handle_recovery misses because there is a risk message that is >3 (like 5) within the last 7 days,
        # but this is not enough to trigger handle_risk_messages. As a result, we end up with a rec level of 3 and a
        # rec level of 0, so we send messages saying we are OK (risk level 0) while maintaining rec level 3.
        if rec_level != 0 and 0.01 == risk_history[0]:
            setattr(human, '_heuristic_rec_level', 0)
        return risk_history

    def debug_heuristic_plot(self, message_rec_level, symptoms_rec_level, human_rec_level, negative_test_rec_level):
        print("A")

    def apply_negative_test(self, rec_level, risk_history, test_risk_history, test_protection_window):
        offset = max(len(test_risk_history) - test_protection_window, 0)
        # pad if we need to pad
        if len(test_risk_history) - test_protection_window > len(risk_history):
            padding = len(test_risk_history) - test_protection_window - len(risk_history)
            risk_history.extend([0.01] * padding)

        for i in range(0, test_protection_window):
            if i >= len(test_risk_history):
                continue
            try:
                risk_history[i + offset] = test_risk_history[i]
            except IndexError:
                risk_history.append(test_risk_history[i])

        # if the test was within the last test_protection_window days, it drives the rec level
        if len(test_risk_history) <= test_protection_window:
            rec_level = 0
        return rec_level, risk_history

    def handle_tests(self, human, test_protection_window=8):
        test_risk_history = []
        test_rec_level = 0

        no_positive_test_result_past_14_days, latest_negative_test_result_num_days = self.extract_test_results(human)
        if no_positive_test_result_past_14_days and latest_negative_test_result_num_days is not None:
            test_risk_history = [self.risk_level_to_risk(1)] * (latest_negative_test_result_num_days + test_protection_window//2)
            test_rec_level = 0

        if human.reported_test_result == "positive":
            test_risk_history = [self.risk_level_to_risk(15)] * human.conf.get("TRACING_N_DAYS_HISTORY")
            test_rec_level = 3

        return test_risk_history, test_rec_level

    def handle_risk_messages(self, human, clusters):
        """ The core idea here is that we want to approximate the day when we would have become infectious, given
            risk signals from other agents. We """

        high_risk_message, high_risk_earliest_day = -1, -1
        moderate_risk_message, moderate_risk_earliest_day = -1, -1
        mild_risk_message, mild_risk_earliest_day = -1, -1
        approx_infectiousness_onset_days = 1
        message_risk_history = []
        message_rec_level = 0

        override = False
        override_limit = 10
        for rel_encounter_day, risk_level, num_encounters in clusters:
            # conservative approach - keep max risk above threshold along with max days in the past
            if (risk_level >= self.high_risk_threshold):
                high_risk_message = max(high_risk_message, risk_level)
                high_risk_earliest_day = max(rel_encounter_day - approx_infectiousness_onset_days, high_risk_earliest_day)
                if num_encounters > override_limit:
                    override = True
            elif (risk_level >= self.moderate_risk_threshold):
                moderate_risk_message = max(moderate_risk_message, risk_level)
                moderate_risk_earliest_day = max(rel_encounter_day - approx_infectiousness_onset_days,
                                                 moderate_risk_earliest_day)
                if num_encounters > override_limit:
                    override = True

            elif (risk_level >= self.mild_risk_threshold):
                mild_risk_message = max(mild_risk_message, risk_level)
                mild_risk_earliest_day = max(rel_encounter_day - approx_infectiousness_onset_days, mild_risk_earliest_day)
                if num_encounters > override_limit:
                    override = True

        if high_risk_message > 0 and high_risk_earliest_day > 0:
            updated_risk = self.risk_level_to_risk(max(human.risk_level, high_risk_message - 5))
            message_risk_history = [updated_risk] * high_risk_earliest_day
            message_rec_level = max(human.rec_level, self.high_risk_rec_level)

        elif moderate_risk_message > 0 and moderate_risk_earliest_day > 0:
            updated_risk = self.risk_level_to_risk(max(human.risk_level, moderate_risk_message - 5))
            message_risk_history = [updated_risk] * moderate_risk_earliest_day
            message_rec_level = max(human.rec_level, self.moderate_risk_rec_level)

        elif mild_risk_message > 0 and mild_risk_earliest_day > 0:
            updated_risk = self.risk_level_to_risk(max(human.risk_level, mild_risk_message - 5))
            message_risk_history = [updated_risk] * min(mild_risk_earliest_day, 5)
            message_rec_level = max(human.rec_level, self.mild_risk_rec_level)
        return message_risk_history, message_rec_level, override

    def extract_test_results(self, human):
        latest_negative_test_result_num_days = None
        no_positive_test_result_past_14_days = True
        for test_result, test_time, _ in human.test_results:
            result_day = (human.env.timestamp - test_time).days
            if result_day >= 0 and result_day < human.conf.get("TRACING_N_DAYS_HISTORY"):
                no_positive_test_result_past_14_days &= (test_result != "positive")
                # keep the date of latest negative test result
                if (test_result == "negative" and ((latest_negative_test_result_num_days is None)
                        or (latest_negative_test_result_num_days > result_day))):
                    latest_negative_test_result_num_days = result_day
        return no_positive_test_result_past_14_days, latest_negative_test_result_num_days

    def handle_symptoms(self, human):
        if not any(human.all_reported_symptoms):
            return [], 0

        # p_covid = self.compute_p_covid_given_symptoms(human, self.conf)

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
        # if len(human.all_reported_symptoms) > 2 * new_rec_level:
        #     new_risk_level = len(human.all_reported_symptoms)
        #     new_rec_level = min(new_risk_level // 2, 3)

        risk_history = [self.risk_level_to_risk(new_risk_level)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)
        return risk_history, new_rec_level

    def handle_recovery(self, human, mailbox):
        # No symptoms in last 7 days
        no_symptoms_past_7_days = \
            not any(islice(human.rolling_all_reported_symptoms, (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)))
        assert human.rec_level == getattr(human, '_heuristic_rec_level'), "rec level mismatch"

        # No positive test results
        no_positive_test_result_past_14_days, _ = self.extract_test_results(human)

        if self.version == 1 or self.version == 2:
            no_high_risk_message = True
            for rel_encounter_day, risk_level, num_encounters in mailbox:
                if (rel_encounter_day < 7) and (risk_level >= 3):
                    no_high_risk_message = False

        elif self.version == 3:
            # No large risk message
            no_high_risk_message = True
            for rel_encounter_day, risk_level, num_encounters in mailbox:
                if (rel_encounter_day < 7) and (risk_level >= 10):
                    no_high_risk_message = False

        risk_history = []
        # set to low risk
        if no_positive_test_result_past_14_days and no_symptoms_past_7_days and no_high_risk_message:
            # Set risk level R = 0 for now and all past 7 days
            risk_history = [self.risk_level_to_risk(0)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)
        return risk_history, 0

    def compute_max_risk_history(self, risk_histories):
        longest_length = max([len(x) for x in risk_histories])
        risk_history = []
        for i in range(longest_length):
            vals = []
            for r in risk_histories:
                try:
                     vals.append(r[i])
                except IndexError:
                    pass

            risk_history.append(max(vals))
        return risk_history
