"""

These are logic engines for doing tracing.

"""
import typing
from itertools import islice
from covid19sim.epidemiology.symptoms import MODERATE, SEVERE, EXTREMELY_SEVERE
from covid19sim.inference.heavy_jobs import DummyMemManager
from covid19sim.epidemiology.symptoms import STR_TO_SYMPTOMS

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
    """
    Implements several Heuristic contact tracing versions and assigns risk_levels to Humans.

    The risk thresholds in this method determine the way that messages are propagated through the system, while the
    "rec_level" is the behavioural recommendation for the user. There are three broad classes of inputs and rules:
    1) RT-PCR test results
    2) Input Symptoms
    3) Risk Messages

    Each version has a different set of thresholds for risk message propagation (e.g. high_risk_threshold) and
    recommendation levels as a result of risky inputs (e.g. high_risk_rec_level).

    Attributes:
            Default Behaviors: can be set in the config as a list of strings corresponding to keys in `create_behavior`

    Args:
        version (Int): which version of the heuristic to use
        conf (dictionary): rendered configuration dictionary (from YAML) determining behavior of simulation

    """

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

        elif self.version == 4:
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
        This function is responsible for applying the recommendation levels from the heuristic to the human after
        the intervention start date.

        Args:
            risk_level (Int): integer representation of a 4-bit risk level (1 is lowest risk, 16 is highest)
        Returns:
            risk (float): a continuously valued representation of the 4-bit risk level.
        """

        if intervention_start:
            setattr(human, "_heuristic_rec_level", 0)
        else:
            assert hasattr(human, '_heuristic_rec_level'), f"heuristic recommendation level not set for {human}"

        return getattr(human, '_heuristic_rec_level')

    def risk_level_to_risk(self, risk_level):
        """ This is a mapping from an integer represention of a 4-bit risk level to a continuously valued risk.

        Args:
            risk_level (Int): integer representation of a 4-bit risk level (1 is lowest risk, 16 is highest)
        Returns:
            risk (float): a continuously valued representation of the 4-bit risk level.
        """
        risk_level = min(risk_level, 15)
        risk = self.risk_mapping[risk_level + 1]
        return risk

    def extract_clusters(self, human):
        """ This function extracts the clustered risk messages such that we can use them to compute risk / over-rides

        Args:
            human (Human): Human object who has an application running the heuristic algorithm
        Returns:
            processed (array): a processed version of the clusters object
        """
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
         Computes risk history and recommendation level according to the heuristic algorithm.

         /!\ Note 0: for heuristic float risk values do not mean anything, therefore, self.risk_level_to_risk is used
         to convert desired risk_level to float risk value.

         /!\ Note 1: Side-effect - we set `_heuristic_rec_level` attribute in this function. This is required because
         heuristic doesn't have a concept of risk_level to rec_level mapping. The function `self.get_recommendations_level`
         will overwrite rec_level attribute of human via `update_recommendations_level`.

        Args:
            human (Human): Human object who has an application running the heuristic algorithm
            clusters (array): An object containing the clustered messages received by this agent
            humans_map (dict): A dictionary mapping string ids to humans (not used in this function)
        Returns:
            risk history (array): an array of length d_max containing predicted risk on those days (in float form)
        """

        # heuristics_reasons is used as an analytics tool to produce plots showing which rules are effective
        human.heuristic_reasons = set()

        # Current risk history is r_{d-1}^i in the epi paper
        cur_risk_history = list(human.risk_history_map.values())

        # This is W in the epi paper, and defines the window where we say a negative test indicates reduced risk
        test_protection_window = 8

        # check whether this individual has received a positive test result, and provide a risk history and recommendation.
        test_risk_history, test_rec_level = self.handle_tests(human)

        # if they have a positive test result, it over-rides everything else, short-circuiting the heuristic algorithm.
        if test_rec_level == 3:
            _heuristic_rec_level = test_rec_level
            setattr(human, '_heuristic_rec_level', _heuristic_rec_level)
            human.heuristic_reasons.add("positive test")
            return test_risk_history

        # Map the clustered message history to a risk history, rec level, and risk over-ride.
        message_risk_history, message_rec_level, risk_override = self.handle_risk_messages(human, clusters)

        # Depending on the version of the heuristic, we handle symptoms differently
        if self.version == 4:
            symptoms_risk_history, symptoms_rec_level = self.handle_symptoms_v4(human)
        else:
            symptoms_risk_history, symptoms_rec_level = self.handle_symptoms(human)

        # record analytics about which rule was causing the current recommendation level
        if message_rec_level > symptoms_rec_level:
            human.heuristic_reasons.add("risk message")
        if symptoms_rec_level > message_rec_level:
            human.heuristic_reasons.add("symptoms")
        if symptoms_rec_level == message_rec_level and message_rec_level != 0:
            human.heuristic_reasons.add("risk message")
            human.heuristic_reasons.add("symptoms")

        # Run the "recovery" rule, which resets risk history and rec level in the event of no incoming risk signals
        recovery_risk_history, recovery_rec_level = self.handle_recovery(human, clusters)

        # if we recovered, ignore the other signals
        if len(recovery_risk_history) == 7:
            setattr(human, '_heuristic_rec_level', recovery_rec_level)
            human.heuristic_reasons.add("recovered")
            return recovery_risk_history

        # compute the element-wise maximum risk history based on each feature type
        risk_history = self.compute_max_risk_history([cur_risk_history, message_risk_history, symptoms_risk_history])
        rec_level = max(message_rec_level,  symptoms_rec_level, human.rec_level)

        # if you have a negative test result, we want to overwrite the above
        if len(test_risk_history) > 0 and not risk_override:
            human.heuristic_reasons.add("negative test")
            rec_level, risk_history = self.apply_negative_test(rec_level, risk_history, test_risk_history, test_protection_window)
        setattr(human, '_heuristic_rec_level', rec_level)

        # Sometimes handle_recovery misses because there is a risk message that is >3 (like 5) within the last 7 days,
        # but this is not enough to trigger handle_risk_messages. As a result, we end up with a rec level of 3 and a
        # rec level of 0, so we send messages saying we are OK (risk level 0) while maintaining rec level 3.
        if rec_level != 0 and 0.01 == risk_history[0]:
            setattr(human, '_heuristic_rec_level', 0)
        return risk_history

    def apply_negative_test(self, rec_level, risk_history, test_risk_history, test_protection_window):
        """ Negative tests can overwrite some risky inputs. For example, if the agent has a fever and cough, but
         goes to the doctor and gets RT-PCR test that is negative, then we overwrite the symptom risk history. I.e.,
         we re-attribute the symptoms to a cold or flu, not Covid-19. However, if there are many risky messages
         (possibly indicating that the person lives with an infected person), then we do not apply this rule.

        Args:
            rec_level (int): current recommendation level
            risk_history (list): the previous day's risk history (elements are floating values)
            test_risk_history (list): The current risk history generated by test inputs (elements are floating values)
            test_protection_window (int): the number of days we reduce risk given a negative test (W)
        Returns:
            risk_history (array): updated risk history after application of negative test rule
            rec_level (array): updated recommendation level after application of negative test rule
        """
        offset = max(len(test_risk_history) - test_protection_window, 0)

        # Add padding to the risk history if required
        if len(test_risk_history) - test_protection_window > len(risk_history):
            padding = len(test_risk_history) - test_protection_window - len(risk_history)
            risk_history.extend([0.01] * padding)

        # we reset to 0 elements of the risk history between negative test result date - window / 2, to date + window /2
        for i in range(0, test_protection_window):
            if i >= len(test_risk_history):
                continue
            try:
                risk_history[i + offset] = test_risk_history[i]
            except IndexError:
                risk_history.append(test_risk_history[i])

        # if the test was within the last test_protection_window days, it drives the rec level to 0
        if len(test_risk_history) <= test_protection_window:
            rec_level = 0
        return rec_level, risk_history

    def handle_tests(self, human, test_protection_window=8):
        """ Check whether a positive test result was received -- if so, set high risk across the entire history and
        recommend quarantine. If a negative test was received, set risk level to 1 for W days around the negative test

        Args:
            human (Human): the agent using the heuristic app
            test_protection_window (int): the number of days we reduce risk given a negative test (W)
        Returns:
            risk_history (array): updated risk history after application of negative test rule
            rec_level (array): updated recommendation level after application of negative test rule

        """
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
        """ This function approximately determines the day when this agent would have become infectious, given
            risk signals from other agents. We create three groups of risk level (mild, moderate, high) with thresholds
            determined by the Heuristic algorithm version. We look through the message history, and determine the
            first time we receive a message in each group, then record this date as a candidate for infection.
            We then use these grouped risk levels and first dates to create a risk history and recommendation level.
            Risk overrides are used to counter-act a negative test result. E.g., if the agent has received many
            (more than 10) risk messages and has a negative test result, then we will ignore the test result because
            RT-PCR tests have relatively high false-negative rates.

        Args:
            human (Human): the agent using the heuristic app
            clusters (array): Clustered messages received by the agent
        Returns:
            risk_history (array): updated risk history after application of negative test rule
            rec_level (array): updated recommendation level after application of negative test rule
"""

        # variables to record key statistics from the clustered risk messages
        high_risk_message, high_risk_earliest_day = -1, -1
        moderate_risk_message, moderate_risk_earliest_day = -1, -1
        mild_risk_message, mild_risk_earliest_day = -1, -1

        # A constant approximation of the amount of time it takes to go from exposed to infectious
        approx_infectiousness_onset_days = 1

        # override variables
        override = False
        override_limit = 10

        # output variables
        message_risk_history = []
        message_rec_level = 0

        # iterate over each cluster, and determine if it represents the earliest messages in a risk group
        for rel_encounter_day, risk_level, num_encounters in clusters:
            # if the risk level in this cluster is above a risk threshold, then...
            if (risk_level >= self.high_risk_threshold):
                # record the maximum of that clusters risk level and all other high-risk threshold crossing levels
                high_risk_message = max(high_risk_message, risk_level)
                # record the earliest day we received this class of risk message
                high_risk_earliest_day = max(rel_encounter_day - approx_infectiousness_onset_days, high_risk_earliest_day)
                # record whether there was enough risk to over-ride a negative test result
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

        # Next, we create a conservative risk history based on the received risk messages
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
        """ This function processes the test results array to extract relevant information: i.e., the outputs which
        indicate whether a positive test result was received within the last 14 days, and when the last negative
        test result was received (which is None, if no negative result was received)
        Args:
            human (Human): the agent using the heuristic app
        Returns:
            no_positive_test_result_past_14_days (Boolean): True if recent positive test result
            latest_negative_test_result_num_days (Int): relative date of negative test result (or None)
        """
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
        """ This function applies symptom rules for versions 1-3 of the heuristic algorithm.

        Args:
            human (Human): the agent using the heuristic app
        Returns:
            risk_history (array): updated risk history after application of handle_symptoms rule
            rec_level (array): updated recommendation level after application of handle_symptoms rule
        """
        # if there are no symptoms, return no risk history and lowest rec level
        if not any(human.all_reported_symptoms):
            return [], 0

        # check the severity of the reported symptoms, and record the relevant risk and rec level
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

        # repeat the selected risk level over the d_max / 2 days to create a risk history
        risk_history = [self.risk_level_to_risk(new_risk_level)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)
        return risk_history, new_rec_level

    def handle_symptoms_v4(self, human):
        """ In version 4 of the heuristic, we group symptoms by how informative they are of covid.

        Args:
            human (Human): the agent using the heuristic app
        Returns:
            risk_history (array): updated risk history after application of handle_symptoms rule
            rec_level (array): updated recommendation level after application of handle_symptoms rule
        """

        # If no symptoms were reported within the last d_max days, then return an empty risk history and rec level 0
        if not any(human.all_reported_symptoms):
            return [], 0

        # group the symptoms into levels based on how informative they are of COVID-19
        low_risk_symptoms = {"mild", "moderate", "fever", "gastro", "sneezing", "runny_nose", "aches", "fatigue"}
        med_risk_symptoms = {"diarrhea", "nausea_vomiting", "cough", "hard_time_waking_up"}
        high_risk_symptoms = {"severe", "extremely-severe", "chills", "unusual", "headache", "confused", "lost_consciousness", "trouble_breathing", "sore_throat", "severe_chest_pain", "loss_of_taste", "light_trouble_breathing", "moderate_trouble_breathing", "heavy_trouble_breathing"}

        # Extract the names of the symptoms
        reported_symptoms = set([x.name for x in human.all_reported_symptoms])

        # If any of the symptoms are in the high-risk group, then set the high risk history
        if reported_symptoms.intersection(high_risk_symptoms):
            new_risk_level = self.severe_symptoms_risk_level
            new_rec_level = self.severe_symptoms_rec_level
        # If the highest risk symptom is in the moderate group, then set the moderate risk and rec levels
        elif MODERATE in reported_symptoms.intersection(med_risk_symptoms):
            new_risk_level = self.moderate_symptoms_risk_level
            new_rec_level = self.moderate_symptoms_rec_level
        # otherwise we know the symptom is in the low risk group
        else:
            new_risk_level = self.mild_symptoms_risk_level
            new_rec_level = self.mild_symptoms_rec_level

        # repeat this risk level over the last d_max / 2 days to create a risk history
        risk_history = [self.risk_level_to_risk(new_risk_level)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)
        return risk_history, new_rec_level

    def handle_recovery(self, human, clusters):
        """ The recovery rule acts as a force to reduce the risk levels if there are no symptoms, risk messages, or
        tests over the last d_max / 2 days.

        Args:
            human (Human): the agent using the heuristic app
            clusters (array): Clustered messages received by the agent
        Returns:
            risk_history (array): updated risk history after application of handle_symptoms_v4 rule
            rec_level (array): updated recommendation level after application of handle_symptoms_v4 rule
        """
        risk_history = []

        # if there are no recent symptoms
        no_symptoms_past_7_days = \
            not any(islice(human.rolling_all_reported_symptoms, (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)))
        assert human.rec_level == getattr(human, '_heuristic_rec_level'), "rec level mismatch"

        # No positive test results
        no_positive_test_result_past_14_days, _ = self.extract_test_results(human)

        # depending on the heuristic algorithm version, we do different things
        # version 1 and 2 checks whether there have been any messages with risk above a low threshold
        # within the last 7 days
        if self.version == 1 or self.version == 2:
            no_high_risk_message = True
            for rel_encounter_day, risk_level, num_encounters in clusters:
                if (rel_encounter_day < 7) and (risk_level >= 3):
                    no_high_risk_message = False

        # version 3 makes it easier to recover by increasing the risk message threshold
        elif self.version == 3:
            # No large risk message
            no_high_risk_message = True
            for rel_encounter_day, risk_level, num_encounters in clusters:
                if (rel_encounter_day < 7) and (risk_level >= 10):
                    no_high_risk_message = False

        # version 4 has different time windows for each group of risk messages. High risk messages stop recovery longer,
        # while low risk messages only stop recovery for 2 days.
        elif self.version == 4:
            # No large risk message
            no_high_risk_message = False
            high_risks = sum([encs for day, level, encs in clusters if level >= self.high_risk_threshold and day < 7])
            med_risks = sum([encs for day, level, encs in clusters if
                             level >= self.moderate_risk_threshold and level < self.high_risk_threshold and day < 4])
            low_risks = sum([encs for day, level, encs in clusters if
                             level >= self.mild_risk_threshold and level < self.moderate_risk_threshold and day < 2])
            if not (high_risks or med_risks or low_risks):
                no_high_risk_message = True

        # If no recent risky signals, then set risk level for now and the past d_max / 2 days to 0
        if no_positive_test_result_past_14_days and no_symptoms_past_7_days and no_high_risk_message:
            risk_history = [self.risk_level_to_risk(0)] * (human.conf.get("TRACING_N_DAYS_HISTORY") // 2)
        return risk_history, 0

    def compute_max_risk_history(self, risk_histories):
        """ Takes the element-wise maximum over all risk histories
        Args:
            risk_histories (list of lists): Each outcome from the previous rules (symptoms, tests, messages)
        Returns:
            risk_history (array): an element-wise maximum over the outcomes
        """
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
