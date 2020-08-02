"""
Implements modification of human attributes at different levels.
"""
import numpy as np
import warnings
from covid19sim.locations.hospital import Hospital, ICU
from covid19sim.utils.constants import SECONDS_PER_DAY

class IntervenedBehavior(object):
    """
    A base class to implement intervened behavior.

    Args:
        human (covid19sim.human.Human): `human` whose behavior needs to be changed
        env (simpy.Environment): environment to schedule events
        conf (dict): yaml configuration of the experiment
    """

    def __init__(self, human, env, conf):
        self.human = human
        self.env = env
        self.conf = conf
        self.rng = human.rng

        assert conf['N_BEHAVIOR_LEVELS'] >= 2, "At least 2 behavior levels are required to model behavior changes"
        assert not conf['RISK_MODEL'] != "" or conf['N_BEHAVIOR_LEVELS'] == 2, "number of behavior levels (N_BEHAVIOR_LEVELS) in unmitigated scenario should be 2"

        self.n_behavior_levels = conf['N_BEHAVIOR_LEVELS']
        self.quarantine_level = self.n_behavior_levels - 1
        # when there is an intervention, behavior levels are increased by 1 to accomodate a baseline beahvior at the start of the intervention
        if conf['RISK_MODEL'] != "":
            self.n_behavior_levels += 1
            self.quarantine_level += 1

        # start filling the reduction levels from the end
        reduction_levels = {
            "HOUSEHOLD": np.zeros(self.n_behavior_levels),
            "WORKPLACE": np.zeros(self.n_behavior_levels),
            "OTHER": np.zeros(self.n_behavior_levels),
            "SCHOOL": np.zeros(self.n_behavior_levels),
        }

        reduction_levels["HOUSEHOLD"][-1] = 1.0
        reduction_levels["WORKPLACE"][-1] = 1.0
        reduction_levels["OTHER"][-1] = 1.0
        reduction_levels["SCHOOL"][-1] = 1.0
        last_filled_index = self.n_behavior_levels - 1

        # if number of behavior levels is 2 and interpolation is with respect to lockdown contacts, it is a Lockdown scenario
        if conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']:
            reduction_levels["HOUSEHOLD"][-2] = conf['LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_HOUSEHOLD']
            reduction_levels["WORKPLACE"][-2] = conf['LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_WORKPLACE']
            reduction_levels["OTHER"][-2] = conf['LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_OTHER']
            reduction_levels["SCHOOL"][-2] = conf['LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_SCHOOL']
            last_filled_index -= 1

        while last_filled_index > 1:
            to_fill_index = last_filled_index - 1
            for location_type in ["HOUSEHOLD", "WORKPLACE", "OTHER", "SCHOOL"]:
                reduction_levels[location_type][to_fill_index] = reduction_levels[location_type][last_filled_index] / 2

            last_filled_index = to_fill_index

        self.reduction_levels = reduction_levels

        # start everyone at the zero level by default (unmitigated scenario i.e. no reduction in contacts)
        self.quarantine_timestamp = None
        self.quarantine_duration = -1
        self.quarantine_reason = ""
        self.set_behavior(level=0, until=None, reason="initialization")

        # dropout
        self._follow_recommendation_today = None
        self.last_date_to_decide_dropout = None

    def initialize(self, check_has_app=False):
        """
        Sets up a baseline behavior.

        Args:
            check_has_app (bool): whether to initialize a baseline beahvior for humans with app
        """
        assert self.conf['INTERVENTION_DAY'] >= 0, "negative intervention day and yet intialization is called."
        assert self.n_behavior_levels >= 2, "with 2 behavior levels and a risk model, behavior level 1 will quarantine everyone"

        if self.conf['RISK_MODEL'] == "":
            self.set_behavior(level = 1, until = None, reason="lockdown-start")
            return

        if check_has_app and self.human.has_app:
            warnings.warn("An unrealistic scenario - initilization of baseline behavior is only for humans with an app")
            self.set_behavior(level = 1, until = None, reason="intervention-start")
            return

        self.set_behavior(level = 1, until = None, reason="intervention-start")

    @property
    def follow_recommendation_today(self):
        """
        Determines whether `human` follows the restrictions today or not
        """
        last_date = self.last_date_to_decide_dropout
        current_date = self.env.timestamp.date()
        if (
            last_date is None
            or (current_date - last_date).days > 0
        ):
            dropout = _get_dropout_rate(self.current_behavior_reason, self.conf)
            self.last_date_to_decide_dropout = current_date
            self._follow_recommendation_today = self.rng.rand() < (1 - dropout)
        return self._follow_recommendation_today

    def daily_interaction_reduction_factor(self, location):
        """
        Returns fraction of contacts that are reduced from the unmitigated scneraio.

        Args:
            location (covid19sim.locations.location.Location): location where `human` is currently at

        Returns:
            (float): fraction by which  unmiitgated contacts should be reduced
        """
        # if (
        #     isinstance(location, (Hospital, ICU))
        #     and self.conf['ASSUME_SAFE_HOSPITAL_DAILY_INTERACTIONS']
        # ):
        #     return 1.0

        # if its an experimental simulatoin where humans are graded based on their risk, but are not allowed to change their behavior
        if (
            self.conf["RISK_MODEL"] != ""
            and not self.conf['SHOULD_MODIFY_BEHAVIOR']
        ):
            return 0.0

        # if `human` is not following any recommendations today, then set the number of interactions to level 0
        if not self.follow_recommendation_today:
            return 0.0

        location_type = _get_location_type(self.human, location)
        return self.reduction_levels[location_type][self.behavior_level]

    def set_behavior(self, level, until, reason):
        """
        Sets `self.behavior_level` to level for duration `until`.

        Args:
            level (int): behvaior level to put `human` on
            until (float): duration for which the restrictions corresponding to level are put on `human` (seconds)
            reason (str): reason for this level.
        """
        assert reason is not None, "reason is None"

        if level == self.quarantine_level:
            assert not reason != "risk-level-update" or until is not None, "quarantine needs to be of a non-zero duration"
            assert until > 0, "non-positive quarantine duration is not allowed"
            self._quarantine(until, reason)

        # `until` is not required for non-quarantine behavior
        self.behavior_level = level
        self.current_behavior_reason = reason

    def _quarantine(self, until, reason):
        """
        Sets quarantine related attributes.

        Args:
            until (float): duration for which to quarantine `human` (seconds)
            reason (str): reason for which `human` is quarantined
        """
        self.quarantine_timestamp = self.env.timestamp
        self.quarantine_duration = until
        self.quarantine_reason = reason

    def _unset_quarantine(self):
        """
        Resets quarantine related attributes.
        """
        self.quarantine_timestamp = None
        self.quarantine_duration = -1
        self.quarantine_reason = ""
        self.set_behavior(level = 1, until = None, reason = "unset-quarantine")

    def _quarantine_household_members(self, until, reason):
        """
        Sets household members to quarantine.

        Args:
            until (float): duration for which to quarantine household members (seconds)
            reason (str): reason for which to quarantine them
        """
        for h in self.human.household.residents:
            h.intervened_behavior.set_behavior(level = self.quarantine_level, until = until, reason = f"{reason}-household-member")

    def is_quarantined(self):
        """
        Returns True if `human` is currently quarantining.
        """
        if self.quarantine_timestamp is not None:
            if (
                self.quarantine_reason != "risk-level-update"
                and (self.quarantine_timestamp - self.env.timestamp).total_seconds() > self.quarantine_duration
            ):
                self._unset_quarantine()
                return False

            if self.follow_recommendation_today:
                return True

        return False

    def trigger_intervention(self, reason, human):
        """
        Changes the necessary attributes in human depending on reason.

        Args:
            reason (str): reason for the change in behavior of human
            human (covid19sim.human.Human): `human` whose behavior needs to change
        """
        # If someone went for a test, they are assumed to be put in quarantine
        # default behavior even in unmitigated case (no app required)
        if reason == "test-taken":
            result = human.hidden_test_result
            duration = human.time_to_test_result
            duration += 0 if result == "negative" else self.conf['QUARANTINE_DAYS_ON_POSITIVE_TEST']
            self.set_behavior(level = self.quarantine_level, until = duration * SECONDS_PER_DAY, reason = f"{reason}-{result}")

            if self.conf['QUARANTINE_HOUSEHOLD_UPON_INDIVIDUAL_POSITIVE_TEST']:
                duration = human.time_to_test_result
                duration += 0 if human.time_to_test_result == "neagative" else self.conf['QUARANTINE_DAYS_HOUSEHOLD_ON_INDIVIDUAL_POSITIVE_TEST']
                self._quarantine_household_members(until = duration * SECONDS_PER_DAY, reason = f"{reason}-{result}")

        # (no app required)
        elif (
            reason == "self-diagnosed-symptoms"
            and self.conf['QUARANTINE_SELF_REPORTED_INDIVIDUALS']
        ):
            duration = self.conf['QUARANTINE_DAYS_ON_SELF_REPORTED_SYMPTOMS']
            self.set_behavior(level = self.quarantine_level, until = duration * SECONDS_PER_DAY, reason = reason)

            if self.conf['QUARANTINE_HOUSEHOLD_UPON_SELF_REPORTED_INDIVIDUAL']:
                duration = self.conf['QUARANTINE_DAYS_HOUSEHOLD_ON_INDIVIDUAL_CONTACT_WITH_POSITIVE_TEST']
                self._quarantine_household_members(until = duration * SECONDS_PER_DAY, reason=reason)

        # tracing based behavioral changes
        elif reason == "risk-level-update":
            assert self.conf['RISK_MODEL'] != "", "risk model is empty but behavior change due to risk changes is being called."
            assert human.has_app, "human doesn't have an app, but the behavior changes are being called."

            # if its a normalization method, we check daily recommendation mappings.
            if self.human.city.daily_rec_level_mapping is None:
                intervention_level = human.rec_level
            else:
                # QKFIX: There are 4 recommendation levels, the value is hard-coded here
                probas = self.human.city.daily_rec_level_mapping[human.rec_level]
                intervention_level = self.rng.choice(4, p=probas)
            human._intervention_level = intervention_level

            behavior_level = intervention_level + 1
            # in digital tracing, human is quarantined once behavior level is max
            # /!\ when tracing will be because of symptoms, a reason will need to be passed
            if (
                self.conf['RISK_MODEL'] == "digital"
                and behavior_level == self.n_behavior_levels
            ):
                duration = self.conf['QUARANTINE_DAYS_ON_TRACED_POSITIVE']
                self.set_behavior(level = behavior_level, until = duration * SECONDS_PER_DAY, reason="0-1-tracing-positive-test")

                if self.conf['QUARANTINE_HOUSEHOLD_UPON_TRACED_POSITIVE_TEST']:
                    duration = self.conf['QUARANTINE_DAYS_HOUSEHOLD_ON_TRACED_POSITIVE']
                    self._quarantine_household_members(until = duration * SECONDS_PER_DAY, reason=reason)

            # TODO - trigger quarantine for self-reported symptoms in digital tracing

            # in alternative methods, max level is still quarantine, but human can be put back in lower levels due to re-evaluation.
            self.set_behavior(level = behavior_level, until = SECONDS_PER_DAY, reason=reason)
            assert 0 < self.behavior_level <= self.n_behavior_levels, f"behavior_level: {self.behavior_level} can't be outside the range [1,{self.n_behavior_levels}]"

        else:
            raise ValueError(f"Unknown reason for intervention:{reason}")


def _get_location_type(human, location):
    """
    Returns the location type to use for contact reduction depending on location and human's attributes

    Args:
        human (covid19sim.human.Human): `human` for whom this factor need to be determined
        location (covid19sim.locations.location.Location): `location` at which human is currently

    Returns:
        (str): location type that should be considered for evaluation number of contacts
    """

    if (
        location == human.workplace
        and location.location_type != "SCHOOL"
    ):
        return "WORKPLACE"

    elif (
        location == human.workplace
        and location.location_type == "SCHOOL"
    ):
        return "SCHOOL"

    elif location == human.household:
        return "HOUSEHOLD"

    else:
        return "OTHER"

def _get_dropout_rate(reason, conf):
    """
    Returns a probability of not following an intervention due to `reason`

    Returns:
        (float): dropout rate for a type of intervention
    """

    if reason in ["initialization", "intervention-start"]:
        return 0.0

    _reason = reason.replace("-household-member", "")
    if _reason in ["test-taken-positive", "test-taken-negative"]:
        return conf['QUARANTINE_DROPOUT_POSITIVE']

    elif _reason == "self-diagnosed-symptoms":
        return conf['QUARANTINE_DROPOUT_SELF_REPORTED_SYMPTOMS']

    elif _reason == "0-1-tracing-symptoms":
        return conf['QUARANTINE_DROPOUT_TRACED_SELF_REPORTED_SYMPTOMS']

    elif _reason == "0-1-tracing-positive-test":
        return conf['QUARANTINE_DROPOUT_TRACED_POSITIVE']

    elif _reason in ["risk-level-update", "unset-quarantine"]:
        return conf['ALL_LEVELS_DROPOUT']

    else:
        raise ValueError(f"Unknown value:{reason}")
