"""
Implements modification of human attributes at different levels.

###################### Quarantining / Behavior change logic ########################

Following orders takes care of the person faced with multiple quarantining triggers (each trigger has a suggested duration for quarantine) -
    (i)   (non-app based) QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT, QUARANTINE_UNTIL_TEST_RESULT
        +ve result: person is quarantined for 14 days from the day that test was taken.
        -ve result: person is quarantined until the test results come out

    (ii)  (non-app based) SELF_DIAGNOSIS
    (iii) (app based) RISK_LEVEL_UPDATE: x->MAX LEVEL
Dropout enables non-adherence to quarantine at any time.

To consider household quarantine, residents are divided into two groups:
    (i) index cases - they have a quarantine trigger i.e. a reason to believe that they should quarantine
    (ii) secondary cases - rest of the residents

non-app quarantining for index cases -
    * A trigger higher in precedence overwrites other triggers i.e. quarantining duration is changed based on the trigger
    * `human` might already be quarantining at the time of this trigger, so the duration is changed only if trigger requirements require so.
    * if there are no  non-app triggers, app-based triggers are checked every `human.time_slot` and behavior levels are adjusted accordingly

non-app quarantining for secondary cases -
    * All of them quarantine for the same duration unless someone is converted to an index case, in which case, they quarantine and influence household quarantine according to their triggers.
    * this duration is defined by the index case who has maximum quarantining restrictions.

app-based recommendations -
Behavior changes for non-app recommendation for household members -
    * if there are no non-app quarantining triggers, humans are put on app recommendation
    * if MAKE_HOUSEHOLD_BEHAVE_SAME_AS_MAX_RISK_RESIDENT is True, other residents follow the same behavior as the max risk individual in the house
########################################################################
"""
import numpy as np
import warnings
import datetime

from covid19sim.locations.hospital import Hospital, ICU
from covid19sim.utils.constants import SECONDS_PER_DAY
from covid19sim.utils.constants import TEST_TAKEN, SELF_DIAGNOSIS, RISK_LEVEL_UPDATE
from covid19sim.utils.constants import NEGATIVE_TEST_RESULT, POSITIVE_TEST_RESULT
from covid19sim.utils.constants import QUARANTINE_UNTIL_TEST_RESULT, QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT, QUARANTINE_DUE_TO_SELF_DIAGNOSIS
from covid19sim.utils.constants import UNSET_QUARANTINE, QUARANTINE_HOUSEHOLD
from covid19sim.utils.constants import INITIALIZED_BEHAVIOR, INTERVENTION_START_BEHAVIOR, IS_IMMUNE_BEHAVIOR

def convert_intervention_to_behavior_level(intervention_level):
    """
    Maps `human._intervention_level` to `IntervenedBehavior.behavior_level`
    """
    return intervention_level + 1 if intervention_level >= 0 else -1

class Quarantine(object):
    """
    Contains logic to handle different combinations of non-app quarantine triggers.

    Args:
        human (covid19sim.human.Human): `human` whose behavior needs to be changed
        env (simpy.Environment): environment to schedule events
        conf (dict): yaml configuration of the experiment
    """
    def __init__(self, intervened_behavior, human, env, conf):
        self.human = human
        self.intervened_behavior = intervened_behavior
        self.env = env
        self.conf = conf
        self.start_timestamp = None
        self.end_timestamp = None
        self.reasons = []

        self.quarantine_idx = self.intervened_behavior.quarantine_idx
        self.baseline_behavior_idx = self.intervened_behavior.baseline_behavior_idx
        self.human_no_longer_needs_quarantining = False # once human has recovered (infered from 14 days after positive test), human no longer quarantines

    def update(self, trigger):
        """
        Updates quarantine start and end timestamp based on the new `trigger` and previous triggers.

        Note 1: `human_no_longer_needs_quarantining` is set in `reset_quarantine`. if its True, all calls to this function  are ignored.
        Note 2: Test results are treated to have conclusive and ultimate say on the duration of quarantine.
        Note 3: There can be quarantining due to several reasons, so all those combinations are treated in this function through rules described in Quaranining Logic at the top.

        Args:
            trigger (str): reason for quarantine trigger.
        """
        if self.human_no_longer_needs_quarantining:
            return

        # if `human` is already quarantining due to TEST_TAKEN, then do not change anything
        if (
            QUARANTINE_UNTIL_TEST_RESULT in self.reasons
            or QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT in self.reasons
        ):
            return

        if (
            trigger == QUARANTINE_HOUSEHOLD
            and self.end_timestamp is not None
            and self.end_timestamp >= self.human.household.quarantine_end_timestamp
        ):
            return

        #
        self.reasons.append(trigger)
        if self.start_timestamp is None:
            self.start_timestamp = self.env.timestamp

        # set end timestamp and behavior levels accordingly
        # negative test result - quarantine until the test result
        if trigger == QUARANTINE_UNTIL_TEST_RESULT:
            duration = self.human.time_to_test_result * SECONDS_PER_DAY
            self.end_timestamp = self.env.timestamp + datetime.timedelta(seconds=duration)
            self._set_quarantine_behavior(self.reasons, test_recommended=False)

            if self.conf['QUARANTINE_HOUSEHOLD_UPON_INDIVIDUAL_TEST_TAKEN']:
                self.human.household.add_to_index_case(self.human, trigger)

        # positive test result - quarantine until max duration
        elif trigger == QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT:
            duration = self.conf['QUARANTINE_DAYS_ON_POSITIVE_TEST'] * SECONDS_PER_DAY
            self.end_timestamp = self.start_timestamp + datetime.timedelta(seconds=duration)
            self._set_quarantine_behavior(self.reasons, test_recommended=False)

            if self.conf['QUARANTINE_HOUSEHOLD_UPON_INDIVIDUAL_POSITIVE_TEST']:
                self.human.household.add_to_index_case(self.human, trigger)

        elif trigger == QUARANTINE_DUE_TO_SELF_DIAGNOSIS:
            assert False, NotImplementedError(f"{trigger} quarantine not implemented")

        elif trigger == QUARANTINE_HOUSEHOLD:
            self.end_timestamp = self.human.household.quarantine_end_timestamp
            self._set_quarantine_behavior(self.reasons, test_recommended=False)

        else:
            raise ValueError(f"Unknown trigger for quarantine: {trigger}")

    def _set_quarantine_behavior(self, reasons, test_recommended):
        """
        Sets behavior level for quarantining and whether a test is recommended or not. Check Quarantine.update for more.

        Note: It is to be called from `Quarantine.update`

        Args:
            reasons (list): reasons for quarantining.
            test_recommended (bool): whether `human` should get a test or not.
        """
        self.intervened_behavior.set_behavior(level=self.quarantine_idx, reasons=reasons)
        self.human._test_recommended = test_recommended

    def _unset_quarantine_behavior(self, to_level):
        """
        Resets `human` from `quarantine_idx` to `to_level`.

        Note: It is to be called from `Quarantine.update`

        Args:
            to_level (int): the level to which `human`s behavior level should be reset to.
        """
        assert to_level != self.quarantine_idx, "unsetting the quarantine to quarantine_level. Something is wrong."
        self.intervened_behavior.set_behavior(level=to_level, reasons=[UNSET_QUARANTINE, f"{UNSET_QUARANTINE}: {self.intervened_behavior._behavior_level}->{to_level}"])
        self.human._test_recommended = False

    def reset_quarantine(self):
        """
        Resets quarantine related attributes and puts `human` into a relevant behavior level.

        Note 1: Specific to non-binary risk tracing, reset doesn't work if the recommendation is still to quarantine.
        Note 2: It also sets the flag for no more need to quarantine once the test results are positive.
        """
        assert self.start_timestamp is not None, "unsetting quarantine twice not allowed"
        assert not self.human_no_longer_needs_quarantining,  f"{self.human} was quarantined while it shouldn't have"

        last_reason = self.reasons[-1]

        #
        if (
            not self.human_no_longer_needs_quarantining
            and (
                self.human.has_had_positive_test
                or last_reason == QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT
                or self.human.test_result == POSITIVE_TEST_RESULT
            )
        ):
            self.human_no_longer_needs_quarantining = True

        self.start_timestamp = None
        self.end_timestamp = None
        self.reasons = []
        self._unset_quarantine_behavior(self.baseline_behavior_idx)
        self.human.household.reset_index_case(self.human)

    def reset_if_its_time(self):
        """
        Resets `timestamp`s.
        It is called everytime a new activity is to be decided or a trigger is added.
        """
        if self.start_timestamp is not None:
            if self.end_timestamp <= self.env.timestamp:
                self.reset_quarantine()


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

        # we reserve 0-index
        self.n_behavior_levels = conf['N_BEHAVIOR_LEVELS'] + 1
        self.quarantine_idx = self.n_behavior_levels - 1
        self.baseline_behavior_idx = 1
        self._behavior_level = 0 # true behavior level
        self.behavior_level = 0 # its a property.setter

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
        last_filled_index = self.quarantine_idx

        # if number of behavior levels is 2 and interpolation is with respect to lockdown contacts, it is a Lockdown scenario
        if conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']:
            reduction_levels["HOUSEHOLD"][-2] = conf['LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_HOUSEHOLD']
            reduction_levels["WORKPLACE"][-2] = conf['LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_WORKPLACE']
            reduction_levels["OTHER"][-2] = conf['LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_OTHER']
            reduction_levels["SCHOOL"][-2] = conf['LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_SCHOOL']
            last_filled_index -= 1
        else:
            # if its a non-tracing scenario, and lockdown is not desired, its an unmitigated scenario with 0% reduction in the first level
            if conf["RISK_MODEL"] == "" and conf['N_BEHAVIOR_LEVELS'] == 2:
                last_filled_index -= 1
                assert last_filled_index == self.baseline_behavior_idx, "unmitigated scenario should not have non-zero reduction in baseline_behavior"

        # in a non-tracing scenario, baseline_behavior is not defined so we populate levels until baseline_behavior
        while last_filled_index > self.baseline_behavior_idx:
            to_fill_index = last_filled_index - 1
            for location_type in ["HOUSEHOLD", "WORKPLACE", "OTHER", "SCHOOL"]:
                reduction_levels[location_type][to_fill_index] = reduction_levels[location_type][last_filled_index] / 2

            last_filled_index = to_fill_index

        self.reduction_levels = reduction_levels

        # start everyone at the zero level by default (unmitigated scenario i.e. no reduction in contacts)
        self.quarantine = Quarantine(self, self.human, self.env, self.conf)
        self.set_behavior(level=0, reasons=[INITIALIZED_BEHAVIOR])

        # dropout
        self._follow_recommendation_today = None
        self.last_date_to_decide_dropout = None

        #
        self.intervention_started = False
        self.pay_no_attention_to_triggers = False

    def initialize(self, check_has_app=False):
        """
        Sets up a baseline behavior on the day intervention starts.

        Args:
            check_has_app (bool): whether to initialize a baseline beahvior only for humans with the app
        """
        assert self.conf['INTERVENTION_DAY'] >= 0, "negative intervention day and yet intialization is called."
        assert self.n_behavior_levels >= 2, "with 2 behavior levels and a risk model, behavior level 1 will quarantine everyone"

        if check_has_app and self.human.has_app:
            warnings.warn("An unrealistic scenario - initilization of baseline behavior is only for humans with an app")
            self.set_behavior(level=self.baseline_behavior_idx, reasons=[INTERVENTION_START_BEHAVIOR])
            return

        self.set_behavior(level=self.baseline_behavior_idx, reasons=[INTERVENTION_START_BEHAVIOR])
        self.intervention_started = True

    def update_and_get_true_behavior_level(self):
        """
        Returns the true underlying behavior of human.
        Updates the underlying behavior level if this function is called past the `quarantine.end_timestamp`.
        (WIP) A true behavior of such kind can only be achieved by using Quarantine as a simpy.Event.

        Note: if `human` uses the app and follows recommendation of someone else in the hosuehold, _behavior_level will be different then what behavior_level is.

        Returns:
            (int): Behavior level of `human` that determines the number of interactions that a human can have
        """
        self.quarantine.reset_if_its_time()
        return self._behavior_level

    @property
    def behavior_level(self):
        """
        Returns appropriate behavior according to which `human` is supposed to act. Dropout, not used here, can further affect this level.

        Note: It updates the underlying behavior level if this function is called past the `quarantine.end_timestamp`.
        It can not be considered as a side effect. A true behavior of such kind can only be achieved by using Quarantine as a simpy.Event.

        Returns:
            (int): Behavior level of `human` that determines the number of interactions that a human can have
        """
        # if currently someone in the house is following app Rx (someone in the house has to have an app)
        if (
            self.quarantine.start_timestamp is None # currently no non-app quarantining
            and not self.pay_no_attention_to_triggers # hasn't had a positive test in the past
            and self.conf['MAKE_HOUSEHOLD_BEHAVE_SAME_AS_MAX_RISK_RESIDENT']
        ):
            # Note: some `human`s in recovery phase who haven't reset their test_results yet will also come here
            return max(resident.intervened_behavior.update_and_get_true_behavior_level() for resident in self.human.household.residents)

        return self.update_and_get_true_behavior_level()

    @behavior_level.setter
    def behavior_level(self, val):
        self._behavior_level = val

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

    @property
    def is_under_quarantine(self):
        """
        Returns True if `human` is under quarantine restrictions. It doesn't account for dropout.
        """
        return self.behavior_level == self.quarantine_idx

    def is_quarantining(self):
        """
        Returns True if `human` is currently quarantining. It accounts for dropout (non-adherence).
        """
        self.quarantine.reset_if_its_time()
        if self.is_under_quarantine:
            if self.follow_recommendation_today:
                return True

        return False

    def daily_interaction_reduction_factor(self, location):
        """
        Returns fraction of contacts that are reduced from the unmitigated scneraio.

        Args:
            location (covid19sim.locations.location.Location): location where `human` is currently at

        Returns:
            (float): fraction by which unmiitgated contacts should be reduced. 1.0 means 0 interactions, and 0.0 means interactions under unmitigated scenario.
        """
        if (
            self.intervention_started
            and isinstance(location, (Hospital, ICU))
            and self.conf['ASSUME_SAFE_HOSPITAL_DAILY_INTERACTIONS_AFTER_INTERVENTION_START']
        ):
            return 1.0

        # if `human` is not following any recommendations today, then set the number of interactions to level 0
        if not self.follow_recommendation_today:
            return 0.0

        location_type = _get_location_type(self.human, location)
        return self.reduction_levels[location_type][self.behavior_level]

    def set_behavior(self, level, reasons):
        """
        Sets `self.behavior_level` to level for duration `until`.

        Args:
            level (int): behvaior level to put `human` on
            reasons (list): reasons for this level.
        """
        assert reasons is not None and type(reasons) == list, f"reasons: {reasons} is None or it is not a list."

        self.behavior_level = level
        self.current_behavior_reason = reasons

        # (debug)
        # if self.human.name in ["human:71", "human:77", "human:34"]:
        #     print(self.env.timestamp, "set behavior level of", self.human, f"to {level}", "because", self.current_behavior_reason, self.quarantine.start_timestamp, self.quarantine.end_timestamp)

    def set_recommended_behavior(self, level):
        """
        All app-based behavior changes happen through here. It sets _test_recommended attribute of human according to the behavior level.

        Args:
            level (int): behvaior level to put `human` on
        """
        if level == self.quarantine_idx:
            self.human._test_recommended = True

        elif (
            level != self.quarantine_idx
            and self._behavior_level == self.quarantine_idx
        ):
            self.human._test_recommended = False

        self.set_behavior(level=level, reasons=[RISK_LEVEL_UPDATE, f"{RISK_LEVEL_UPDATE}: {self._behavior_level}->{level}"])

    def trigger_intervention(self, reason):
        """
        Changes the necessary attributes in `human`, `self.quarantine`, and `self` depending on the reason.

        Args:
            reason (str): reason for the change in behavior of human
        """

        # if `human` knows about immunity, there is no need to follow any recommendations/quarantining
        if self.pay_no_attention_to_triggers:
            return

        if (
            not self.pay_no_attention_to_triggers
            and self.human.has_had_positive_test
            and self.quarantine.start_timestamp is None
        ):
            self.pay_no_attention_to_triggers = True
            self.set_behavior(level=self.baseline_behavior_idx, reasons=[IS_IMMUNE_BEHAVIOR])
            return

        # (no app required)
        # If someone went for a test, they need to quarantine
        if reason == TEST_TAKEN:
            result = self.human.hidden_test_result

            if result == NEGATIVE_TEST_RESULT:
                self.quarantine.update(QUARANTINE_UNTIL_TEST_RESULT)
            elif result == POSITIVE_TEST_RESULT:
                self.quarantine.update(QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT)
            else:
                raise ValueError(f"Unknown test result:{result}")

        # (no app required)
        elif reason == SELF_DIAGNOSIS:
            assert self.conf['QUARANTINE_SELF_REPORTED_INDIVIDUALS'], "configs do not allow for quarantining self-reported individuals"
            self.quarantine.update(QUARANTINE_DUE_TO_SELF_DIAGNOSIS)

        # (app required) tracing based behavioral changes
        elif reason == RISK_LEVEL_UPDATE:
            assert self.conf['RISK_MODEL'] != "", "risk model is empty but behavior change due to risk changes is being called."
            assert self.human.has_app, "human doesn't have an app, but the behavior changes are being called."

            # if currently quarantining because of non-app triggers, don't do anything
            self.quarantine.reset_if_its_time()
            if self.quarantine.start_timestamp is not None:
                return

            # determine recommendation of the app
            normalized_model = False
            if self.human.city.daily_rec_level_mapping is None:
                intervention_level = self.human.rec_level
            else:
                # QKFIX: There are 4 recommendation levels, the value is hard-coded here
                probas = self.human.city.daily_rec_level_mapping[human.rec_level]
                intervention_level = self.rng.choice(4, p=probas)
                normalized_model = True
            self.human._intervention_level = intervention_level

            # map intervention level to behavior levels by shifting them by 1 (because 1st index is reserved for no reduction in contacts)
            behavior_level = convert_intervention_to_behavior_level(intervention_level)
            assert 0 < behavior_level < self.n_behavior_levels, f"behavior_level: {behavior_level} can't be outside the range [1,{self.n_behavior_levels}]. Total number of levels:{self.n_behavior_levels}"

            # if there is no change in the recommendation, don't do anything
            if (
                RISK_LEVEL_UPDATE in self.current_behavior_reason
                and self._behavior_level == behavior_level
            ):
                return

            # (debug)
            # if self.human.name == "human:71" and self._behavior_level==1 and behavior_level==4:
            #     breakpoint()

            self.set_recommended_behavior(level=behavior_level)
            # if self.conf['MAKE_HOUSEHOLD_BEHAVE_SAME_AS_MAX_RISK_RESIDENT']:
            #     self.human.household.update_max_behavior_level()
        else:
            raise ValueError(f"Unknown reason for intervention:{reason}")

    def __repr__(self):
        return f"IntervenedBehavior for {self.human}"

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

def _get_dropout_rate(reasons, conf):
    """
    Returns a probability of not following an intervention due to `reasons`

    Args:
        reasons (list): list of strings that define the current behavior
        conf (dict): yaml configuration of the experiment

    Returns:
        (float): dropout rate for the current behavior
    """
    _reason = reasons[-1]
    _reason = UNSET_QUARANTINE if UNSET_QUARANTINE in _reason else _reason
    _reason = RISK_LEVEL_UPDATE if RISK_LEVEL_UPDATE in _reason else _reason

    if _reason in [INITIALIZED_BEHAVIOR, INTERVENTION_START_BEHAVIOR, UNSET_QUARANTINE, IS_IMMUNE_BEHAVIOR]:
        return 0.0

    if _reason in [QUARANTINE_UNTIL_TEST_RESULT, QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT]:
        return conf['QUARANTINE_DROPOUT_TEST']

    elif _reason == QUARANTINE_DUE_TO_SELF_DIAGNOSIS:
        return conf['QUARANTINE_DROPOUT_SELF_REPORTED_SYMPTOMS']

    elif _reason == QUARANTINE_HOUSEHOLD:
        return conf['QUARANTINE_DROPOUT_HOUSEHOLD']

    elif _reason == RISK_LEVEL_UPDATE:
        return conf['ALL_LEVELS_DROPOUT']

    else:
        raise ValueError(f"Unknown value:{reasons}")
