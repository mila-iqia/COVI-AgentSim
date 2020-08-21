"""
Implements modification of human attributes at different levels.

###################### Quarantining logic ########################

There are two types of quarantining triggers:
    (i) Conclusive - due to test results
    (ii) inconclusive - rest. E.g. getting traced or being recommended to quarantine by a non-binary tracing method.
Each trigger has a suggested duration for quarantine.
To consider household quarantine, residents are divided into two groups:
    (i) index cases - they have a quarantine trigger i.e. a reason to believe that they should quarantine
    (ii) secondary cases - rest.

Anyone who has had a positive test in the past is not quarantined after 14 days of the positive test result.
For index cases -
    * A conclusive quarantining trigger has a final say on the duration of quarantining.
    * If the index case is already quarantining, an inconclusive trigger is used only if the total time alraedy spent quarantining is less than the suggested duration.

For secondary cases -
    * All of them quarantine for the same duration unless someone is converted to an index case, in which case, they quarantine and influence household quarantine according to their triggers.
    * Duration is defined by the index case who has maximum quarantining restrictions.

Scenarios -
    * (binary-tracing) Secondary case is coming out of a quarantine of 2 days due to negative test result of the index. This person has also been traced.
      We put this secondary case back in quarantine for a maximum of duration required for a traced index.
    * (no-tracing) Secondary case who is also infected goes for a test. Test result turn out to be negative. This case is released from quarantine because test-results are taken as conclusive evidence.

Dropout enables non-adherence to quarantine at any time.

########################################################################
"""
import numpy as np
import warnings
import datetime

from covid19sim.locations.hospital import Hospital, ICU
from covid19sim.utils.constants import SECONDS_PER_DAY
from covid19sim.utils.constants import TEST_TAKEN, SELF_DIAGNOSIS, RISK_LEVEL_UPDATE
from covid19sim.utils.constants import NEGATIVE_TEST_RESULT, POSITIVE_TEST_RESULT
from covid19sim.utils.constants import QUARANTINE_UNTIL_TEST_RESULT, QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT
from covid19sim.utils.constants import TRACED_BY_POSITIVE_TEST, TRACED_BY_SELF_REPORTED_SYMPTOMS, MAX_RISK_LEVEL_TRACED
from covid19sim.utils.constants import UNSET_QUARANTINE, QUARANTINE_HOUSEHOLD
from covid19sim.utils.constants import INITIALIZED_BEHAVIOR, INTERVENTION_START_BEHAVIOR

def convert_intervention_to_behavior_level(intervention_level):
    """
    Maps `human._intervention_level` to `IntervenedBehavior.behavior_level`
    """
    return intervention_level + 1 if intervention_level >= 0 else -1

class Quarantine(object):
    """
    Contains logic to handle different combinations of quarantine triggers.

    Args:
        human (covid19sim.human.Human): `human` whose behavior needs to be changed
        env (simpy.Environment): environment to schedule events
        conf (dict): yaml configuration of the experiment
    """
    def __init__(self, human, env, conf):
        self.human = human
        self.env = env
        self.conf = conf
        self.start_timestamp = None
        self.end_timestamp = None
        self.reasons = []

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
        assert trigger != MAX_RISK_LEVEL_TRACED, "quarantine due to risk-level-update is implemented in `set_recommended_behavior`"
        if self.human_no_longer_needs_quarantining:
            return

        # if `human` is already quarantining due to TEST_TAKEN, then do not change anything
        if (
            QUARANTINE_UNTIL_TEST_RESULT in self.reasons
            or QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT in self.reasons
        ):
            return

        #
        if self.start_timestamp is None:
            self.start_timestamp = self.env.timestamp

        # traced positive test
        if trigger == TRACED_BY_POSITIVE_TEST:
            duration = self.conf['QUARANTINE_DAYS_ON_TRACED_POSITIVE_TEST']
            self.end_timestamp = self.start_timestamp + datetime.timedelta(seconds=duration * SECONDS_PER_DAY)
            self.reasons.append(trigger)
            self.human.intervened_behavior._set_quarantine_behavior(self.reasons, test_recommended=True)

            if self.conf['QUARANTINE_HOUSEHOLD_UPON_TRACED_POSITIVE_TEST']:
                self.human.household.add_to_index_case(self.human, trigger)

        # negative test result - quarantine until the test result
        elif trigger == QUARANTINE_UNTIL_TEST_RESULT:
            duration = self.human.time_to_test_result * SECONDS_PER_DAY
            self.end_timestamp = self.env.timestamp + datetime.timedelta(seconds=duration)
            self.reasons.append(QUARANTINE_UNTIL_TEST_RESULT)
            self.human.intervened_behavior._set_quarantine_behavior(self.reasons, test_recommended=False)

            if self.conf['QUARANTINE_HOUSEHOLD_UPON_INDIVIDUAL_TEST_TAKEN']:
                self.human.household.add_to_index_case(self.human, trigger)

        # positive test result - quarantine until max duration
        elif trigger == QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT:
            duration = self.conf['QUARANTINE_DAYS_ON_POSITIVE_TEST'] * SECONDS_PER_DAY
            self.end_timestamp = self.start_timestamp + datetime.timedelta(seconds=duration)
            self.reasons.append(QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT)
            self.human.intervened_behavior._set_quarantine_behavior(self.reasons, test_recommended=False)

            if self.conf['QUARANTINE_HOUSEHOLD_UPON_INDIVIDUAL_POSITIVE_TEST']:
                self.human.household.add_to_index_case(self.human, trigger)

        elif trigger == QUARANTINE_HOUSEHOLD:
            if (
                self.end_timestamp is not None
                and self.end_timestamp >= self.human.household.quarantine_end_timestamp
            ):
                return
            self.end_timestamp = self.human.household.quarantine_end_timestamp
            self.reasons.append(QUARANTINE_HOUSEHOLD)
            self.human.intervened_behavior._set_quarantine_behavior(self.reasons, test_recommended=False)

        elif trigger == TRACED_BY_SELF_REPORTED_SYMPTOMS:
            assert False, NotImplementedError(f"{trigger} quarantine not implemented")

        elif trigger == SELF_DIAGNOSIS:
            assert False, NotImplementedError(f"{trigger} quarantine not implemented")
            duration = self.conf['QUARANTINE_DAYS_ON_SELF_REPORTED_SYMPTOMS']
            if self.conf['QUARANTINE_HOUSEHOLD_UPON_SELF_REPORTED_INDIVIDUAL']:
                self.human.household.add_to_index_case(self.human, trigger)

        else:
            raise ValueError(f"Unknown trigger for quarantine: {trigger}")

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
        self.human.intervened_behavior._unset_quarantine_behavior(self.human.intervened_behavior.baseline_behavior_idx)
        self.human.household.reset_index_case(self.human)

        # if `human` uses an app, level is reset to the recommended level
        # only if the last reason for quarantining was not TEST_TAKEN
        if (
            not last_reason == QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT
            and not last_reason == QUARANTINE_UNTIL_TEST_RESULT
            and not self.human._intervention_level == -1
        ):
            to_level = convert_intervention_to_behavior_level(self.human._intervention_level)

            if to_level == self.human.intervened_behavior.quarantine_idx:
                if (
                    self.conf['RISK_MODEL'] != "digital"
                    or self.human.city.daily_rec_level_mapping is not None
                ):
                    self.set_recommended_quarantine(force=True)
                else:
                    self.update(TRACED_BY_POSITIVE_TEST)

    def set_recommended_quarantine(self, force=False):
        """
        Quarantining for non-binary tracing is set through this function call. It is separate from other `update` because end_timestamp is set to max.

        Note 1: Any non-app based quarantining like TEST_TAKEN or SELF_DIAGNOSIS takes priority over the recommendations.

        Args:
            force (bool): if True, it doesn't look at non-app based quarantining. Called from `reset_quarantine` when quarantining is being reset from inconclusive triggers however the app still recommends quarantine.
        """
        if not force:
            assert MAX_RISK_LEVEL_TRACED not in self.reasons, f"{self.human} is already quarantining under max recommendation"

            # any non-app quarantining takes priority over the app
            if len(self.reasons) > 0:
                return

            self.start_timestamp = self.env.timestamp

        self.reasons.apppend(MAX_RISK_LEVEL_TRACED)
        self.end_timestamp = datetime.datetime.max
        self.human.intervened_behavior._set_quarantine_behavior(self.reasons, test_recommended=True)

        if self.conf['QUARANTINE_HOUSEHOLD_UPON_INDIVIDUAL_MAX_RISK_LEVEL_TRACED']:
            assert False

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
        self.behavior_level = 0

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
        self.quarantine = Quarantine(self.human, self.env, self.conf)
        self.set_behavior(level=0, reasons=[INITIALIZED_BEHAVIOR])

        # dropout
        self._follow_recommendation_today = None
        self.last_date_to_decide_dropout = None

        #
        self.intervention_started = False

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
        if self.human.name in ["human:71", "human:77", "human:34"]:
            print(self.env.timestamp, "set behavior level of", self.human, f"to {level}", "because", self.current_behavior_reason, self.quarantine.start_timestamp, self.quarantine.end_timestamp)

    def _set_quarantine_behavior(self, reasons, test_recommended):
        """
        Sets behavior level for quarantining and whether test is recommended or not. Check Quarantine.update for more.

        Note: It is to be called from `Quarantine.update`

        Args:
            reasons (list): reasons for quarantining.
            test_recommended (bool): whether `human` should get a test or not.
        """
        self.set_behavior(level=self.quarantine_idx, reasons=reasons)
        self.human._test_recommended = test_recommended

    def _unset_quarantine_behavior(self, to_level):
        """
        Resets `human` from `quarantine_idx` to `to_level`.

        Note: It is to be called from `Quarantine.update`

        Args:
            to_level (int): the level to which `human`s behavior level should be reset to.
        """
        assert to_level != self.quarantine_idx, "unsetting the quarantine to quarantine_level. Something is wrong."
        self.set_behavior(level=to_level, reasons=[UNSET_QUARANTINE])
        self.human._test_recommended = False

    def set_recommended_behavior(self, level):
        """
        Non-binary tracing methods use recommendation level to control `human`s behavior.
        This function is called to set `behavior_level` recommended by those methods.

        Args:
            level (int): the level to which `human`s behavior should be set to.
        """
        if level == self.quarantine_idx:
            self.quarantine.set_recommended_quarantine()

        elif (
            self.behavior_level == self.quarantine_idx
            and level != self.quarantine_idx
        ):
            self.quarantine.reset_quarantine()

        else:
            self.set_behavior(level=level, reasons=[RISK_LEVEL_UPDATE])
            if self.conf['SET_HOUSEHOLD_BEHAVIOR_UPON_INDIVIDUAL_RISK_LEVEL_UPDATES']:
                assert False

    def is_quarantined(self):
        """
        Returns True if `human` is currently quarantining. It accounts for dropout (non-adherence).
        """
        self.quarantine.reset_if_its_time()
        if self.quarantine.start_timestamp is not None:
            if self.follow_recommendation_today:
                return True

        return False

    def trigger_intervention(self, reason):
        """
        Changes the necessary attributes in `human`, `self.quarantine`, and `self` depending on the reason.

        Args:
            reason (str): reason for the change in behavior of human
        """
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
            self.quarantine.update(SELF_DIAGNOSIS)

        # (app required) tracing based behavioral changes
        elif reason == RISK_LEVEL_UPDATE:
            assert self.conf['RISK_MODEL'] != "", "risk model is empty but behavior change due to risk changes is being called."
            assert self.human.has_app, "human doesn't have an app, but the behavior changes are being called."

            normalized_model = False
            if self.human.city.daily_rec_level_mapping is None:
                intervention_level = self.human.rec_level
            else:
                # QKFIX: There are 4 recommendation levels, the value is hard-coded here
                probas = self.human.city.daily_rec_level_mapping[human.rec_level]
                intervention_level = self.rng.choice(4, p=probas)
                normalized_model = True
            self.human._intervention_level = intervention_level

            # map rec levels to intervention levels by shifting them by 1 (because 1st index is reserved for no reduction)
            behavior_level = convert_intervention_to_behavior_level(intervention_level)
            assert 0 < behavior_level < self.n_behavior_levels, f"behavior_level: {self.behavior_level} can't be outside the range [1,{self.n_behavior_levels}]. Total number of levels:{self.n_behavior_levels}"

            #
            if self.behavior_level == behavior_level:
                return

            # (non-binary tracing) normalized model has RISK_MODEL == "digital"
            if (
                self.conf['RISK_MODEL'] != "digital"
                or normalized_model
            ):
                # in alternative methods, max level is still quarantine, but human can be put in lower levels due to re-evaluation.
                self.set_recommended_behavior(level=behavior_level)
                return

            # (binary tracing)
            # in binary tracing, human is quarantined once behavior level is max
            if behavior_level == self.quarantine_idx:
                self.quarantine.update(TRACED_BY_POSITIVE_TEST)
            elif behavior_level == self.baseline_behavior_idx:
                self.quarantine.reset_quarantine()
            else:
                raise ValueError(f"found non-binary recommendations in binary tracing. Behavior level:{behavior_level}. Allowed values: 1 and {self.quarantine_idx}")

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

    if (
        INITIALIZED_BEHAVIOR == _reason
        or INTERVENTION_START_BEHAVIOR == _reason
    ):
        return 0.0

    if _reason in [QUARANTINE_UNTIL_TEST_RESULT, QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT]:
        return conf['QUARANTINE_DROPOUT_TEST']

    elif _reason == TRACED_BY_POSITIVE_TEST:
        return conf['QUARANTINE_DROPOUT_TRACED_POSITIVE']

    elif _reason in [QUARANTINE_HOUSEHOLD, UNSET_QUARANTINE, RISK_LEVEL_UPDATE]:
        return conf['ALL_LEVELS_DROPOUT']

    elif _reason == MAX_RISK_LEVEL_TRACED:
        return conf['QUARANTINE_DROPOUT_MAX_RISK_LEVEL_TRACED']

    elif _reason == SELF_DIAGNOSIS:
        return conf['QUARANTINE_DROPOUT_SELF_REPORTED_SYMPTOMS']

    elif _reason == TRACED_BY_SELF_REPORTED_SYMPTOMS:
        return conf['QUARANTINE_DROPOUT_TRACED_SELF_REPORTED_SYMPTOMS']

    else:
        raise ValueError(f"Unknown value:{reason}")
