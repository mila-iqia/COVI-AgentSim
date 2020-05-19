"""
Implements human behavior/government policy changes.
"""
from orderedset import OrderedSet
import numpy as np

from covid19sim.models.run import integrated_risk_pred
from covid19sim.constants import BIG_NUMBER
class BehaviorInterventions(object):
    """
    A base class to modify behavior based on the type of intervention.
    """
    def __init__(self):
        """
        [summary]
        """
        pass

    def modify_behavior(self, human):
        """
        Changes the behavior attributes of `Human`.
        This function can add new attributes to `Human`.
        If the name of the attribute being changed is `attr`, a new attribute
        is `_attr`.
        `_attr` stores the `attribute` value of `Human` before the change will be made.
        `attr` will store new value.

        Args:
            human (Human): `Human` object.
        """
        pass

    def revert_behavior(self, human):
        """
        Resets the behavior attributes of `Human`.
        It changes `attr` back to what it was before modifying the `attribute`.
        deletes `_attr` from `Human`.

        Args:
            human (Human): `Human` object.
        """
        pass

    def __repr__(self):
        return "BehaviorInterventions"


class StayHome(BehaviorInterventions):
    """
    TODO.
    Not currently being used.
    """
    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._max_misc_per_week = human.max_misc_per_week
        human._max_shop_per_week = human.max_shop_per_week

        human.max_misc_per_week = 1
        human.max_shop_per_week = 1

    def revert_behavior(self, human):
        human.max_misc_per_week = human._max_misc_per_week
        human.max_shop_per_week = human._max_shop_per_week
        delattr(human, "_max_misc_per_week")
        delattr(human, "_max_shop_per_week")

    def __repr__(self):
        return "Stay Home"

class LimitContact (BehaviorInterventions):
    """
    TODO.
    Not currently being used.
    """
    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._maintain_distance = human.maintain_distance
        human._max_misc_per_week = human.max_misc_per_week
        human._max_shop_per_week = human.max_shop_per_week

        human.maintain_distance = human.conf.get("DEFAULT_DISTANCE") + 100 * (human.carefulness - 0.5)
        human.max_misc_per_week = 1
        human.max_shop_per_week = 1

    def revert_behavior(self, human):
        human.maintain_distance = human._maintain_distance
        human.max_misc_per_week = human._max_misc_per_week
        human.max_shop_per_week = human._max_shop_per_week
        delattr(human, "_maintain_distance")
        delattr(human, "_max_misc_per_week")
        delattr(human, "_max_shop_per_week")

    def __repr__(self):
        return "Limit Contact"


class Stand2M(BehaviorInterventions):
    """
    `Human` should maintain an extra distance with other people.
    It adds `_maintain_extra_distance_2m` because of the conflict with a same named attribute in
    `SocialDistancing`
    """
    def __init__(self):
        pass

    def modify_behavior(self, human):
        # FIXME : Social distancing also has this parameter
        human._maintain_extra_distance_2m = human.maintain_extra_distance
        human.maintain_extra_distance = 200 # cms

    def revert_behavior(self, human):
        human.maintain_extra_distance = human._maintain_extra_distance_2m
        delattr(human, "_maintain_extra_distance_2m")

    def __repr__(self):
        return "Stand 2M"

class WashHands(BehaviorInterventions):
    """
    Increases `Human.hygeine`.
    This factor is used to decay likelihood of getting infected/infecting others exponentially.
    """

    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._hygiene = human.hygiene
        human.hygiene = human.rng.uniform(human.carefulness, 2)

    def revert_behavior(self, human):
        human.hygiene = human._hygiene
        delattr(human, "_hygiene")

    def __repr__(self):
        return "Wash Hands"

class Quarantine(BehaviorInterventions):
    """
    Implements quarantining for `Human`. Following is included -
        1. work from home (changes `Human.workplace` to `Human.household`)
        2. rest at home (not go out unless)
        3. stay at home unless hospitalized (so there can still be household infections)
        4. go out with a reduce probability of 0.10 to stores/parks/miscs, but every time `Human` goes out
            they do not explore i.e. do not go to more than one location. (reduce RHO and GAMMA)

    Adds an attribute `_quarantine` to be used as a flag.
    """
    _RHO = 0.1
    _GAMMA = 1

    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._workplace = human.workplace
        human.workplace = human.household
        human.rho = self._RHO
        human.gamma = self._GAMMA
        human.rest_at_home = True
        human._quarantine = True
        # print(f"{human} quarantined {human.tracing_method}")

    def revert_behavior(self, human):
        human.workplace = human._workplace
        human.rho = human.conf.get("RHO")
        human.gamma = human.conf.get("GAMMA")
        human.rest_at_home = False
        human._quarantine = False
        delattr(human, "_workplace")

    def __repr__(self):
        return f"Quarantine"

# FIXME: Lockdown should be a mix of CityBasedIntervention and BehaviorInterventions.
class Lockdown(BehaviorInterventions):
    """
    Implements lockdown. Needs some more work.
    It only implements behvior modification for `Human`. Ideally, it should close down stores/parks/etc.

    Following behavior modifications are included -
        1. reducde mobility through RHO and GAMMA. Enables minimal exploration if going out.
            i.e. `Human` revisits the previously visited location with increased probability.
            If `Human` is on a leisure trip, it visits only a few location.
        2. work from home (changes `Human.workplace` to `Human.household`)
    """
    _RHO = 0.1
    _GAMMA = 1

    def modify_behavior(self, human):
        human._workplace = human.workplace
        human.workplace = human.household
        human.rho = self._RHO
        human.gamma = self._GAMMA

    def revert_behavior(self, human):
        human.workplace = human._workplace
        human.rho = human.conf.get("RHO")
        human.gamma = human.conf.get("GAMMA")
        delattr(human, "_workplace")

    def __repr__(self):
        return f"Lockdown"

class SocialDistancing(BehaviorInterventions):
    """
    Implements social distancing. Following is included -
        1. maintain a distance of 200 cms with other people.
        2. Reduce the time of encounter by 0.5 than what one would do without this intervention.
        3. Reduced mobility (using RHO and GAMMA)

    """
    DEFAULT_SOCIAL_DISTANCE = 200 # cm
    TIME_ENCOUNTER_REDUCTION_FACTOR = 0.5
    _RHO = 0.2
    _GAMMA = 0.5

    def modify_behavior(self, human):
        human._maintain_extra_distance = human.maintain_extra_distance
        human._time_encounter_reduction_factor = human.time_encounter_reduction_factor

        human.maintain_extra_distance = self.DEFAULT_SOCIAL_DISTANCE + 100 * (human.carefulness - 0.5)
        human.time_encounter_reduction_factor = self.TIME_ENCOUNTER_REDUCTION_FACTOR
        human.rho = self._RHO
        human.gamma = self._GAMMA

    def revert_behavior(self, human):
        human.maintain_extra_distance = human._maintain_extra_distance
        human.time_encounter_reduction_factor = human._time_encounter_reduction_factor
        human.rho = human.conf.get("RHO")
        human.gamma = human.conf.get("GAMMA")
        delattr(human, "_maintain_extra_distance")
        delattr(human, "_time_encounter_reduction_factor")

    def __repr__(self):
        """
        [summary]
        """
        return f"Social Distancing"

class BinaryTracing(BehaviorInterventions):
    """
    Implements two recommendations for binary tracing.
    There are only two levels, i.e., 0 and 1.
    At the start of this intervention, everyone is initialized with recommendations
    in the level 0.
    """
    def __init__(self):
        super(BinaryTracing, self).__init__()

    def modify_behavior(self, human):
        if human.rec_level == 0:
            recommendations = []
        else:
            recommendations = [Quarantine()]
        self.revert_behavior(human)
        for rec in recommendations:
            if isinstance(rec, BehaviorInterventions) and human.rng.rand() < human.how_much_I_follow_recommendations:
                rec.modify_behavior(human)
                human.recommendations_to_follow.add(rec)

    def revert_behavior(self, human):
        for rec in human.recommendations_to_follow:
            rec.revert_behavior(human)
        human.recommendations_to_follow = OrderedSet()


class WearMask(BehaviorInterventions):
    """
    `Human` wears a mask according to `Human.wear_mask()`.
    Sets `Human.WEAR_MASK` to True.
    """

    def __init__(self, available=None):
        super(WearMask, self).__init__()
        self.available = available

    def modify_behavior(self, human):
        if self.available is None:
            human.WEAR_MASK = True
            return

        elif self.available > 0:
            human.WEAR_MASK = True
            self.available -= 1

    def revert_behavior(self, human):
        human.WEAR_MASK = False

    def __repr__(self):
        return f"Wear Mask"

def get_recommendations(level):
    """
    Maps recommendation level to a list `BehaviorInterventions`.

    Args:
        level (int): recommendation level.

    Returns:
        list: a list of `BehaviorInterventions`.
    """
    if level == 0:
        return [WashHands()]
    if level == 1:
        return [WashHands(), Stand2M(), WearMask()]
    if level == 2:
        return [WashHands(), SocialDistancing(), WearMask(), 'monitor_symptoms']

    return [WashHands(), SocialDistancing(), WearMask(), 'monitor_symptoms', GetTested("recommendations"), Quarantine()]

class RiskBasedRecommendations(BehaviorInterventions):
    """
    Implements recommendation based behavior modifications.
    The risk level is mapped to a recommendation level. The thresholds to do so are fixed.
    These thresholds are decided using a heuristic, which is outside the scope of this class.
    Each recommendation level is a list of different `BehaviorInterventions`.

    It uses `Human.recommendations_to_follow` to keep a record of various recommendations
    that `Human` is currently following.
    """
    UPPER_GREEN = 1
    UPPER_BLUE = 3
    UPPER_ORANGE = 5
    UPPER_RED = 15

    def __init__(self):
        super(RiskBasedRecommendations, self).__init__()

    @staticmethod
    def get_recommendations_level(risk_level):
        """
        Converts the risk level to recommendation level.

        Args:
            risk_level (int): quantized risk level of range 0-15. sent in encounter messages and update messages.

        Returns:
            recommendation level (int): App recommendation level which takes on a range of 0-3.
        """
        if risk_level <= RiskBasedRecommendations.UPPER_GREEN:
            return 0
        elif RiskBasedRecommendations.UPPER_GREEN < risk_level <= RiskBasedRecommendations.UPPER_BLUE:
            return 1
        elif RiskBasedRecommendations.UPPER_BLUE < risk_level <= RiskBasedRecommendations.UPPER_ORANGE:
            return 2
        elif RiskBasedRecommendations.UPPER_ORANGE < risk_level <= RiskBasedRecommendations.UPPER_RED:
            return 3
        else:
            raise

    def modify_behavior(self, human):
        recommendations = get_recommendations(human.rec_level)
        self.revert_behavior(human)
        for rec in recommendations:
            if isinstance(rec, BehaviorInterventions) and human.rng.rand() < human.how_much_I_follow_recommendations:
                rec.modify_behavior(human)
                human.recommendations_to_follow.add(rec)

    def revert_behavior(self, human):
        for rec in human.recommendations_to_follow:
            rec.revert_behavior(human)
        human.recommendations_to_follow = OrderedSet()

class GetTested(BehaviorInterventions):
    """
    `Human` should get tested.
    """
    # FIXME: can't be called as a stand alone class. Needs human.recommendations_to_follow to work
    # FIXME: test_recommended should be _test_recommended. Make it a convention that any attribute added here,
    # starts with _
    def __init__(self, source):
        """
        Args:
            source (str): reason behind getting tested e.g. recommendation, diagnosis, etc.
        """
        self.source = source

    def modify_behavior(self, human):
        human.test_recommended  = True

    def revert_behavior(self, human):
        human.test_recommended  = False

    def __repr__(self):
        return "Get Tested"

class Tracing(object):
    """
    Implements tracing. It relies on categorization of `Human` according to risk_levels.
    """
    def __init__(self, risk_model, max_depth=1, symptoms=False, risk=False, should_modify_behavior=True):
        """
        risk_levels are determined based on risk, which is computed using `Tracing.compute_risk`.
        `Human.message_info` carries information about all the contacts that person had in the past `configs.blah.yml --> TRACING_N_DAYS_HISTORY`.
        Depending on the type of risk_model, `Human.risk` is computed using the contents of this `Human.message_info`.

        Note 0: `Human.message_info` is used for all `risk_model`s except for `transformer`, a type of `retrospective-risk` tracing method.
        Note 1: `risk`, `symptoms`, `max_depth` do not mean anything for `retrospective-risk` tracing. This method relies on changes in risk_levels, i.e., every time
        the risk level changes, a message is sent to the contacts.
        Note 2: `retrospective-risk` tracing relies on `Human.messages`, `Human.messages_by_day`, etc.

        Following attributes are further used -
            1. p_contact (float) - adds a noise to the tracing procedure. For example, it is not possible to contact everyone in manual tracing.
            2. delay (bool) - if there should be a delay between the time when someone triggers tracing and someone is traced. For example, its 0 for digital tracing.
            3. app (bool) - If an app is required for this tracing. For example, manual tracing doesn't use app.
            4. dont_trace_traced (bool) - If `Human.message_info['traced']` is True, no need to send him a message. If True, reduces the number of messages being passed around.

        Args:
            risk_model (str): Type of tracing to implement. Following methods are currently available - digital, manual, naive, other, transformer.
            max_depth (int, optional): The number of hops away from the source. The term `order` is also used for this. Defaults to 1.
            symptoms (bool, optional): If tracing is to be triggered when someone reports symptoms? Defaults to False.
            risk (bool, optional): If tracing is to be triggered when someone changes risk level?. Defaults to False. Note: this is not to be confused with risk_model="transformer".
            should_modify_behavior (bool, optional): If behavior should be modified or not? Used for conunterfactual studies. Defaults to True.
        """
        self.risk_model = risk_model
        if risk_model in ['manual', 'digital']:
            self.intervention = BinaryTracing()
        else:
            # risk based
            self.intervention = RiskBasedRecommendations()

        self.max_depth = max_depth
        self.propagate_symptoms = symptoms
        self.propagate_risk = risk
        self.propagate_postive_test = True # bare minimum
        self.should_modify_behavior = should_modify_behavior

        self.p_contact = 1
        self.delay = 0
        self.app = True
        if risk_model == "manual":
            self.p_contact = self.conf.get("MANUAL_TRACING_P_CONTACT")
            self.delay = 1
            self.app = False

        self.dont_trace_traced = False
        if risk_model in ["manual", "digital"]:
            self.dont_trace_traced = True

        self.propagate_risk_max_depth = max_depth
        # more than 3 will slow down the simulation too much
        if self.propagate_risk:
            self.propage_risk_max_depth = 3

        if risk_model == "transformer":
            self.propagate_risk_max_depth = BIG_NUMBER
            self.propagate_risk = False
            self.propagate_symptoms = False


    def modify_behavior(self, human):
        if not self.should_modify_behavior and (not human.app and self.app):
            return

        return self.intervention.modify_behavior(human)

    def process_messages(self, human):
        """
        Extract relevant information from `Human.message_info`

        Note 0: one can use any combination of these contacts across `order`s. We have used exponential decay (experimental)
        Note 1: not used by `risk_model = transformer`.

        Args:
            human (Human): Human object.

        Returns:
            t (int): Number of past registered contacts that are tested positive in the last `configs.blah.yml --> TRACING_N_DAYS_HISTORY`.
            s (int): Number of past registered contacts that have shown symptoms in the last `configs.blah.yml --> TRACING_N_DAYS_HISTORY`.
            (r_up, v_up, r_down, v_down) (tuple):
                r_up: Number of past registered contacts that increased their risk levels in the last `configs.blah.yml --> TRACING_N_DAYS_HISTORY`.
                v_up: Averge increase in magnitude of risk levels of contacts that increased their risk levels in the last `configs.blah.yml --> TRACING_N_DAYS_HISTORY`.
                r_down: Number of past registered contacts that decreased their risk levels in the last `configs.blah.yml --> TRACING_N_DAYS_HISTORY`.
                v_down: Averge decrease in magnitude of risk levels of contacts that increased their risk levels in the last `configs.blah.yml --> TRACING_N_DAYS_HISTORY`.
        """
        # total test messages
        t = 0
        for order in human.message_info['n_contacts_tested_positive']:
            t += sum(human.message_info['n_contacts_tested_positive'][order]) * np.exp(-2*(order-1))

        # total symptoms messages
        s = 0
        if self.propagate_symptoms:
            for order in human.message_info['n_contacts_symptoms']:
                s += sum(human.message_info['n_contacts_symptoms'][order]) * np.exp(-2*(order-1))

        r_up, r_down, v_up, v_down = 0, 0, 0, 0
        if self.propagate_risk:
            for order in human.message_info['n_risk_increased']:
                xy = zip(human.message_info['n_risk_mag_increased'][order], human.message_info['n_risk_increased'][order])
                r_up += sum(human.message_info['n_risk_increased'][order])
                z = [1.0*x/y for x,y in xy if y]
                if z:
                    v_up += np.mean(z)

            for order in human.message_info['n_risk_decreased']:
                xy = zip(human.message_info['n_risk_mag_decreased'][order], human.message_info['n_risk_decreased'][order])
                r_down += sum(human.message_info['n_risk_decreased'][order])
                z = [1.0*x/y for x,y in xy if y]
                if z:
                    v_down += np.mean(z)

        return t,s,(r_up, v_up, r_down, v_down)

    def compute_risk(self, t, s, r):
        """
        computes risk value based on the statistics of past contacts.
        Note 1: not used by `risk_model = transformer`.

        Args:
            Output of `self.process_messages`

        Returns:
            [float]: a scalar value.
        """
        if self.risk_model in ['manual', 'digital']:
            if t + s > 0:
                risk = 1.0
            else:
                risk = 0.0

        elif self.risk_model == "naive":
            risk = 1.0 - (1.0 - self.conf.get("RISK_TRANSMISSION_PROBA")) ** (t+s)

        elif self.risk_model == "other":
            r_up, v_up, r_down, v_down = r
            r_score = 2*v_up - v_down
            risk = 1.0 - (1.0 - self.conf.get("RISK_TRANSMISSION_PROBA")) ** (t + 0.5*s + r_score)

        return risk

    def update_human_risks(self, **kwargs):
        """
        Updates the risk value based on the `risk_model`.
        Calls an external server if the `risk_model` relies on a machine learning server.
        """
        city = kwargs.get("city")
        all_possible_symptoms = kwargs.get("symptoms")
        port = kwargs.get("port")
        n_jobs = kwargs.get("n_jobs")
        data_path = kwargs.get("data_path")
        COLLECT_TRAINING_DATA = kwargs.get("COLLECT_TRAINING_DATA")

        if self.risk_model == "transformer":
            all_possible_symptoms = kwargs.get("symptoms")
            port = kwargs.get("port")
            n_jobs = kwargs.get("n_jobs")
            data_path = kwargs.get("data_path")
            city.humans = integrated_risk_pred(city.humans, city.start_time, city.current_day, city.env.timestamp.hour, all_possible_symptoms, port=port, n_jobs=n_jobs, data_path=data_path, conf=city.conf)
        else:
            for human in city.humans:
                cur_day = (human.env.timestamp - human.env.initial_timestamp).days
                if (human.env.timestamp - human.message_info['receipt']).days >= human.message_info['delay'] or self.risk_model != "manual":
                    old_risk = human.risk
                    if not human.is_removed and human.test_result != "positive":
                        t, s, r = self.process_messages(human)
                        human.risk = self.compute_risk(t, s, r)

                    if human.risk != old_risk:
                        self.modify_behavior(human)

                    human.risk_history_map[cur_day] = human.risk

            if COLLECT_TRAINING_DATA:
                city.humans = integrated_risk_pred(city.humans, city.start_time, city.current_day, city.env.timestamp.hour, all_possible_symptoms, port=port, n_jobs=n_jobs, data_path=data_path, conf=city.conf)


    def compute_tracing_delay(self, human):
        """
        Computes delay for tracing. NOT IMPLEMENTED.

        Args:
            human (Human): `Human` object
        """
        pass # FIXME: circualr imports issue; can't import _draw_random_discreet_gaussian

    def __repr__(self):
        if self.risk_model == "transformer":
            return f"Tracing: {self.risk_model}"

        return f"Tracing: {self.risk_model} order {self.max_depth} symptoms: {self.propagate_symptoms} risk: {self.propagate_risk} modify:{self.should_modify_behavior}"


class CityInterventions(object):
    """
    Implements city based interventions such as opening or closing of stores/parks/miscs etc.
    """
    def __init__(self):
        pass

    def modify_city(self, city):
        """
        Modify attributes of city.

        Args:
            city (City): `City` object
        """
        pass

    def revert_city(self, city):
        """
        resets attributes of the city.

        Args:
            city (City): `City` object
        """
        pass


class TestCapacity(CityInterventions):
    """
    Change the test capacity of the city.
    """

    def modify_city(self, city):
        raise NotImplementedError

    def revert_city(self, city):
        raise NotImplementedError

def get_intervention(key, RISK_MODEL=None, TRACING_ORDER=None, TRACE_SYMPTOMS=None, TRACE_RISK_UPDATE=None, SHOULD_MODIFY_BEHAVIOR=True,MASKS_SUPPLY=0):
    """
    Returns appropriate class of intervention.

    Args:
        key (str): type of intervention
        RISK_MODEL (str, optional): passed to `Tracing.risk_model`. Defaults to None.
        TRACING_ORDER (int, optional): passed to `Tracing.max_depth`. Defaults to None.
        TRACE_SYMPTOMS (bool, optional): passed to `Tracing.symptoms`. Defaults to None.
        TRACE_RISK_UPDATE (bool, optional): passed to `Tracing.risk`. Defaults to None.
        SHOULD_MODIFY_BEHAVIOR (bool, optional): passed to `Tracing.should_modify_behavior`. Defaults to True.

    Raises:
        NotImplementedError: If intervention has not been implemented.

    Returns:
        [BehaviorInterventions]: `BehaviorInterventions` corresponding to the arguments.
    """
    if key == "Lockdown":
        return Lockdown()
    elif key == "WearMask":
        return WearMask(MASKS_SUPPLY)
    elif key == "SocialDistancing":
        return SocialDistancing()
    elif key == "Quarantine":
        return Quarantine()
    elif key == "Tracing":
        # there's a global variable somewhere called 'Tracing'
        import covid19sim.interventions
        return covid19sim.interventions.Tracing(
            RISK_MODEL,
            TRACING_ORDER,
            TRACE_SYMPTOMS,
            TRACE_RISK_UPDATE,
            SHOULD_MODIFY_BEHAVIOR,
        )
    elif key == "WashHands":
        return WashHands()
    elif key == "Stand2M":
        return Stand2M()
    elif key == "StayHome":
        return StayHome()
    elif key == "GetTested":
        raise NotImplementedError
    else:
        raise
