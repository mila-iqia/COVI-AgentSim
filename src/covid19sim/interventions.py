from orderedset import OrderedSet
import numpy as np

from covid19sim.config import RHO, GAMMA, MANUAL_TRACING_P_CONTACT,\
    RISK_TRANSMISSION_PROBA, BIG_NUMBER, USE_INFERENCE_SERVER
from covid19sim.models.run import integrated_risk_pred


class BehaviorInterventions(object):
    def __init__(self):
        pass

    def modify_behavior(self, human):
        pass

    def revert_behavior(self, human):
        pass


class StayHome(BehaviorInterventions):

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

class LimitContact (BehaviorInterventions):

    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._maintain_distance = human.maintain_distance
        human._max_misc_per_week = human.max_misc_per_week
        human._max_shop_per_week = human.max_shop_per_week

        human.maintain_distance = DEFAULT_DISTANCE + 100 * (human.carefulness - 0.5)
        human.max_misc_per_week = 1
        human.max_shop_per_week = 1

    def revert_behavior(self, human):
        human.maintain_distance = human._maintain_distance
        human.max_misc_per_week = human._max_misc_per_week
        human.max_shop_per_week = human._max_shop_per_week
        delattr(human, "_maintain_distance")
        delattr(human, "_max_misc_per_week")
        delattr(human, "_max_shop_per_week")

class Stand2M (BehaviorInterventions):

    def __init__(self):
        pass

    def modify_behavior(self, human):
        # FIXME : Social distancing also has this parameter
        human._maintain_extra_distance_2m = human.maintain_extra_distance
        human.maintain_extra_distance = 100 # cms

    def revert_behavior(self, human):
        human.maintain_extra_distance = human._maintain_extra_distance_2m
        delattr(human, "_maintain_extra_distance_2m")

    def __repr__(self):
        return "Stand 2M"

class WashHands(BehaviorInterventions):

    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._hygiene = human.hygiene
        human.hygiene = human.rng.uniform(human.carefulness, 1)

    def revert_behavior(self, human):
        human.hygiene = human._hygiene
        delattr(human, "_hygiene")

    def __repr__(self):
        return "Wash Hands"

class Quarantine(BehaviorInterventions):
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
        human.rho = RHO
        human.gamma = GAMMA
        human.rest_at_home = False
        human._quarantine = False
        delattr(human, "_workplace")

    def __repr__(self):
        return f"Quarantine"

class Lockdown(BehaviorInterventions):
    _RHO = 0.1
    _GAMMA = 1

    def modify_behavior(self, human):
        human._workplace = human.workplace
        human.workplace = human.household
        human.rho = self._RHO
        human.gamma = self._GAMMA

    def revert_behavior(self, human):
        human.workplace = human._workplace
        human.rho = RHO
        human.gamma = GAMMA
        delattr(human, "_workplace")

    def __repr__(self):
        return f"Lockdown"

class SocialDistancing(BehaviorInterventions):
    DEFAULT_SOCIAL_DISTANCE = 100 # cm
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
        human.rho = RHO
        human.gamma = GAMMA
        delattr(human, "_maintain_extra_distance")
        delattr(human, "_time_encounter_reduction_factor")

    def __repr__(self):
        return f"Social Distancing"

class WearMask(BehaviorInterventions):

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
    if level == 0:
        return [WashHands()]
    if level == 1:
        return [WashHands(), WearMask()]
    if level == 2:
        return [WashHands(), SocialDistancing(), Stand2M(), WearMask(), 'monitor_symptoms']

    return [WashHands(), SocialDistancing(), WearMask(), 'monitor_symptoms', GetTested("recommendations"), Quarantine()]

class RiskBasedRecommendations(BehaviorInterventions):
    UPPER_GREEN = 1
    UPPER_BLUE = 3
    UPPER_ORANGE = 5
    UPPER_RED = 15

    def __init__(self):
        super(RiskBasedRecommendations, self).__init__()

    def get_recommendations_level(self, risk_level):
        if risk_level <= self.UPPER_GREEN:
            return 0
        elif self.UPPER_GREEN < risk_level <= self.UPPER_BLUE:
            return 1
        elif self.UPPER_BLUE < risk_level <= self.UPPER_ORANGE:
            return 2
        elif self.UPPER_ORANGE < risk_level <= self.UPPER_RED:
            return 3
        else:
            raise

    def modify_behavior(self, human):
        rec_level = self.get_recommendations_level(human.risk_level)
        human.rec_level = rec_level # FIXME: Shoudl rec_level be a part of human?
        recommendations = get_recommendations(rec_level)
        # print(f"chaging {human} from {human.rec_level} to {rec_level} {human.risk} {human.risk_level}")
        self.revert_behavior(human)
        for rec in recommendations:
            if isinstance(rec, BehaviorInterventions) and human.rng.rand() < human.how_much_I_follow_recommendations:
                rec.modify_behavior(human)
                human.recommendations_to_follow.add(rec)

    def revert_behavior(self, human):
        # print(f"chaging back {human}")
        for rec in human.recommendations_to_follow:
            rec.revert_behavior(human)
        human.recommendations_to_follow = OrderedSet()

class GetTested(BehaviorInterventions):
    # FIXME: can't be called as a stand alone class. Needs human.recommendations_to_follow to work
    def __init__(self, source):
        self.source = source

    def modify_behavior(self, human):
        human.test_recommended  = True

    def revert_behavior(self, human):
        human.test_recommended  = False

    def __repr__(self):
        return "Get Tested"

class Tracing(object):
    def __init__(self, risk_model, max_depth = None, symptoms = False, risk = False, should_modify_behavior=True):
        self.risk_model = risk_model
        if risk_model in ['manual', 'digital']:
            self.intervention = Quarantine()
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
            self.p_contact = MANUAL_TRACING_P_CONTACT
            self.delay = 1
            self.app = False

        self.dont_trace_traced = False
        if risk_model in ["manual", "digital"]:
            self.dont_trace_traced = True

        self.propagate_risk_max_depth = max_depth # too slow
        if risk_model == "transformer":
            self.propagate_risk_max_depth = BIG_NUMBER
            self.propagate_risk = True
            # self.propagate_symptoms = True

        # if self.propagate_risk:
        #     self.propage_risk_max_depth = 3

    def modify_behavior(self, human):
        if not self.should_modify_behavior:
            return

        # FIXME: maybe merge Quarantine in RiskBasedRecommendations with 2 levels
        if self.risk_model in ["manual", "digital"]:
            if human.risk == 1.0:
                human.rec_level = 3 # required for the calculation of mobility
            else:
                human.rec_level = 0

        return self.intervention.modify_behavior(human)

    def process_messages(self, human):
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

    def compute_risk(self, human):
        t,s,r = self.process_messages(human)

        if self.risk_model in ['manual', 'digital'] and t + s > 0:
            human.risk  = 1.0

        elif self.risk_model == "naive":
            human.risk = 1.0 - (1.0 - RISK_TRANSMISSION_PROBA) ** (t+s)

        elif self.risk_model == "other":
            r_up, v_up, r_down, v_down = r
            r_score = 2*v_up - v_down
            human.risk = 1.0 - (1.0 - RISK_TRANSMISSION_PROBA) ** (t + 0.5*s + r_score)

        elif self.risk_model == "transformer":
            pass # risks are computed using the server

    def update_human_risks(self, **kwargs):
        city = kwargs.get("city")

        if self.risk_model == "transformer":
            assert USE_INFERENCE_SERVER == True, "can't run transformer without the server..."
            all_possible_symptoms = kwargs.get("symptoms")
            port = kwargs.get("port")
            n_jobs = kwargs.get("n_jobs")
            data_path = kwargs.get("data_path")
            city.humans = integrated_risk_pred(city.humans, city.start_time, city.current_day, all_possible_symptoms, port=port, n_jobs=n_jobs, data_path=data_path)
            for h in city.humans:
                # same as naive
                if h.is_removed:
                    h.risk = 0.0
                if h.test_result == "positive":
                    h.risk = 1.0
                elif h.test_result == "negative":
                    h.risk = 0.2

        else:
            for human in city.humans:
                if (human.env.timestamp - human.message_info['receipt']).days >= human.message_info['delay']:
                    self.compute_risk(human)
                    human.update_risk_level()

    def compute_tracing_delay(self, human):
        pass # FIXME: circualr imports issue; can't import _draw_random_discreet_gaussian

    def __repr__(self):
        if self.risk_model == "transformer":
            return f"Tracing: {self.risk_model}"

        return f"Tracing: {self.risk_model} order {self.max_depth} symptoms: {self.propagate_symptoms} risk: {self.propagate_risk}"


class CityInterventions(object):
    def __init__(self):
        pass

    def modify_city(self, city):
        pass

    def revert_city(self, city):
        pass


class TestCapacity(CityInterventions):

    def modify_city(self, city):
        pass

    def revert_city(self, city):
        pass

class TransformerTracing(object):
    def modify_behavior(self, human):
        pass
