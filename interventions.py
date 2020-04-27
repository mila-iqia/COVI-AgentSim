from config import RHO, GAMMA, DEFAULT_DISTANCE

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

class Stand2M (BehaviorInterventions):

    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._maintain_distance = human.maintain_distance
        human.maintain_distance = 200 # cms

    def revert_behavior(self, human):
        human.maintain_distance = self._maintain_distance

class WashHands(BehaviorInterventions):

    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._hygiene = human.hygiene
        human.hygiene = 1

    def revert_behavior(self, human):
        human.hygiene = human._hygiene

class Quarantine(BehaviorInterventions):
    _RHO = 0.01
    _GAMMA = 2

    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._workplace = human.workplace
        human.workplace = human.household
        human.rho = self._RHO
        human.gamma = self._GAMMA
        human.rest_at_home = True

    def revert_behavior(self, human):
        human.workplace = human._workplace
        human.rho = RHO
        human.gamma = GAMMA
        human.rest_at_home = False

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

    def __repr__(self):
        return f"Lockdown"

class SocialDistancing(BehaviorInterventions):
    MIN_DISTANCE_ENCOUNTER = 200 # cm
    TIME_ENCOUNTER_REDUCTION_FACTOR = 0.5
    _RHO = 0.2
    _GAMMA = 0.5

    def modify_behavior(self, human):
        human._maintain_distance = human.maintain_distance
        human._time_encounter_reduction_factor = human.time_encounter_reduction_factor

        human.maintain_distance = DEFAULT_DISTANCE + 100 * (human.carefulness - 0.5)
        human.time_encounter_reduction_factor = self.TIME_ENCOUNTER_REDUCTION_FACTOR
        human.rho = self._RHO
        human.gamma = self._GAMMA

    def revert_behavior(self, human):
        human.maintain_distance = human._maintain_distance
        human.time_encounter_reduction_factor = human._time_encounter_reduction_factor
        human.rho = RHO
        human.gamma = GAMMA

    def __repr__(self):
        return f"Social Distancing"

class WearMask(BehaviorInterventions):

    def __init__(self, available=None):
        super(WearMask, self).__init__()
        self.available = available

    def modify_behavior(self, human):
        if self.available is None:
            human.WEAR_MASK = True

        if self.available > 0:
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
        return [WashHands(), SocialDistancing()]
    if level == 2:
        return [WashHands(), SocialDistancing(), WearMask(), 'monitor_symptoms']
    else:
        return [WashHands(), SocialDistancing(), WearMask(), 'monitor_symptoms', GetTested("recommendations"), Quarantine()]

class RiskBasedRecommendations(BehaviorInterventions):
    UPPER_GREEN = 1
    UPPER_BLUE = 4
    UPPER_ORANGE = 12
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
            return 4
        else:
            raise

    def modify_behavior(self, human):
        # print(f"chaging {human}")
        rec_level = self.get_recommendations_level(human.risk_level)
        recommendations = get_recommendations(rec_level)

        self.revert_behavior(human)
        for rec in recommendations:
            if isinstance(rec, BehaviorInterventions) and human.rng.rand() < human.how_much_I_follow_recommendations:
                rec.modify_behavior(human)
                if human.name == "human:93":print(f"{rec}")
                human.recommendations_to_follow.add(rec)

    def revert_behavior(self, human):
        # print(f"chaging back {human}")
        try:
            for rec in human.recommendations_to_follow:
                rec.revert_behavior(human)
                if human.name == "human:93":print(f"{rec}")
        except:
            import pdb; pdb.set_trace()

class GetTested(BehaviorInterventions):
    def __init__(self, source):
        self.source = source

    def modify_behavior(self, human):
        pass

    def revert_behavior(self, human):
        pass

    def __repr__(self):
        return "Get Tested"

class Tracing(object):
    def __init__(self, risk_model):
        self.risk_model = risk_model
        if risk_model in ['manual tracing', 'digital tracing']:
            self.intervention = Quarantine()
        elif risk_model == "first order probabilistic tracing":
            self.intervention = RiskBasedRecommendations()

    def modify_behavior(self, human):
        return self.intervention.modify_behavior(human)

    def __repr__(self):
        return f"Tracing: {self.risk_model}"

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
