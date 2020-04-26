from config import RHO, GAMMA

class BehaviorInterventions(object):
    def __init__(self):
        pass

    def modify_behavior(self, human):
        pass

    def revert_behavior(self, human):
        pass

class StayHome (BehaviorInterventions):

    def __init__(self):
        self.premod_exercise = None
        self.premod_misc = None
        self.premod_shop = None

    def modify_behavior(self, human):
        self.premod_exercise = human.max_exercise_per_week
        self.premod_misc = human.max_misc_per_week
        self.premod_shop = human.max_shop_per_week

        human.max_exercise_per_week = 7
        human.max_misc_per_week = 1
        human.max_shop_per_week = 1

    def revert_behavior(self, human):
        human.maintain_distance = self.premod_distance
        human.max_exercise_per_week = self.premod_exercise
        human.max_misc_per_week = self.premod_misc
        human.max_shop_per_week = self.premod_shop


class LimitContact (BehaviorInterventions):

    def __init__(self):
        self.premod_distance = None
        self.premod_exercise = None
        self.premod_misc = None
        self.premod_shop = None

    def modify_behavior(self, human):
        self.premod_distance = human.maintain_distance
        self.premod_exercise = human.max_exercise_per_week
        self.premod_misc = human.max_misc_per_week
        self.premod_shop = human.max_shop_per_week

        human.maintain_distance = 3
        human.max_exercise_per_week = 7
        human.max_misc_per_week = 0
        human.max_shop_per_week = 1

    def revert_behavior(self, human):
        human.maintain_distance = self.premod_distance
        human.max_exercise_per_week = self.premod_exercise
        human.max_misc_per_week = self.premod_misc
        human.max_shop_per_week = self.premod_shop

class Stand2M (BehaviorInterventions):

    def __init__(self):
        self.premod_distance = None

    def modify_behavior(self, human):
        self.premod_distance = human.maintain_distance
        human.maintain_distance = 2

    def revert_behavior(self, human):
        human.maintain_distance = self.premod_distance

class WashHands(BehaviorInterventions):

    def __init__(self):
        self.premod_hygiene = None

    def modify_behavior(self, human):
        self.premod_hygiene = human.hygiene
        human.hygiene = 1

    def revert_behavior(self, human):
        human.hygiene = self.original_hygiene

class Quarantine(BehaviorInterventions):

    def __init__(self):
        self.premod_rest = None
        self.premod_RHO = 0.01
        self.premod_GAMMA = 2

    @staticmethod
    def modify_behavior(self, human):
        human._workplace = human.workplace
        human.workplace = human.household
        human.rho = self._RHO
        human.gamma = self._GAMMA
        human.rest_at_home = True

    def revert_behavior(self, human):
        human.workpalce = human._workplace
        human.rho = RHO
        human.gamma = GAMMA
        human.rest_at_home = self.premod_rest

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
        human.workpalce = human._workplace
        human.rho = RHO
        human.gamma = GAMMA

    def __repr__(self):
        return f"Lockdown"

class SocialDistancing(BehaviorInterventions):
    MIN_DISTANCE_ENCOUNTER = 200 # cm
    TIME_ENCOUNTER_FACTOR = 0.5
    _RHO = 0.2
    _GAMMA = 0.5

    def modify_behavior(self, human):
        human.min_distance_encounter = self.MIN_DISTANCE_ENCOUNTER
        human.time_encounter_reduction_factor = self.TIME_REDUCTION_FACTOR
        human.rho = self._RHO
        human.gamma = self._GAMMA

    def revert_behavior(self, human):
        human.min_distance_encounter = 0
        human.time_encounter_reduction_factor = 1
        human.rho = RHO
        human.gamma = GAMMA

    def __repr__(self):
        return f"Social Distancing"

class WearMask(BehaviorInterventions):

    def __init__(self, available):
        super(WearMask, self).__init__()
        self.available = available

    def modify_behavior(self, human):
        if self.available > 0:
            human.WEAR_MASK = True
            self.available -= 1

    def revert_behavior(self, human):
        human.WEAR_MASK = False

    def __repr__(self):
        return f"Wear Mask"

class RiskBasedRecommendations(BehaviorInterventions):
    UPPER_GREEN = 1
    UPPER_BLUE = 4
    UPPER_ORANGE = 12
    UPPER_RED = 15

    def __init__(self, RISK_MODEL):
        super(RiskBasedRecommendations, self).__init__()
        self.RISK_MODEL = RISK_MODEL

    def get_recommendations_level(self, risk_level):
        if risk_level <= UPPER_GREEN:
            return 0
        elif UPPER_GREEN < risk_level <= UPPER_BLUE:
            return 1
        elif UPPER_BLUE < risk_level <= UPPER_ORANGE:
            return 2
        elif UPPER_ORANGE < risk_level <= UPPER_RED:
            return 4
        else:
            raise

    @staticmethod
    def modify_behavior(self, human):
        rec_level = self.get_recommendations_level(human.risk_level)
        if rec_level == 1:
            return

        if rec_level == 2:
            return SocialDistancing.modify_behavior(human)

        if rec_level == 3:
            return Quarantine.modify_behavior(human)

    def revert_behavior(self, human):
        pass

class Tracing(object):
    def __init__(self, risk_model):
        self.risk_model = risk_model
        if risk_model in ['manual tracing', 'digital tracing']:
            self.intervention = Quarantine()
        elif risk_model == "first order probabilistic tracing":
            self.intervention = RiskBasedRecommendations()

    def modify_behavior(self, human):
        return self.intervention.modify_behavior

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
