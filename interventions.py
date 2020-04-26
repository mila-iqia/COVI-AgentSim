from config import RHO, GAMMA

class BehaviorInterventions(object):
    def __init__(self):
        pass

    def modify_behavior(self, human):
        pass

    def revert_behavior(self, human):
        pass

class WashHands(BehaviorInterventions):

    def modify_behavior(self, human):
        pass

    def revert_behavior(self, human):
        pass

class Quarantine(BehaviorInterventions):
    _RHO = 0.01
    _GAMMA = 2

    @staticmethod
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
