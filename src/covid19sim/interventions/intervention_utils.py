from orderedset import OrderedSet
from covid19sim.interventions.behaviors import *

def get_intervention(conf):
    """
    Returns the appropriate class of intervention.

    Args:
        conf (dict): configuration to send to intervention object.

    Raises:
        NotImplementedError: If intervention has not been implemented.

    Returns:
        `Behavior`: `Behavior` corresponding to the arguments.
    """
    key = conf.get("INTERVENTION")
    if key == "Lockdown":
        return Lockdown()
    elif key == "WearMask":
        return WearMask(conf.get("MASKS_SUPPLY"))
    elif key == "SocialDistancing":
        return SocialDistancing()
    elif key == "Quarantine":
        return Quarantine()
    elif key == "Tracing":
        from covid19sim.interventions.recommendation_manager import NonMLRiskComputer
        return NonMLRiskComputer(conf)
    elif key == "WashHands":
        return WashHands()
    elif key == "StandApart":
        return StandApart()
    elif key == "GetTested":
        raise NotImplementedError
    elif key == "BundledInterventions":
        return BundledInterventions(conf["BUNDLED_INTERVENTION_RECOMMENDATION_LEVEL"])
    else:
        raise

class BundledInterventions(Behavior):
    """
    Used for tuning the "strength" of parameters associated with interventions.
    At the start of this intervention, everyone is initialized with these interventions.
    DROPOUT might affect their ability to follow.
    """

    def __init__(self, level):
        super(BundledInterventions, self).__init__()
        self.recommendations = _get_tracing_recommendations(level)

    def modify(self, human):
        self.revert(human)
        for rec in self.recommendations:
            if isinstance(rec, Behavior) and human.follows_recommendations_today:
                rec.modify(human)
                human.recommendations_to_follow.add(rec)

    def revert(self, human):
        for rec in human.recommendations_to_follow:
            rec.revert(human)
        human.recommendations_to_follow = OrderedSet()

    def __repr__(self):
        return "\t".join([str(x) for x in self.recommendations])


def _get_tracing_recommendations(level):
    """
    Maps recommendation level to a list `Behavior`.

    Args:
        level (int): recommendation level.

    Returns:
        list: a list of `Behavior`.
    """
    assert level in [0, 1, 2, 3]
    if level == 0:
        return [WashHands(), StandApart(default_distance=25)]
    if level == 1:
        return [WashHands(), StandApart(default_distance=75), WearMask()]
    if level == 2:
        return [WashHands(), SocialDistancing(default_distance=100), WearMask()]

    return [WashHands(), SocialDistancing(default_distance=150), WearMask(), GetTested("recommendations"), Quarantine()]

