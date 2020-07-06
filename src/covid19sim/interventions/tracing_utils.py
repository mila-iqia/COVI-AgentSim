from orderedset import OrderedSet
from covid19sim.interventions.behaviors import *


def get_intervention(conf):
    from covid19sim.interventions.tracing import BaseMethod, Heuristic, BinaryDigitalTracing
    key = conf.get("RISK_MODEL")
    if key == "" or key == "transformer":
        return BaseMethod(conf)
    elif key == "heuristicv1":
        return Heuristic(version=1, conf=conf)
    elif key == "heuristicv2":
        return Heuristic(version=2, conf=conf)
    elif key == "digital":
        return BinaryDigitalTracing(conf)
    elif key == "BundledInterventions":
        return BundledInterventions(conf["BUNDLED_INTERVENTION_RECOMMENDATION_LEVEL"])
    else:
        raise NotImplementedError


def create_behavior(key, conf):
    if key == "WearMask":
        return WearMask(conf.get("MASKS_SUPPLY"))
    elif key == "SocialDistancing":
        return SocialDistancing()
    elif key == "Quarantine":
        return Quarantine()
    elif key == "WashHands":
        return WashHands()
    elif key == "StandApart":
        return StandApart()
    else:
        raise NotImplementedError


class BundledInterventions(Behavior):
    """
    Used for tuning the "strength" of parameters associated with interventions.
    At the start of this intervention, everyone is initialized with these interventions.
    DROPOUT might affect their ability to follow.
    """

    def __init__(self, level):
        super(BundledInterventions, self).__init__()
        self.recommendations = _get_behaviors_for_level(level)

    def modify(self, human):
        self.revert(human)
        for rec in self.recommendations:
            if isinstance(rec, Behavior) and human.follows_recommendations_today:
                rec.modify(human)
                human.recommendations_to_follow.add(rec)

    def revert(self, human):
        for rec in reversed(human.recommendations_to_follow):
            rec.revert(human)
        human.recommendations_to_follow = OrderedSet()

    def __repr__(self):
        return "\t".join([str(x) for x in self.recommendations])


def _get_behaviors_for_level(level):
    """
    Maps recommendation level to a list of `Behavior` objects.

    Args:
        level (int): recommendation level.

    Returns:
        list: a list of `Behavior`.
    """
    if level == 0:
        return [WashHands(), StandApart(default_distance=25)]
    if level == 1:
        return [WashHands(), StandApart(default_distance=75), WearMask()]
    if level == 2:
        return [WashHands(), SocialDistancing(default_distance=100), WearMask()]
    if level == 3:
        return [WashHands(), SocialDistancing(default_distance=150), WearMask(), GetTested(), Quarantine()]
    raise AssertionError(f"cannot generate behavior modifiers for rec level: {level}")
