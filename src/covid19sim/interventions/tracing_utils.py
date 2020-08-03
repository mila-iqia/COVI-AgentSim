from orderedset import OrderedSet
# from covid19sim.interventions.behaviors import *


def get_tracing_method(risk_model, conf):
    """
    Returns an appropriate `BaseMethod` corresponding to `risk_model`

    Args:
        risk_model (str): risk model that tracing method uses
        conf (dict): yaml configuration of the experiment

    Returns:
        (covid19sim.interventions.tracing.BaseMethod): Tracing method with relevant implementatoin of compute_risk
    """
    from covid19sim.interventions.tracing import BaseMethod, Heuristic, BinaryDigitalTracing
    if risk_model == "transformer":
        return BaseMethod(conf)
    elif risk_model == "heuristicv1":
        return Heuristic(version=1, conf=conf)
    elif risk_model == "heuristicv2":
        return Heuristic(version=2, conf=conf)
    elif risk_model == "digital":
        return BinaryDigitalTracing(conf)
    else:
        raise NotImplementedError


# def create_behavior(key, conf):
#     if key == "WearMask":
#         return WearMask(conf.get("MASKS_SUPPLY"))
#     elif key == "SocialDistancing":
#         return SocialDistancing()
#     elif key == "Quarantine":
#         return Quarantine()
#     elif key == "WashHands":
#         return WashHands()
#     elif key == "StandApart":
#         return StandApart()
#     else:
#         raise NotImplementedError

# def _get_behaviors_for_level(level):
#     """
#     Maps recommendation level to a list of `Behavior` objects.
#
#     Args:
#         level (int): recommendation level.
#
#     Returns:
#         list: a list of `Behavior`.
#     """
#     if level == 0:
#         return [WashHands(), StandApart(default_distance=25)]
#     if level == 1:
#         return [WashHands(), StandApart(default_distance=75), WearMask()]
#     if level == 2:
#         return [WashHands(), SocialDistancing(default_distance=100), WearMask()]
#     if level == 3:
#         return [WashHands(), SocialDistancing(default_distance=150), WearMask(), GetTested(), Quarantine()]
#     raise AssertionError(f"cannot generate behavior modifiers for rec level: {level}")
