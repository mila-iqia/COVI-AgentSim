"""
Utility functions to interface between interventions and rest of the code.
"""

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
    elif risk_model == "heuristicv3":
        return Heuristic(version=3, conf=conf)
    elif risk_model == "digital":
        return BinaryDigitalTracing(conf)
    else:
        raise NotImplementedError
