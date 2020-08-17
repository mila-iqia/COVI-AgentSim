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


def compute_p_covid_given_symptoms(human, conf, correction_factor=2):
    """
    Computes probability of COVID given symptoms being experienced by `human`.

    Args:
        human (covid19sim.human.Human): `human` object
        correction_factor (float): factor to correct for underestimation of prevalence via hospitalization, deaths or testing. Needs to be calibrated.

    Returns:
        (float): probability of having COVID
    """
    symptoms = human.all_reported_symptoms
    p_each_symptom_given_covid = conf['P_REPORTED_SYMPTOM_GIVEN_COVID']
    p_each_symptom = conf['P_REPORTED_SYMPTOM']

    # (experimental) joint probability of observing symptoms
    p_symptoms_given_covid = max(p_each_symptom_given_covid.get(x.name, 0) for x in symptoms)
    p_symptoms = max(p_each_symptom.get(x.name, 0) for x in symptoms)
    p_symptoms_given_not_covid = max(0, p_symptoms - p_symptoms_given_covid)

    # probability of having covid
    covid_prevalence = human.city.tracker.get_estimated_covid_prevalence()
    p_covid = correction_factor * max(covid_prevalence['estimation_by_test'], covid_prevalence['estimation_by_hospitalization'])

    # probability of covid given symptoms
    p_covid_given_symptoms = p_symptoms_given_covid * p_covid
    p_not_covid_given_symptoms = p_symptoms_given_not_covid * (1 - p_covid)

    return p_covid_given_symptoms / (p_covid_given_symptoms + p_not_covid_given_symptoms + 1e-6)
