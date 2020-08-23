"""
Utility functions to interface between interventions and rest of the code.
"""
from covid19sim.utils.constants import TEST_TAKEN, SELF_DIAGNOSIS, RISK_LEVEL_UPDATE
from covid19sim.utils.constants import QUARANTINE_UNTIL_TEST_RESULT, QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT
from covid19sim.utils.constants import UNSET_QUARANTINE, QUARANTINE_HOUSEHOLD


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


def get_household_quarantine_duration(human, triggers, conf):
    """
    Returns quarantine duration required for secondary cases in household. It is determined based on the quarantine triggers.

    Args:
        human (covid19sim.human.Human): `human` for whom quarantine duration for household members need to be determined
        triggers (list): list of quarantine reasons that `human` has had in the past
        conf (dict): yaml configuration of the experiment

    Returns:
        (float): duration (days) for which household needs to quarantine due to the triggers of `human`
    """
    assert len(triggers) > 0, "no quarantine trigger for index case found"
    assert QUARANTINE_UNTIL_TEST_RESULT not in triggers, "quarantining for negative test reult should be handled separately"
    assert not QUARANTINE_HOUSEHOLD in triggers, f"no household quarantine requirements for household quarantine trigger are defined . Triggers:{triggers}"
    assert not UNSET_QUARANTINE in triggers, f"no household quarantine requirements for unset quarantine trigger are defined. Triggers:{triggers}"

    if QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT in triggers:
        return conf['QUARANTINE_DAYS_HOUSEHOLD_ON_INDIVIDUAL_POSITIVE_TEST']

    if QUARANTINE_DUE_TO_SELF_DIAGNOSIS in triggers:
        return conf['QUARANTINE_DAYS_HOUSEHOLD_ON_INDIVIDUAL_SELF_REPORTED_SYMPTOMS']

    return duration


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
