"""
Utility functions to interface between interventions and rest of the code.
"""
import numpy as np

ALL_INTERVENTIONS = ['NO_INTERVENTION', 'LOCKDOWN', 'BDT1', 'BDT2', 'BDT1S', 'BDT2S', 'ORACLE', 'TRANSFORMER']

def get_intervention_conf(conf, name=None):
    """
    Prepares and returns an appropriate dictionary corresponding to `name` by selectively picking variables from `conf`.
    Parameters like DROPOUT and QUARANTINE_DAYS are read directly from conf, however, we return their flags here depending on `name`.

    Args:
        name (str): name of the intervention. None or empty string if its "UNMITIGATED".
        conf (dict): yaml configuration of the experiment

    Returns:
        (dict): configuration dictionary holding variables to implement intervention for `name`
    """
    base_parameters = {
        "NAME": name,
        # default risk model is none
        "RISK_MODEL": "",
        # quarantine parameters
        "QUARANTINE_HOUSEHOLD_UPON_INDIVIDUAL_POSITIVE_TEST" : conf["QUARANTINE_HOUSEHOLD_UPON_INDIVIDUAL_POSITIVE_TEST"],
        "QUARANTINE_SELF_REPORTED_INDIVIDUALS": conf["QUARANTINE_SELF_REPORTED_INDIVIDUALS"],
        "QUARANTINE_HOUSEHOLD_UPON_SELF_REPORTED_INDIVIDUAL": conf["QUARANTINE_HOUSEHOLD_UPON_SELF_REPORTED_INDIVIDUAL"],
        # rest after intervention parameters are set to 0 for unmitigated
        "ASSUME_NO_ENVIRONMENTAL_INFECTION_AFTER_INTERVENTION_START": False,
        "ASSUME_NO_UNKNOWN_INTERACTIONS_AFTER_INTERVENTION_START": False,
        "ASSUME_SAFE_HOSPITAL_DAILY_INTERACTIONS_AFTER_INTERVENTION_START": False
    }

    # unmitigated scenario
    if (
        name is None
        or name == ""
        or name.upper() == "UNMITIGATED"
    ):
        base_parameters.update({
            "NAME": "UNMITIGATED",
            "N_BEHAVIOR_LEVELS": 2,
            "INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS": False,
            "SHOULD_MODIFY_BEHAVIOR": True,
            "APP_REQUIRED": False
        })
        return base_parameters

    #
    base_parameters.update({
        "ASSUME_NO_ENVIRONMENTAL_INFECTION_AFTER_INTERVENTION_START": conf['ASSUME_NO_ENVIRONMENTAL_INFECTION_AFTER_INTERVENTION_START'],
        "ASSUME_NO_UNKNOWN_INTERACTIONS_AFTER_INTERVENTION_START": conf['ASSUME_NO_UNKNOWN_INTERACTIONS_AFTER_INTERVENTION_START'],
        "ASSUME_SAFE_HOSPITAL_DAILY_INTERACTIONS_AFTER_INTERVENTION_START": conf['ASSUME_SAFE_HOSPITAL_DAILY_INTERACTIONS_AFTER_INTERVENTION_START']
    })

    if name == "LOCKDOWN":
        base_parameters.update({
            "N_BEHAVIOR_LEVELS": 2,
            "INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS": True,
            "SHOULD_MODIFY_BEHAVIOR": True,
            "APP_REQUIRED": False
        })
        return base_parameters

    # load base parameters for tracing
    base_parameters.update({
        # (behavior specification)
        "N_BEHAVIOR_LEVELS": conf['N_BEHAVIOR_LEVELS'],
        "INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS": conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS'],
    })

    if name == "POST_LOCKDOWN_NO_TRACING":
        base_parameters.update({
            "SHOULD_MODIFY_BEHAVIOR": True,
            "APP_REQUIRED": False
        })
        return base_parameters

    # load base parameters for tracing
    base_parameters.update({
        "SHOULD_MODIFY_BEHAVIOR": conf['SHOULD_MODIFY_BEHAVIOR'],
        # risk /rec levels
        "REC_LEVEL_THRESHOLDS" : conf['REC_LEVEL_THRESHOLDS'],
        "MAX_RISK_LEVEL": conf["MAX_RISK_LEVEL"],
        "RISK_MAPPING": conf["RISK_MAPPING"],
        #
        "TRACING_N_DAYS_HISTORY": conf["TRACING_N_DAYS_HISTORY"],
        #
        "APP_UPTAKE": conf["APP_UPTAKE"]
    })

    # update the base parameters with risk compuation
    if name == "BDT":
        base_parameters.update({
                    # (risk model)
                    "RISK_MODEL": "digital",
                    # (digital tracing parameters)
                    "TRACING_ORDER": conf['TRACING_ORDER'],
                    "TRACE_SELF_REPORTED_INDIVIDUAL": conf['TRACE_SELF_REPORTED_INDIVIDUAL'],
                    "TRACED_DAYS_FOR_SELF_REPORTED_INDIVIDUAL": conf["TRACED_DAYS_FOR_SELF_REPORTED_INDIVIDUAL"],
                    # (recommendation levels)
                    "REC_LEVEL_THRESHOLDS": [0, 0, 1],
                    # (app flag)
                    "APP_REQUIRED": True
                })
        return base_parameters

    if name == "HEURISTIC":
        base_parameters.update({
                    # (risk model)
                    "RISK_MODEL": "heuristic",
                    # (heuristic parameters)
                    "HEURISTIC_VERSION": conf['HEURISTIC_VERSION'],
                    # (app flag)
                    "APP_REQUIRED": True
                })
        return base_parameters

    if name == "TRANSFORMER":
        base_parameters.update({
                    # (risk model)
                    "RISK_MODEL": "transformer",
                    # (transformer parameters)
                    "USE_ORACLE": False,
                    # (app flag)
                    "APP_REQUIRED": True
                })

    if name == "ORACLE":
        base_parameters.update({
                    # (risk model)
                    "RISK_MODEL": "transformer",
                    # (transformer parameters)
                    "USE_ORACLE": True,
                    # (app flag)
                    "APP_REQUIRED": True
                })

        return base_parameters

    raise ValueError(f"Unknown intervention name:{name}")

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

    elif risk_model == "heuristic":
        return Heuristic(version=conf['HEURISTIC_VERSION'], conf=conf)

    elif risk_model == "digital":
        return BinaryDigitalTracing(conf)

    else:
        raise NotImplementedError

def get_intervention_string(conf):
    """
    Consolidates all the parameters to one single string.

    Args:
        conf (dict): yaml configuration of the experiment

    Returns:
        (str): a string to identify type of intervention being run

    Raises:
        (ValueError): if RISK_MODEL is unknown
    """
    risk_model = conf['RISK_MODEL']
    n_behavior_levels = conf['N_BEHAVIOR_LEVELS']

    #
    if risk_model == "":
        type_of_run = f"{conf['NAME']} | N_BEHAVIOR_LEVELS:{n_behavior_levels} |"
        type_of_run += f" N_LEVELS_USED: 2 (1st and last) |"
        return type_of_run

    #
    type_of_run = f"{conf['NAME']} | RISK_MODEL: {risk_model} | N_BEHAVIOR_LEVELS:{n_behavior_levels} |"
    if risk_model == "digital":
        type_of_run += f" N_LEVELS_USED: 2 (1st and last) |"
        type_of_run += f" TRACING_ORDER:{conf['TRACING_ORDER']} |"
        type_of_run += f" TRACE_SELF_REPORTED_INDIVIDUAL: {conf['TRACE_SELF_REPORTED_INDIVIDUAL']} |"
        type_of_run += f" INTERPOLATE_USING_LOCKDOWN_CONTACTS:{conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']} |"
        type_of_run += f" MODIFY_BEHAVIOR: {conf['SHOULD_MODIFY_BEHAVIOR']}"
        return type_of_run

    if risk_model == "transformer":
        type_of_run += f" USE_ORACLE: {conf['USE_ORACLE']}"
        type_of_run += f" N_LEVELS_USED: {n_behavior_levels} |"
        type_of_run += f" INTERPOLATE_USING_LOCKDOWN_CONTACTS:{conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']} |"
        type_of_run += f" REC_LEVEL_THRESHOLDS: {conf['REC_LEVEL_THRESHOLDS']} |"
        type_of_run += f" MAX_RISK_LEVEL: {conf['MAX_RISK_LEVEL']} |"
        type_of_run += f" MODIFY_BEHAVIOR: {conf['SHOULD_MODIFY_BEHAVIOR']} "
        type_of_run += f"\n RISK_MAPPING: {np.array(conf['RISK_MAPPING'])}\n"
        return type_of_run

    if risk_model == "heuristic":
        type_of_run += f" HEURISTIC_VERSION: {conf['HEURISTIC_VERSION']} |"
        type_of_run += f" N_LEVELS_USED: {n_behavior_levels} |"
        type_of_run += f" INTERPOLATE_USING_LOCKDOWN_CONTACTS:{conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']} |"
        type_of_run += f" MAX_RISK_LEVEL: {conf['MAX_RISK_LEVEL']} |"
        type_of_run += f" MODIFY_BEHAVIOR: {conf['SHOULD_MODIFY_BEHAVIOR']}"
        return type_of_run

    raise ValueError(f"Unknown risk model:{risk_model}")

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
