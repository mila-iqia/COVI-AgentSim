import covid19sim.epidemiology._refactored_symptom_helpers as sh
from covid19sim.epidemiology._refactored_symptom_helpers import HealthState


HEALTHY = HealthState.make("undefined", "no_symptoms", "healthy", "healthy")

# Flu
FATIGUE_AT_FLU_ONSET = HealthState.parse("fatigue at flu onset")
FEVER_AT_FLU_ONSET = HealthState.parse("fever at flu onset")
ACHES_AT_FLU_ONSET = HealthState.parse("aches at flu onset")
HARD_TIME_WAKING_UP_AT_FLU_ONSET = HealthState.parse("hard_time_waking_up at flu onset")
