"""
A list of constants to be consistent across different scripts.
"""

# age bins
AGE_BIN_WIDTH_5           = [(0,4), (5,9), (10,14), (15,19), (20,24), (25,29), (30,34), (35,39), (40,44), (45,49), (50,54), (55,59), (60,64), (65,69), (70,74), (75,110)]
AGE_BIN_WIDTH_10          = [(0,9), (10,19), (20,29), (30,39), (40,49), (50,59), (60,69), (70,79), (80,110)]

# locations
ALL_LOCATIONS             = ["HOUSEHOLD", "SENIOR_RESIDENCE", "WORKPLACE", "STORE", "MISC", "HOSPITAL", "PARK", "SCHOOL"]

# day of week lists
WEEKDAYS = [0, 1, 2, 3, 4]
WEEKENDS = [5, 6]
ALL_DAYS = [0, 1, 2, 3, 4, 5, 6]

# triggers for taking test
TAKE_TEST_DUE_TO_SELF_DIAGNOSIS = "self-diagnosis"
TAKE_TEST_DUE_TO_RANDOM_REASON = "random"
TAKE_TEST_DUE_TO_RECOMMENDATION = "recommended"

# triggers for behavior changes
TEST_TAKEN = "test-taken"
SELF_DIAGNOSIS = "self-diagnosis"
RISK_LEVEL_UPDATE = "risk-level-update"

# test results
NEGATIVE_TEST_RESULT = "negative"
POSITIVE_TEST_RESULT = "positive"

# quarantine behaviors
QUARANTINE_UNTIL_TEST_RESULT = f"{TEST_TAKEN}-{NEGATIVE_TEST_RESULT}"
QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT = f"{TEST_TAKEN}-{POSITIVE_TEST_RESULT}"

TRACED_BY_POSITIVE_TEST = "traced-positive-test"
TRACED_BY_SELF_REPORTED_SYMPTOMS = "traced-self-reported"
MAX_RISK_LEVEL_TRACED = "max-risk-traced"

ALL_QUARANTINE_TRIGGERS = [QUARANTINE_UNTIL_TEST_RESULT, QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT, SELF_DIAGNOSIS, TRACED_BY_SELF_REPORTED_SYMPTOMS, TRACED_BY_SELF_REPORTED_SYMPTOMS, MAX_RISK_LEVEL_TRACED]
UNSET_QUARANTINE = "unset-quarantine"
QUARANTINE_HOUSEHOLD = "quarantine-household"

# other behaviors
INITIALIZED_BEHAVIOR = "initialization"
INTERVENTION_START_BEHAVIOR = "intervention_start"

# See commentary in covid19sim.native._native.
from covid19sim.native._native import (SECONDS_PER_MINUTE,
                                       SECONDS_PER_HOUR,
                                       SECONDS_PER_DAY,
                                       SECONDS_PER_EPHEMERIS_DAY,
                                       SECONDS_PER_WEEK,
                                       SECONDS_PER_YEAR,
                                       SECONDS_PER_TROPICAL_YEAR,
                                       SECONDS_PER_LEAP_YEAR)
