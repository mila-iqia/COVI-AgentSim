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

# See commentary in covid19sim.native._native.
from covid19sim.native._native import (SECONDS_PER_MINUTE,
                                       SECONDS_PER_HOUR,
                                       SECONDS_PER_DAY,
                                       SECONDS_PER_EPHEMERIS_DAY,
                                       SECONDS_PER_WEEK,
                                       SECONDS_PER_YEAR,
                                       SECONDS_PER_TROPICAL_YEAR,
                                       SECONDS_PER_LEAP_YEAR)
