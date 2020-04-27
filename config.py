# NOISE IN SIM PARAMETERS
LOCATION_TECH = 'gps' # &location-tech

# CITY PARAMETERS
MIN_AVG_HOUSE_AGE = 15

# n - people per location
LOCATION_DISTRIBUTION = {
    "store":{
        "n" : 50,
        "area": 0.15,
        "social_contact_factor": 0.6,
        "surface_prob": [0.1, 0.1, 0.3, 0.2, 0.3],
        "rnd_capacity": (30, 50),
    },
    "workplace": {
        "n" : 50,
        "area": 0.2,
        "social_contact_factor": 0.3,
        "surface_prob": [0.1, 0.1, 0.3, 0.2, 0.3],
        "rnd_capacity": None,
    },
    "school":{
        "n" : 100,
        "area": 0.05,
        "social_contact_factor": 0.8,
        "surface_prob": [0.1, 0.1, 0.3, 0.2, 0.3],
        "rnd_capacity": None,
    },
    "senior_residency":{
        "n" : 100,
        "area": 0.05,
        "social_contact_factor": 0.8,
        "surface_prob": [0.1, 0.1, 0.3, 0.2, 0.3],
        "rnd_capacity": None,
    },
    "household":{
        "n" : 2.6,
        "area": 0.30,
        "social_contact_factor": 1.0,
        "surface_prob": [0.1, 0.1, 0.3, 0.2, 0.3],
        "rnd_capacity": None,
    },
    "park":{
        "n" : 50,
        "area": 0.05,
        "social_contact_factor": 0.2,
        "surface_prob": [0.8, 0.05, 0.05, 0.05, 0.05],
        "rnd_capacity": None,
    },
    "misc":{
        "n" : 30,
        "area": 0.15,
        "social_contact_factor": 0.8,
        "surface_prob": [0.1, 0.1, 0.3, 0.2, 0.3],
        "rnd_capacity": (30,50),
    },
    "hospital":{
        "n": 100,
        "area": 0.05,
        "social_contact_factor": 0.4,
        "surface_prob": [0.0, 0.0, 0.0, 0.0, 1.0],
        "rnd_capacity": (40,100)
    }
}

# house_size: 1 2 3 4 5
OTHERS_WORKPLACE_CHOICE=[1.0, 0.0, 0.0]
HOUSE_SIZE_PREFERENCE = [0.30, 0.30, 0.15, 0.15, 0.1]
HUMAN_DISTRIBUTION = {
    (1,15): {
        "p":0.04,
        "residence_preference":{
            "house_size":[0.0, 0.2, 0.3, 0.3, 0.2],
            "senior_residency":0.0
        },
        "profession_profile":{
            "healthcare":0.0,
            "school":1.0,
            "others":0.0,
            "retired":0.0
        }
    },
    (15,20):{
        "p":0.01,
        "residence_preference":{
            "house_size":[0.05, 0.05, 0.1, 0.3, 0.5],
            "senior_residency":0.0
        },
        "profession_profile":{
            "healthcare":0.0,
            "school":0.9,
            "others":0.1,
            "retired":0.0
        }
    },
    (20,40):{
        "p":0.28,
        "residence_preference":{
            "house_size":[0.2, 0.3, 0.25, 0.15, 0.1],
            "senior_residency":0.0
        },
        "profession_profile": {
                "healthcare":0.1,
                "school":0.1,
                "others":0.8,
                "retired":0.0
        },

    },
    (40,60):{
        "p":0.36,
        "residence_preference":{
            "house_size":[0.05, 0.3, 0.3, 0.15, 0.2],
            "senior_residency":0.0
        },
        "profession_profile": {
                "healthcare":0.1,
                "school":0.05,
                "others":0.85,
                "retired":0.0
        },

    },
    (60,80):{
        "p":0.24,
        "residence_preference":{
            "house_size":[0.1, 0.4, 0.2, 0.2, 0.1],
            "senior_residency":0.7
        },
        "profession_profile": {
                "healthcare":0.05,
                "school":0.0,
                "others":0.9,
                "retired":0.05
        },

    },
    (80,100):{
        "p":0.07,
        "residence_preference":{
            "house_size":[0.05, 0.5, 0.1, 0.25, 0.1],
            "senior_residency":0.9
        },
        "profession_profile":{
                "healthcare":0.0,
                "school":0.0,
                "others":0.0,
                "retired":1.0
        },

    }
}

# INDIVIDUAL DIFFERENCES PARAMETERS
WORK_FROM_HOME = False
P_HAS_APP = 0.5 # &has_app
P_CAREFUL_PERSON = 0.3 # &carefulness

# DISEASE PARAMETERS
AVG_INCUBATION_DAYS = 5 # &avg-incubation-days
SCALE_INCUBATION_DAYS = 0.5
INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET = 2.5 # relative to incubation days
AVG_RECOVERY_DAYS = 14
SCALE_RECOVERY_DAYS = 4
INFECTION_RADIUS = 200 # cms
INFECTION_DURATION = 15 # minutes
#                   0-9 10-19 20-29  30-39  40-49  50-59 60-69 70-79  80-  # Assuming dath rate to be same for 80 and above
P_NEVER_RECOVERS = [0, 0.002, 0.002, 0.002, 0.004, 0.02, 0.04, 0.08, 0.15] # &never_recovers
REINFECTION_POSSIBLE = False # [0, 1]
# aerosol    copper      cardboard       steel       plastic
MAX_DAYS_CONTAMINATION = [0.125, 1.0/3.0, 1, 2, 3] # &envrionmental contamination
VIRAL_LOAD_MIN = 0.0001


# convex combination - if it sums to 1
INFECTION_DISTANCE_FACTOR = 0.0
INFECTION_DURATION_FACTOR = 0.0

#TESTING
# capacity is per day; time_to_result is per day
TEST_TYPES = {
    "lab": {
        "capacity": 100,
        "time_to_result":2,
        "P_FALSE_NEGATIVE":0.1, #&false-negative,
        "preference":1
    }
}
P_TEST = 0.5
TEST_DAYS = 3

# VIRAL LOAD PARAMS
MIN_VIRAL_LOAD = 0.1
MAX_VIRAL_LOAD = 0.4

PLATEAU_START_MEAN=2.5
PLATEAU_START_STD=0.25
PLATEAU_START_CLIP_LOW = 2
PLATEAU_START_CLIP_HIGH = 3

PLATEAU_DURATION_MEAN=5.5
PLEATEAU_DURATION_STD=1
PLATEAU_DURATION_CLIP_LOW = 3.
PLATEAU_DURATION_CLIP_HIGH = 9.

RECOVERY_MEAN = 6
RECOVERY_STD = 1
RECOVERY_CLIP_LOW = 2.5
RECOVERY_CLIP_HIGH = 10

# INCUBATION PARAMS
SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_AVG = 0.6 # DAYS
SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_STD = 0.1
INFECTIOUSNESS_ONSET_DAYS_AVG = 2.5
INFECTIOUSNESS_ONSET_DAYS_STD = 0.1

# ASYMTPOMATIC
BASELINE_P_ASYMPTOMATIC = 15 # &p-asymptomatic
ASYMPTOMATIC_INFECTION_RATIO = 0.1 # &prob_infectious

# OTHER TRANSMISSIBLE DISEASES
P_FLU = 0.05 # &p-flu
FLU_CONTAGIOUSNESS = 0.05
FLU_INCUBATION = 1
FLU_RECOVERY_START = 7
FLU_RECOVERY_DURATION = 14

P_COLD = 0.1 # &p-cold
COLD_CONTAGIOUSNESS = 0.05
COLD_INCUBATION = 1
COLD_RECOVERY_START = 5
COLD_RECOVERY_DURATION = 10

# MASK
MASK_INTERVENTION = False
MASK_EFFICACY_NORMIE = 0.32
MASK_EFFICACY_HEALTHWORKER = 0.98
BASELINE_P_MASK = 0.5

# SIMULATION PARAMETERS
TICK_MINUTE = 2  # @param increment
SIMULATION_DAYS = 30  # @param
SYMPTOM_DAYS = 5  # @param
COLLECT_LOGS = True

# LIFESTYLE PARAMETERS
## SHOP
AVG_SHOP_TIME_MINUTES = 30 # @param
SCALE_SHOP_TIME_MINUTES = 15
AVG_SCALE_SHOP_TIME_MINUTES =  10
SCALE_SCALE_SHOP_TIME_MINUTES = 5
NUM_WEEKLY_GROCERY_RUNS = 2 # @param

AVG_MAX_NUM_SHOP_PER_WEEK = 5
SCALE_MAX_NUM_SHOP_PER_WEEK = 1

AVG_NUM_SHOPPING_DAYS = 3
SCALE_NUM_SHOPPING_DAYS = 1
AVG_NUM_SHOPPING_HOURS = 3
SCALE_NUM_SHOPPING_HOURS = 1

## WORK
AVG_WORKING_MINUTES = 8 * 60
SCALE_WORKING_MINUTES = 1 * 60
AVG_SCALE_WORKING_MINUTES = 2 * 60
SCALE_SCALE_WORKING_MINUTES = 1 * 60

## HOSPITAL
AVG_HOSPITAL_HOURS = 7 * 24
SCALE_HOSPITAL_HOURS = 24
AVG_SCALE_HOSPITAL_HOURS = 12
SCALE_SCALE_HOSPITAL_HOURS = 6

## EXERCISE
AVG_EXERCISE_MINUTES = 60
SCALE_EXERCISE_MINUTES = 15
AVG_SCALE_EXERCISE_MINUTES = 15
SCALE_SCALE_EXERCISE_MINUTES = 5

AVG_MAX_NUM_EXERCISE_PER_WEEK = 5
SCALE_MAX_NUM_EXERCISE_PER_WEEK = 2
AVG_NUM_EXERCISE_DAYS = 3
SCALE_NUM_EXERCISE_DAYS = 1
AVG_NUM_EXERCISE_HOURS = 3
SCALE_NUM_EXERCISE_HOURS = 1

## MISC
AVG_MISC_MINUTES = 60
SCALE_MISC_MINUTES = 15
AVG_SCALE_MISC_MINUTES = 15
SCALE_SCALE_MISC_MINUTES = 5

# DISTANCE_ENCOUNTER PARAMETERS
MIN_DIST_ENCOUNTER = 20
MAX_DIST_ENCOUNTER = 400

# RISK MODEL PARAMETERS
RISK_TRANSMISSION_PROBA = 0.01
RISK_WITH_TRUE_SYMPTOMS = False
CLIP_RISK = False

# KNOBS
CONTAGION_KNOB = 1.0
ENVIRONMENTAL_INFECTION_KNOB = 0.0005
