# -*- coding: utf-8 -*-
import simpy

import itertools
import numpy as np
from collections import defaultdict, namedtuple
import datetime
from bitarray import bitarray
import operator
import math

from utils import _normalize_scores, _get_random_sex, _get_covid_symptoms, \
    _get_preexisting_conditions, _draw_random_discreet_gaussian, _json_serialize, _sample_viral_load_piecewise, \
    _get_random_area, _encode_message, _decode_message, float_to_binary, binary_to_float, _reported_symptoms, \
    _get_mask_wearing,  _get_cold_symptoms_v2, _get_flu_symptoms_v2, _reported_symptoms, proba_to_risk_fn

from config import *
from base import *
from interventions import *
if COLLECT_LOGS is False:
    Event = DummyEvent

_proba_to_risk_level = proba_to_risk_fn(np.exp(np.load(RISK_MAPPING_FILE)))

class Visits(object):

    def __init__(self):
        self.parks = defaultdict(int)
        self.stores = defaultdict(int)
        self.hospitals = defaultdict(int)
        self.miscs = defaultdict(int)

    @property
    def n_parks(self):
        return len(self.parks)

    @property
    def n_stores(self):
        return len(self.stores)

    @property
    def n_hospitals(self):
        return len(self.hospitals)

    @property
    def n_miscs(self):
        return len(self.miscs)


class Human(object):

    def __init__(self, env, name, age, rng, infection_timestamp, household, workplace, profession, rho=0.3, gamma=0.21, symptoms=[],
                 test_results=None):

        self.env = env
        self._events = []
        self.name = f"human:{name}"
        self.rng = rng
        self.profession = profession
        self.is_healthcare_worker = True if profession == "healthcare" else False
        self.assign_household(household)
        self.workplace = workplace
        self.rho = rho
        self.gamma = gamma

        self.age = age
        self.sex = _get_random_sex(self.rng)
        self.preexisting_conditions = _get_preexisting_conditions(self.age, self.sex, self.rng)

        age_modifier = 2 if self.age > 40 or self.age < 12 else 2
        # &carefulness
        if self.rng.rand() < P_CAREFUL_PERSON:
            self.carefulness = (round(self.rng.normal(55, 10)) + self.age/2) / 100
        else:
            self.carefulness = (round(self.rng.normal(25, 10)) + self.age/2) / 100

        self.has_app = self.rng.rand() < (P_HAS_APP / age_modifier) + (self.carefulness / 2)

        # logged info can be quite different
        self.has_logged_info = self.has_app and self.rng.rand() < self.carefulness
        self.obs_is_healthcare_worker = True if self.is_healthcare_worker and rng.random()<0.9 else False # 90% of the time, healthcare workers will declare it
        self.obs_age = self.age if self.has_app and self.has_logged_info else None
        self.obs_sex = self.sex if self.has_app and self.has_logged_info else None
        self.obs_preexisting_conditions = self.preexisting_conditions if self.has_app and self.has_logged_info else None

        self.rest_at_home = False # to track mobility due to symptoms
        self.visits = Visits()
        self.travelled_recently = self.rng.rand() > 0.9

        # &symptoms, &viral-load
        # probability of being asymptomatic is basically 50%, but a bit less if you're older and a bit more if you're younger
        self.is_asymptomatic = self.rng.rand() < (BASELINE_P_ASYMPTOMATIC - (self.age - 50) * 0.5) / 100
        self.asymptomatic_infection_ratio = ASYMPTOMATIC_INFECTION_RATIO if self.is_asymptomatic else 0.0 # draw a beta with the distribution in documents
        self.viral_load_plateau_height, self.viral_load_plateau_start, self.viral_load_plateau_end, self.viral_load_recovered = _sample_viral_load_piecewise(rng, age=age)
        self.infectiousness_onset_days = self.rng.normal(loc=INFECTIOUSNESS_ONSET_DAYS_AVG, scale=INFECTIOUSNESS_ONSET_DAYS_STD)
        self.incubation_days = self.infectiousness_onset_days + self.viral_load_plateau_start + self.rng.normal(loc=SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_AVG, scale=SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_STD)
        self.recovery_days = self.infectiousness_onset_days + self.viral_load_recovered
        self.test_result, self.test_type = None, None

        # Indicates whether this person will show severe signs of illness.
        self.infection_timestamp = infection_timestamp
        self.cold_timestamp = self.env.timestamp if self.rng.random() < P_COLD else None
        self.flu_timestamp = self.env.timestamp if self.rng.random() < P_FLU else None # different from asymptomatic
        self.recovered_timestamp = datetime.datetime.min
        self.is_immune = False # different from asymptomatic
        self.can_get_really_sick = self.rng.random() >= 0.8 + (age/100)
        self.can_get_extremely_sick = self.can_get_really_sick and self.rng.random() >= 0.7 # &severe; 30% of severe cases need ICU
        self.never_recovers = self.rng.random() <= P_NEVER_RECOVERS[min(math.floor(self.age/10),8)]
        self.obs_hospitalized = False
        self.obs_in_icu = False
        if self.infection_timestamp is not None: print(f"{self} is infected")

        # counters and memory
        self.r0 = []
        self.has_logged_symptoms = False
        self.has_been_tested = False
        self.last_state = self.state
        self.n_infectious_contacts = 0
        self.last_date = defaultdict(lambda : self.env.initial_timestamp.date())
        self.last_location = self.location
        self.last_duration = 0

        # interventions & risk prediction
        self.tracing = False
        self.WEAR_MASK = False

        # risk prediction
        self.risk = BASELINE_RISK_VALUE
        self.risk_level = _proba_to_risk_level(self.risk)
        self.past_N_days_contacts = [OrderedSet()]
        self.n_contacts_tested_positive = 0
        self.contact_book = Contacts(self.has_app)
        self.message_info = {'traced': False, 'receipt':datetime.datetime.max, 'delay':BIG_NUMBER}

        # symptoms
        self.symptom_start_time = None
        self.all_cold_symptoms = _get_cold_symptoms_v2(self.age, self.rng, self.carefulness, self.preexisting_conditions, self.can_get_really_sick, self.can_get_extremely_sick)
        self.all_flu_symptoms = _get_flu_symptoms_v2(self.age, self.rng, self.carefulness, self.preexisting_conditions, self.can_get_really_sick, self.can_get_extremely_sick)
        self.all_covid_symptoms = _get_covid_symptoms(self.viral_load_plateau_start, self.viral_load_plateau_end,
                                        self.viral_load_recovered, age=self.age, incubation_days=self.incubation_days,
                                                          really_sick=self.can_get_really_sick, extremely_sick=self.can_get_extremely_sick,
                          rng=self.rng, preexisting_conditions=self.preexisting_conditions)
        self.all_symptoms, self.cold_symptoms, self.flu_symptoms, self.covid_symptoms = [], [], [], []

        # habits
        self.avg_shopping_time = _draw_random_discreet_gaussian(AVG_SHOP_TIME_MINUTES, SCALE_SHOP_TIME_MINUTES, self.rng)
        self.scale_shopping_time = _draw_random_discreet_gaussian(AVG_SCALE_SHOP_TIME_MINUTES,
                                                                  SCALE_SCALE_SHOP_TIME_MINUTES, self.rng)

        self.avg_exercise_time = _draw_random_discreet_gaussian(AVG_EXERCISE_MINUTES, SCALE_EXERCISE_MINUTES, self.rng)
        self.scale_exercise_time = _draw_random_discreet_gaussian(AVG_SCALE_EXERCISE_MINUTES,
                                                                  SCALE_SCALE_EXERCISE_MINUTES, self.rng)

        self.avg_working_minutes = _draw_random_discreet_gaussian(AVG_WORKING_MINUTES, SCALE_WORKING_MINUTES, self.rng)
        self.scale_working_minutes = _draw_random_discreet_gaussian(AVG_SCALE_WORKING_MINUTES, SCALE_SCALE_WORKING_MINUTES, self.rng)

        self.avg_misc_time = _draw_random_discreet_gaussian(AVG_MISC_MINUTES, SCALE_MISC_MINUTES, self.rng)
        self.scale_misc_time = _draw_random_discreet_gaussian(AVG_SCALE_MISC_MINUTES, SCALE_SCALE_MISC_MINUTES, self.rng)

        #getting the number of shopping days and hours from a distribution
        self.number_of_shopping_days = _draw_random_discreet_gaussian(AVG_NUM_SHOPPING_DAYS, SCALE_NUM_SHOPPING_DAYS, self.rng)
        self.number_of_shopping_hours = _draw_random_discreet_gaussian(AVG_NUM_SHOPPING_HOURS, SCALE_NUM_SHOPPING_HOURS, self.rng)

        #getting the number of exercise days and hours from a distribution
        self.number_of_exercise_days = _draw_random_discreet_gaussian(AVG_NUM_EXERCISE_DAYS, SCALE_NUM_EXERCISE_DAYS, self.rng)
        self.number_of_exercise_hours = _draw_random_discreet_gaussian(AVG_NUM_EXERCISE_HOURS, SCALE_NUM_EXERCISE_HOURS, self.rng)

        #Multiple shopping days and hours
        self.shopping_days = self.rng.choice(range(7), self.number_of_shopping_days)
        self.shopping_hours = self.rng.choice(range(7, 20), self.number_of_shopping_hours)

        #Multiple exercise days and hours
        self.exercise_days = self.rng.choice(range(7), self.number_of_exercise_days)
        self.exercise_hours = self.rng.choice(range(7, 20), self.number_of_exercise_hours)

        #Limiting the number of hours spent shopping per week
        self.max_shop_per_week = _draw_random_discreet_gaussian(AVG_MAX_NUM_SHOP_PER_WEEK, SCALE_MAX_NUM_SHOP_PER_WEEK, self.rng)
        self.count_shop=0

        # Limiting the number of hours spent exercising per week
        self.max_exercise_per_week = _draw_random_discreet_gaussian(AVG_MAX_NUM_EXERCISE_PER_WEEK, SCALE_MAX_NUM_EXERCISE_PER_WEEK, self.rng)
        self.count_exercise=0

        self.work_start_hour = self.rng.choice(range(7, 12), 3)

    def assign_household(self, location):
        self.household = location
        self.location = location
        if self.profession == "retired":
            self.workplace = location

    def __repr__(self):
        return f"H:{self.name}, SEIR:{int(self.is_susceptible)}{int(self.is_exposed)}{int(self.is_infectious)}{int(self.is_removed)}"

    ########### MEMORY OPTIMIZATION ###########
    @property
    def events(self):
        return self._events

    def pull_events(self):
        if self._events:
            events = self._events
            self._events = []
        else:
            events = self._events
        return events

    ########### EPI ###########

    @property
    def is_susceptible(self):
        return not self.is_exposed and not self.is_infectious and not self.is_removed and not self.is_immune

    @property
    def is_exposed(self):
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp < datetime.timedelta(days=self.infectiousness_onset_days)

    @property
    def is_infectious(self):
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.infectiousness_onset_days)

    @property
    def is_removed(self):
        return self.recovered_timestamp == datetime.datetime.max

    @property
    def is_incubated(self):
        return (not self.is_asymptomatic and self.infection_timestamp is not None and
                self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.incubation_days))

    @property
    def state(self):
        return [int(self.is_susceptible), int(self.is_exposed), int(self.is_infectious), int(self.is_removed)]

    @property
    def days_since_exposed(self):
        if self.infection_timestamp is None:
            return
        return (self.env.timestamp-self.infection_timestamp ).days

    @property
    def is_really_sick(self):
        return self.can_get_really_sick and 'severe' in self.symptoms

    @property
    def is_extremely_sick(self):
        return self.can_get_extremely_sick and 'severe' in self.symptoms

    @property
    def viral_load(self):
        """ Calculates the elapsed time since infection, returning this person's current viral load"""
        if not self.infection_timestamp:
            return 0.
        # calculates the time since infection in days
        days_infectious = (self.env.timestamp - self.infection_timestamp).total_seconds() / 86400 - self.infectiousness_onset_days

        if days_infectious < 0:
            return 0.

        # implements the piecewise linear function
        if days_infectious < self.viral_load_plateau_start:
            cur_viral_load = self.viral_load_plateau_height * days_infectious / self.viral_load_plateau_start
        elif days_infectious < self.viral_load_plateau_end:
            cur_viral_load = self.viral_load_plateau_height
        else:
            cur_viral_load = self.viral_load_plateau_height - self.viral_load_plateau_height * (days_infectious - self.viral_load_plateau_end) / (self.viral_load_recovered - self.viral_load_plateau_end)

        # the viral load cannot be negative
        if cur_viral_load < 0:
            cur_viral_load = 0.

        return cur_viral_load

    @property
    def infectiousness(self):
        severity_multiplier = 1
        if self.is_infectious:
            if self.can_get_really_sick:
              severity_multiplier = 1.25
            if self.is_extremely_sick:
              severity_multiplier = 1.5
            if 'immuno-compromised' in self.preexisting_conditions:
              severity_multiplier += 0.2
            if 'cough' in self.symptoms:
              severity_multiplier += 0.25
        return self.viral_load * severity_multiplier

    @property
    def has_cold(self):
        return self.cold_timestamp is not None

    @property
    def has_flu(self):
        return self.flu_timestamp is not None

    @property
    def symptoms(self):
        if self.last_date['symptoms'] != self.env.timestamp.date():
            self.last_date['symptoms'] = self.env.timestamp.date()
            self.update_symptoms()
        return self.all_symptoms

    @property
    def all_reported_symptoms(self):
        if not self.has_app:
            return []

        reported_symptoms = []
        for symptom in self.symptoms:
            if self.rng.random() < self.carefulness:
                reported_symptoms.append(symptom)
        return reported_symptoms

    def update_symptoms(self):
        symptoms = []
        if self.cold_timestamp is not None:
            t = (self.env.timestamp - self.cold_timestamp).days
            if t >= len(self.all_cold_symptoms):
                self.cold_symptoms = []
            else:
                self.cold_symptoms = self.all_cold_symptoms[t]

        if self.flu_timestamp is not None:
            t = (self.env.timestamp - self.flu_timestamp).days
            if t >= len(self.all_flu_symptoms):
                self.flu_symptoms = []
            else:
                self.flu_symptoms = self.all_flu_symptoms[t]

        if self.is_incubated and not self.is_asymptomatic:
            days_since_infectious = math.floor(self.days_since_exposed - self.infectiousness_onset_days)
            self.covid_symptoms = self.all_covid_symptoms[days_since_infectious]

        all_symptoms = set(self.flu_symptoms + self.cold_symptoms + self.covid_symptoms)
        # self.new_symptoms = list(all_symptoms - set(self.all_symptoms))
        self.all_symptoms = list(all_symptoms)

    def get_tested(self, city):
        if not city.tests_available:
            return False

        # TODO: get observed data on testing / who gets tested when??
        if any(self.symptoms) and self.rng.rand() > P_TEST:
            self.test_type = city.get_available_test()
            if self.rng.rand() > TEST_TYPES[self.test_type]['P_FALSE_NEGATIVE']:
                self.test_result =  'negative'
            else:
                self.test_result =  'positive'

            if self.test_type == "lab":
                self.test_result_validated = True
            else:
                self.test_result_validated = False

            if self.has_app and self.rng.random() < self.carefulness:
                self.reported_test_result = self.test_result
                self.reported_test_type = self.test_type
            else:
                self.reported_test_result = None
                self.reported_test_type = None

            return True

        return False

    def wear_mask(self):
        if not self.WEAR_MASK:
            self.wearing_mask, self.mask_efficacy = False, 0
            return

        self.wearing_mask = False
        if self.location == self.household:
            self.wearing_mask = False

        if self.location.location_type == 'store':
            if self.carefulness > 0.6:
                self.wearing_mask = True
            elif self.rng.rand() < self.carefulness * BASELINE_P_MASK:
                self.wearing_mask = True
        elif self.rng.rand() < self.carefulness * BASELINE_P_MASK :
            self.wearing_mask = True

        # efficacy - people do not wear it properly
        if self.wearing_mask:
            if  self.workplace.location_type == 'hospital':
              self.mask_efficacy = MASK_EFFICACY_HEALTHWORKER
            else:
              self.mask_efficacy = MASK_EFFICACY_NORMIE
        else:
            self.mask_efficacy = 0

    def recover_from_cold_and_flu(self):
        if (self.cold_timestamp is not None and
            (self.env.timestamp - self.cold_timestamp) >= datetime.timedelta(days=len(self.all_cold_symptoms))):
            self.cold_timestamp = None
            self.cold_symptoms = []

        if (self.flu_timestamp is not None and
            (self.env.timestamp - self.flu_timestamp) >= datetime.timedelta(days=len(self.all_flu_symptoms))):
            self.flu_timestamp = None
            self.flu_symptoms = []

    def how_am_I_feeling(self):
        current_symptoms = self.symptoms
        if current_symptoms == []:
            return 1.0

        if sum(x in current_symptoms for x in ["severe", "extremely_severe"]) > 0:
            return 0.0

        elif self.test_result == "positive":
            return 0.05

        elif sum(x in current_symptoms for x in ["trouble_breathing"]) > 0:
            return 0.3 * (1 + self.carefulness)

        elif sum(x in current_symptoms for x in ["moderate", "mild", "fever"]) > 0:
            return self.rng.uniform(0, self.carefulness)
        #
        # elif sum(x in current_symptoms for x in ["cough", "fatigue", "gastro", "aches"]) > 0:
        #     return self.rng.random(0, self.carefulness)

        return 1.0

    def assert_state_changes(self):
        next_state = {0:[1], 1:[2], 2:[0, 3], 3:[3]}
        assert sum(self.state) == 1, f"invalid compartment for {self.name}: {self.state}"
        if self.last_state != self.state:
            # can skip the compartment if hospitalized in exposed
            if not self.obs_hospitalized:
                assert self.state.index(1) in next_state[self.last_state.index(1)], f"invalid compartment transition for {self.name}: {self.last_state} to {self.state}"
            self.last_state = self.state

    def notify(self, intervention):
        if intervention is not None and not self.notified:
            print(f"Intervention: {intervention}")
            self.tracing = False
            if isinstance(intervention, Tracing):
                self.tracing = True
                self.tracing_method = intervention
            else:
                intervention.modify_behavior(self)
            self.notified = True

    def run(self, city):
        """
           1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
           State  h h h h h h h h h sh sh h  h  h  ac h  h  h  h  h  h  h  h  h
        """
        self.household.humans.add(self)
        while True:
            hour, day = self.env.hour_of_day(), self.env.day_of_week()
            if day==0:
                self.count_exercise=0
                self.count_shop=0

            if self.last_date['run'] != self.env.timestamp.date():
                self.last_date['run'] = self.env.timestamp.date()
                Event.log_daily(self, self.env.timestamp)
                self.update_symptoms()
                city.tracker.track_symptoms(self)
                self.notify(city.intervention)

            if self.tracing and self.message_info['traced']:
                if (self.env.timestamp - self.message_info['receipt']).days > self.message_info['delay']:
                    self.update_risk_level()

            # recover health
            self.recover_from_cold_and_flu()

            # track symptoms
            if self.is_incubated and self.symptom_start_time is None :
                self.symptom_start_time = self.env.timestamp
                city.tracker.track_generation_times(self.name) # it doesn't count environmental infection or primary case or asymptomatic/presymptomatic infections; refer the definition

            # log test
            # TODO: needs better conditions; log test based on some condition on symptoms
            if (self.is_incubated and
                self.test_result != "positive" and
                self.env.timestamp - self.symptom_start_time >= datetime.timedelta(days=TEST_DAYS)):
                # make testing a function of age/hospitalization/travel
                if self.get_tested(city):
                    Event.log_test(self, self.env.timestamp)
                    self.has_been_tested = True
                    city.tracker.track_tested_results(self, self.test_result, self.test_type)
                    self.update_risk(test_results=True)

            # recover
            if self.is_infectious and self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.recovery_days):
                city.tracker.track_recovery(self.n_infectious_contacts, self.recovery_days)

                self.infection_timestamp = None # indicates they are no longer infected
                if self.never_recovers:
                    self.recovered_timestamp = datetime.datetime.max
                    self.dead = True
                else:
                    if not REINFECTION_POSSIBLE:
                        self.recovered_timestamp = datetime.datetime.max
                        self.is_immune = not REINFECTION_POSSIBLE
                    else:
                        self.recovered_timestamp = self.env.timestamp
                        self.test_result, self.test_type = None, None
                    self.never_recovers = self.rng.random() <= P_NEVER_RECOVERS[min(math.floor(self.age/10),8)]
                    self.dead = False

                self.obs_hospitalized = True
                self.all_symptoms, self.covid_symptoms = [], []
                self.update_risk(recovery=True)
                Event.log_recovery(self, self.env.timestamp, self.dead)
                if self.dead:
                    yield self.env.timeout(np.inf)

            self.assert_state_changes()

            # Mobility
            # self.how_am_I_feeling = 1.0 (great) ---> rest_at_home = False
            if not self.rest_at_home:
                # if self.is_infectious : print(f"I am feeling {self.how_am_I_feeling()}")
                # set it once for the rest of the disease path
                if self.rng.random() > self.how_am_I_feeling():
                    self.rest_at_home = True

            # happens when recovered
            elif self.rest_at_home and self.how_am_I_feeling() == 1.0:
                self.rest_at_home = False

            # if self.name == "human:69":print(f"{self} rest_at_home: {self.rest_at_home} S:{len(self.symptoms)} flu:{self.has_flu} cold:{self.has_cold}")

            if self.is_extremely_sick:
                yield self.env.process(self.excursion(city, "hospital-icu"))

            elif self.is_really_sick:
                yield self.env.process(self.excursion(city, "hospital"))

            if (not WORK_FROM_HOME and
                not self.env.is_weekend() and
                hour in self.work_start_hour and
                not self.rest_at_home):
                yield self.env.process(self.excursion(city, "work"))

            elif ( hour in self.shopping_hours and
                   day in self.shopping_days and
                   self.count_shop<=self.max_shop_per_week and
                   not self.rest_at_home):
                self.count_shop+=1
                yield self.env.process(self.excursion(city, "shopping"))

            elif ( hour in self.exercise_hours and
                    day in self.exercise_days and
                    self.count_exercise<=self.max_exercise_per_week and
                    not self.rest_at_home):
                self.count_exercise+=1
                yield  self.env.process(self.excursion(city, "exercise"))

            elif (self.env.is_weekend() and
                    self.rng.random() < 0.5 and
                    not self.rest_at_home):
                yield  self.env.process(self.excursion(city, "leisure"))

            # start from house all the time
            yield self.env.process(self.at(self.household, city, 60))

    ############################## MOBILITY ##################################
    @property
    def lat(self):
        return self.location.lat if self.location else self.household.lat

    @property
    def lon(self):
        return self.location.lon if self.location else self.household.lon


    @property
    def obs_lat(self):
        if LOCATION_TECH == 'bluetooth':
            return round(self.lat + self.rng.normal(0, 2))
        else:
            return round(self.lat + self.rng.normal(0, 10))

    @property
    def obs_lon(self):
        if LOCATION_TECH == 'bluetooth':
            return round(self.lon + self.rng.normal(0, 2))
        else:
            return round(self.lon + self.rng.normal(0, 10))

    def excursion(self, city, type):

        if type == "shopping":
            grocery_store = self._select_location(location_type="stores", city=city)
            t = _draw_random_discreet_gaussian(self.avg_shopping_time, self.scale_shopping_time, self.rng)
            with grocery_store.request() as request:
                yield request
                yield self.env.process(self.at(grocery_store, city, t))

        elif type == "exercise":
            park = self._select_location(location_type="park", city=city)
            t = _draw_random_discreet_gaussian(self.avg_exercise_time, self.scale_exercise_time, self.rng)
            yield self.env.process(self.at(park, city, t))

        elif type == "work":
            t = _draw_random_discreet_gaussian(self.avg_working_minutes, self.scale_working_minutes, self.rng)
            yield self.env.process(self.at(self.workplace, city, t))

        elif type == "hospital":
            print(f"{self} got hospitalized")
            hospital = self._select_location(location_type=type, city=city)
            if hospital is None: # no more hospitals
                self.dead = True
                self.recovered_timestamp = datetime.datetime.max
                yield self.env.timeout(np.inf)

            self.obs_hospitalized = True
            t = self.recovery_days - (self.env.timestamp - self.infection_timestamp).total_seconds() / 86400 # DAYS
            yield self.env.process(self.at(hospital, city, t * 24 * 60))

        elif type == "hospital-icu":
            print(f"{self} got icu-ed")
            icu = self._select_location(location_type=type, city=city)
            if icu is None:
                self.dead = True
                self.recovered_timestamp = datetime.datetime.max
                yield self.env.timeout(np.inf)

            if len(self.preexisting_conditions) < 2:
                extra_time = self.rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            else:
                extra_time = self.rng.choice([1, 2, 3], p=[0.2, 0.3, 0.5]) # DAYS
            t = self.viral_load_plateau_end - self.viral_load_plateau_start + extra_time

            yield self.env.process(self.at(icu, city, t * 24 * 60))

        elif type == "leisure":
            S = 0
            p_exp = 1.0
            while True:
                if self.rng.random() > p_exp:  # return home
                    yield self.env.process(self.at(self.household, city, 60))
                    break

                loc = self._select_location(location_type='miscs', city=city)
                S += 1
                p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
                with loc.request() as request:
                    yield request
                    t = _draw_random_discreet_gaussian(self.avg_misc_time, self.scale_misc_time, self.rng)
                    yield self.env.process(self.at(loc, city, t))
        else:
            raise ValueError(f'Unknown excursion type:{type}')

    def at(self, location, city, duration):
        city.tracker.track_trip(from_location=self.location.location_type, to_location=location.location_type, age=self.age, hour=self.env.hour_of_day())

        # if self.name == "human:69":
        #     print(f"{self} {self.location} --> {location}")

        self.wear_mask()

        # add the human to the location
        self.location = location
        location.add_human(self)
        self.leaving_time = duration + self.env.now
        self.start_time = self.env.now
        area = self.location.area

        # accumulate time at household
        if location == self.household:
            if self.last_location != self.household:
                self.last_duration = duration
                self.last_location = location
            else:
                self.last_duration += duration
        else:
            if self.last_location == self.household:
                city.tracker.track_social_mixing(location=self.household, duration=self.last_duration)

            self.last_location = location
            city.tracker.track_social_mixing(location=location, duration=self.last_duration)

        # Report all the encounters (epi transmission)
        for h in location.humans:
            if h == self:
                continue

            # age mixing
            if not self.rng.random() < (0.1 * abs(self.age - h.age) + 1) ** -1:
                continue

            distance =  np.sqrt(int(area/len(self.location.humans))) + self.rng.randint(MIN_DIST_ENCOUNTER, MAX_DIST_ENCOUNTER)
            # risk model
            # TODO: Add GPS measurements as conditions; refer JF's docs
            if self.tracing and MIN_MESSAGE_PASSING_DISTANCE < distance <  MAX_MESSAGE_PASSING_DISTANCE:
                self.contact_book.add(human=h, timestamp=self.env.timestamp)
                h.contact_book.add(human=self, timestamp=self.env.timestamp)

            t_overlap = min(self.leaving_time, getattr(h, "leaving_time", 60)) - max(self.start_time, getattr(h, "start_time", 60))
            t_near = self.rng.random() * t_overlap

            city.tracker.track_social_mixing(human1=self, human2=h, duration=t_near, timestamp = self.env.timestamp)
            contact_condition = distance <= INFECTION_RADIUS and t_near > INFECTION_DURATION
            if contact_condition:
                proximity_factor = 1
                if INFECTION_DISTANCE_FACTOR or INFECTION_DURATION_FACTOR:
                    proximity_factor = INFECTION_DISTANCE_FACTOR * (1 - distance/INFECTION_RADIUS) + INFECTION_DURATION_FACTOR * min((t_near - INFECTION_DURATION)/INFECTION_DURATION, 1)
                mask_efficacy = self.mask_efficacy * h.mask_efficacy

                # TODO: merge thw two clauses into one (follow cold and flu)
                infectee = None
                if self.is_infectious:
                    ratio = self.asymptomatic_infection_ratio  if self.is_asymptomatic else 1.0
                    p_infection = self.infectiousness * ratio * (1-mask_efficacy) * proximity_factor
                    x_human = self.rng.random() < p_infection * CONTAGION_KNOB

                    if x_human and h.is_susceptible:
                        h.infection_timestamp = self.env.timestamp
                        self.n_infectious_contacts+=1
                        Event.log_exposed(h, self, self.env.timestamp)
                        city.tracker.track_infection('human', from_human=self, to_human=h, location=location, timestamp=self.env.timestamp)
                        # print(f"{self.name} infected {h.name} at {location}")
                        infectee = h.name

                elif h.is_infectious:
                    ratio = h.asymptomatic_infection_ratio  if h.is_asymptomatic else 1.0
                    p_infection = h.infectiousness * ratio * (1-mask_efficacy) * proximity_factor # &prob_infectious
                    x_human = self.rng.random() < p_infection * CONTAGION_KNOB

                    if x_human and self.is_susceptible:
                        self.infection_timestamp = self.env.timestamp
                        h.n_infectious_contacts+=1
                        Event.log_exposed(self, h, self.env.timestamp)
                        city.tracker.track_infection('human', from_human=h, to_human=self, location=location, timestamp=self.env.timestamp)
                        # print(f"{h.name} infected {self.name} at {location}")
                        infectee = self.name

                # cold_and_flu_transmission(self, h)
                if self.cold_timestamp is not None or h.cold_timestamp is not None:
                    cold_infector, cold_infectee = h, self
                    if self.cold_timestamp is not None:
                        cold_infector, cold_infectee = self, h

                    if self.rng.random() < COLD_CONTAGIOUSNESS:
                        cold_infectee.cold_timestamp = self.env.timestamp

                if self.flu_timestamp is not None or h.flu_timestamp is not None:
                    flu_infector, flu_infectee = h, self
                    if self.cold_timestamp is not None:
                        flu_infector, flu_infectee = self, h

                    if self.rng.random() < FLU_CONTAGIOUSNESS:
                        flu_infectee.flu_timestamp = self.env.timestamp

                city.tracker.track_encounter_events(human1=self, human2=h, location=location, distance=distance, duration=t_near)
                Event.log_encounter(self, h,
                                    location=location,
                                    duration=t_near,
                                    distance=distance,
                                    infectee=infectee,
                                    time=self.env.timestamp
                                    )

        yield self.env.timeout(duration / TICK_MINUTE)

        # environmental transmission
        p_infection = ENVIRONMENTAL_INFECTION_KNOB * location.contamination_probability * (1-self.mask_efficacy) # &prob_infection
        x_environment = location.contamination_probability > 0 and self.rng.random() < p_infection
        if x_environment and self.is_susceptible:
            self.infection_timestamp = self.env.timestamp
            Event.log_exposed(self, location,  self.env.timestamp)
            city.tracker.track_infection('env', from_human=None, to_human=self, location=location, timestamp=self.env.timestamp)
            self.historical_infection_timestamp = self.env.timestamp
            # print(f"{self.name} is enfected at {location}")


        if self.cold_timestamp is None and self.rng.random() < P_COLD:
            self.cold_timestamp  = self.env.timestamp

        if self.flu_timestamp is None and self.rng.random() < P_FLU:
            self.flu_timestamp = self.env.timestamp

        location.remove_human(self)

    def _select_location(self, location_type, city):
        """
        Preferential exploration treatment to visit places
        rho, gamma are treated in the paper for normal trips
        Here gamma is multiplied by a factor to supress exploration for parks, stores.
        """
        if location_type == "park":
            S = self.visits.n_parks
            self.adjust_gamma = 1.0
            pool_pref = self.parks_preferences
            locs = city.parks
            visited_locs = self.visits.parks

        elif location_type == "stores":
            S = self.visits.n_stores
            self.adjust_gamma = 1.0
            pool_pref = self.stores_preferences
            locs = city.stores
            visited_locs = self.visits.stores

        elif location_type == "hospital":
            hospital = None
            for hospital in sorted(city.hospitals, key=lambda x:compute_distance(self.location, x)):
                if len(hospital.humans) < hospital.capacity:
                    return hospital
            return None

        elif location_type == "hospital-icu":
            icu = None
            for hospital in sorted(city.hospitals, key=lambda x:compute_distance(self.location, x)):
                if len(hospital.icu.humans) < hospital.icu.capacity:
                    return hospital.icu
            return None

        elif location_type == "miscs":
            S = self.visits.n_miscs
            self.adjust_gamma = 1.0
            pool_pref = [(compute_distance(self.location, m) + 1e-1) ** -1 for m in city.miscs if
                         m != self.location]
            pool_locs = [m for m in city.miscs if m != self.location]
            locs = city.miscs
            visited_locs = self.visits.miscs

        else:
            raise ValueError(f'Unknown location_type:{location_type}')

        if S == 0:
            p_exp = 1.0
        else:
            p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)

        if self.rng.random() < p_exp and S != len(locs):
            # explore
            cands = [i for i in locs if i not in visited_locs]
            cands = [(loc, pool_pref[i]) for i, loc in enumerate(cands)]
        else:
            # exploit
            cands = [(i, count) for i, count in visited_locs.items()]

        cands, scores = zip(*cands)
        loc = self.rng.choice(cands, p=_normalize_scores(scores))
        visited_locs[loc] += 1
        return loc

    ############################## RISK PREDICTION #################################

    def update_risk_level(self):
        new_risk_level = _proba_to_risk_level(self.risk)
        if new_risk_level != self.risk_level:
            # print(f"{self} changed to {self.risk_level} to {new_risk_level}")
            # modify behavior
            self.risk_level = new_risk_level
            change = self.tracing_method.modify_behavior(self)

    def update_risk(self, recovery=False, test_results=False, update_messages=None):
        if recovery:
            if self.is_removed:
                self.risk = 0.0
            else:
                self.risk = BASELINE_RISK_VALUE

        if test_results:
            if self.test_result == "positive":
                self.risk = 1.0
                self.contact_book.send_message(self, RISK_MODEL)
            elif self.test_result == "negative":
                self.risk = 0.20

        if (update_messages and
            not self.is_removed and
            self.test_result != "positive"):

            self.message_info = {
                                    'traced': True,
                                    'receipt':min(self.env.timestamp, self.message_info['receipt']),
                                    'delay':min(update_messages['delay'], self.message_info['delay'])
                                }

            if self.tracing_method.risk_model == "first order probabilistic tracing":
                self.n_contacts_tested_positive += update_messages['n']
                self.risk = 1.0 - (1.0 - RISK_TRANSMISSION_PROBA) ** self.n_contacts_tested_positive
            elif self.tracing_method.risk_model == "manual tracing":
                self.risk = 1.0
            elif self.tracing_method.risk_model == "digital tracing":
                self.risk = 1.0
            elif self.tracing_method.risk_model == "smart tracing":
                pass # Martin's code
            else:
                raise
