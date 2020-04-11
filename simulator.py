# -*- coding: utf-8 -*-
import simpy

import itertools
import numpy as np
from collections import defaultdict, namedtuple
import datetime
from bitarray import bitarray
import operator
import math

from utils import _normalize_scores, _get_random_age, _get_random_sex, _get_all_symptoms_array, \
    _get_preexisting_conditions, _draw_random_discreet_gaussian, _json_serialize, _sample_viral_load_piecewise, \
    _get_random_area, _encode_message, _decode_message, float_to_binary, binary_to_float
from config import *  # PARAMETERS
from base import *


class Visits(object):

    def __init__(self):
        self.parks = defaultdict(int)
        self.stores = defaultdict(int)
        self.miscs = defaultdict(int)

    @property
    def n_parks(self):
        return len(self.parks)

    @property
    def n_stores(self):
        return len(self.stores)

    @property
    def n_miscs(self):
        return len(self.miscs)


class Human(object):

    def __init__(self, env, name, age, rng, infection_timestamp, household, workplace, rho=0.3, gamma=0.21, symptoms=[],
                 test_results=None):
        self.env = env
        self.events = []
        self.name = name
        self.rng = rng

        self.age = _get_random_age(self.rng)
        self.sex = _get_random_sex(self.rng)
        self.preexisting_conditions = _get_preexisting_conditions(self.age, self.sex, self.rng)

        self.household = household
        self.workplace = workplace
        self.location = household
        self.rho = rho
        self.gamma = gamma

        self.visits = Visits()
        self.travelled_recently = self.rng.rand() > 0.9


        # &carefullness
        if self.rng.rand() < P_CAREFUL_PERSON:
            self.carefullness = round(self.rng.normal(55, 10)) + self.age/2
        else:
            self.carefullness = round(self.rng.normal(25, 10)) + self.age/2

        age_modifier = 1
        if self.age > 40 or self.age < 12:
            age_modifier = 2
        self.has_cold = self.rng.rand() < P_COLD * age_modifier
        self.has_flu = self.rng.rand() < P_FLU * age_modifier
        self.has_app = self.rng.rand() < (P_HAS_APP / age_modifier) + (self.carefullness / 2)
        self.incubation_days = _draw_random_discreet_gaussian(AVG_INCUBATION_DAYS, SCALE_INCUBATION_DAYS, self.rng)


        # Indicates whether this person will show severe signs of illness.
        self.infection_timestamp = infection_timestamp
        self.recovered_timestamp = datetime.datetime.min
        self.really_sick = self.is_exposed and self.rng.random() >= 0.9
        self.extremely_sick = self.really_sick and self.rng.random() >= 0.7 # &severe; 30% of severe cases need ICU
        self.never_recovers = self.rng.random() <= P_NEVER_RECOVERS[min(math.floor(self.age/10),8)]
        
        # &symptoms, &viral-load
        # probability of being asymptomatic is basically 50%, but a bit less if you're older
        # and a bit more if you're younger
        self.is_asymptomatic = self.rng.rand() > (BASELINE_P_ASYMPTOMATIC - (self.age - 50) * 0.5) / 100
        self.asymptomatic_infection_ratio = 0.0
        if self.is_asymptomatic:
            self.asymptomatic_infection_ratio = ASYMPTOMATIC_INFECTION_RATIO # draw a beta with the distribution in documents
        self.recovery_days = _draw_random_discreet_gaussian(AVG_RECOVERY_DAYS, SCALE_RECOVERY_DAYS, self.rng) # make it IQR &recovery
        self.viral_load_plateau_height, self.viral_load_plateau_start, self.viral_load_plateau_end, self.viral_load_recovered = _sample_viral_load_piecewise(rng, age=age)
        self.all_symptoms_array = _get_all_symptoms_array(
                          np.ndarray.item(self.viral_load_plateau_start), np.ndarray.item(self.viral_load_plateau_end),
                          np.ndarray.item(self.viral_load_recovered), age=self.age, incubation_days=self.incubation_days, 
                                                          really_sick=self.really_sick, extremely_sick=self.extremely_sick, 
                          rng=self.rng, preexisting_conditions=self.preexisting_conditions)

        # counters and memory
        self.r0 = []
        self.has_logged_symptoms = False
        self.has_logged_test = False
        self.n_infectious_contacts = 0
        self.last_state = self.state

        # privacy
        self.M = []
        self.A = {}
        self.cur_num_messages = 0
        self.pending_messages = []
        self.cur_day = -1

        # risk
        self.risk = 0
        self.contact_history = {}

        # habits
        self.avg_shopping_time = _draw_random_discreet_gaussian(AVG_SHOP_TIME_MINUTES, SCALE_SHOP_TIME_MINUTES, self.rng)
        self.scale_shopping_time = _draw_random_discreet_gaussian(AVG_SCALE_SHOP_TIME_MINUTES,
                                                                  SCALE_SCALE_SHOP_TIME_MINUTES, self.rng)

        self.avg_exercise_time = _draw_random_discreet_gaussian(AVG_EXERCISE_MINUTES, SCALE_EXERCISE_MINUTES, self.rng)
        self.scale_exercise_time = _draw_random_discreet_gaussian(AVG_SCALE_EXERCISE_MINUTES,
                                                                  SCALE_SCALE_EXERCISE_MINUTES, self.rng)

        self.avg_working_hours = _draw_random_discreet_gaussian(AVG_WORKING_MINUTES, SCALE_WORKING_MINUTES, self.rng)
        self.scale_working_hours = _draw_random_discreet_gaussian(AVG_SCALE_WORKING_MINUTES, SCALE_SCALE_WORKING_MINUTES, self.rng)

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

        #Limiting the number of hours spent exercising per week
        self.max_exercise_per_week = _draw_random_discreet_gaussian(AVG_MAX_NUM_EXERCISE_PER_WEEK, SCALE_MAX_NUM_EXERCISE_PER_WEEK, self.rng)
        self.count_exercise=0

        self.work_start_hour = self.rng.choice(range(7, 12))

    def __repr__(self):
        return f"H:{self.name}, SEIR:{int(self.is_susceptible)}{int(self.is_exposed)}{int(self.is_infectious)}{int(self.is_removed)}"


    def handle_message(self, m_i):
        m_i_enc = _encode_message(m_i)
        temp_cur_num_messages = self.cur_num_messages
        # if first message received, it's group 0
        if self.cur_num_messages == 0:
            self.A[m_i_enc] = 0
            self.M.append(m_i_enc)
            self.cur_num_messages += 1
            return

        scores = {}
        for m_enc in self.M:
            m = _decode_message(m_enc)
            if m_i[0] == m[0] and m_i[2].day == m[2].day:
                scores[m_enc] = 3
            elif m_i[0][:3] == m[0][:3] and m_i[2].day - 1 == m[2].day:
                scores[m_enc] = 2
            elif m_i[0][:2] == m[0][:2] and m_i[2].day - 2 == m[2].day:
                scores[m_enc] = 1
            elif m_i[0][:1] == m[0][:1] and m_i[2].day - 2 == m[2].day:
                scores[m_enc] = 0

        if scores:
            max_score_message = max(scores.items(), key=operator.itemgetter(1))[0]
            self.A[m_i_enc] = self.A[max_score_message]
            self.cur_num_messages += 1
            self.M.append(m_i_enc)
            return

        if temp_cur_num_messages == self.cur_num_messages:
            self.A[m_i_enc] = max(self.A.values()) + 1
            self.cur_num_messages += 1
            self.M.append(m_i_enc)
            return

    @property
    def uid(self):
        return self._uid

    def update_uid(self):
        try:
            self._uid.pop()
            self._uid.extend([self.rng.choice([True, False])])
        except AttributeError:
            self._uid = bitarray()
            self._uid.extend(self.rng.choice([True, False], 4)) # generate a random 4-bit code

    def risk_for_symptoms(self):
        sickness_day = (self.env.timestamp - self.infection_timestamp).days
        symptoms = []
        for day in range(sickness_day + 1):
            if RISK_WITH_TRUE_SYMPTOMS and self.rng.rand() < self.carefullness:
                    symptoms.extend(self.all_symptoms_array[day-1])
            else:
                symptoms.extend(self.all_symptoms_array[day-1])
        if self.test_results == 'positive':
            return 1
        elif 'severe' in symptoms:
            return 0.75
        elif 'moderate' in symptoms:
            return 0.5
        elif 'mild' in symptoms:
            return 0.25
        else:
            return 0


    def update_risk(self, other):
        if self.infection_timestamp and (self.env.timestamp - self.infection_timestamp).days >= 0 and (self.env.timestamp - self.infection_timestamp).days <= len(self.all_symptoms_array)-1:
            self.risk = self.risk_for_symptoms()
        if self.recovered_timestamp and (self.env.timestamp - self.recovered_timestamp).days >= 0:
            self.risk = 0
        m_risk = binary_to_float("".join([str(x) for x in np.array(other[1].tolist()).astype(int)]), 0, 4)
        m_uid = other[0]

        if RISK_MODEL == 'yoshua':
            if self.risk < m_risk:
                update = (m_risk - m_risk * self.risk) * RISK_TRANSMISSION_PROBA
            else:
                update = 0
        elif RISK_MODEL == 'lenka':
            update = m_risk * RISK_TRANSMISSION_PROBA
        elif RISK_MODEL == 'eilif':
            if other.name not in self.contact_history:
                # update is delta_risk
                update = m_risk * RISK_TRANSMISSION_PROBA
            else:
                previous_risk = self.contact_history[other.name].previous_risk
                carry_over_transmission_proba = self.contact_history[other.name].carry_over_transmission_proba
                update = ((m_risk - previous_risk) * RISK_TRANSMISSION_PROBA +
                          previous_risk * carry_over_transmission_proba)
            # Update contact history
            self.contact_history[m_uid].previous_risk = m_risk
            self.contact_history[m_uid].carry_over_transmission_proba = \
                RISK_TRANSMISSION_PROBA * (1 - update)
        else:
            update = 0
        self.risk += update
        if CLIP_RISK:
            self.risk = min(self.risk, 1.)


    def to_sick_to_move(self):
        # Assume 2 weeks incubation time ; in 10% of cases person becomes to sick
        # to go shopping after 2 weeks for at least 10 days and in 1% of the cases
        # never goes shopping again.
        time_since_sick_delta = (env.timestamp - self.infection_timestamp).days
        in_peak_illness_time = (
                time_since_sick >= self.incubation_days and
                time_since_sick <= (self.incubation_days + NUM_DAYS_SICK))
        return (in_peak_illness_time or self.never_recovers) and self.really_sick

    @property
    def is_susceptible(self):
        return not self.is_exposed and not self.is_infectious and not self.is_removed
        # return self.infection_timestamp is None and not self.recovered_timestamp == datetime.datetime.max


    @property
    def is_exposed(self):
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp < datetime.timedelta(days=self.incubation_days)

    @property
    def is_infectious(self):
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.incubation_days)

    @property
    def is_removed(self):
        return self.recovered_timestamp != datetime.datetime.min

    @property
    def test_results(self):
        if not any(self.symptoms):
            return None
        else:
            tested = self.rng.rand() > P_TEST
            if tested:
                if self.is_infectious:
                    return 'positive'
                else:
                    if self.rng.rand() > P_FALSE_NEGATIVE:
                        return 'negative'
                    else:
                        return 'positive'
            else:
                return None

    @property
    def symptoms(self):
        if not self.infection_timestamp:
            return []
        sickness_day = (self.env.timestamp - self.infection_timestamp).days
        return self.all_symptoms_array[sickness_day]


    @property
    def viral_load(self):
        """ Calculates the elapsed time since infection, returning this person's current viral load"""
        if not self.infection_timestamp:
            return 0.
        # calculates the time since infection in days
        time_exposed = (self.env.timestamp - self.infection_timestamp)
        time_exposed_days = time_exposed.days + time_exposed.seconds / 86400 #(seconds in a day)

        # implements the piecewise linear function
        if time_exposed_days < self.viral_load_plateau_start:
            cur_viral_load = self.viral_load_plateau_height * time_exposed_days / self.viral_load_plateau_start
        elif time_exposed_days < self.viral_load_plateau_end:
            cur_viral_load = self.viral_load_plateau_height
        else:
            cur_viral_load = self.viral_load_plateau_height - self.viral_load_plateau_height * (time_exposed_days -self.viral_load_plateau_end) / (self.viral_load_recovered - self.viral_load_plateau_end)

        # the viral load cannot be negative
        if cur_viral_load < 0:
            cur_viral_load = 0.
        return cur_viral_load

    @property
    def infectiousness(self):
        severity_multiplier = 1
        if self.is_infectious:
            if self.really_sick:
              severity_multiplier = 1.25
            if self.extremely_sick:
              severity_multiplier = 1.5
            if 'immuno-compromised' in self.preexisting_conditions:
              severity_multiplier += 0.2
            if 'cough' in self.symptoms:
              severity_multiplier += 0.25
        return self.viral_load * severity_multiplier

    @property
    def wearing_mask(self):
        mask = False
        if not self.location == self.household:
            mask = self.rng.rand() < self.carefullness
        return mask

    @property
    def reported_symptoms(self):
        if not any(self.symptoms) or self.test_results is None or not self.has_app:
            return []
        else:
            if self.rng.rand() < self.carefullness:
                return self.symptoms
            else:
                return []

    def update_r(self, timedelta):
        timedelta /= datetime.timedelta(days=1) # convert to float days
        self.r0.append(self.n_infectious_contacts/timedelta)
        self.n_infectious_contacts = 0

    @property
    def state(self):
        return [int(self.is_susceptible), int(self.is_exposed), int(self.is_infectious), int(self.is_removed)]

    def assert_state_changes(self):
        next_state = {0:1, 1:2, 2:3}
        assert sum(self.state) == 1, f"invalid compartment for human:{self.name}"
        if self.last_state != self.state:
            assert next_state[self.last_state.index(1)] == self.state.index(1), f"invalid compartment transition for human:{self.name}"
            self.last_state = self.state

    @property
    def message_risk(self):
        if self.risk == 1.0:
            return bitarray('1111')
        return bitarray(float_to_binary(self.risk, 0, 4))

    def cur_message(self, time):
        Message = namedtuple('message', 'uid risk time unobs_id')
        message = Message(self.uid, self.message_risk, time, self.name)
        return message

    def run(self, city):
        """
           1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
           State  h h h h h h h h h sh sh h  h  h  ac h  h  h  h  h  h  h  h  h
        """
        self.household.humans.add(self)
        while True:
            if self.is_infectious and self.has_logged_symptoms is False:
                Event.log_symptom_start(self, True, self.env.timestamp)
                self.has_logged_symptoms = True

            if self.is_infectious and self.env.timestamp - self.infection_timestamp > datetime.timedelta(days=TEST_DAYS) and not self.has_logged_test:
                result = self.rng.random() > 0.8
                Event.log_test(self, result, self.env.timestamp)
                self.has_logged_test = True
                assert self.has_logged_symptoms is True # FIXME: assumption might not hold

            if self.is_infectious and self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.recovery_days):
                if self.never_recovers: # re-infection assumed negligble
                    self.recovered_timestamp = datetime.datetime.max
                    dead = True
                else:
                    self.recovered_timestamp = self.env.timestamp
                    dead = False

                self.update_r(self.env.timestamp - self.infection_timestamp)
                self.infection_timestamp = None # indicates they are no longer infected
                Event.log_recovery(self, self.env.timestamp, dead)
                if dead:
                    yield self.env.timeout(np.inf)


            self.assert_state_changes()

            # Mobility

            hour, day = self.env.hour_of_day(), self.env.day_of_week()

            if day==0:
                self.count_exercise=0
                self.count_shop=0

            if not WORK_FROM_HOME and not self.env.is_weekend() and hour == self.work_start_hour:
                yield self.env.process(self.excursion(city, "work"))

            elif hour in self.shopping_hours and day in self.shopping_days and self.count_shop<=self.max_shop_per_week:
                self.count_shop+=1
                yield self.env.process(self.excursion(city, "shopping"))

            elif hour in self.exercise_hours and day in self.exercise_days and self.count_exercise<=self.max_exercise_per_week:
                self.count_exercise+=1
                yield  self.env.process(self.excursion(city, "exercise"))

            elif self.rng.random() < 0.05 and self.env.is_weekend():
                yield  self.env.process(self.excursion(city, "leisure"))

            # start from house all the time
            yield self.env.process(self.at(self.household, 60))

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
                yield self.env.process(self.at(grocery_store, t))

        elif type == "exercise":
            park = self._select_location(location_type="park", city=city)
            t = _draw_random_discreet_gaussian(self.avg_exercise_time, self.scale_exercise_time, self.rng)
            yield self.env.process(self.at(park, t))

        elif type == "work":
            t = _draw_random_discreet_gaussian(self.avg_working_hours, self.scale_working_hours, self.rng)
            yield self.env.process(self.at(self.workplace, t))

        elif type == "leisure":
            S = 0
            p_exp = 1.0
            while True:
                if self.rng.random() > p_exp:  # return home
                    yield self.env.process(self.at(self.household, 60))
                    break

                loc = self._select_location(location_type='miscs', city=city)
                S += 1
                p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
                with loc.request() as request:
                    yield request
                    t = _draw_random_discreet_gaussian(self.avg_misc_time, self.scale_misc_time, self.rng)
                    yield self.env.process(self.at(loc, t))
        else:
            raise ValueError(f'Unknown excursion type:{type}')


    def at(self, location, duration):
        if self.name == 1:
            # print(self, self.env.timestamp.strftime("%b %d, %H %M"), self.location)
            # print(self.env.timestamp.strftime("%b %d, %H %M"), self.location._name, "-->", location._name, duration)
            pass

        self.location = location
        location.add_human(self)
        self.leaving_time = duration + self.env.now
        self.start_time = self.env.now
        area = self.location.area
        # Report all the encounters
        for h in location.humans:
            if h == self or self.location.location_type == 'household':
                continue

            distance = np.sqrt(int(area/len(self.location.humans))) + self.rng.randint(MIN_DIST_ENCOUNTER, MAX_DIST_ENCOUNTER)
            t_near = min(self.leaving_time, h.leaving_time) - max(self.start_time, h.start_time)
            is_exposed = False
            # FIXME: This is a hack to take into account the difference between asymptomatic transmission rate and symptomatic transmission rate.
            # The fix should be handled by better modelling the infectiousness of a person as a function of viral_load
            p_infection = h.viral_load * (h.is_asymptomatic * h.asymptomatic_infection_ratio + 1.0 * (not h.is_asymptomatic))
            x_human = distance <= INFECTION_RADIUS and t_near * TICK_MINUTE > INFECTION_DURATION and self.rng.random() < p_infection
            x_environment = self.rng.random() < location.contamination_probability # &prob_infection
            if x_human or x_environment:
                if self.is_susceptible:
                    is_exposed = True
                    h.n_infectious_contacts+=1
                    Event.log_exposed(self, self.env.timestamp)

            if self.is_susceptible and is_exposed:
                self.infection_timestamp = self.env.timestamp
                self.historical_infection_timestamp = self.env.timestamp

            Event.log_encounter(self, h,
                                location=location,
                                duration=t_near,
                                distance=distance,
                                # cm  #TODO: prop to Area and inv. prop to capacity
                                time=self.env.timestamp,
                                )

        yield self.env.timeout(duration / TICK_MINUTE)
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

    def serialize(self):
        del self.env
        del self.events
        del self.rng
        del self.visits
        del self.leaving_time
        del self.start_time
        del self.household
        del self.location
        del self.workplace
        del self.viral_load_plateau_start
        del self.viral_load_plateau_end
        del self.viral_load_recovered
        del self.exercise_hours
        del self.exercise_days
        del self.shopping_days
        del self.shopping_hours
        del self.work_start_hour
        try:
            print(f"{self.name}, {self.historical_infection_timestamp}")
            self.infection_timestamp = str(self.historical_infection_timestamp)
        except Exception:
            self.infection_timestamp = None

        try:
            self.recovered_timestamp = str(self.recovered_timestamp)
        except Exception:
            self.recovered_timestamp = None
        return self