# -*- coding: utf-8 -*-
import simpy

import itertools
import numpy as np
from collections import defaultdict
import datetime
import math

from utils import _normalize_scores, _get_random_age, _draw_random_discreet_gaussian, _json_serialize
from config import *  # PARAMETERS
from base import *

class Visits:
    parks = defaultdict(int)
    stores = defaultdict(int)
    miscs = defaultdict(int)

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

    def __init__(self, env, name, age, rng, infection_timestamp, household, workplace, rho=0.3, gamma=0.21, symptoms=None,
                 test_results=None):
        self.env = env
        self.events = []
        self.name = name
        self.age = age
        self.rng = rng

        # probability of being asymptomatic is basically 50%, but a bit less if you're older
        # and a bit more if you're younger
        self.is_asymptomatic = self.rng.rand() > (BASELINE_P_ASYMPTOMATIC - (self.age - 50) * 0.5) / 100
        self.asymptomatic_infection_ratio = 0.0
        if self.is_asymptomatic:
            self.asymptomatic_infection_ratio = ASYMPTOMATIC_INFECTION_RATIO # draw a beta with the distribution in documents
        self.incubation_days = _draw_random_discreet_gaussian(AVG_INCUBATION_DAYS, SCALE_INCUBATION_DAYS, self.rng)
        self.recovery_days = _draw_random_discreet_gaussian(AVG_RECOVERY_DAYS, SCALE_RECOVERY_DAYS, self.rng) # make it IQR &recovery

        self.household = household
        self.workplace = workplace
        self.location = household
        self.rho = rho
        self.gamma = gamma

        self.visits = Visits()
        self.travelled_recently = self.rng.rand() > 0.9

        # &carefullness
        if self.rng.rand() < P_CAREFUL_PERSON:
            self.carefullness = round(self.rng.normal(75, 10))
        else:
            self.carefullness = round(self.rng.normal(35, 10))

        age_modifier = 1
        if self.age > 40 or self.age < 12:
            age_modifier = 2
        self.has_cold = self.rng.rand() < P_COLD * age_modifier
        self.has_flu = self.rng.rand() < P_FLU * age_modifier
        self.has_app = self.rng.rand() < (P_HAS_APP / age_modifier) + (self.carefullness / 2)

        # Indicates whether this person will show severe signs of illness.
        self.infection_timestamp = infection_timestamp
        self.recovered_timestamp = datetime.datetime.min
        self.really_sick = self.is_exposed and self.rng.random() >= 0.9
        self.never_recovers = self.rng.random() <= P_NEVER_RECOVERS[min(math.floor(self.age/10),8)]

        # counters and memory
        self.r0 = []
        self.has_logged_symptoms = False
        self.has_logged_test = False
        self.n_infectious_contacts = 0
        self.last_state = self.state

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
        return self.recovered_timestamp == datetime.datetime.max

    @property
    def is_contagious(self):
        return self.infectiousness

    @property
    def test_results(self):
        if self.symptoms == None:
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
        # probability of being asymptomatic is basically 50%, but a bit less if you're older
        # and a bit more if you're younger
        symptoms = None
        if self.is_asymptomatic or self.is_susceptible:
            pass
        else:
            time_since_exposed = self.env.timestamp - self.infection_timestamp
            symptom_start = datetime.timedelta(abs(self.rng.normal(SYMPTOM_DAYS, 2.5)))
            #  print (time_since_sick)
            #  print (symptom_start)
            if time_since_exposed >= symptom_start:
                symptoms = ['mild']
                if self.really_sick:
                    symptoms.append('severe')
                if self.rng.rand() < 0.9:
                    symptoms.append('fever')
                if self.rng.rand() < 0.85:
                    symptoms.append('cough')
                if self.rng.rand() < 0.8:
                    symptoms.append('fatigue')
                if self.rng.rand() < 0.7:
                    symptoms.append('trouble_breathing')
                if self.rng.rand() < 0.1:
                    symptoms.append('runny_nose')
                if self.rng.rand() < 0.4:
                    symptoms.append('loss_of_taste')
                if self.rng.rand() < 0.4:
                    symptoms.append('gastro')
        if self.has_cold:
            if symptoms is None:
                symptoms = ['mild', 'runny_nose']
            if self.rng.rand() < 0.2:
                symptoms.append('fever')
            if self.rng.rand() < 0.6:
                symptoms.append('cough')
        if self.has_flu:
            if symptoms is None:
                symptoms = ['mild']
            if self.rng.rand() < 0.2:
                symptoms.append('severe')
            if self.rng.rand() < 0.8:
                symptoms.append('fever')
            if self.rng.rand() < 0.4:
                symptoms.append('cough')
            if self.rng.rand() < 0.8:
                symptoms.append('fatigue')
            if self.rng.rand() < 0.8:
                symptoms.append('aches')
            if self.rng.rand() < 0.5:
                symptoms.append('gastro')
        return symptoms

    @property
    def infectiousness(self):
        if self.is_infectious:
            days_exposed = (self.env.timestamp - self.infection_timestamp).days
            if days_exposed > len(INFECTIOUSNESS_CURVE):
                return 0
            else:
                return INFECTIOUSNESS_CURVE[days_exposed - 1]
        else:
            return 0

    @property
    def wearing_mask(self):
        mask = False
        if not self.location == self.household:
            mask = self.rng.rand() < self.carefullness
        return mask

    @property
    def reported_symptoms(self):
        if self.symptoms is None or self.test_results is None or not self.has_app:
            return None
        else:
            if self.rng.rand() < self.carefullness:
                return self.symptoms
            else:
                return None

    def update_r(self, timedelta):
        timedelta /= datetime.timedelta(days=1) # convert to float days
        self.r0.append(self.n_infectious_contacts/timedelta)
        self.n_infectious_contacts = 0

    @property
    def state(self):
        return [int(self.is_susceptible), int(self.is_exposed), int(self.is_infectious), int(self.is_removed)]

    def assert_state_changes(self):
        next_state = {0:1, 1:2, 2:0}
        assert sum(self.state) == 1, f"invalid compartment for human:{self.name}"
        if self.last_state != self.state:
            assert next_state[self.last_state.index(1)] == self.state.index(1), f"invalid compartment transition for human:{self.name}"
            self.last_state = self.state

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
                if self.never_recovers or True: # re-infection assumed negligble
                    self.recovered_timestamp = datetime.datetime.max
                    dead = True
                else:
                    self.recovered_timestamp = self.env.timestamp
                    dead = False

                self.update_r(self.env.timestamp - self.infection_timestamp)
                self.infection_timestamp = None
                if dead:
                    yield self.env.timeout(np.inf)

                Event.log_recovery(self, self.env.timestamp, dead)

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

            distance =  np.sqrt(int(area/len(self.location.humans))) + self.rng.randint(MIN_DIST_ENCOUNTER, MAX_DIST_ENCOUNTER)
            t_near = min(self.leaving_time, h.leaving_time) - max(self.start_time, h.start_time)
            is_exposed = False
            p_infection = h.infectiousness * (h.is_asymptomatic  * h.asymptomatic_infection_ratio  + 1.0 * (not h.is_asymptomatic)) # &prob_infectious
            x_human = distance <= INFECTION_RADIUS and t_near * TICK_MINUTE > INFECTION_DURATION and self.rng.random() < p_infection
            x_environment = self.rng.random() < location.contamination_probability # &prob_infection
            if x_human or x_environment:
                if self.is_susceptible:
                    is_exposed = True
                    h.n_infectious_contacts+=1
                    Event.log_exposed(self, self.env.timestamp)

            if self.is_susceptible and is_exposed:
                self.infection_timestamp = self.env.timestamp

            Event.log_encounter(self, h,
                                location=location,
                                duration=t_near,
                                distance=distance,
                                # cm  #TODO: prop to Area and inv. prop to capacity
                                time=self.env.timestamp,
                                # latent={"infected":self.is_exposed}
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
