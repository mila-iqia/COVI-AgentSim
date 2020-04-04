# -*- coding: utf-8 -*-
import itertools
import numpy as np
from collections import defaultdict
import datetime

from utils import _draw_random_discreet_gaussian, _normalize_scores, _json_serialize, compute_distance
from config import * # PARAMETERS
from base import Event

class Event:
    test = 'test'
    encounter = 'encounter'
    symptom_start = 'symptom_start'
    contamination = 'contamination'

    @staticmethod
    def members():
        return [Event.test, Event.encounter, Event.symptom_start, Event.contamination]

    @staticmethod
    def log_encounter(human1, human2, location, duration, distance, time):
        pass

    @staticmethod
    def log_test(human, result, time):
        pass

    @staticmethod
    def log_symptom_start(human, time, covid=True):
        pass

    @staticmethod
    def log_exposed(human, time):
        pass

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

    def __init__(self, env, rng, name, infection_timestamp, household, workplace, age, rho=0.3, gamma=0.21, symptoms=None, test_results=None):
        self.env = env
        self.events = []
        self.name = name
        self.rng = rng
        self.visits = Visits()
        self.age=age

        self.household = household
        self.workplace = workplace

        self.rho = rho
        self.gamma = gamma
        self.location = household

        # Indicates whether this person will show severe signs of illness.
        # probability of being asymptomatic is basically 50%, but a bit less if you're older
        # and a bit more if you're younger
        self.asymptomatic = rng.random() > (BASELINE_P_ASYMPTOMATIC - (self.age - 50)*0.5)/100
        self.infection_timestamp = infection_timestamp
        self.never_recovers = rng.random() >= 0.99
        self.recovered_timestamp =  datetime.datetime.min
        self.r0 = []
        self.has_logged_symptoms = False

        self.last_state = None
        # metrics
        self.n_infectious_contacts = 0

        # habits
        self.avg_shopping_time = _draw_random_discreet_gaussian(AVG_SHOP_TIME_MINUTES, SCALE_SHOP_TIME_MINUTES, rng)
        self.scale_shopping_time = _draw_random_discreet_gaussian(AVG_SCALE_SHOP_TIME_MINUTES, SCALE_SCALE_SHOP_TIME_MINUTES, rng)

        self.avg_exercise_time = _draw_random_discreet_gaussian(AVG_EXERCISE_MINUTES, SCALE_EXERCISE_MINUTES, rng)
        self.scale_exercise_time = _draw_random_discreet_gaussian(AVG_SCALE_EXERCISE_MINUTES, SCALE_SCALE_EXERCISE_MINUTES, rng)

        self.avg_working_hours = _draw_random_discreet_gaussian(AVG_WORKING_MINUTES, SCALE_WORKING_MINUTES, rng)
        self.scale_working_hours = _draw_random_discreet_gaussian(AVG_SCALE_WORKING_MINUTES, SCALE_SCALE_WORKING_MINUTES, rng)

        self.avg_misc_time = _draw_random_discreet_gaussian(AVG_MISC_MINUTES, SCALE_MISC_MINUTES, rng)
        self.scale_misc_time = _draw_random_discreet_gaussian(AVG_SCALE_MISC_MINUTES, SCALE_SCALE_MISC_MINUTES, rng)

        # TODO: multiple possible days and times & limit these activities in a week
        self.shopping_days = rng.choice(range(7))
        self.shopping_hours = rng.choice(range(7, 20))

        self.exercise_days = rng.choice(range(7))
        self.exercise_hours = rng.choice(range(7, 20))

        self.work_start_hour = rng.choice(range(7, 12))

    def __repr__(self):
        return f"H:{self.name}, SEIR:{int(self.is_susceptible)}{int(self.is_exposed)}{int(self.is_infectious)}{int(self.is_removed)}"

    @property
    def is_susceptible(self):
        return not self.is_exposed and not self.is_infectious and not self.is_removed
        # return self.infection_timestamp is None and not self.recovered_timestamp == datetime.datetime.max

    @property
    def is_exposed(self):
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp < datetime.timedelta(days=AVG_INCUBATION_DAYS)

    @property
    def is_infectious(self):
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=AVG_INCUBATION_DAYS)

    @property
    def is_removed(self):
        return self.recovered_timestamp == datetime.datetime.max

    @property
    def state(self):
        return f"{int(self.is_susceptible)}{int(self.is_exposed)}{int(self.is_infectious)}{int(self.is_removed)}"


    def run(self, city):
        self.household.humans.add(self)
        while True:
            if self.name == 1:
                # to check the source of randomness
                if self.last_state != self.state:
                    print(self.env.timestamp, self.state)
                    self.last_state = self.state

            if self.is_infectious and self.has_logged_symptoms is False:
                Event.log_symptom_start(self, self.env.timestamp, True)
                self.has_logged_symptoms = True

            if self.is_infectious and self.env.timestamp - self.infection_timestamp > datetime.timedelta(days=TEST_DAYS):
                Event.log_test(self, self.env.timestamp, True)
                assert self.has_logged_symptoms is True

            if self.is_infectious and self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=AVG_RECOVERY_DAYS):
                # self.recovered_timestamp = self.env.timestamp
                self.recovered_timestamp = datetime.datetime.max
                self.update_r(self.env.timestamp - self.infection_timestamp)
                self.infection_timestamp = None
                yield self.env.timeout(np.inf)
                # yield self.env.process(self.at(self.grave, np.inf))

            # Mobility
            hour = self.env.hour_of_day()
            if not WORK_FROM_HOME and not self.env.is_weekend() and hour == self.work_start_hour:
                yield self.env.process(self.excursion(city, "work"))

            elif hour == self.shopping_hours and self.env.day_of_week() == self.shopping_days:
                yield self.env.process(self.excursion(city, "shopping"))

            elif hour == self.exercise_hours and self.env.day_of_week() == self.exercise_days:
                yield  self.env.process(self.excursion(city, "exercise"))

            elif self.rng.random() < 0.05 and self.env.is_weekend():
                yield  self.env.process(self.excursion(city, "leisure"))

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
        location.humans.add(self)
        self.leaving_time = duration + self.env.now
        self.start_time = self.env.now

        # Report all the encounters
        for h in location.humans:
            if h == self or self.location.location_type == 'household':
                continue

            distance = self.rng.randint(50, 1000)
            t_near = min(self.leaving_time, h.leaving_time) - max(self.start_time, h.start_time)
            is_exposed = False
            if h.is_infectious and distance <= 200 and t_near * TICK_MINUTE > 2 :
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
        location.humans.remove(self)

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

    def update_r(self, timedelta):
        timedelta /= datetime.timedelta(days=1) # convert to float days
        self.r0.append(self.n_infectious_contacts/timedelta)
