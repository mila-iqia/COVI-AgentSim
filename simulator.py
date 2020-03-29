# -*- coding: utf-8 -*-
import simpy
import random
import json
from matplotlib import pyplot as plt
import pylab as pl
import itertools
import enum
import numpy as np
from scipy.stats import truncnorm
from collections import defaultdict
import pickle
import datetime

from utils import _normalize_scores, _draw_random_discreet_gaussian, _json_serialize
from config import * # PARAMETERS

class Env(simpy.Environment):
  def __init__(self, initial_timestamp):
    super().__init__()
    self.initial_timestamp = initial_timestamp

  def time(self):
    return self.now

  @property
  def timestamp(self):
    return self.initial_timestamp + datetime.timedelta(
        minutes=self.now * TICK_MINUTE)

  def minutes(self):
    return self.timestamp.minute

  def hour_of_day(self):
    return self.timestamp.hour

  def day_of_week(self):
    return self.timestamp.weekday()

  def is_weekend(self):
      return self.day_of_week() in [0, 6]

  def time_of_day(self):
    return self.timestamp.isoformat()

class City(object):

  def __init__(self, stores, parks, humans, miscs):
    self.stores = stores
    self.parks = parks
    self.humans = humans
    self.miscs = miscs
    self._compute_preferences()

  @property
  def events(self):
    return list(itertools.chain(*[h.events for h in self.humans]))

  @staticmethod
  def compute_distance(loc1, loc2):
      return np.sqrt((loc1.lat - loc2.lat)**2 + (loc1.lon - loc2.lon)**2)

  def _compute_preferences(self):
      """ compute preferred distribution of each human for park, stores, etc."""
      for h in self.humans:
          h.stores_preferences = [(self.compute_distance(h.household, s)+1e-1)**-1 for s in self.stores]
          h.parks_preferences = [(self.compute_distance(h.household, s)+1e-1)**-1 for s in self.parks]


class Location(simpy.Resource):

  def __init__(self, env, capacity=simpy.core.Infinity, name='Safeway', location_type='stores', lat=None, lon=None, cont_prob=None):
    super().__init__(env, capacity)
    self.humans = set()
    self.name = name
    self.lat = lat
    self.lon = lon
    self.location_type = location_type
    self.cont_prob = cont_prob

  def sick_human(self):
    return any([h.is_sick for h in self.humans])

  def __repr__(self):
    return f"{self.location_type}:{self.name} - Total number of people in {self.location_type}:{len(self.humans)} - sick:{self.sick_human()}"

  def contamination_proba(self):
    if not self.sick_human():
      return 0
    return self.cont_prob

  def __hash__(self):
      return hash(self.name)

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

      human1.events.append(
          {
              'human_id': human1.name,
              'time': time,
              'event_type': Event.encounter,
              'payload': {
                  'encounter_human_id': human2.name,
                  'other_human_is_infected': human2.is_sick,
                  'other_human_symptoms': human2.symptoms,
                  'other_human_test_results': human2.test_results,
                  'duration': duration,
                  'distance': distance,
                  'location_type': location.location_type,
                  'contamination_prob': location.cont_prob,
                  'lat': location.lat,
                  'lon': location.lon,
              }
          }
      )

      human2.events.append(
          {
              'human_id': human2.name,
              'time': time,
              'event_type': Event.encounter,
              'payload': {
                  'encounter_human_id': human1.name,
                  'other_human_is_infected': human1.is_sick,
                  'other_human_symptoms': human1.symptoms,
                  'other_human_test_results': human1.test_results,
                  'duration': duration,
                  'distance': distance,
                  'location_type': location.location_type,
                  'contamination_prob': location.cont_prob,
                  'lat': location.lat,
                  'lon': location.lon,
              }
          }
      )

  @staticmethod
  def log_test(human, result, time):

      human.events.append(
              {
                  'human_id': human.name,
                  'event_type': Event.test,
                  'time': time,
                  'payload': {
                    'result': result,
                  }
              }
        )

  @staticmethod
  def log_symptom_start(human, time, covid=True):
      human.events.append(
              {
                  'human_id': human.name,
                  'event_type': Event.symptom_start,
                  'time': time,
                  'payload': {
                    'covid': covid
                  }
              }
          )

  @staticmethod
  def log_contaminate(human, time):
      human.events.append(
          {
              'human_id': human.name,
              'event_type': Event.contamination,
              'time': time,
              'payload': {}
          }
      )



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
  actions = {
      'shopping': 1,
      'at_home': 3,
      'exercise': 4
  }

  def __init__(self, name, infection_timestamp, household, workplace, rho=0.3, gamma=0.21, symptoms=None, test_results=None):
      self.events = []
      self.name = name

      self.household = household
      self.workplace = workplace
      self.location = household
      self.rho = rho
      self.gamma = gamma

      self.action = Human.actions['at_home']
      self.visits = Visits()

      # Indicates whether this person will show severe signs of illness.
      self.infection_timestamp = infection_timestamp
      self.really_sick = self.is_sick and random.random() >= 0.9
      self.symptoms = symptoms
      self.test_results = test_results
      self.never_recovers = random.random() >= 0.99

      # habits
      self.avg_shopping_time = _draw_random_discreet_gaussian(AVERAGE_SHOP_TIME_MINUTES, SCALE_SHOP_TIME_MINUTES)
      self.scale_shopping_time = _draw_random_discreet_gaussian(AVG_SCALE_SHOP_TIME_MINUTES, SCALE_SCALE_SHOP_TIME_MINUTES)

      self.avg_exercise_time = _draw_random_discreet_gaussian(AVG_EXERCISE_MINUTES, SCALE_EXERCISE_MINUTES)
      self.scale_exercise_time = _draw_random_discreet_gaussian(AVG_SCALE_EXERCISE_MINUTES, SCALE_SCALE_EXERCISE_MINUTES)

      self.avg_working_hours = _draw_random_discreet_gaussian(AVG_WORKING_HOURS, SCALE_WORKING_HOURS)
      self.scale_working_hours = _draw_random_discreet_gaussian(AVG_SCALE_WORKING_HOURS, SCALE_SCALE_WORKING_HOURS)

      self.avg_misc_time = _draw_random_discreet_gaussian(AVG_MISC_MINUTES, SCALE_MISC_MINUTES)
      self.scale_misc_time = _draw_random_discreet_gaussian(AVG_SCALE_MISC_MINUTES, SCALE_SCALE_MISC_MINUTES)

      # TODO: multiple possible days and times & limit these activities in a week
      self.shopping_days = np.random.choice(range(7))
      self.shopping_hours = np.random.choice(range(7, 20))

      self.exercise_days = np.random.choice(range(7))
      self.exercise_hours = np.random.choice(range(7, 20))

      self.work_start_hour = np.random.choice(range(7, 12))

  def to_sick_to_shop(self):
    # Assume 2 weeks incubation time ; in 10% of cases person becomes to sick
    # to go shopping after 2 weeks for at least 10 days and in 1% of the cases
    # never goes shopping again.
    time_since_sick_delta = env.timestamp - self.infection_timestamp
    in_peak_illness_time = (
        time_since_sick >= INCUBATION_DAYS * 24 * 60 and
        time_since_sick <= (INCUBATION_DAYS + NUM_DAYS_SICK) * 24 * 60)
    return (in_peak_illness_time or self.never_recovers) and self.really_sick

  def lat(self):
    return self.location.lat if self.location else self.household.lat

  def lon(self):
    return self.location.lon if self.location else self.household.lon

  @property
  def is_contagious(self):
    return self.is_sick

  @property
  def is_sick(self):
    return self.infection_timestamp is not None

  def __repr__(self):
    return f"person:{self.name}, sick:{self.is_sick}"

  def run(self, env, city):
    """
       1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
       State  h h h h h h h h h sh sh h  h  h  ac h  h  h  h  h  h  h  h  h
    """
    self.household.humans.add(self)
    while True:
      # Simulate some tests
      if self.is_sick and env.timestamp - self.infection_timestamp > datetime.timedelta(days=INCUBATION_DAYS):
        # Todo ensure it only happen once
        result = random.random() > 0.8
        Event.log_test(self, time=env.timestamp, result=result)
        #Fixme: After a user get tested positive, assume no more activity
        break

      elif env.hour_of_day() == self.work_start_hour and not env.is_weekend() and not WORK_FROM_HOME:
        yield env.process(self.go_to_work(env))

      elif env.hour_of_day() == self.shopping_hours and env.day_of_week() == self.shopping_days:
        yield env.process(self.shop(env, city))
      elif env.hour_of_day() == self.exercise_hours and env.day_of_week() == self.exercise_days: ##LIMIT AND VARIABLE
        yield env.process(self.exercise(env, city))
      elif np.random.random() < 0.05 and env.is_weekend():
        yield env.process(self.take_a_trip(env, city))
      elif self.is_sick and env.timestamp - self.infection_timestamp > datetime.timedelta(days=SYMPTOM_DAYS):
        # Stay home after symptoms
        # TODO: ensure it only happen once
        # Event.log_symptom_start(self, time=env.timestamp)
        pass
      self.location = self.household
      yield env.process(self.stay_at_home(env))

  def stay_at_home(self, env):
    self.action = Human.actions['at_home']
    yield env.process(self.at(self.household, env, 60))

  def go_to_work(self, env):
      t = _draw_random_discreet_gaussian(self.avg_working_hours, self.scale_working_hours)
      yield env.process(self.at(self.workplace, env, t))

  def take_a_trip(self, env, city):
      S = 0
      p_exp = 1.0
      while True:
          if np.random.random() > p_exp: # return home
            yield env.process(self.at(self.household, env, 60))
            break

          loc = self._select_location(location_type='miscs', city=city)
          S += 1
          p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
          with loc.request() as request:
            yield request
            t = _draw_random_discreet_gaussian(self.avg_misc_time, self.scale_misc_time)
            yield env.process(self.at(loc, env, t))

  def shop(self, env, city):
    self.action = Human.actions['shopping']
    grocery_store = self._select_location(location_type="stores", city=city) ## MAKE IT EPR

    with grocery_store.request() as request:
      yield request
      t = _draw_random_discreet_gaussian(self.avg_shopping_time, self.scale_shopping_time)
      yield env.process(self.at(grocery_store, env, t))

  def exercise(self, env, city):
    self.action = Human.actions['exercise']
    park = self._select_location(location_type="park", city=city)
    t = _draw_random_discreet_gaussian(self.avg_shopping_time, self.scale_shopping_time)
    yield env.process(self.at(park, env, t))

  def _select_location(self, location_type, city):
      """
      Preferential exploration treatment of visiting places
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
          pool_pref = [(city.compute_distance(self.location, m)+1e-1)**-1 for m in city.miscs if m != self.location]
          pool_locs = [m for m in city.miscs if m != self.location]
          locs = city.miscs
          visited_locs = self.visits.miscs

      else:
          raise

      if S == 0:
          p_exp = 1.0
      else:
          p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)

      if np.random.random() < p_exp and S != len(locs):
          # explore
          cands = [i for i in locs if i not in visited_locs]
          cands = [(loc, pool_pref[i]) for i,loc in enumerate(cands)]
      else:
          # exploit
          cands = [(i, count)  for i, count in visited_locs.items()]

      cands, scores = zip(*cands)
      loc = np.random.choice(cands, p=_normalize_scores(scores))
      visited_locs[loc] += 1
      return loc

  def at(self, location, env, duration):
      self.location = location
      location.humans.add(self)
      self.leaving_time = duration + env.now
      self.start_time = env.now

      # Report all the encounters
      for h in location.humans:
        if h == self or location.location_type == 'household':
          continue
        Event.log_encounter( self, h,
            location=location,
            duration=min(self.leaving_time, h.leaving_time) - max(self.start_time, h.start_time),
            distance=np.random.randint(50, 1000), # cm  #TODO: prop to Area and inv. prop to capacity
            time=env.timestamp,
        )

      if not self.is_sick:
        if random.random() < location.contamination_proba():
          self.infection_timestamp = env.timestamp
          Event.log_contaminate(self, env.timestamp)
      yield env.timeout(duration/TICK_MINUTE)
      location.humans.remove(self)


####### MONITORING
class EventMonitor(object):
  def __init__(self, f=None):
    self.data = []
    self.f = f or 60

  def run(self, env, city: City):
    while True:
      self.data = city.events
      yield env.timeout(self.f/TICK_MINUTE)
      # self.plot()


  def plot(self):
    display.clear_output(wait=True)
    pl.clf()
    for e in Event.members():
        event_series = len([d for d in self.data if d['event_type'] == e])
        pl.plot(time_series, event_series, label=k)

    pl.title(f"City at {self.data[-1]['time']}")
    pl.legend()
    display.display(pl.gcf())

  def dump(self, dest:str=None):
    if dest is None:
        print(json.dumps(self.data, indent=1, default=_json_serialize))
    else:
        with open(f"{dest}.pkl", 'wb') as f:
            pickle.dump(self.data, f)

class TimeMonitor(object):
  def __init__(self, f=None):
    self.data = []
    self.f = f or 60

  def run(self, env, city: City):
    while True:
      print(env.timestamp)
      yield env.timeout(self.f/TICK_MINUTE)
