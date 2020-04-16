import simpy
import copy
import datetime
import itertools
from orderedset import OrderedSet
from config import TICK_MINUTE, MAX_DAYS_CONTAMINATION
from utils import compute_distance

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

    def __init__(self, stores, parks, hospitals, humans, miscs):
        self.stores = stores
        self.parks = parks
        self.hospitals = hospitals
        self.humans = humans
        self.miscs = miscs
        self._compute_preferences()

    @property
    def events(self):
        return list(itertools.chain(*[h.events for h in self.humans]))

    def _compute_preferences(self):
        """ compute preferred distribution of each human for park, stores, etc."""
        for h in self.humans:
            h.stores_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.stores]
            h.parks_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.parks]


class Location(simpy.Resource):

    def __init__(self, env, rng, capacity=simpy.core.Infinity, name='Safeway', location_type='stores', lat=None,
                 lon=None, area=None, cont_prob=None, surface_prob=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super().__init__(env, capacity)
        self.humans = OrderedSet() #OrderedSet instead of set for determinism when iterating
        self.name = name
        self.rng = rng
        self.lat = lat
        self.lon = lon
        self.area = area
        self.location_type = location_type
        self.social_contact_factor = cont_prob
        self.env = env
        self.contamination_timestamp = datetime.datetime.min
        self.contaminated_surface_probability = surface_prob
        self.max_day_contamination = 0
        self.location_contamination = cont_prob

    def infectious_human(self):
        return any([h.is_infectious for h in self.humans])

    def __repr__(self):
        return f"{self.name} - occ:{len(self.humans)}/{self.capacity} - I:{self.infectious_human()}"

    def add_human(self, human):
        self.humans.add(human)
        if human.is_infectious:
            self.contamination_timestamp = self.env.timestamp
            rnd_surface = float(self.rng.choice(a=MAX_DAYS_CONTAMINATION, size=1, p=self.contaminated_surface_probability))
            self.max_day_contamination = max(self.max_day_contamination, rnd_surface)

    def remove_human(self, human):
        self.humans.remove(human)

    @property
    def is_contaminated(self):
        return self.env.timestamp - self.contamination_timestamp <= datetime.timedelta(days=self.max_day_contamination)


    @property
    def contamination_probability(self):
        if self.is_contaminated:
            lag = (self.env.timestamp - self.contamination_timestamp)
            lag /= datetime.timedelta(days=1)
            p_infection = 1 - lag / self.max_day_contamination # linear decay; &envrionmental_contamination
            return self.social_contact_factor * p_infection
        return 0.0

    def __hash__(self):
        return hash(self.name)

    def serialize(self):
        """ This function serializes the location object"""
        s = copy.copy(self.__dict__)
        if s.get('env'):
            del s['env']
        if s.get('rng'):
            del s['rng']
        if s.get('_env'):
            del s['_env']
        if s.get('contamination_timestamp'):
            del s['contamination_timestamp']
        del s['humans']
        return s

class Hospital(Location):

    def __init__(self, env, rng, capacity=simpy.core.Infinity, name='vgh', location_type='hospital', lat=None,
                 lon=None, area=None, cont_prob=None, surface_prob=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super().__init__(env, rng, capacity, name, location_type, lat, lon, cont_prob)
        self.location_contamination = 1
        self.icu = ICU(env, rng, self, capacity=capacity/50, name=f"{name}_icu", location_type='icu', lat=lat, lon=lon,
                            area=None, cont_prob=None)

    def add_human(self, human):
        human.obs_hospitalized = True
        super().add_human(human)

    def remove_human(self, human):
        human.obs_hospitalized = False
        super().remove_human(human)


class ICU(Location):

    def __init__(self, env, rng, hospital, capacity=simpy.core.Infinity, name='icu', location_type='icu', lat=None,
                 lon=None, area=None, cont_prob=None, surface_prob=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super().__init__(env, rng, capacity, name, location_type, lat, lon, cont_prob)
        self.hospital = hospital

    def add_human(self, human):
        human.obs_hospitalized = True
        human.obs_in_icu = True
        super().add_human(human)

    def remove_human(self, human):
        human.obs_hospitalized = False
        human.obs_in_icu = False
        super().remove_human(human)


class Event:
    test = 'test'
    encounter = 'encounter'
    symptom_start = 'symptom_start'
    contamination = 'contamination'
    recovered = 'recovered'

    @staticmethod
    def members():
        return [Event.test, Event.encounter, Event.symptom_start, Event.contamination]

    @staticmethod
    def log_encounter(human1, human2, location, duration, distance, time):
        h_obs_keys   = ['obs_age', 'has_app', 'obs_preexisting_conditions',
                        'obs_symptoms', 'obs_test_result', 'obs_test_type',
                        'obs_hospitalized', 'obs_in_icu', 'wearing_mask',
                        'obs_test_validated', 'obs_lat', 'obs_lon']

        h_unobs_keys = ['age', 'carefullness', 'viral_load', 'infectiousness', 
                        'symptoms', 'is_exposed', 'is_infectious',
                        'household', 'workplace', 'infection_timestamp', 'really_sick',
                        'extremely_sick', 'preexisting_conditions']

        loc_obs_keys = ['location_type', 'lat', 'lon']
        loc_unobs_keys = ['contamination_probability', 'social_contact_factor']

        obs, unobs = [], []
        for human in [human1, human2]:
            o = {key:getattr(human, key) for key in h_obs_keys}
            obs.append(o)
            u = {key:getattr(human, key) for key in h_unobs_keys}
            u['is_infected'] = human.is_exposed or human.is_infectious
            u['human_id'] = human.name
            if u['household']:
                u['household'] = u['household'].serialize()
            if u['workplace']:
                u['workplace'] = u['workplace'].serialize()
            unobs.append(u)
        loc_obs = {key:getattr(location, key) for key in loc_obs_keys}
        loc_unobs = {key:getattr(location, key) for key in loc_unobs_keys}
        loc_unobs['location_p_infection'] = location.contamination_probability / location.social_contact_factor
        other_obs = {'duration':duration, 'distance':distance}
        both_have_app = human1.has_app and human2.has_app
        for i, human in [(0, human1), (1, human2)]:
            if both_have_app:
                obs_payload = {**loc_obs, **other_obs, 'human1':obs[i], 'human2':obs[1-i]}
                unobs_payload = {**loc_unobs, 'human1':unobs[i], 'human2':unobs[1-i]}
            else:
                obs_payload = {}
                unobs_payload = { **loc_obs, **loc_unobs, **other_obs, 'human1':{**obs[i], **unobs[i]},
                                    'human2': {**obs[1-i], **unobs[1-i]} }
                
            human.events.append({
                'human_id':human.name,
                'event_type':Event.encounter,
                'time':time,
                'payload':{'observed':obs_payload, 'unobserved':unobs_payload}
            })


    @staticmethod
    def log_test(human, result, test_type, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.test,
                'time': time,
                'payload': {
                    'observed':{
                        'result': result,
                        'type': test_type
                    },
                    'unobserved':{
                    }

                }
            }
        )

    @staticmethod
    def log_symptom_start(human, covid, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.symptom_start,
                'time': time,
                'payload': {
                    'observed':{
                        "reported_symptoms": human.all_reported_symptoms
                    },
                    'unobserved':{
                        'covid': covid,
                        "all_symptoms": human.all_symptoms
                    }

                }
            }
        )

    @staticmethod
    def log_exposed(human, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.contamination,
                'time': time,
                'payload': {
                    'unobserved':{
                      'exposed': True
                    }

                }
            }
        )

    @staticmethod
    def log_recovery(human, time, death):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.recovered,
                'time': time,
                'payload': {
                    'observed':{
                    },
                    'unobserved':{
                        'recovered': not death,
                        'death': death
                    }
                }
            }
        )
