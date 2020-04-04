import simpy
import datetime
import itertools
from config import TICK_MINUTE
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

    def __init__(self, stores, parks, humans, miscs):
        self.stores = stores
        self.parks = parks
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

    def __init__(self, env, capacity=simpy.core.Infinity, name='Safeway', location_type='stores', lat=None, lon=None,
                 cont_prob=None):
        super().__init__(env, capacity)
        self.humans = set()
        self.name = name
        self.lat = lat
        self.lon = lon
        self.location_type = location_type
        self.cont_prob = cont_prob

    def infectious_human(self):
        return any([h.is_infectious for h in self.humans])

    def __repr__(self):
        return f"{self.name} - occ:{len(self.humans)}/{self.capacity} - I:{self.infectious_human()}"

    @property
    def contamination_probability(self):
        if not self.infectious_human():
            return 0
        return self.cont_prob

    def __hash__(self):
        return hash(self.name)


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
        human1.events.append(
            {
                'human_id': human1.name,
                'event_type': Event.encounter,
                'time': time,
                'payload':{
                    'observed':{
                        'duration': duration,
                        'distance': distance,
                        'location_type': location.location_type,
                        'lat': location.lat,
                        'lon': location.lon,
                        'human1':{
                            'obs_lat': human1.obs_lat,
                            'obs_lon': human1.obs_lon,
                            'age': human1.age,
                            'reported_symptoms': human1.reported_symptoms,
                            'test_results': human1.test_results,
                        },
                        'human2':{ ## FIXME: i don't think human1 can see this information--> should go in unobserved??
                            'obs_lat': human2.obs_lat,
                            'obs_lon': human2.obs_lon,
                            'age': human2.age,
                            'reported_symptoms': human2.reported_symptoms,
                            'test_results': human2.test_results,
                        }

                    },
                    'unobserved':{
                        'contamination_prob': location.cont_prob,
                        'human1':{
                            'carefullness': human1.carefullness,
                            'is_infected': human1.is_exposed or human1.is_infectious,
                            'infectiousness': human1.infectiousness,
                            'symptoms': human1.symptoms,
                            'has_app': human1.has_app,
                        },
                        'human2':{
                            'carefullness': human2.carefullness,
                            'is_infected': human2.is_exposed or human2.is_infectious,
                            'infectiousness': human2.infectiousness,
                            'symptoms': human2.symptoms,
                            'has_app': human2.has_app,
                        }

                    }
                }
            }
        )

        human2.events.append(
            {
                'time': time,
                'event_type': Event.encounter,
                'human_id': human2.name,
                'payload':{
                    'observed':{
                        'duration': duration,
                        'distance': distance,
                        'location_type': location.location_type,
                        'lat': location.lat,
                        'lon': location.lon,
                        'human1':{
                            'obs_lat': human2.obs_lat,
                            'obs_lon': human2.obs_lon,
                            'age': human2.age,
                            'reported_symptoms': human2.reported_symptoms,
                            'test_results': human2.test_results,
                        },
                        'human2':{ ## FIXME: i don't think human1 can see this information--> should go in unobserved??
                            'obs_lat': human1.obs_lat,
                            'obs_lon': human1.obs_lon,
                            'age': human1.age,
                            'reported_symptoms': human1.reported_symptoms,
                            'test_results': human1.test_results,
                        }

                    },
                    'unobserved':{
                        'contamination_prob': location.cont_prob,
                        'human1':{
                            'carefullness': human2.carefullness,
                            'is_infected': human2.is_exposed or human2.is_infectious,
                            'infectiousness': human2.infectiousness,
                            'symptoms': human2.symptoms,
                            'has_app': human2.has_app,
                        },
                        'human2':{
                            'carefullness': human1.carefullness,
                            'is_infected': human1.is_exposed or human1.is_infectious,
                            'infectiousness': human1.infectiousness,
                            'symptoms': human1.symptoms,
                            'has_app': human1.has_app,
                        }

                    }
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
                    'observed':{
                        'result': result,
                    },
                    'unobserved':{
                    }

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
                    'observed':{
                    },
                    'unobserved':{
                        'covid': covid
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
                    'observed':{
                    },
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
