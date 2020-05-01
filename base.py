import simpy
import math
import copy
import datetime
import itertools
import numpy as np
from collections import defaultdict
from orderedset import OrderedSet
import copy
import zipfile
from config import *
from utils import compute_distance, _get_random_area, _draw_random_discreet_gaussian, get_intervention
from track import Tracker
from models.run import integrated_risk_pred
from interventions import *

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

class City(simpy.Environment):

    def __init__(self, env, n_people, rng, x_range, y_range, start_time, init_percent_sick, Human):
        self.env = env
        self.rng = rng
        self.x_range = x_range
        self.y_range = y_range
        self.total_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        self.n_people = n_people
        self.start_time = start_time
        self.init_percent_sick = init_percent_sick
        self.last_date_to_check_tests = self.env.timestamp.date()
        self.test_count_today = defaultdict(int)
        self.test_type_preference = list(zip(*sorted(TEST_TYPES.items(), key=lambda x:x[1]['preference'])))[0]
        print("Initializing locations ...")
        self.initialize_locations()

        self.humans = []
        self.households = OrderedSet()
        print("Initializing humans ...")
        self.initialize_humans(Human)

        self.log_static_info()

        print("Computing their preferences")
        self._compute_preferences()
        self.tracker = Tracker(env, self)
        # self.tracker.track_initialized_covid_params(self.humans)

        self.intervention = None

    def create_location(self, specs, type, name, area=None):
        _cls = Location
        if type in ['household', 'senior_residency']:
            _cls = Household
        if type == 'hospital':
            _cls = Hospital

        return   _cls(
                        env=self.env,
                        rng=self.rng,
                        name=f"{type}:{name}",
                        location_type=type,
                        lat=self.rng.randint(*self.x_range),
                        lon=self.rng.randint(*self.y_range),
                        area=area,
                        social_contact_factor=specs['social_contact_factor'],
                        capacity= None if not specs['rnd_capacity'] else self.rng.randint(*specs['rnd_capacity']),
                        surface_prob = specs['surface_prob']
                        )
    @property
    def tests_available(self):
        if self.last_date_to_check_tests != self.env.timestamp.date():
            self.last_date_to_check_tests = self.env.timestamp.date()
            for k in self.test_count_today.keys():
                self.test_count_today[k] = 0
        return any(self.test_count_today[test_type] < TEST_TYPES[test_type]['capacity'] for test_type in self.test_type_preference)

    def get_available_test(self):
        for test_type in self.test_type_preference:
            if self.test_count_today[test_type] < TEST_TYPES[test_type]['capacity']:
                self.test_count_today[test_type] += 1
                return test_type

    def initialize_locations(self):
        for location, specs in LOCATION_DISTRIBUTION.items():
            if location in ['household']:
                continue

            n = math.ceil(self.n_people/specs["n"])
            area = _get_random_area(n, specs['area'] * self.total_area, self.rng)
            locs = [self.create_location(specs, location, i, area[i]) for i in range(n)]
            setattr(self, f"{location}s", locs)

    def initialize_humans(self, Human):
        # allocate humans to houses such that (unsolved)
        # 1. average number of residents in a house is (approx.) 2.6
        # 2. not all residents are below 15 years of age
        # 3. age occupancy distribution follows HUMAN_DSITRIBUTION.residence_preference.house_size

        # current implementation is an approximate heuristic

        # make humans
        count_humans = 0
        house_allocations = {2:[], 3:[], 4:[], 5:[]}
        n_houses = 0
        for age_bin, specs in HUMAN_DISTRIBUTION.items():
            n = math.ceil(specs['p'] * self.n_people)
            ages = self.rng.randint(*age_bin, size=n)

            senior_residency_preference = specs['residence_preference']['senior_residency']

            professions = ['healthcare', 'school', 'others', 'retired']
            p = [specs['profession_profile'][x] for x in professions]
            profession = self.rng.choice(professions, p=p, size=n)

            for i in range(n):
                count_humans += 1
                age = ages[i]

                # residence
                res = None
                if self.rng.random() < senior_residency_preference:
                    res = self.rng.choice(self.senior_residencys)
                # workplace
                if profession[i] == "healthcare":
                    workplace = self.rng.choice(self.hospitals + self.senior_residencys)
                elif profession[i] == 'school':
                    workplace = self.rng.choice(self.schools)
                elif profession[i] == 'others':
                    type_of_workplace = self.rng.choice([0,1,2], p=OTHERS_WORKPLACE_CHOICE, size=1).item()
                    type_of_workplace = [self.workplaces, self.stores, self.miscs][type_of_workplace]
                    workplace = self.rng.choice(type_of_workplace)
                else:
                    workplace = res

                self.humans.append(Human(
                        env=self.env,
                        city=self,
                        rng=self.rng,
                        name=count_humans,
                        age=age,
                        household=res,
                        workplace=workplace,
                        profession=profession[i],
                        rho=RHO,
                        gamma=GAMMA,
                        infection_timestamp=self.start_time if self.rng.random() < self.init_percent_sick else None
                        )
                    )

        # assign houses
        # stores tuples - (location, current number of residents, maximum number of residents allowed)
        remaining_houses = []
        for human in self.humans:
            if human.household is not None:
                continue
            if len(remaining_houses) == 0:
                cap = self.rng.choice(range(1,6), p=HOUSE_SIZE_PREFERENCE, size=1)
                x = self.create_location(LOCATION_DISTRIBUTION['household'], 'household', len(self.households))

                remaining_houses.append((x, cap))

            # get_best_match
            res = None
            for  c, (house, n_vacancy) in enumerate(remaining_houses):
                new_avg_age = (human.age + sum(x.age for x in house.residents))/(len(house.residents) + 1)
                if new_avg_age > MIN_AVG_HOUSE_AGE:
                    res = house
                    n_vacancy -= 1
                    if n_vacancy == 0:
                        remaining_houses = remaining_houses[:c] + remaining_houses[c+1:]
                    break

            if res is None:
                for i, (l,u) in enumerate(HUMAN_DISTRIBUTION.keys()):
                    if l <= human.age < u:
                        bin = (l,u)
                        break

                house_size_preference = HUMAN_DISTRIBUTION[(l,u)]['residence_preference']['house_size']
                cap = self.rng.choice(range(1,6), p=house_size_preference, size=1)
                res = self.create_location(LOCATION_DISTRIBUTION['household'], 'household', len(self.households))
                if cap - 1 > 0:
                    remaining_houses.append((res, cap-1))

            # FIXME: there is some circular reference here
            res.residents.append(human)
            human.assign_household(res)
            self.households.add(res)

        # assign area to house
        area = _get_random_area(len(self.households), LOCATION_DISTRIBUTION['household']['area'] * self.total_area, self.rng)
        for i,house in enumerate(self.households):
            house.area = area[i]

        # this allows for easy O(1) access of humans for message passing
        self.hd = {human.name: human for human in self.humans}

    def log_static_info(self):
        for h in self.humans:
            Event.log_static_info(self, h, self.env.timestamp)

    @property
    def events(self):
        return list(itertools.chain(*[h.events for h in self.humans]))

    def pull_events(self):
        return list(itertools.chain(*[h.pull_events() for h in self.humans]))

    def _compute_preferences(self):
        """ compute preferred distribution of each human for park, stores, etc."""
        for h in self.humans:
            h.stores_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.stores]
            h.parks_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.parks]

    def run(self, duration, outfile, start_time, all_possible_symptoms, port, n_jobs):
        self.current_day = 0

        while True:

            #
            if INTERVENTION_DAY >= 0 and self.current_day == INTERVENTION_DAY:
                self.intervention = get_intervention(INTERVENTION)
                _ = [h.notify(self.intervention) for h in self.humans]
                print(self.intervention)

            # update risk every day
            if isinstance(self.intervention, Tracing):
                self.intervention.update_human_risks(city=self,
                                symptoms=all_possible_symptoms, port=port,
                                n_jobs=n_jobs, data_path=outfile)

            #
            # if (COLLECT_TRAINING_DATA or GET_RISK_PREDICTOR_METRICS) and (self.current_day == 0 and INTERVENTION_DAY < 0):
            #     _ = [h.notify(collect_training_data=True) for h in self.humans]
            #     print("naive risk calculation without changing behavior... Humans notified!")

            self.tracker.increment_day()
            self.current_day += 1

            yield self.env.timeout(duration / TICK_MINUTE)

class Location(simpy.Resource):

    def __init__(self, env, rng, area, name, location_type, lat, lon,
            social_contact_factor, capacity, surface_prob):

        if capacity is None:
            capacity = simpy.core.Infinity

        super().__init__(env, capacity)
        self.humans = OrderedSet() #OrderedSet instead of set for determinism when iterating
        self.name = name
        self.rng = rng
        self.lat = lat
        self.lon = lon
        self.area = area
        self.location_type = location_type
        self.social_contact_factor = social_contact_factor
        self.env = env
        self.contamination_timestamp = datetime.datetime.min
        self.contaminated_surface_probability = surface_prob
        self.max_day_contamination = 0

    def infectious_human(self):
        return any([h.is_infectious for h in self.humans])

    def __repr__(self):
        return f"{self.name} - occ:{len(self.humans)}/{self.capacity} - I:{self.is_contaminated}"

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
        s = self.__dict__
        if s.get('env'):
            del s['env']
        if s.get('rng'):
            del s['rng']
        if s.get('_env'):
            del s['_env']
        if s.get('contamination_timestamp'):
            del s['contamination_timestamp']
        if s.get('residents'):
            del s['residents']
        if s.get('humans'):
            del s['humans']
        return s

class Household(Location):
    def __init__(self, **kwargs):
        super(Household, self).__init__(**kwargs)
        self.residents = []


class Hospital(Location):
    ICU_AREA = 0.10
    ICU_CAPACITY = 0.10
    def __init__(self, **kwargs):
        env = kwargs.get('env')
        rng = kwargs.get('rng')
        capacity = kwargs.get('capacity')
        name = kwargs.get("name")
        lat = kwargs.get('lat')
        lon = kwargs.get('lon')
        area = kwargs.get('area')
        surface_prob = kwargs.get('surface_prob')
        social_contact_factor = kwargs.get('social_contact_factor')

        super(Hospital, self).__init__( env=env,
                                        rng=rng,
                                        area=area * (1-self.ICU_AREA),
                                        name=name,
                                        location_type="hospital",
                                        lat=lat,
                                        lon=lon,
                                        social_contact_factor=social_contact_factor,
                                        capacity=int(capacity* (1- self.ICU_CAPACITY)),
                                        surface_prob=surface_prob,
                                        )
        self.location_contamination = 1
        self.icu = ICU( env=env,
                        rng=rng,
                        area=area * (self.ICU_AREA),
                        name=f"{name}-icu",
                        location_type="hospital-icu",
                        lat=lat,
                        lon=lon,
                        social_contact_factor=social_contact_factor,
                        capacity=int(capacity* (self.ICU_CAPACITY)),
                        surface_prob=surface_prob,
                        )

    def add_human(self, human):
        human.obs_hospitalized = True
        super().add_human(human)

    def remove_human(self, human):
        human.obs_hospitalized = False
        super().remove_human(human)


class ICU(Location):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    contamination = 'contamination'
    recovered = 'recovered'
    static_info = 'static_info'
    visit = 'visit'
    daily = 'daily'

    @staticmethod
    def members():
        return [Event.test, Event.encounter, Event.contamination, Event.static_info, Event.visit, Event.daily]

    @staticmethod
    def log_encounter(human1, human2, location, duration, distance, infectee, time):

        h_obs_keys   = ['obs_hospitalized', 'obs_in_icu',
                        'obs_lat', 'obs_lon']

        h_unobs_keys = ['carefulness', 'viral_load', 'infectiousness',
                        'symptoms', 'is_exposed', 'is_infectious',
                        'infection_timestamp', 'is_really_sick',
                        'is_extremely_sick', 'sex',  'wearing_mask', 'mask_efficacy']

        loc_obs_keys = ['location_type', 'lat', 'lon']
        loc_unobs_keys = ['contamination_probability', 'social_contact_factor']

        obs, unobs = [], []

        same_household = (human1.household.name == human2.household.name) & (location.name == human1.household.name)
        for human in [human1, human2]:
            o = {key:getattr(human, key) for key in h_obs_keys}
            obs.append(o)
            u = {key:getattr(human, key) for key in h_unobs_keys}
            u['human_id'] = human.name
            u['location_is_residence'] = human.household == location
            u['got_exposed'] = infectee == human.name if infectee else False
            u['exposed_other'] = infectee != human.name if infectee else False
            u['same_household'] = same_household
            u['infectiousness_start_time'] = None if not u['got_exposed'] else human.infection_timestamp + datetime.timedelta(days=human.infectiousness_onset_days)
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
    def log_test(human, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.test,
                'time': time,
                'payload': {
                    'observed':{
                        'result': human.reported_test_result,
                        'test_type':human.reported_test_type,
                        'validated_test_result':human.test_result_validated
                    },
                    'unobserved':{
                        'test_type':human.test_type,
                        'result': human.test_result
                    }

                }
            }
        )

    @staticmethod
    def log_daily(human, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.daily,
                'time': time,
                'payload': {
                    'observed':{
                        "reported_symptoms": human.obs_symptoms
                    },
                    'unobserved':{
                        'infectiousness': human.infectiousness,
                        "viral_load": human.viral_load,
                        "all_symptoms": human.all_symptoms,
                        "covid_symptoms":human.covid_symptoms,
                        "flu_symptoms":human.flu_symptoms,
                        "cold_symptoms":human.cold_symptoms

                    }
                }
            }
        )

    @staticmethod
    def log_exposed(human, source, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.contamination,
                'time': time,
                'payload': {
                    'observed':{
                    },
                    'unobserved':{
                      'exposed': True,
                      'source':source.name,
                      'source_is_location': 'human' not in source.name,
                      'source_is_human': 'human' in source.name,
                      'infectiousness_start_time': human.infection_timestamp + datetime.timedelta(days=human.infectiousness_onset_days)
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


    @staticmethod
    def log_static_info(city, human, time):
        h_obs_keys = ['obs_preexisting_conditions',  "obs_age", "obs_sex", "obs_is_healthcare_worker"]
        h_unobs_keys = ['preexisting_conditions', "age", "sex", "is_healthcare_worker"]
        obs_payload = {key:getattr(human, key) for key in h_obs_keys}
        unobs_payload = {key:getattr(human, key) for key in h_unobs_keys}

        if human.workplace.location_type in ['healthcare', 'store', 'misc', 'senior_residency']:
            obs_payload['n_people_workplace'] = 'many people'
        elif "workplace" == human.workplace.location_type:
            obs_payload['n_people_workplace'] = 'few people'
        else:
            obs_payload['n_people_workplace'] = 'no people outside my household'

        obs_payload['household_size'] = len(human.household.residents)

        human.events.append(
            {
                'human_id': human.name,
                'event_type':Event.static_info,
                'time':time,
                'payload':{
                    'observed': obs_payload,
                    'unobserved':unobs_payload
                }

            }
        )

class DummyEvent:
    @staticmethod
    def log_encounter(*args, **kwargs):
        pass

    @staticmethod
    def log_test(*args, **kwargs):
        pass

    @staticmethod
    def log_recovery(*args, **kwargs):
        pass

    @staticmethod
    def log_exposed(*args, **kwargs):
        pass

    @staticmethod
    def log_static_info(*args, **kwargs):
        pass

    @staticmethod
    def log_visit(*args, **kwargs):
        pass

    @staticmethod
    def log_daily(*args, **kwargs):
        pass

class Contacts(object):
    def __init__(self, has_app):
        self.messages = []
        self.messages_by_day = defaultdict(list)
        self.update_messages = []
        # human --> [[date, counts], ...]
        self.book = {}
        self.has_app = has_app
        self.risk_level_history = []

    def add(self, **kwargs):
        human = kwargs.get("human")
        self_human = kwargs.get("self_human")
        timestamp = kwargs.get("timestamp")
        current_risk = kwargs.get("current_risk")
        cur_day = (timestamp - human.env.initial_timestamp).days
        cur_message = human.cur_message(cur_day)
        if not human.has_app or not self_human.has_app:
            self.messages.append(cur_message)
            self.messages_by_day[cur_day].append(cur_message)

        if human not in self.book:
            self.book[human] = [[timestamp.date(), 1]]
            return
        if timestamp.date() != self.book[human][-1][0]:
            self.book[human].append([timestamp.date(), 1])
        else:
            self.book[human][-1][1] += 1
        self.update_book(human, timestamp.date())

    def update_book(self, human, date=None, risk_level = None):
        # keep the history of risk levels (transformers)
        if risk_level:
            if len(self.risk_level_history) > TRACING_N_DAYS_HISTORY:
                self.risk_level_history = self.risk_level_history[1:]
            self.risk_level_history.append(human.risk_level)

        if date is None:
            date = self.book[human][-1][0] # last contact date

        remove_idx = -1
        for history in self.book[human]:
            if (date - history[0]).days > TRACING_N_DAYS_HISTORY:
                remove_idx += 1
            else:
                break
        self.book[human] = self.book[human][remove_idx:]

        # TODO: this should contain only todays info; clean up history should happen once per day
        if False:
            remove_idx = 0
            for historical_message in self.messages:
                if (human.env.timestamp - human.env.initial_timestamp).days - historical_message.day > TRACING_N_DAYS_HISTORY:
                    remove_idx += 1
                else:
                    break
            self.messages = self.messages[remove_idx:]

    def send_message(self, owner, tracing_method, order=1, reason="test", payload=None):
        p_contact = tracing_method.p_contact
        delay = tracing_method.delay
        app = tracing_method.app
        today = (owner.env.timestamp - owner.env.initial_timestamp).days
        if app and not owner.has_app:
            return

        for idx, human in enumerate(self.book):

            redundant_tracing = human.message_info['traced'] and tracing_method.dont_trace_traced
            if redundant_tracing: # manual and digital - no effect of new messages
                continue

            if not app or (app and human.has_app):
                if human.rng.random() < p_contact:
                    self.update_book(human)
                    t = 0
                    if delay:
                        t = _draw_random_discreet_gaussian(MANUAL_TRACING_DELAY_AVG, MANUAL_TRACING_DELAY_STD, human.rng)

                    total_contacts = sum(map(lambda x:x[1], self.book[human]))
                    human.update_risk(update_messages={'n':total_contacts, 'delay': t, 'order':order, 'reason':reason, 'payload':payload})
                    sent_at = human.env.timestamp + datetime.timedelta(minutes=idx)
                    for i in range(total_contacts):
                        # FIXME when we have messages sent hourly with a bucketed set of users sending messages
                        human.update_messages.append(owner.cur_message_risk_update(today, owner.uid, owner.risk, sent_at))
