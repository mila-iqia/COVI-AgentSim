"""
[summary]
"""
import simpy
import math
import datetime
import itertools
from collections import defaultdict

from covid19sim.configs.config import *
from covid19sim.utils import compute_distance, _get_random_area, _draw_random_discreet_gaussian, get_intervention, calculate_average_infectiousness
from covid19sim.track import Tracker
from covid19sim.interventions import *
from covid19sim.frozen.utils import update_uid
from covid19sim.configs.constants import *
from covid19sim.configs.exp_config import ExpConfig


class Env(simpy.Environment):
    """
    Custom simpy.Environment
    """

    def __init__(self, initial_timestamp):
        """
        Args:
            initial_timestamp (datetime.datetime): The environment's initial timestamp
        """
        self.initial_timestamp = datetime.datetime.combine(initial_timestamp.date(),
                                                           datetime.time())
        self.ts_initial = int(self.initial_timestamp.timestamp())
        super().__init__(self.ts_initial)

    @property
    def timestamp(self):
        """
        Returns:
            datetime.datetime: Current date.
        """
        #
        ##
        ## The following is preferable, but not equivalent to the initial
        ## version, because timedelta ignores Daylight Saving Time.
        ##
        #
        #return datetime.datetime.fromtimestamp(int(self.now))
        #
        return self.initial_timestamp + datetime.timedelta(
            seconds=self.now-self.ts_initial)

    def minutes(self):
        """
        Returns:
            int: Current timestamp minute
        """
        return self.timestamp.minute

    def hour_of_day(self):
        """
        Returns:
            int: Current timestamp hour
        """
        return self.timestamp.hour

    def day_of_week(self):
        """
        Returns:
            int: Current timestamp day of the week
        """
        return self.timestamp.weekday()

    def is_weekend(self):
        """
        Returns:
            bool: Current timestamp day is a weekend day
        """
        return self.day_of_week() >= 5

    def time_of_day(self):
        """
        Time of day in iso format
        datetime(2020, 2, 28, 0, 0) => '2020-02-28T00:00:00'

        Returns:
            str: iso string representing current timestamp
        """
        return self.timestamp.isoformat()

class City:
    """
    City
    """

    def __init__(self, env, n_people, init_percent_sick, rng, x_range, y_range, Human):
        """
        Args:
            env (simpy.Environment): [description]
            n_people (int): Number of people in the city
            init_percent_sick: % of population to be infected on day 0
            rng (np.random.RandomState): Random number generator
            x_range (tuple): (min_x, max_x)
            y_range (tuple): (min_y, max_y)
            init_percent_sick (float): % of humans sick at the start of the simulation
            Human (covid19.simulator.Human): Class for the city's human instances
        """
        self.env = env
        self.rng = rng
        self.x_range = x_range
        self.y_range = y_range
        self.total_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        self.n_people = n_people
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

    @property
    def start_time(self):
        return datetime.datetime.fromtimestamp(self.env.ts_initial)

    def create_location(self, specs, type, name, area=None):
        """
        Create a location instance based on `type`

        Specs is a dict like:
        {
            "n" : (int) number of such locations,
            "area": (float) locations' typical area,
            "social_contact_factor": (float(0:1)) how much people are close to each other
                see contamination_probability(),
            "surface_prob": [0.1, 0.1, 0.3, 0.2, 0.3], distribution over types of surfaces
                in that location
            "rnd_capacity": (tuple, optional) Either None or a tuple of ints (min, max)
            describing the args of np.random.randint,
        }

        Args:
            specs (dict): location's parameters
            type (str): "household" and "senior_residency" create a Household instance,
                "hospital" creates a Hospital, other strings create a generic Location
            name (str): Location's name, created as `type:name`
            area (float, optional): Location's area. Defaults to None.

        Returns:
            Location | Household | Hospital: new location instance
        """
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
        """
        Returns:
            bool: tests are available
        """
        if self.last_date_to_check_tests != self.env.timestamp.date():
            self.last_date_to_check_tests = self.env.timestamp.date()
            for k in self.test_count_today.keys():
                self.test_count_today[k] = 0
        return any(self.test_count_today[test_type] < TEST_TYPES[test_type]['capacity'] for test_type in self.test_type_preference)

    def get_available_test(self):
        """
        Returns a test_type: the first type that is available according to preference
        hierarchy (TEST_TYPES[test_type]['preference']).

        See TEST_TYPES in config.py

        Returns:
            str: available test_type
        """
        for test_type in self.test_type_preference:
            if self.test_count_today[test_type] < TEST_TYPES[test_type]['capacity']:
                self.test_count_today[test_type] += 1
                return test_type

    def initialize_locations(self):
        """
        Create locations according to config.py / LOCATION_DISTRIBUTION.
        The City instance will have attributes <location_type>s = list(location(*args))
        """
        for location, specs in LOCATION_DISTRIBUTION.items():
            if location in ['household']:
                continue

            n = math.ceil(self.n_people/specs["n"])
            area = _get_random_area(n, specs['area'] * self.total_area, self.rng)
            locs = [self.create_location(specs, location, i, area[i]) for i in range(n)]
            setattr(self, f"{location}s", locs)

    def initialize_humans(self, Human):
        """
        allocate humans to houses such that (unsolved)
        1. average number of residents in a house is (approx.) 2.6
        2. not all residents are below 15 years of age
        3. age occupancy distribution follows HUMAN_DSITRIBUTION.residence_preference.house_size

        current implementation is an approximate heuristic

        Args:
            Human (Class): Class for the city's human instances
        """


        # make humans
        count_humans = 0
        house_allocations = {2:[], 3:[], 4:[], 5:[]}
        n_houses = 0

        # initial infections
        init_infected = math.ceil(self.init_percent_sick * self.n_people)
        chosen_infected = set(self.rng.choice(self.n_people, init_infected, replace=False).tolist())

        # app users
        all_has_app = ExpConfig.get('P_HAS_APP') < 0
        n_apps = ExpConfig.get('P_HAS_APP') * self.n_people if ExpConfig.get('P_HAS_APP') > 0 else self.n_people
        n_apps_per_age = {k:math.ceil(v * n_apps) for k,v in APP_USERS_FRACTION_BY_AGE.items()}

        for age_bin, specs in HUMAN_DISTRIBUTION.items():
            n = math.ceil(specs['p'] * self.n_people)
            ages = self.rng.randint(*age_bin, size=n)

            senior_residency_preference = specs['residence_preference']['senior_residency']

            professions = ['healthcare', 'school', 'others', 'retired']
            p = [specs['profession_profile'][x] for x in professions]
            profession = self.rng.choice(professions, p=p, size=n)

            # select who should have app based on APP_USERS_FRACTION_BY_AGE
            chosen_app_user_bin = []
            for my_age in ages:
                for x, frac in APP_USERS_FRACTION_BY_AGE.items():
                    if x[0] <= my_age <= x[1]:
                        chosen_app_user_bin.append(x)
                        break

            for i in range(n):
                count_humans += 1
                age = ages[i]

                # should the person has an app?
                current_app_bin = chosen_app_user_bin[i]
                if n_apps_per_age[current_app_bin] > 0:
                    has_app = True
                    n_apps_per_age[current_app_bin] -= 1
                else:
                    has_app = False

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
                        has_app=has_app or all_has_app,
                        name=count_humans,
                        age=age,
                        household=res,
                        workplace=workplace,
                        profession=profession[i],
                        rho=RHO,
                        gamma=GAMMA,
                        infection_timestamp=self.start_time if count_humans - 1 in chosen_infected else None
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
        """
        Logs events for all humans in the city
        """
        for h in self.humans:
            Event.log_static_info(self, h, self.env.timestamp)

    @property
    def events(self):
        """
        Get all events of all humans in the city

        Returns:
            list: all of everyone's events
        """
        return list(itertools.chain(*[h.events for h in self.humans]))

    def events_slice(self, begin, end):
        """
        Get all sliced events of all humans in the city

        Args:
            begin (datetime.datetime): minimum time of events
            end (int): maximum time of events

        Returns:
            list: The list each human's events, restricted to a slice
        """
        return list(itertools.chain(*[h.events_slice(begin, end) for h in self.humans]))

    def pull_events_slice(self, end):
        """
        Get the list of all human's events before `end`.
        /!\ Modifies each human's events

        Args:
            end (datetime.datetime): maximum time of pulled events

        Returns:
            list: All the events which occured before `end`
        """
        return list(itertools.chain(*[h.pull_events_slice(end) for h in self.humans]))

    def _compute_preferences(self):
        """
        Compute preferred distribution of each human for park, stores, etc.
        /!\ Modifies each human's stores_preferences and parks_preferences
        """
        for h in self.humans:
            h.stores_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.stores]
            h.parks_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.parks]

    def run(self, duration, outfile, all_possible_symptoms, port, n_jobs):
        """
        Run the City DOCTODO(improve this)

        Args:
            duration (int): duration of a step, in seconds.
            outfile (str): may be None, the run's output file to write to
            all_possible_symptoms (dict): copy of SYMPTOMS_META (config.py)
            port (int): the port for integrated_risk_pred when updating the humans'
                risk
            n_jobs (int): the number of jobs for integrated_risk_pred when updating
                the humans' risk

        Yields:
            simpy.Timeout
        """
        self.current_day = 0
        humans_notified = False

        while True:
            # Notify humans to follow interventions on intervention day
            if self.current_day == ExpConfig.get('INTERVENTION_DAY') and not humans_notified:
                self.intervention = get_intervention(key=ExpConfig.get('INTERVENTION'),
                                                     RISK_MODEL=ExpConfig.get('RISK_MODEL'),
                                                     TRACING_ORDER=ExpConfig.get('TRACING_ORDER'),
                                                     TRACE_SYMPTOMS=ExpConfig.get('TRACE_SYMPTOMS'),
                                                     TRACE_RISK_UPDATE=ExpConfig.get('TRACE_RISK_UPDATE'),
                                                     SHOULD_MODIFY_BEHAVIOR=ExpConfig.get('SHOULD_MODIFY_BEHAVIOR'))

                _ = [h.notify(self.intervention) for h in self.humans]
                print(self.intervention)
                if ExpConfig.get('COLLECT_TRAINING_DATA'):
                    print("naive risk calculation without changing behavior... Humans notified!")

                humans_notified = True

            # iterate over humans, and if it's their timeslot, then update their infectionsness, symptoms, and message info
            for human in self.humans:
                # if it's your time to update,
                if self.env.timestamp.hour not in human.time_slots:
                    continue

                # And you haven't updated today
                if human.last_date.get('symptoms_updated') == self.env.timestamp.date():
                    continue

                human.last_date['symptoms_updated'] = self.env.timestamp.date()
                human.update_symptoms()
                human.update_reported_symptoms()
                human.update_risk(symptoms=human.symptoms)
                human.infectiousnesses.appendleft(calculate_average_infectiousness(human))

                Event.log_daily(human, human.env.timestamp)
                self.tracker.track_symptoms(human)

                # keep only past N_DAYS contacts
                if human.tracing:
                    for type_contacts in ['n_contacts_tested_positive', 'n_contacts_symptoms', \
                                          'n_risk_increased', 'n_risk_decreased', "n_risk_mag_decreased",
                                          "n_risk_mag_increased"]:
                        for order in human.message_info[type_contacts]:
                            if len(human.message_info[type_contacts][order]) > ExpConfig.get('TRACING_N_DAYS_HISTORY'):
                                human.message_info[type_contacts][order] = human.message_info[type_contacts][order][1:]
                            human.message_info[type_contacts][order].append(0)

            if isinstance(self.intervention, Tracing):
                self.intervention.update_human_risks(city=self,
                                symptoms=all_possible_symptoms, port=port,
                                n_jobs=n_jobs, data_path=outfile)
                self.tracker.track_risk_attributes(self.humans)

            # Let the hour pass
            yield self.env.timeout(int(duration))

            # increment the day / update uids if we start the timeslot 0
            if self.env.timestamp.hour == 0 and self.env.timestamp != self.env.initial_timestamp:
                # TODO: this is an assumption which will break in reality, instead of updating once per day everyone at the same time, it should be throughout the day
                for human in self.humans:
                    human.uid = update_uid(human.uid, human.rng)
                self.current_day += 1
                self.tracker.increment_day()


class Location(simpy.Resource):
    """
    Class representing generic locations used in the simulator
    """

    def __init__(self, env, rng, area, name, location_type, lat, lon,
            social_contact_factor, capacity, surface_prob):
        """
        Locations are created with city.create_location(), not instantiated directly

        Args:
            env (covid19sim.Env): Shared environment
            rng (np.random.RandomState): Random number generator
            area (float): Area of the location
            name (str): The location's name
            location_type (str): Location's type, see
            lat (float): Location's latitude
            lon (float): Location's longitude
            social_contact_factor (float): how much people are close to each other
                see contamination_probability() (this scales the contamination pbty)
            capacity (int): Daily intake capacity for the location (infinity if None).
            surface_prob (float): distribution of surface types in the Location. As
                different surface types have different contamination probabilities
                and virus "survival" durations, this will influence the contamination
                of humans at this location.
                Surfaces: aerosol, copper, cardboard, steel, plastic
        """


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
        """
        Returns:
            bool: Is there an infectious human currently at that location
        """
        return any([h.is_infectious for h in self.humans])

    def __repr__(self):
        """
        Returns:
            str: Representation of the Location
        """
        return f"{self.name} - occ:{len(self.humans)}/{self.capacity} - I:{self.is_contaminated}"

    def add_human(self, human):
        """
        Adds a human instance to the OrderedSet of humans at the location.
        If they are infectious, then location.contamination_timestamp is set to the
        env's timestamp and the duration of this contamination is set
        (location.max_day_contamination) according to the distribution of surfaces
        (location.contaminated_surface_probability) and the survival of the virus
        per surface type (MAX_DAYS_CONTAMINATION)

        Args:
            human (covid19sim.simulator.Human): The human to add.
        """
        self.humans.add(human)
        if human.is_infectious:
            self.contamination_timestamp = self.env.timestamp
            rnd_surface = float(self.rng.choice(a=MAX_DAYS_CONTAMINATION, size=1, p=self.contaminated_surface_probability))
            self.max_day_contamination = max(self.max_day_contamination, rnd_surface)

    def remove_human(self, human):
        """
        Remove a given human from location.human
        /!\ Human is not returned

        Args:
            human (covid19sim.simulator.Human): The human to remove
        """
        self.humans.remove(human)

    @property
    def is_contaminated(self):
        """
        Is the location currently contaminated? This depends on the moment
        when it got contaminated (see add_human()), current time and the
        duration of the contamination (location.max_day_contamination)

        Returns:
            bool: Is the place currently contaminating?
        """
        return self.env.timestamp - self.contamination_timestamp <= datetime.timedelta(days=self.max_day_contamination)

    @property
    def contamination_probability(self):
        """
        Contamination depends on the time the virus has been sitting on a given surface
        (location.max_day_contamination) and is linearly decayed over time.
        Then it is scaled by location.social_contact_factor

        If not location.is_contaminated, return 0.0

        Returns:
            float: probability that a human is contaminated when going to this location.
        """
        if self.is_contaminated:
            lag = (self.env.timestamp - self.contamination_timestamp)
            lag /= datetime.timedelta(days=1)
            p_infection = 1 - lag / self.max_day_contamination # linear decay; &envrionmental_contamination
            return self.social_contact_factor * p_infection
        return 0.0

    def __hash__(self):
        """
        Hash of the location is the hash of its name

        Returns:
            int: hash
        """
        return hash(self.name)

    def serialize(self):
        """
        This function serializes the location object by deleting
        non-serializable keys

        Returns:
            dict: serialized location
        """
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
    """
    Household location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(Household, self).__init__(**kwargs)
        self.residents = []


class Hospital(Location):
    """
    Hospital location class, inheriting from covid19sim.base.Location
    """
    ICU_AREA = 0.10
    ICU_CAPACITY = 0.10
    def __init__(self, **kwargs):
        """
        Create the Hospital and its ICU

        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
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
        """
        Add a human to the Hospital's OrderedSet through the Location's
        default add_human() method + set the human's obs_hospitalized attribute
        is set to True

        Args:
            human (covid19sim.simulator.Human): human to add
        """
        human.obs_hospitalized = True
        super().add_human(human)

    def remove_human(self, human):
        """
        Remove a human from the Hospital's Ordered set.
        On top of Location.remove_human(), the human's obs_hospitalized attribute is
        set to False

        Args:
            human (covid19sim.simulator.Human): human to remove
        """
        human.obs_hospitalized = False
        super().remove_human(human)


class ICU(Location):
    """
    Hospital location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Create a Hospital's ICU Location

        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super().__init__(**kwargs)

    def add_human(self, human):
        """
        Add a human to the ICU's OrderedSet through the Location's
        default add_human() method + set the human's obs_hospitalized and
        obs_in_icu attributes are set to True

        Args:
            human (covid19sim.simulator.Human): human to add
        """
        human.obs_hospitalized = True
        human.obs_in_icu = True
        super().add_human(human)

    def remove_human(self, human):
        """
        Remove a human from the ICU's Ordered set.
        On top of Location.remove_human(), the human's obs_hospitalized and
        obs_in_icu attributes are set to False

        Args:
            human (covid19sim.simulator.Human): human to remove
        """
        human.obs_hospitalized = False
        human.obs_in_icu = False
        super().remove_human(human)

class Event:
    """
    [summary]
    """
    test = 'test'
    encounter = 'encounter'
    contamination = 'contamination'
    recovered = 'recovered'
    static_info = 'static_info'
    visit = 'visit'
    daily = 'daily'

    @staticmethod
    def members():
        """
        DEPRECATED
        """
        return [Event.test, Event.encounter, Event.contamination, Event.static_info, Event.visit, Event.daily]

    @staticmethod
    def log_encounter(human1, human2, location, duration, distance, infectee, time):
        """
        Logs the encounter between `human1` and `human2` at `location` for `duration`
        while staying at `distance` from each other. If infectee is not None, it is
        either human1.name or human2.name.

        Each of the two humans gets its `events` attribute appended whit a dictionnary
        describing the encounter:

        human.events.append({
                'human_id':human.name,
                'event_type':Event.encounter,
                'time':time,
                'payload':{
                    'observed': obs_payload,  # None if one of the humans does not have
                                              # the app. Otherwise contains the observed
                                              # data: lat, lon, location_type
                    'unobserved':unobs_payload  # unobserved data, see loc_unobs_keys and
                                                # h_unobs_keys
                }
        })


        Args:
            human1 (covid19sim.simulator.Human): One of the encounter's 2 humans
            human2 (covid19sim.simulator.Human): One of the encounter's 2 humans
            location (covid19sim.base.Location): Where the encounter happened
            duration (int): duration of encounter
            distance (float): distance between people (TODO: meters? cm?)
            infectee (str | None): name of the human which is infected, if any.
                None otherwise
            time (datetime.datetime): timestamp of encounter
        """
        if ExpConfig.get('COLLECT_LOGS') is False:
            return

        h_obs_keys   = ['obs_hospitalized', 'obs_in_icu',
                        'obs_lat', 'obs_lon']

        h_unobs_keys = ['carefulness', 'viral_load', 'infectiousness',
                        'symptoms', 'is_exposed', 'is_infectious',
                        'infection_timestamp', 'is_really_sick',
                        'is_extremely_sick', 'sex',  'wearing_mask', 'mask_efficacy',
                        'risk', 'risk_level', 'rec_level']

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
        """
        Adds an event to a human's `events` list if COLLECT_LOGS is True.
        Events contains the test resuts time, reported_test_result,
        reported_test_type, test_result_validated, test_type, test_result
        split across observed and unobserved data.

        Args:
            human (covid19sim.simulator.Human): Human whose test should be logged
            time (datetime.datetime): Event's time
        """
        if ExpConfig.get('COLLECT_LOGS') is False:
            return

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
        """
        Adds an event to a human's `events` list containing daily health information
        like symptoms, infectiousness and viral_load.

        Args:
            human (covid19sim.simulator.Human): Human who's health should be logged
            time (datetime.datetime): Event time
        """
        if ExpConfig.get('COLLECT_LOGS') is False:
            return

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
        """
        [summary]

        Args:
            human ([type]): [description]
            source ([type]): [description]
            time ([type]): [description]
        """
        if ExpConfig.get('COLLECT_LOGS') is False:
            return

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
        """
        [summary]

        Args:
            human ([type]): [description]
            time ([type]): [description]
            death ([type]): [description]
        """
        if ExpConfig.get('COLLECT_LOGS') is False:
            return

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
        """
        [summary]

        Args:
            city ([type]): [description]
            human ([type]): [description]
            time ([type]): [description]
        """
        if ExpConfig.get('COLLECT_LOGS') is False:
            return

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


class Contacts(object):
    """
    [summary]
    """
    def __init__(self, has_app):
        """
        [summary]

        Args:
            has_app (bool): [description]
        """
        self.messages = []
        self.sent_messages_by_day = defaultdict(list)
        self.messages_by_day = defaultdict(list)
        self.update_messages = []
        # human --> [[date, counts], ...]
        self.book = {}
        self.has_app = has_app

    def add(self, **kwargs):
        """
        [summary]
        """
        human = kwargs.get("human")
        timestamp = kwargs.get("timestamp")

        if human not in self.book:
            self.book[human] = [[timestamp.date(), 1]]
            return

        if timestamp.date() != self.book[human][-1][0]:
            self.book[human].append([timestamp.date(), 1])
        else:
            self.book[human][-1][1] += 1
        self.update_book(human, timestamp.date())

    def update_book(self, human, date=None, risk_level = None):
        """
        [summary]

        Args:
            human ([type]): [description]
            date ([type], optional): [description]. Defaults to None.
            risk_level ([type], optional): [description]. Defaults to None.
        """
        # keep the history of risk levels (transformers)
        if date is None:
            date = self.book[human][-1][0] # last contact date

        remove_idx = -1
        for history in self.book[human]:
            if (date - history[0]).days > ExpConfig.get('TRACING_N_DAYS_HISTORY'):
                remove_idx += 1
            else:
                break
        self.book[human] = self.book[human][remove_idx:]

        # remove that human from the book
        if len(self.book[human]) == 0:
            self.book.pop(human)

    def send_message(self, owner, tracing_method, order=1, reason="test", payload=None):
        """
        [summary]

        Args:
            owner ([type]): [description]
            tracing_method ([type]): [description]
            order (int, optional): [description]. Defaults to 1.
            reason (str, optional): [description]. Defaults to "test".
            payload ([type], optional): [description]. Defaults to None.
        """
        p_contact = tracing_method.p_contact
        delay = tracing_method.delay
        app = tracing_method.app
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
                    owner.city.tracker.track_update_messages(owner, human, {'reason':reason})
