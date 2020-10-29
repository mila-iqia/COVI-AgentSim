"""
This module implements the `City` class which is responsible for running the environment in which all
humans will interact. Its `run` loop also contains the tracing application logic that runs every hour.
"""

import numpy as np
import copy
import datetime
import itertools
import math
import time
import os
import typing
from collections import defaultdict, Counter
from orderedset import OrderedSet

from covid19sim.utils.utils import compute_distance, _get_random_area, relativefreq2absolutefreq, _convert_bin_5s_to_bin_10s, log
from covid19sim.utils.demographics import get_humans_with_age, assign_households_to_humans, create_locations_and_assign_workplace_to_humans
from covid19sim.log.track import Tracker
from covid19sim.interventions.tracing import BaseMethod
from covid19sim.inference.message_utils import UIDType, UpdateMessage, RealUserIDType
from covid19sim.distribution_normalization.dist_utils import get_rec_level_transition_matrix
from covid19sim.interventions.tracing_utils import get_tracing_method
from covid19sim.locations.test_facility import TestFacility
from covid19sim.inference.server_utils import TransformerInferenceEngine
from covid19sim.utils.lmdb import LMDBSortedMap
from covid19sim.utils.mmap import MMAPArray
from covid19sim.locations.district import District
from covid19sim.locations.location import Location


if typing.TYPE_CHECKING:
    from covid19sim.human import Human
    from covid19sim.utils.env import Env

SimulatorMailboxType = typing.Dict[RealUserIDType, typing.List[UpdateMessage]]


class City:
    """
    City agent/environment class. Currently, a single city object will be instantiated at the start of
    a simulation. In the future, multiple 'cities' may be executed in parallel to simulate larger populations.
    """

    def __init__(
            self,
            env: "Env",
            n_people: int,
            init_fraction_sick: float,
            rng: np.random.RandomState,
            x_range: typing.Tuple,
            y_range: typing.Tuple,
            conf: typing.Dict,
            logfile: str = None,
    ):
        """
        Constructs a city object.

        Args:
            env (simpy.Environment): Keeps track of events and their schedule
            n_people (int): Number of people in the city
            init_fraction_sick (float): fraction of population to be infected on day 0
            rng (np.random.RandomState): Random number generator
            x_range (tuple): (min_x, max_x)
            y_range (tuple): (min_y, max_y)
            human_type (covid19.simulator.Human): Class for the city's human instances
            conf (dict): yaml configuration of the experiment
            logfile (str): filepath where the console output and final tracked metrics will be logged. Prints to the console only if None.
        """
        self.conf = conf
        self.logfile = logfile
        self.env = env
        self.rng = np.random.RandomState(rng.randint(2 ** 16))
        self.x_range = x_range
        self.y_range = y_range
        self.total_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        self.n_people = n_people
        self.init_fraction_sick = init_fraction_sick
        self.hash = int(time.time_ns())  # real-life time used as hash for inference server data hashing

        self.test_type_preference = list(zip(*sorted(conf.get("TEST_TYPES").items(), key=lambda x:x[1]['preference'])))[0]
        assert len(self.test_type_preference) == 1, "WARNING: Do not know how to handle multiple test types"
        self.max_capacity_per_test_type = {
            test_type: max(int(self.conf['PROPORTION_LAB_TEST_PER_DAY'] * self.n_people), 1)
            for test_type in self.test_type_preference
        }

        if 'DAILY_TARGET_REC_LEVEL_DIST' in conf:
            # QKFIX: There are 4 recommendation levels, value is hard-coded here
            self.daily_target_rec_level_dists = (np.asarray(conf['DAILY_TARGET_REC_LEVEL_DIST'], dtype=np.float_)
                                                   .reshape((-1, 4)))
        else:
            self.daily_target_rec_level_dists = None
        self.daily_rec_level_mapping = None
        self.covid_testing_facility = TestFacility(self.test_type_preference, self.max_capacity_per_test_type, env, conf)

        self.humans = MMAPArray(num_items=n_people, item_size=100)
        self.district_queues = LMDBSortedMap()
        self.hd = {}
        self.households = OrderedSet()
        self.age_histogram = None

        log("Initializing humans ...", self.logfile)
        self.initialize_humans_and_locations()

        log("Computing their preferences", self.logfile)
        self._compute_preferences()
        self.tracing_method = None

        # GAEN summary statistics that enable the individual to determine whether they should send their info
        self.risk_change_histogram = Counter()
        self.risk_change_histogram_sum = 0
        self.sent_messages_by_day: typing.Dict[int, int] = {}

        # note: for good efficiency in the simulator, we will not allow humans to 'download'
        # database diffs between their last timeslot and their current timeslot; instead, we
        # will give them the global mailbox object (a dictionary) and have them 'pop' all
        # messages they consume from their own (simulation-only!) personal mailbox
        self.global_mailbox: SimulatorMailboxType = LMDBSortedMap()

        # create a global inference engine
        self.inference_engine = None # TransformerInferenceEngine(conf)

        # split the location objects such as households between district objects
        self.num_districts = self.conf['NUM_DISTRICTS']
        self.split_locations_into_districts()

    @property
    def start_time(self):
        return datetime.datetime.fromtimestamp(self.env.ts_initial)

    def initialize_humans_and_locations(self):
        """
        Samples a synthetic population along with their dwellings and workplaces according to census.
        """
        self.age_histogram = relativefreq2absolutefreq(
            bins_fractions={(x[0], x[1]): x[2] for x in self.conf.get('P_AGE_REGION')},
            n_elements=self.n_people,
            rng=self.rng,
        )

        # initalize human objects
        self.humans = get_humans_with_age(self, self.age_histogram, self.conf, self.rng)

        # find best grouping to put humans together in a house
        # /!\ households are created at the time of allocation.
        # self.households is initialized within this function through calls to `self.create_location`
        self.humans = assign_households_to_humans(self.humans, self, self.conf, self.logfile)

        # assign workplace to humans
        # self.`location_type`s are created in this function
        self.humans, self = create_locations_and_assign_workplace_to_humans(self.humans, self, self.conf, self.logfile)

        # prepare schedule
        log("Preparing schedule ... ")
        start_time = datetime.datetime.now()
        # TODO - parallelize this for speedup in initialization
        for human in self.humans:
            human.mobility_planner.initialize()

        timedelta = (datetime.datetime.now() - start_time).total_seconds()
        log(f"Schedule prepared (Took {timedelta:2.3f}s)", self.logfile)
        self.hd = {human.name: human for human in self.humans}

    def add_to_test_queue(self, human):
        self.covid_testing_facility.add_to_test_queue(human)

    def log_static_info(self):
        """
        Logs events for all humans in the city
        """
        for h in self.humans:
            Event.log_static_info(self.conf['COLLECT_LOGS'], self, h, self.env.timestamp)

    def _compute_preferences(self):
        """
        Compute preferred distribution of each human for park, stores, etc.
        /!\ Modifies each human's stores_preferences and parks_preferences
        """
        for h in self.humans:
            h.stores_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.stores]
            h.parks_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.parks]

    def location_district_id(self, location: Location) -> int:
        """
        maps location id to the district id containing the location
        """
        return location.id % self.num_districts

    def split_locations_into_districts(self):
        """
        splits the location objects between district processes
        """
        self.districts: typing.List[District] = [None] * self.num_districts
        ind: int = 0
        # parent_pid = os.getpid()
        self.district = None

        def split_locs(locs: typing.List[Location]):
            district_locs = [[]] * self.num_districts
            for loc in locs:
                district_locs[self.location_district_id(loc)].append(loc)
            return district_locs

        district_households = split_locs(self.households)
        district_humans = [household.residents for households in district_households for household in households]
        district_stores = split_locs(self.stores)
        district_senior_residences = split_locs(self.senior_residences)
        district_hospitals = split_locs(self.hospitals)
        district_miscs = split_locs(self.miscs)
        district_parks = split_locs(self.parks)
        district_schools = split_locs(self.schools)
        district_workplaces = split_locs(self.workplaces)
        for ind in range(self.num_districts):
            if os.fork() == 0: # in the child process
                self.district = District(
                        ind,
                        district_humans[ind],
                        district_households[ind],
                        district_stores[ind],
                        district_senior_residences[ind],
                        district_hospitals[ind],
                        district_miscs[ind],
                        district_parks[ind],
                        district_schools[ind],
                        district_workplaces[ind],
                        env=self.env,
                        conf=self.conf,
                        city=self
                    )
                break
        else:
            class Monitor():
                def __init__(self):
                    pass
                def run(self, *args, **kwargs):
                    yield from ()
            self.district = Monitor()

        del self.households, self.stores, self.senior_residences, self.hospitals, \
            self.miscs, self.parks, self.schools, self.workplaces
        # self.humans
        

class EmptyCity(City):
    """
    An empty City environment (no humans or locations) that the user can build with
    externally defined code.  Useful for controlled scenarios and functional testing
    """

    def __init__(self, env, rng, x_range, y_range, conf):
        """

        Args:
            env (simpy.Environment): [description]
            rng (np.random.RandomState): Random number generator
            x_range (tuple): (min_x, max_x)
            y_range (tuple): (min_y, max_y)
            conf (dict): yaml experiment configuration
        """
        self.conf = conf
        self.env = env
        self.rng = np.random.RandomState(rng.randint(2 ** 16))
        self.x_range = x_range
        self.y_range = y_range
        self.total_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        self.n_people = 0
        self.logfile = None

        self.test_type_preference = list(zip(*sorted(conf.get("TEST_TYPES").items(), key=lambda x:x[1]['preference'])))[0]
        self.max_capacity_per_test_type = {
            test_type: max(int(self.conf['PROPORTION_LAB_TEST_PER_DAY'] * self.n_people), 1)
            for test_type in self.test_type_preference
        }

        self.daily_target_rec_level_dists = None
        self.daily_rec_level_mapping = None
        self.covid_testing_facility = TestFacility(self.test_type_preference, self.max_capacity_per_test_type, env, conf)

        # Get the test type with the lowest preference?
        # TODO - EM: Should this rather sort on 'preference' in descending order?
        self.test_type_preference = list(
            zip(
                *sorted(
                    self.conf.get("TEST_TYPES").items(),
                    key=lambda x:x[1]['preference']
                )
            )
        )[0]

        self.humans = []
        self.hd = {}
        self.households = OrderedSet()
        self.stores = []
        self.senior_residences = []
        self.hospitals = []
        self.miscs = []
        self.parks = []
        self.schools = []
        self.workplaces = []
        self.global_mailbox: SimulatorMailboxType = {}
        self.n_init_infected  = 0
        self.init_fraction_sick = 0

    @property
    def start_time(self):
        return datetime.datetime.fromtimestamp(self.env.ts_initial)

    def initWorld(self):
        """
        After adding humans and locations to the city, execute this function to finalize the City
        object in preparation for simulation.
        """
        self.n_people = len(self.humans)
        self.n_init_infected = sum(1 for h in self.humans if h.infection_timestamp is not None)
        self.init_fraction_sick = self.n_init_infected /  self.n_people
        print("Computing preferences")
        # self.initialize_humans_and_locations()
        # assign workplace to humans
        self.humans, self = create_locations_and_assign_workplace_to_humans(self.humans, self, self.conf, self.logfile)

        self._compute_preferences()
        self.tracker = Tracker(self.env, self, self.conf, None)
        self.tracker.initialize()
        # self.tracker.track_initialized_covid_params(self.humans)
        self.tracing_method = BaseMethod(self.conf)
        self.age_histogram = relativefreq2absolutefreq(
            bins_fractions={(x[0], x[1]): x[2] for x in self.conf.get('P_AGE_REGION')},
            n_elements=self.n_people,
            rng=self.rng,
        )
