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
        self.tracker = Tracker(env, self, conf, logfile)

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
        self.hd = {}
        self.households = OrderedSet()
        self.age_histogram = None

        log("Initializing humans ...", self.logfile)
        self.initialize_humans_and_locations()

        # self.log_static_info()
        self.tracker.track_static_info()


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
        self.tracker.initialize()

        # create a global inference engine
        self.inference_engine = TransformerInferenceEngine(conf)

        # split the location objects such as households between district objects
        self.split_locations_into_districts(self.conf['NUM_DISTRICTS'])

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

    def split_locations_into_districts(self, n_districts: int):
        """
<<<<<<< Updated upstream
        humans_notified, infections_seeded = False, False
        last_day_idx = 0
        while True:
            current_day = (self.env.timestamp - self.start_time).days

            # seed infections and change mixing constants (end of burn-in period)
            if (
                not infections_seeded
                and self.env.timestamp == self.conf['COVID_SPREAD_START_TIME']
            ):
                self._initiate_infection_spread_and_modify_mixing_if_needed()
                infections_seeded = True

            # Notify humans to follow interventions on intervention day
            if (
                not humans_notified
                and self.env.timestamp == self.conf.get('INTERVENTION_START_TIME')
            ):
                log("\n *** ****** *** ****** *** INITIATING INTERVENTION *** *** ****** *** ******\n", self.logfile)
                log(self.conf['INTERVENTION'], self.logfile)

                # if its a tracing method, load the class that can compute risk
                if self.conf['RISK_MODEL'] != "":
                    self.tracing_method = get_tracing_method(risk_model=self.conf['RISK_MODEL'], conf=self.conf)
                    self.have_some_humans_download_the_app()

                # initialize everyone from the baseline behavior
                for human in self.humans:
                    human.intervened_behavior.initialize()
                    if self.tracing_method is not None:
                        human.set_tracing_method(self.tracing_method)

                # log reduction levels
                log("\nCONTACT REDUCTION LEVELS (first one is not used) -", self.logfile)
                for location_type, value in human.intervened_behavior.reduction_levels.items():
                    log(f"{location_type}: {value} ", self.logfile)

                humans_notified = True
                if self.tracing_method is not None:
                    self.tracker.track_daily_recommendation_levels(set_tracing_started_true=True)

                # modify knobs because now people are more aware
                if self.conf['ASSUME_NO_ENVIRONMENTAL_INFECTION_AFTER_INTERVENTION_START']:
                    self.conf['_ENVIRONMENTAL_INFECTION_KNOB'] = 0.0

                if self.conf['ASSUME_NO_UNKNOWN_INTERACTIONS_AFTER_INTERVENTION_START']:
                    self.conf['_MEAN_DAILY_UNKNOWN_CONTACTS'] = 0.0

                log("\n*** *** ****** *** ****** *** ****** *** ****** *** ****** *** ****** *** ****** *** ***\n", self.logfile)
            # run city testing routine, providing test results for those who need them
            # TODO: running this every hour of the day might not be correct.
            # TODO: testing budget is used up at hour 0 if its small
            self.covid_testing_facility.clear_test_queue()

            alive_humans = []

            # run non-app-related-stuff for all humans here (test seeking, infectiousness updates)
            for human in self.humans:
                if not human.is_dead:
                    human.check_if_needs_covid_test()  # humans can decide to get tested whenever
                    human.check_covid_symptom_start()
                    human.check_covid_recovery()
                    human.fill_infectiousness_history_map(current_day)
                    alive_humans.append(human)

            # now, run app-related stuff (risk assessment, message preparation, ...)
            prev_risk_history_maps, update_messages = self.run_app(current_day, outfile, alive_humans)

            # update messages may not be sent if the distribution strategy (e.g. GAEN) chooses to filter them
            self.register_new_messages(
                current_day_idx=current_day,
                current_timestamp=self.env.timestamp,
                update_messages=update_messages,
                prev_human_risk_history_maps=prev_risk_history_maps,
                new_human_risk_history_maps={h: h.risk_history_map for h in self.humans},
            )

            # for debugging/plotting a posteriori, track all human/location attributes...
            self.tracker.track_humans(hd=self.hd, current_timestamp=self.env.timestamp)
            # self.tracker.track_locations() # TODO

            yield self.env.timeout(int(duration))
            # finally, run end-of-day activities (if possible); these include mailbox cleanups, symptom updates, ...
            if current_day != last_day_idx:
                alive_humans = [human for human in self.humans if not human.is_dead]
                last_day_idx = current_day
                if self.conf.get("DIRECT_INTERVENTION", -1) == current_day:
                    self.conf['GLOBAL_MOBILITY_SCALING_FACTOR'] = self.conf['GLOBAL_MOBILITY_SCALING_FACTOR']  / 2
                self.do_daily_activies(current_day, alive_humans)

    def do_daily_activies(
            self,
            current_day: int,
            alive_humans: typing.Iterable["Human"],
    ):
        """Runs all activities that should be completed only once per day."""
        # Compute the transition matrix of recommendation levels to
        # target distribution of recommendation levels
        self.daily_rec_level_mapping = self.compute_daily_rec_level_mapping(current_day)
        self.cleanup_global_mailbox(self.env.timestamp)
        # TODO: this is an assumption which will break in reality, instead of updating once per day everyone
        #       at the same time, it should be throughout the day
        for human in alive_humans:
            human.recover_health() # recover from cold/flu/allergies if it's time
            human.catch_other_disease_at_random() # catch cold/flu/allergies at random
            human.update_symptoms()
            human.increment_healthy_day()
            human.check_if_test_results_should_be_reset() # reset test results if its time
            human.mobility_planner.send_social_invites()
        self.tracker.increment_day()
        if self.conf.get("USE_GAEN"):
            print(
                "cur_day: {}, budget spent: {} / {} ".format(
                    current_day,
                    self.sent_messages_by_day.get(current_day, 0),
                    int(self.conf["n_people"] * self.conf["MESSAGE_BUDGET_GAEN"])
                ),
            )

    def run_app(
            self,
            current_day: int,
            outfile: typing.AnyStr,
            alive_humans: typing.Iterable["Human"]
    ) -> typing.Tuple[typing.Dict, typing.List[UpdateMessage]]:
        """Runs the application logic for all humans that are still alive.

        The logic is split into three parts. First, 'lightweight' jobs will run. These include
        daily risk level initialization, symptoms reporting updates, and digital (binary) contact
        tracing (if necessary). Then, if a risk inference model is being used or if we are collecting
        training data, batches of humans will be used to do clustering and to call the model. Finally,
        the recommendation level of all humans will be updated, they will generate update messages
        (if necessary), and the tracker will be updated with the state of all humans.
=======
        splits the location objects between district processes
>>>>>>> Stashed changes
        """
        self.districts: List[District] = [None] * n_districts
        ind: int = 0
        parent_pid = os.getpid()
        self.district = None
        while (os.getpid() == parent_pid) and (ind < n_districts):
            if os.fork() == 0: # in the child process
                s = slice(ind, None, n_districts)
                self.district = District(
                        os.getpid() - parent_pid,
                        self.humans,
                        self.households[s],
                        self.stores[s],
                        self.senior_residences[s],
                        self.hospitals[s],
                        self.miscs[s],
                        self.parks[s],
                        self.schools[s],
                        self.workplaces[s]
                    )
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
