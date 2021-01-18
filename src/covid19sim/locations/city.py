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
from covid19sim.inference.heavy_jobs import batch_run_timeslot_heavy_jobs
from covid19sim.interventions.tracing import BaseMethod
from covid19sim.inference.message_utils import UIDType, UpdateMessage, RealUserIDType
from covid19sim.distribution_normalization.dist_utils import get_rec_level_transition_matrix
from covid19sim.interventions.tracing_utils import get_tracing_method
from covid19sim.locations.test_facility import TestFacility


if typing.TYPE_CHECKING:
    from covid19sim.human import Human
    from covid19sim.utils.env import Env

PersonalMailboxType = typing.Dict[UIDType, typing.List[UpdateMessage]]
SimulatorMailboxType = typing.Dict[RealUserIDType, PersonalMailboxType]


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

        self.humans = []
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
        self.global_mailbox: SimulatorMailboxType = defaultdict(dict)
        self.tracker.initialize()

    def cleanup_global_mailbox(
            self,
            current_timestamp: datetime.datetime,
    ):
        """Removes all messages older than 14 days from the global mailbox."""
        # note that to keep the simulator efficient, users will directly pop the update messages that
        # they consume, so this is only necessary for edge cases (e.g. dead people can't update)
        max_encounter_age = self.conf.get('TRACING_N_DAYS_HISTORY')
        new_global_mailbox = defaultdict(dict)
        for user_key, personal_mailbox in self.global_mailbox.items():
            new_personal_mailbox = {}
            for mailbox_key, messages in personal_mailbox.items():
                for message in messages:
                    if (current_timestamp - message.encounter_time).days <= max_encounter_age:
                        if mailbox_key not in new_personal_mailbox:
                            new_personal_mailbox[mailbox_key] = []
                        new_personal_mailbox[mailbox_key].append(message)
            new_global_mailbox[user_key] = new_personal_mailbox
        self.global_mailbox = new_global_mailbox

    def register_new_messages(
            self,
            current_day_idx: int,
            current_timestamp: datetime.datetime,
            update_messages: typing.List[UpdateMessage],
            prev_human_risk_history_maps: typing.Dict["Human", typing.Dict[int, float]],
            new_human_risk_history_maps: typing.Dict["Human", typing.Dict[int, float]],
    ):
        """Adds new update messages to the global mailbox, passing them to trackers if needed.

        This function may choose to discard messages based on the mailbox server protocol being used.
        For this purpose, the risk history level maps of all users are provided as input. Note that
        these risk history maps may not actually be transfered to the server, and we have to be
        careful how we use this data in a safe fashion so that (in real life) only non-PII info
        needs to be transmitted to the server for filtering.
        """
        humans_with_updates = {self.hd[m._sender_uid] for m in update_messages}

        if self.conf.get("USE_GAEN"):
            # update risk level change histogram using scores of new updaters
            updater_scores = {
                h: h.contact_book.get_risk_level_change_score(
                    prev_risk_history_map=prev_human_risk_history_maps[h],
                    curr_risk_history_map=new_human_risk_history_maps[h],
                    proba_to_risk_level_map=h.proba_to_risk_level_map,
                ) for h in humans_with_updates
            }
            for score in updater_scores.values():
                self.risk_change_histogram[score] += 1
            self.risk_change_histogram_sum += len(updater_scores)
        else:
            # TODO: add filtering steps (if any) for other protocols
            pass

        # note that to keep the simulator efficient, users will have their own private mailbox inside
        # the global mailbox (this replaces the database diff logic & allows faster access)
        for update_message in update_messages:
            source_human = self.hd[update_message._sender_uid]
            destination_human = self.hd[update_message._receiver_uid]

            if self.conf.get("USE_GAEN"):
                should_send = self._check_should_send_message_gaen(
                    current_day_idx=current_day_idx,
                    current_timestamp=current_timestamp,
                    human=source_human,
                    risk_change_score=updater_scores[source_human],
                )
                if not should_send:
                    continue
            else:
                # TODO: add filtering steps (if any) for other protocols
                pass

            # if we get here, it's because we actually chose to send the message
            self.tracker.track_update_messages(
                from_human=source_human,
                to_human=destination_human,
                payload={
                    "reason": update_message._update_reason,
                    "new_risk_level": update_message.new_risk_level
                },
            )
            if current_day_idx not in self.sent_messages_by_day:
                self.sent_messages_by_day[current_day_idx] = 0
            self.sent_messages_by_day[current_day_idx] += 1
            source_human.contact_book.latest_update_time = \
                max(source_human.contact_book.latest_update_time, current_timestamp)
            mailbox_key = update_message.uid  # mailbox key = message uid
            personal_mailbox = self.global_mailbox[destination_human.name]
            if mailbox_key not in personal_mailbox:
                personal_mailbox[mailbox_key] = []
            personal_mailbox[mailbox_key].append(update_message)

    def _check_should_send_message_gaen(
            self,
            current_day_idx: int,
            current_timestamp: datetime.datetime,
            human: "Human",
            risk_change_score: int,
    ) -> bool:
        """Returns whether a specific human with a given GAEN impact score should send updates or not."""
        message_budget = self.conf.get("MESSAGE_BUDGET_GAEN")
        # give at least BURN_IN_DAYS days to get risk histograms
        if current_day_idx - self.conf.get("INTERVENTION_DAY") < self.conf.get("BURN_IN_DAYS"):
            return False
        days_between_messages = self.conf.get("DAYS_BETWEEN_MESSAGES")
        # don't send messages if you sent recently
        if (current_timestamp - human.contact_book.latest_update_time).days < days_between_messages:
            return False
        # don't exceed the message budget
        already_sent_messages = self.sent_messages_by_day.get(current_day_idx, 0)
        if already_sent_messages >= message_budget * self.conf.get("n_people"):
            return False

        # if people uniformly send messages in the population, then 1 / days_between_messages people
        # won't send messages today anyway
        message_budget = days_between_messages * message_budget
        # but if we don't have that we underestimate the number of available messages in the budget
        # because some people will have already sent a message the day before and won't be able to
        # on this day

        # reduce due to multiple updates per day so uniformly distribute messages
        message_budget = message_budget / self.conf.get("UPDATES_PER_DAY")

        # sorted list (decreasing order) of tuples (risk_change, proportion of people with that risk_change)
        scaled_risks = sorted(
            {
                k: v / self.risk_change_histogram_sum
                for k, v in self.risk_change_histogram.items()
            }.items(),
            reverse=True
        )
        summed_percentiles = 0
        for rc, percentile in scaled_risks:
            # there's room for this person's message
            if risk_change_score == rc and summed_percentiles < message_budget:
                # coin flip if you're in this bucket but this is the "last" bucket
                if summed_percentiles + percentile > message_budget:
                    # Out of all the available messages (MAX_NUM_MESSAGES_GAEN), summed_percentile
                    # have been sent. There remains (MAX - summed) messages to split across percentile people
                    if self.rng.random() < (message_budget - summed_percentiles):  # / percentile:
                        return True
                # otherwise there is room for messages so you should send
                else:
                    return True
            summed_percentiles += percentile
        return False

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

    def _initiate_infection_spread_and_modify_mixing_if_needed(self):
        """
        Seeds infection in the population and sets other attributes corresponding to social mixing dynamics.
        """
        # seed infection
        self.n_init_infected = math.ceil(self.init_fraction_sick * self.n_people)
        chosen_infected = self.rng.choice(self.humans, size=self.n_init_infected, replace=False)
        for human in chosen_infected:
            human._get_infected(initial_viral_load=human.rng.random())

        # modify likelihood of meeting new people
        self.conf['_CURRENT_PREFERENTIAL_ATTACHMENT_FACTOR'] = self.conf['END_PREFERENTIAL_ATTACHMENT_FACTOR']

        self.tracker.log_seed_infections()

    def have_some_humans_download_the_app(self):
        """
        Simulates the process of downloading the app on for smartphone users according to `APP_UPTAKE` and `SMARTPHONE_OWNER_FRACTION_BY_AGE`.
        """
        def _log_app_info(human):
            if not human.has_app:
                return
            # (assumption) 90% of the time, healthcare workers will declare it
            human.obs_is_healthcare_worker = False
            if (
                human.workplace is not None
                and human.workplace.location_type == "HOSPITAL"
                and human.rng.random() < 0.9
            ):
                human.obs_is_healthcare_worker = True

            human.has_logged_info = human.rng.rand() < human.proba_report_age_and_sex
            human.obs_age = human.age if human.has_logged_info else None
            human.obs_sex = human.sex if human.has_logged_info else None
            human.obs_preexisting_conditions = human.preexisting_conditions if human.has_logged_info else []

        log("Downloading the app...", self.logfile)
        # app users
        all_has_app = self.conf.get('APP_UPTAKE') < 0
        # The dict below keeps track of an app quota for each age group
        age_histogram_bin_10s = _convert_bin_5s_to_bin_10s(self.age_histogram)
        n_apps_per_age = {
            (x[0], x[1]): math.ceil(age_histogram_bin_10s[(x[0], x[1])] * x[2] * self.conf.get('APP_UPTAKE'))
            for x in self.conf.get("SMARTPHONE_OWNER_FRACTION_BY_AGE")
        }
        for human in self.humans:
            if all_has_app:
                # i get an app, you get an app, everyone gets an app
                human.has_app = True
                _log_app_info(human)
                continue

            age_bin = human.age_bin_width_10.bin
            if n_apps_per_age[age_bin] > 0:
                # This human gets an app If there's quota left in his age group
                human.has_app = True
                _log_app_info(human)
                n_apps_per_age[age_bin] -= 1
                continue

        self.tracker.track_app_adoption()

    def add_to_test_queue(self, human):
        self.covid_testing_facility.add_to_test_queue(human)

    @property
    def events(self):
        """
        Get all events of all humans in the city

        Returns:
            list: all of everyone's events
        """
        return list(itertools.chain(*[h.events for h in self.humans]))

    def compute_daily_rec_level_distribution(self):
        # QKFIX: There are 4 recommendation levels, the value is hard-coded here
        counts = np.zeros((4,), dtype=np.float_)
        for human in self.humans:
            if human.has_app:
                counts[human.rec_level] += 1

        if np.sum(counts) > 0:
            counts /= np.sum(counts)

        return counts

    def compute_daily_rec_level_mapping(self, current_day):
        # If there is no target recommendation level distribution
        if self.daily_target_rec_level_dists is None:
            return None

        if self.conf.get('INTERVENTION_DAY', -1) < 0:
            return None
        else:
            current_day -= self.conf.get('INTERVENTION_DAY')
            if current_day < 0:
                return None

        daily_rec_level_dist = self.compute_daily_rec_level_distribution()

        index = min(current_day, len(self.daily_target_rec_level_dists) - 1)
        daily_target_rec_level_dist = self.daily_target_rec_level_dists[index]

        return get_rec_level_transition_matrix(daily_rec_level_dist,
                                               daily_target_rec_level_dist)

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

    def run(self, duration, outfile):
        """
        Run the City.
        Several daily tasks take place here.
        Examples include - (1) modifying behavior of humans based on an intervention
        (2) gathering daily statistics on humans

        Args:
            duration (int): duration of a step, in seconds.
            outfile (str): may be None, the run's output file to write to

        Yields:
            simpy.Timeout
        """
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
        """
        backup_human_init_risks = {}  # backs up human risks before any update takes place

        # iterate over humans, and if it's their timeslot, then update their state
        for human in alive_humans:
            if (
                not human.has_app
                or self.env.timestamp.hour not in human.time_slots
            ):
                continue
            # set the human's risk to a correct value for the day (if it does not exist already)
            human.initialize_daily_risk(current_day)
            # keep a backup of the current risk map before infering anything, in case GAEN needs it
            backup_human_init_risks[human] = copy.deepcopy(human.risk_history_map)
            # run 'lightweight' app jobs (e.g. contact book cleanup, symptoms reporting, bdt) without batching
            human.run_timeslot_lightweight_jobs(
                init_timestamp=self.start_time,
                current_timestamp=self.env.timestamp,
                personal_mailbox=self.global_mailbox[human.name],
            )

        # now, batch-run the clustering + risk prediction using an ML model (if we need it)
        if (
            self.conf.get("RISK_MODEL") == "transformer"
            or "heuristic" in self.conf.get("RISK_MODEL")
            or self.conf.get("COLLECT_TRAINING_DATA")
        ):
            self.humans = batch_run_timeslot_heavy_jobs(
                humans=self.humans,
                init_timestamp=self.start_time,
                current_timestamp=self.env.timestamp,
                global_mailbox=self.global_mailbox,
                time_slot=self.env.timestamp.hour,
                conf=self.conf,
                city_hash=self.hash,
            )

        # iterate over humans again, and if it's their timeslot, then prepare risk update messages
        update_messages = []
        for human in alive_humans:
            if not human.has_app or self.env.timestamp.hour not in human.time_slots:
                continue

            # overwrite risk values to 1.0 if human has positive test result (for all tracing methods)
            if human.reported_test_result == "positive":
                for day_offset in range(self.conf.get("TRACING_N_DAYS_HISTORY")):
                    human.risk_history_map[current_day - day_offset] = 1.0

            # using the risk history map & the risk level map, update the human's rec level
            human.update_recommendations_level()

            # if we had any encounters for which we have not sent an initial message, do it now
            update_messages.extend(human.contact_book.generate_initial_updates(
                current_day_idx=current_day,
                current_timestamp=self.env.timestamp,
                risk_history_map=human.risk_history_map,
                proba_to_risk_level_map=human.proba_to_risk_level_map,
                intervention=human.intervention,
            ))

            # then, generate risk level update messages for all other encounters (if needed)
            update_messages.extend(human.contact_book.generate_updates(
                current_day_idx=current_day,
                current_timestamp=self.env.timestamp,
                prev_risk_history_map=human.prev_risk_history_map,
                curr_risk_history_map=human.risk_history_map,
                proba_to_risk_level_map=human.proba_to_risk_level_map,
                update_reason="unknown",
                intervention=human.intervention,
            ))

            # finally, override the 'previous' risk history map with the updated values of the current
            # map so that the next call can look at the proper difference between the two
            for day_idx, risk_val in human.risk_history_map.items():
                human.prev_risk_history_map[day_idx] = risk_val

        return backup_human_init_risks, update_messages


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
        self.global_mailbox: SimulatorMailboxType = defaultdict(dict)
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
