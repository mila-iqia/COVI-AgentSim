"""
This module implements the `Human` class which is the focal point of the agent-based simulation.
"""

import math
import datetime
import logging
import numpy as np
import scipy
import typing
import warnings
from collections import defaultdict
from orderedset import OrderedSet

from covid19sim.utils.mobility_planner import MobilityPlanner
from covid19sim.utils.utils import compute_distance, proba_to_risk_fn
from covid19sim.locations.city import PersonalMailboxType
from covid19sim.locations.hospital import Hospital, ICU
from covid19sim.log.event import Event
from collections import deque

from covid19sim.utils.utils import _normalize_scores, draw_random_discrete_gaussian, filter_open, filter_queue_max, calculate_average_infectiousness
from covid19sim.epidemiology.human_properties import may_develop_severe_illness, _get_inflammatory_disease_level,\
    _get_preexisting_conditions, _get_random_sex, get_carefulness, get_age_bin
from covid19sim.epidemiology.viral_load import compute_covid_properties, viral_load_for_day
from covid19sim.epidemiology.symptoms import _get_cold_progression, _get_flu_progression, \
    _get_allergy_progression, \
    MILD, MODERATE, SEVERE, EXTREMELY_SEVERE, \
    ACHES, COUGH, FATIGUE, FEVER, GASTRO, TROUBLE_BREATHING
from covid19sim.epidemiology.p_infection import get_human_human_p_transmission, infectiousness_delta
from covid19sim.utils.constants import SECONDS_PER_MINUTE, SECONDS_PER_HOUR, SECONDS_PER_DAY
from covid19sim.inference.message_utils import ContactBook, exchange_encounter_messages, RealUserIDType
from covid19sim.utils.visits import Visits
from covid19sim.native._native import BaseHuman
from covid19sim.interventions.intervened_behavior import IntervenedBehavior

if typing.TYPE_CHECKING:
    from covid19sim.utils.env import Env
    from covid19sim.locations.city import City
    from covid19sim.locations.location import Location


class Human(BaseHuman):
    """
    Defines various attributes of `human` concerned with COVID spread and contact patterns.
    Human agent class. Human objects can only be instantiated by a city at the start of a simulation.
    See `covid19sim.locations.city.py` for more information.

    Args:
        env (simpy.Environment): environment to schedule events
        city (covid19sim.locations.city.City): `City` to carry out regular checks on human and update its attributes
        name (str): identifier for this `human`
        age (int): age of the `human`
        rng (np.random.RandomState): Random number generator
        conf (dict): yaml configuration of the experiment
    """

    def __init__(self, env, city, name, age, rng, conf):
        super().__init__(env)

        # Utility References
        self.conf = conf  # Simulation-level Configurations
        self.env = env  # Simpy Environment (primarily used for timing / syncronization)
        self.city = city  # Manages lots of things inc. initializing humans and locations, tracking/updating humans

        # SEIR Tracking
        self.recovered_timestamp = datetime.datetime.min  # Time of recovery from covid -- min if not yet recovered
        self._infection_timestamp = None  # private time of infection with covid - implemented this way to ensure only infected 1 time
        self.infection_timestamp = None  # time of infection with covid
        self.n_infectious_contacts = 0  # number of high-risk (infected someone) contacts with an infected individual.
        self.exposure_source = None  # source that exposed this human to covid (and infected them). None if not infected.

        # rng stuff
        # note: the seed is the important part, if one is not given directly, it will be generated...
        # note2: keeping the initial seed value is important for when we serialize/deserialize this object
        # note3: any call to self.rng after construction is not guaranteed to return the same behavior as during the run
        if isinstance(rng, np.random.RandomState):
            self.init_seed = rng.randint(2 ** 16)
        else:
            assert isinstance(rng, int)
            self.init_seed = rng
        self.rng = np.random.RandomState(self.init_seed)  # RNG for this particular human

        # Human-related properties
        self.name: RealUserIDType = f"human:{name}"  # Name of this human
        self.known_connections = set() # keeps track of all other humans that this human knows of
        self.does_not_work = False # to identify those who weren't assigned any workplace from the beginning
        self.work_start_time, self.work_end_time, self.working_days = None, None, []
        self.workplace = None  # we sometimes modify human's workplace to WFH if in quarantine, then go back to work when released
        self.household = None  # assigned later
        self.location = None  # assigned later

        # Logging data
        self._events = []

        """ Biological Properties """
        # Individual Characteristics
        self.sex = _get_random_sex(self.rng, self.conf)  # The sex of this person conforming with Canadian statistics
        self.age = age  # The age of this person, conforming with Canadian statistics
        self.age_bin_width_10 = get_age_bin(age, width=10)  # Age bins of width 10 are required for Oxford-like COVID-19 infection model and social mixing tracker
        self.normalized_susceptibility = self.conf['NORMALIZED_SUSCEPTIBILITY_BY_AGE'][self.age_bin_width_10.index][2]  # Susceptibility to Covid-19 by age
        self.mean_daily_interaction_age_group = self.conf['MEAN_DAILY_INTERACTION_FOR_AGE_GROUP'][self.age_bin_width_10.index][2]  # Social mixing is determined by age
        self.age_bin_width_5 = get_age_bin(age, width=5)
        self.preexisting_conditions = _get_preexisting_conditions(self.age, self.sex, self.rng)  # Which pre-existing conditions does this person have? E.g. COPD, asthma
        self.inflammatory_disease_level = _get_inflammatory_disease_level(self.rng, self.preexisting_conditions, self.conf.get("INFLAMMATORY_CONDITIONS"))  # how many pre-existing conditions are inflammatory (e.g. smoker)
        self.carefulness = get_carefulness(self.age, self.rng, self.conf)  # How careful is this person? Determines their liklihood of contracting Covid / getting really sick, etc

        # Illness Properties
        self.is_asymptomatic = self.rng.rand() < self.conf.get("BASELINE_P_ASYMPTOMATIC") - (self.age - 50) * 0.5 / 100  # e.g. 70: baseline-0.1, 20: baseline+0.15
        self.infection_ratio = None  # Ratio that helps make asymptomatic people less infectious than symptomatic. see `ASYMPTOMATIC_INFECTION_RATIO` in core.yaml
        self.cold_timestamp = None  # time when this person was infected with cold
        self.flu_timestamp = None  # time when this person was infected with flu
        self.allergy_timestamp = None  # time when this person started having allergy symptoms

        # Allergies
        len_allergies = self.rng.normal(1/self.carefulness, 1)   # determines the number of symptoms this persons allergies would present with (if they start experiencing symptoms)
        self.len_allergies = 7 if len_allergies > 7 else math.ceil(len_allergies)
        self.allergy_progression = _get_allergy_progression(self.rng)  # if this human starts having allergy symptoms, then there is a progression of symptoms over one or multiple days

        """ Covid-19 """
        # Covid-19 properties
        self.viral_load_plateau_height, self.viral_load_plateau_start, self.viral_load_plateau_end, self.viral_load_peak_start, self.viral_load_peak_height = None, None, None, None, None  # Determines aspects of the piece-wise linear viral load curve for this human
        self.incubation_days = 0  # number of days the virus takes to incubate before the person becomes infectious
        self.recovery_days = None  # number of recovery days post viral load plateau
        self.infectiousness_onset_days = 0  # number of days after exposure that this person becomes infectious
        self.can_get_really_sick = may_develop_severe_illness(self.age, self.sex, self.rng)  # boolean representing whether this person may need to go to the hospital
        self.can_get_extremely_sick = self.can_get_really_sick and self.rng.random() >= 0.7  # &severe; 30% of severe cases need ICU
        self.never_recovers = self.rng.random() <= self.conf.get("P_NEVER_RECOVERS")[min(math.floor(self.age/10), 8)]  # boolean representing that this person will die if they are infected with Covid-19
        self.is_immune = False  # whether this person is immune to Covid-19 (happens after recovery)

        # Covid-19 testing
        self.test_type = None  # E.g. PCR, Antibody, Physician
        self.test_time = None  # Time when this person was tested
        self.hidden_test_result = None  # Test results (that will be reported to this person but have not yet been)
        self._will_report_test_result = None  # Determines whether this individual will report their test (given that they received a test result)
        self.time_to_test_result = None  # How long does it take for this person to receive their test after it has been administered
        self.test_result_validated = None  # Represents whether a test result is validated by some public health agency (True for PCR Tests, some antiody tests)
        self._test_results = deque()  # History of test results (e.g. if you get a negative PCR test, you can still get more tests)
        self.denied_icu = None  # Used because some older people have been denied use of ICU for younger / better candidates
        self.denied_icu_days = None  # number of days the person would be denied ICU access (Note: denied ICU logic could probably be improved)

        # Symptoms
        self.covid_symptom_start_time = None  # The time when this persons covid symptoms start (requires that they are in infectious state)
        self.cold_progression = _get_cold_progression(self.age, self.rng, self.carefulness, self.preexisting_conditions, self.can_get_really_sick, self.can_get_extremely_sick) # determines the symptoms that this person would have if they had a cold
        self.flu_progression = _get_flu_progression(
            self.age, self.rng, self.carefulness, self.preexisting_conditions,
            self.can_get_really_sick, self.can_get_extremely_sick, self.conf.get("AVG_FLU_DURATION")
        )  # determines the symptoms this person would have if they had the flu
        self.all_symptoms, self.cold_symptoms, self.flu_symptoms, self.covid_symptoms, self.allergy_symptoms = [], [], [], [], []
        self.rolling_all_symptoms = deque(
            [tuple()] * self.conf.get('TRACING_N_DAYS_HISTORY'),
            maxlen=self.conf.get('TRACING_N_DAYS_HISTORY')
        )  # stores the ground-truth Covid-19 symptoms this person has on day D (used for our ML predictor and other symptom-based predictors)
        self.rolling_all_reported_symptoms = deque(
            [tuple()] * self.conf.get('TRACING_N_DAYS_HISTORY'),
            maxlen=self.conf.get('TRACING_N_DAYS_HISTORY')
        )  # stores the Covid-19 symptoms this person had reported in the app until the current simulation day (empty if they do not have the app)

        """App-related"""
        self.has_app = False  # Does this prson have the app
        time_slot = self.rng.randint(0, 24)  # Assign this person to some timeslot
        self.time_slots = [
            int((time_slot + i * 24 / self.conf.get('UPDATES_PER_DAY')) % 24)
            for i in range(self.conf.get('UPDATES_PER_DAY'))
        ]  # If people update their risk predictions 4 times per day (every 6 hours) then this code assigns the specific times _this_ person will update
        self.phone_bluetooth_noise = self.rng.rand()  # Error in distance estimation using Bluetooth with a specific type of phone is sampled from a uniform distribution between 0 and 1

        # Observed attributes; whether people enter stuff in the app
        self.has_logged_info = False # Determines whether this person writes their demographic data into the app
        self.obs_is_healthcare_worker = None # 90% of the time, healthcare workers will declare it
        self.obs_age = None  # The age of this human reported to the app
        self.obs_sex = None  # The sex of this human reported to the app
        self.obs_preexisting_conditions = []  # the preexisting conditions of this human reported to the app

        """ Interventions """
        self.intervention = None  # Type of contact tracing to do, e.g. Transformer or binary contact tracing or heuristic
        self._rec_level = -1  # Recommendation level used for Heuristic / ML methods
        self._intervention_level = -1  # Intervention level (level of behaviour modification to apply), for logging purposes
        self.recommendations_to_follow = OrderedSet()  # which recommendations this person will follow now
        self._test_recommended = False  # does the app recommend that this person should get a covid-19 test
        self.effective_contacts = 0  # A scaled number of the high-risk contacts (under 2m for over 15 minutes) that this person had
        self.healthy_effective_contacts = 0  # A scaled number of the high-risk contacts (under 2m for over 15 minutes) that this person had while healthy
        self.healthy_days = 0
        self.num_contacts = 0  # unscaled number of high-risk contacts
        self.intervened_behavior = IntervenedBehavior(self, self.env, self.conf) # keeps track of behavior level of human

        """Risk prediction"""
        self.contact_book = ContactBook(tracing_n_days_history=self.conf.get("TRACING_N_DAYS_HISTORY"))  # Used for tracking high-risk contacts (for app-based contact tracing methods)
        self.infectiousness_history_map = dict()  # Stores the (predicted) 14-day history of Covid-19 infectiousness (based on viral load and symptoms)
        self.risk_history_map = dict()  # 14-day risk history (estimated infectiousness) updated inside the human's (current) timeslot
        self.prev_risk_history_map = dict()  # used to check how the risk changed since the last timeslot
        self.last_sent_update_gaen = 0  # Used for modelling the Googe-Apple Exposure Notification protocol
        risk_mapping_array = np.array(self.conf.get('RISK_MAPPING'))  # mapping from float risk value to risk level
        assert len(risk_mapping_array) > 0, "risk mapping must always be defined!"
        self.proba_to_risk_level_map = proba_to_risk_fn(risk_mapping_array)

        """Mobility"""
        self.rho = conf['RHO']  # controls mobility (how often this person goes out and visits new places)
        self.gamma = conf['GAMMA']  # controls mobility (how often this person goes out and visits new places)

        self.household, self.location = None, None
        self.obs_hospitalized, self.obs_in_icu = None, None
        self.visits = Visits()  # used to help implement mobility
        self.last_date = defaultdict(lambda : self.env.initial_timestamp.date())  # used to track the last time this person did various things (like record smptoms)
        self.mobility_planner = MobilityPlanner(self, self.env, self.conf)

        self.location_leaving_time = self.env.ts_initial + SECONDS_PER_HOUR
        self.location_start_time = self.env.ts_initial

    def assign_household(self, location):
        if location is not None:
            self.household = location
            self.location = location
            location.add_human(self)

    def assign_workplace(self, workplace):
        """
        Initializes work related attributes for `human`
        """
        N_WORKING_DAYS = self.conf['N_WORKING_DAYS']
        AVERAGE_TIME_SPENT_WORK = self.conf['AVERAGE_TIME_SPENT_WORK']
        WORKING_START_HOUR = self.conf['WORKING_START_HOUR']

        # /!\ all humans are given a work start time same as workplace opening time
        self.work_start_time = workplace.opening_time
        if workplace.opening_time == 0:
            self.work_start_time = WORKING_START_HOUR * SECONDS_PER_HOUR

        self.work_end_time = workplace.closing_time
        if workplace.closing_time == SECONDS_PER_DAY:
            self.work_end_time = self.work_start_time + AVERAGE_TIME_SPENT_WORK * SECONDS_PER_HOUR

        self.working_days = self.rng.choice(workplace.open_days, size=N_WORKING_DAYS, replace=False)
        self.workplace = workplace

    ########### MEMORY OPTIMIZATION ###########
    @property
    def events(self):
        return self._events

    def events_slice(self, begin, end):
        end_i = len(self._events)
        begin_i = end_i
        for i, event in enumerate(self._events):
            if i < begin_i and event['time'] >= begin:
                begin_i = i
            elif event['time'] > end:
                end_i = i
                break

        return self._events[begin_i:end_i]

    def pull_events_slice(self, end):
        end_i = len(self._events)
        for i, event in enumerate(self._events):
            if event['time'] >= end:
                end_i = i
                break
        events_slice, self._events = self._events[:end_i], self._events[end_i:]
        return events_slice

    ########### EPI ###########

    @property
    def state(self):
        """
        The state (SEIR) that this person is in (True if in state, False otherwise)

        Returns:
            bool: True for the state this person is in (Susceptible, Exposed, Infectious, Removed)
        """
        return [int(self.is_susceptible), int(self.is_exposed), int(self.is_infectious), int(self.is_removed)]

    @property
    def is_really_sick(self):
        return self.can_get_really_sick and SEVERE in self.symptoms

    @property
    def is_extremely_sick(self):
        return self.can_get_extremely_sick and (SEVERE in self.symptoms or
                                                EXTREMELY_SEVERE in self.symptoms)

    @property
    def viral_load(self):
        """
        Calculates the elapsed time since infection, returning this person's current viral load

        Returns:
            Float: Returns a real valued number between 0. and 1. indicating amount of viral load (proportional to infectiousness)
        """
        return viral_load_for_day(self, self.env.now)

    def get_infectiousness_for_day(self, timestamp, is_infectious):
        """
        Scales the infectiousness value using pre-existing conditions

        Returns:
            [type]: [description]
        """
        infectiousness = 0.
        if is_infectious:
            infectiousness = viral_load_for_day(self, timestamp) * self.viral_load_to_infectiousness_multiplier
        return infectiousness

    @property
    def infectiousness_severity_multiplier(self):
        severity_multiplier = 1
        if 'immuno-compromised' in self.preexisting_conditions:
            severity_multiplier += self.conf['IMMUNOCOMPROMISED_SEVERITY_MULTIPLIER_ADDITION']
        if COUGH in self.symptoms:
            severity_multiplier += self.conf['COUGH_SEVERITY_MULTIPLIER_ADDITION']
        return severity_multiplier

    @property
    def viral_load_to_infectiousness_multiplier(self):
        """Final multiplier that converts viral-load to infectiousness."""
        if self.infection_ratio is None:
            return None
        else:
            return (self.infectiousness_severity_multiplier * self.infection_ratio /
                    self.conf['INFECTIOUSNESS_NORMALIZATION_CONST'])

    @property
    def infectiousness(self):
        return self.get_infectiousness_for_day(self.env.now, self.is_infectious)

    @property
    def symptoms(self):
        # TODO: symptoms should not be updated here.
        #  Explicit call to Human.update_symptoms() should be required
        self.update_symptoms()
        return self.rolling_all_symptoms[0]

    @property
    def reported_symptoms(self):
        self.update_symptoms()
        return self.rolling_all_reported_symptoms[0]

    @property
    def all_reported_symptoms(self):
        """
        returns all symptoms reported in the past TRACING_N_DAYS_HISTORY days

        Returns:
            list: list of symptoms
        """
        if not self.has_app:
            return []

        # TODO: symptoms should not be updated here.
        # Explicit call to Human.update_reported_symptoms() should be required
        self.update_reported_symptoms()
        return set(symptom for symptoms in self.rolling_all_reported_symptoms for symptom in symptoms)

    def update_symptoms(self):
        """
        [summary]
        """
        current_date = self.env.timestamp.date()
        if self.last_date['symptoms'] == current_date:
            return

        self.last_date['symptoms'] = current_date

        if self.has_cold:
            t = self.days_since_cold
            if t < len(self.cold_progression):
                self.cold_symptoms = self.cold_progression[t]
            else:
                self.cold_symptoms = []

        if self.has_flu:
            t = self.days_since_flu
            if t < len(self.flu_progression):
                self.flu_symptoms = self.flu_progression[t]
            else:
                self.flu_symptoms = []

        if self.has_covid and not self.is_asymptomatic:
            t = self.days_since_covid
            if self.is_removed or t >= len(self.covid_progression):
                self.covid_symptoms = []
            else:
                self.covid_symptoms = self.covid_progression[t]

        if self.has_allergy_symptoms:
            self.allergy_symptoms = self.allergy_progression[0]

        all_symptoms = self.flu_symptoms + self.cold_symptoms + self.allergy_symptoms + self.covid_symptoms
        # self.new_symptoms = list(all_symptoms - set(self.all_symptoms))
        # TODO: remove self.all_symptoms in favor of self.rolling_all_symptoms[0]
        self.all_symptoms = OrderedSet(all_symptoms)
        self.rolling_all_symptoms.appendleft(self.all_symptoms)
        self.city.tracker.track_symptoms(self)

    def update_reported_symptoms(self):
        """
        [summary]
        """
        self.update_symptoms()
        current_date = self.env.timestamp.date()
        if self.last_date['reported_symptoms'] == current_date:
            return

        self.last_date['reported_symptoms'] = current_date

        reported_symptoms = [s for s in self.rolling_all_symptoms[0] if self.rng.random() < self.carefulness]
        self.rolling_all_reported_symptoms.appendleft(reported_symptoms)
        self.city.tracker.track_symptoms(self)

    @property
    def test_result(self):
        if self.test_time is None:
            return None

        # for every new test, this will return None until the test results arrive
        if (self.env.timestamp - self.test_time).days < self.time_to_test_result:
            return None

        return self.hidden_test_result

    @property
    def reported_test_result(self):
        if self.will_report_test_result:
            return self.test_result
        return None

    @property
    def reported_test_type(self):
        if self.will_report_test_result:
            return self.test_type
        return None

    def reset_test_result(self):
        self.test_type = None
        self.test_time = None
        self.hidden_test_result = None
        self._will_report_test_result = None
        self.time_to_test_result = None
        self.test_result_validated = None

    @property
    def will_report_test_result(self):
        if self._will_report_test_result is None:
            return None
        return self.has_app and self._will_report_test_result

    @property
    def test_results(self):
        test_results = []
        for hidden_test_result, will_report, test_timestamp, test_delay in self._test_results:
            test_results.append((
                hidden_test_result if (self.has_app and will_report) else None,
                test_timestamp,
                test_delay
            ))
        return test_results

    def set_test_info(self, test_type, unobserved_result):
        """
        sets test related attributes such as
            test_type (str): type of test used
            test_time (str): time of testing
            time_to_test_result (str): delay in getting results back
            hidden_test_result (str): test results are not immediately available
            test_result_validated (str): whether these results will be validated by an agency
            reported_test_result (str): test result reported by self
            reported_test_type (str): test type reported by self

        NOTE: these values can be overwritten by subsequent test results
        """
        self.test_type = test_type
        self.test_time = self.env.timestamp
        self.hidden_test_result = unobserved_result
        self._will_report_test_result = self.rng.random() < self.conf.get("TEST_REPORT_PROB")
        if isinstance(self.location, (Hospital, ICU)):
            self.time_to_test_result = self.conf['DAYS_TO_LAB_TEST_RESULT_IN_PATIENT']
        else:
            self.time_to_test_result = self.conf['DAYS_TO_LAB_TEST_RESULT_OUT_PATIENT']
        self.test_result_validated = self.test_type == "lab"

        self._test_results.appendleft((
            self.hidden_test_result,
            self._will_report_test_result,
            self.env.timestamp,  # for result availability checking later
            self.time_to_test_result,  # in days
        ))

        self.intervened_behavior.trigger_intervention(reason="test-taken", human=self)

        # log
        self.city.tracker.track_tested_results(self)
        Event.log_test(self.conf.get('COLLECT_LOGS'), self, self.test_time)

    def check_if_needs_covid_test(self, at_hospital=False):
        """
        Checks whether self needs a test or not. Note: this only adds self to the test queue (not administer a test yet) of City.
        It is called every time symptoms are updated. It is also called from GetTested intervention.

        It depends upon the following factors -
            1. if `Human` is at a hospital, TEST_SYMPTOMS_FOR_HOSPITAL are checked for
            2. elsewhere there is a proability related to whether symptoms are "severe", "moderate", or "mild"
            3. if _test_recommended is true (set by app recommendations)
            4. if the `Human` is careful enough to check symptoms itself

        Args:
            at_hospital (bool, optional): follows the check for testing needs at a hospital.
        """
        # has been tested positive already
        if self.test_result == "positive":
            return

        # waiting for the results. no need to test again.
        if self.test_time is not None and self.test_result is None:
            return

        # already in test queue, bail out
        if self in self.city.covid_testing_facility.test_queue:
            return

        should_get_test = False
        if at_hospital:
            assert isinstance(self.location, (Hospital, ICU)), "Not at hospital; wrong argument"
            # Is in a hospital and has symptoms that hospitals check for
            TEST_SYMPTOMS_FOR_HOSPITAL = set(self.conf['GET_TESTED_SYMPTOMS_CHECKED_IN_HOSPITAL'])
            should_get_test = any(TEST_SYMPTOMS_FOR_HOSPITAL & set(self.symptoms))
        else:
            if SEVERE in self.symptoms or EXTREMELY_SEVERE in self.symptoms:
                should_get_test = self.rng.rand() < self.conf['P_TEST_SEVERE']

            elif MODERATE in self.symptoms:
                should_get_test = self.rng.rand() < self.conf['P_TEST_MODERATE']

            elif MILD in self.symptoms:
                should_get_test = self.rng.rand() < self.conf['P_TEST_MILD']

            # has been recommended the test by an intervention
            if not should_get_test and self._test_recommended:
                should_get_test = self.intervened_behavior.follow_recommendation_today

            if not should_get_test:
                # Has symptoms that a careful person would fear to be covid
                SUSPICIOUS_SYMPTOMS = set(self.conf['GET_TESTED_SYMPTOMS_CHECKED_BY_SELF'])
                if set(self.symptoms) & SUSPICIOUS_SYMPTOMS:
                    should_get_test = self.rng.rand() < self.carefulness
                    if should_get_test:
                        # self.intervened_behavior.trigger_intervention("self-diagnosed-symptoms", human=self)
                        pass

        if should_get_test:
            self.city.add_to_test_queue(self)

    def check_covid_symptom_start(self):
        """
        records the first time when symptoms show up to compute serial intervals.
        """
        # used for tracking serial interval
        # person needs to show symptoms in order for this to be true.
        # is_incubated checks for asymptomaticity and whether the days since exposure is
        # greater than incubation_days.
        # Note: it doesn't count symptom start time from environmental infection or asymptomatic/presymptomatic infections
        # reference is in city.tracker.track_serial_interval.__doc__
        if self.is_incubated and self.covid_symptom_start_time is None and any(self.symptoms):
            self.covid_symptom_start_time = self.env.timestamp
            self.city.tracker.track_serial_interval(self.name)

    def check_covid_recovery(self):
        """
        If `self` has covid, this function will check when can `self` recover and set necessary variables accordingly.
        """
        if self.is_infectious and (self.env.timestamp - self.infection_timestamp).total_seconds() >= self.recovery_days * SECONDS_PER_DAY:
            self.city.tracker.track_recovery(self)

            # TO DISCUSS: Should the test result be reset here? We don't know in reality
            # when the person has recovered; currently not reset
            # self.reset_test_result()
            self.infection_timestamp = None
            self.all_symptoms, self.covid_symptoms = [], []
            if self.never_recovers:
                self.mobility_planner.human_dies_in_next_activity = True
                return
            else:
                self.recovered_timestamp = self.env.timestamp
                self.is_immune = not self.conf.get("REINFECTION_POSSIBLE")

                # "resample" the chance probability of never recovering again (experimental)
                if not self.is_immune:
                    self.never_recovers = self.rng.random() < self.conf.get("P_NEVER_RECOVERS")[
                        min(math.floor(self.age / 10), 8)]

                Event.log_recovery(self.conf.get('COLLECT_LOGS'), self, self.env.timestamp, death=False)

    def check_cold_and_flu_contagion(self, other_human):
        """
        Detects whether cold or flu contagion occurs.
        initializes the respective timestamp if it does.
        NOTE: `other_human` can transmit to `self` or vice-versa.

        Args:
            other_human (covid19sim.human.Human): `human` who happened to be near `self`
        """

        # cold transmission
        if self.has_cold ^ other_human.has_cold:
            cold_infector, cold_infectee = other_human, self
            if self.cold_timestamp is not None:
                cold_infector, cold_infectee = self, other_human

            # (assumption) no overloading with covid
            if cold_infectee.infection_timestamp is None:
                if self.rng.random() < self.conf.get("COLD_CONTAGIOUSNESS"):
                    cold_infectee.cold_timestamp = self.env.timestamp

        # flu tansmission
        if self.has_flu ^ other_human.has_flu:
            flu_infector, flu_infectee = other_human, self
            if self.flu_timestamp is not None:
                flu_infector, flu_infectee = self, other_human

            # (assumption) no overloading with covid
            if flu_infectee.infection_timestamp is not None:
                if self.rng.random() < self.conf.get("FLU_CONTAGIOUSNESS"):
                    flu_infectee.flu_timestamp = self.env.timestamp

    def check_covid_contagion(self, other_human, t_near, h1_msg, h2_msg):
        """
        Determines if covid contagion takes place.
        If yes, intializes the appropriate variables needed for covid progression.

        Args:
            other_human (covid19sim.human.Human): other_human with whom the encounter took place
            t_near (float): duration for which this encounter took place (seconds)
            h1_msg ():
            h2_msg ():

        Returns:
            infector (covid19sim.human.Human): one who infected the infectee. None if contagion didn't occur.
            infectee (covid19sim.human.Human): one who got infected by the infector. None if contagion didn't occur.
        """
        if not (
            (self.is_infectious and other_human.is_susceptible)
            or (self.is_susceptible and other_human.is_infectious)
        ):
            return None, None, 0

        infector, infectee, p_infection = None, None, None
        if self.is_infectious:
            infector, infectee = self, other_human
            infectee_msg = h2_msg
        else:
            assert other_human.is_infectious, "expected other_human to be infectious"
            infector, infectee = other_human, self
            infectee_msg = h1_msg

        infectiousness = infectiousness_delta(infector, t_near)
        p_infection = get_human_human_p_transmission(infector,
                                      infectiousness,
                                      infectee,
                                      self.location.social_contact_factor,
                                      self.conf['CONTAGION_KNOB'],
                                      self,
                                      other_human)

        x_human = infector.rng.random() < p_infection

        # track infection related parameters
        self.city.tracker.track_infection(source="human",
                                    from_human=infector,
                                    to_human=infectee,
                                    location=self.location,
                                    timestamp=self.env.timestamp,
                                    p_infection=p_infection,
                                    success=x_human,
                                    viral_load=infector.viral_load,
                                    infectiousness=infectiousness)
        # infection
        if x_human and infectee.is_susceptible:
            infector.n_infectious_contacts += 1
            infectee._get_infected(initial_viral_load=infector.rng.random())
            if infectee_msg is not None:  # could be None if we are not currently tracing
                infectee_msg._exposition_event = True

            # log
            Event.log_exposed(self.conf.get('COLLECT_LOGS'), infectee, infector, p_infection, self.env.timestamp)
        else:
            infector, infectee = None, None

        return infector, infectee, p_infection

    def _get_infected(self, initial_viral_load):
        """
        Initializes necessary attributes for COVID progression.

        Args:
            initial_viral_load (float): initial value of viral load
        """
        self.infection_timestamp = self.env.timestamp
        self.initial_viral_load = initial_viral_load
        compute_covid_properties(self)

    def initialize_daily_risk(self, current_day_idx: int):
        """Initializes the risk history map with a new/copied risk value for the given day, if needed.

        Will also drop old unnecessary entries in the current & previous risk history maps.
        """
        if self.test_result:
            days_since_test_result = (self.env.timestamp - self.test_time -
                                      datetime.timedelta(days=self.time_to_test_result)).days
            if self.test_result == "negative" and \
                    days_since_test_result >= self.conf.get("RESET_NEGATIVE_TEST_RESULT_DELAY", 2):
                self.reset_test_result()
            elif self.test_result == "positive" and \
                    days_since_test_result >= self.conf.get("RESET_POSITIVE_TEST_RESULT_DELAY",
                                                            self.conf.get("TRACING_N_DAYS_HISTORY") + 1):
                self.reset_test_result()

        if not self.risk_history_map:  # if we're on day zero, add a baseline risk value in
            self.risk_history_map[current_day_idx] = self.baseline_risk
        elif current_day_idx not in self.risk_history_map:
            assert (current_day_idx - 1) in self.risk_history_map, \
                "humans should never skip a day worth of risk refresh"
            self.risk_history_map[current_day_idx] = self.risk_history_map[current_day_idx - 1]

        curr_day_set = set(self.risk_history_map.keys())
        prev_day_set = set(self.prev_risk_history_map.keys())
        day_set_diff = curr_day_set.symmetric_difference(prev_day_set)
        assert not day_set_diff or day_set_diff == {current_day_idx}, \
            "1st timeslot should have single-day-diff, otherwise no diff, what is this?"
        history_day_idxs = curr_day_set | prev_day_set
        expected_history_len = self.conf.get("TRACING_N_DAYS_HISTORY")
        for day_idx in history_day_idxs:
            assert day_idx <= current_day_idx, "...we're looking into the future now?"
            if current_day_idx - day_idx > expected_history_len:
                del self.risk_history_map[day_idx]
                if day_idx in self.prev_risk_history_map:
                    del self.prev_risk_history_map[day_idx]
        # ready for the day now; prepare the prev risk entry in case we need a quick diff
        self.prev_risk_history_map[current_day_idx] = self.risk_history_map[current_day_idx]
        logging.debug(f"Initializing the risk of {self.name} to "
                      f"{self.risk_history_map[current_day_idx]}")

    def check_if_latest_risk_level_changed(self):
        """Returns whether the latest risk level stored in the current/previous risk history maps match."""
        previous_latest_day = max(self.prev_risk_history_map.keys())
        new_latest_day = max(self.risk_history_map.keys())
        prev_risk_level = min(self.proba_to_risk_level_map(self.prev_risk_history_map[previous_latest_day]), 15)
        curr_risk_level = min(self.proba_to_risk_level_map(self.risk_history_map[new_latest_day]), 15)
        return prev_risk_level != curr_risk_level

    def run_timeslot_lightweight_jobs(
            self,
            init_timestamp: datetime.datetime,
            current_timestamp: datetime.datetime,
            personal_mailbox: PersonalMailboxType,
    ):
        """Runs the first half of processes that should happen when the app is woken up at the
        human's timeslot. These include symptom updates, initial risk updates & contact tracing,
        but not clustering+risk level inference, as that is done elsewhere (batched).

        This function will change the underlying state of lots of things owned by the human object.
        This includes stuff related to infectiousness, symptoms, and risk level update messages. The
        function will return the update messages that should be sent out to past contacts (if any).

        Args:
            init_timestamp: initialization timestamp of the simulation.
            current_timestamp: the current timestamp of the simulation.
            personal_mailbox: centralized mailbox with all recent update messages.

        Returns:
            A list of update messages to send out to contacts (if any).
        """
        assert current_timestamp.hour in self.time_slots
        assert self.last_date["symptoms_updated"] <= current_timestamp.date()
        current_day_idx = (current_timestamp - init_timestamp).days
        assert current_day_idx >= 0
        self.contact_book.cleanup_contacts(init_timestamp, current_timestamp)
        self.last_date["symptoms_updated"] = current_timestamp.date()
        self.update_reported_symptoms()
        # baseline risk might be updated by methods below (if enabled)
        self.risk_history_map[current_day_idx] = self.baseline_risk

        # if you're dead, not tracing, or using the transformer you don't need to update your risk here
        if self.is_dead or self.conf.get("RISK_MODEL") == "transformer":
            return

        # All tracing methods that are _not ML_ (heuristic, bdt1, bdt2, etc) will compute new risks here
        risks = self.intervention.compute_risk(self, personal_mailbox, self.city.hd)
        for day_offset, risk in enumerate(risks):
            if current_day_idx - day_offset in self.risk_history_map:
                self.risk_history_map[current_day_idx - day_offset] = risk

    def apply_transformer_risk_updates(
            self,
            current_day_idx: int,
            risk_history: typing.List[float],
    ):
        """Applies a vector of risk values predicted by the transformer to this human.

        Args:
            current_day_idx: the current day index inside the simulation.
            current_timestamp: the current timestamp of the simulation.
            risk_history: the risk history vector predicted by the transformer.
        """
        assert self.conf.get("RISK_MODEL") == "transformer"
        assert len(risk_history) == self.contact_book.tracing_n_days_history, \
            "unexpected transformer history coverage; what's going on?"
        for day_offset_idx in range(len(risk_history)):  # note: idx:0 == today
            self.risk_history_map[current_day_idx - day_offset_idx] = risk_history[day_offset_idx]

    def recover_health(self):
        """
        Implements basic functionalities to recover from non-COVID diseases
        """
        if (self.has_cold and
            self.days_since_cold >= len(self.cold_progression)):
            self.cold_timestamp = None
            self.cold_symptoms = []

        if (self.has_flu and
            self.days_since_flu >= len(self.flu_progression)):
            self.flu_timestamp = None
            self.flu_symptoms = []

        if (self.has_allergy_symptoms and
            self.days_since_allergies >= len(self.allergy_progression)):
            self.allergy_timestamp = None
            self.allergy_symptoms = []

    def catch_other_disease_at_random(self):
        # BUG: Is it intentional that if a random cold is caught, then one
        #      cannot also catch a random flu, due to early return?
        #
        # # assumption: no other infection if already infected with covid
        # if self.infection_timestamp is not None:
        #     return

        # Catch a random cold
        if not self.has_cold and self.rng.random() < self.conf["P_COLD_TODAY"]:
            self.ts_cold_symptomatic = self.env.now
            # print("caught cold")
            return

        # Catch a random flu (TODO: model seasonality through P_FLU_TODAY)
        if not self.has_flu and self.rng.random() < self.conf["P_FLU_TODAY"]:
            self.ts_flu_symptomatic = self.env.now
            return

        # Have random allergy symptoms
        if "allergies" in self.preexisting_conditions and self.rng.random() < self.conf["P_HAS_ALLERGIES_TODAY"]:
            self.ts_allergy_symptomatic = self.env.now
            # print("caught allergy")
            return

    def expire(self):
        """
        This function (generator) will cause the human to expire, after which self.is_dead==True.
        Yields self.env.timeout(np.inf), which when passed to env.procces will inactivate self
        for the remainder of the simulation.

        Yields:
            generator
        """
        self.infection_timestamp = None
        self.recovered_timestamp = datetime.datetime.max
        self.all_symptoms, self.covid_symptoms = [], []
        Event.log_recovery(self.conf.get('COLLECT_LOGS'), self, self.env.timestamp, death=True)
        # important to remove this human from the location or else there will be sampled interactions
        if self in self.location.humans:
            self.location.remove_human(self)
        self.household.residents.remove(self) # remove from the house
        self.mobility_planner.cancel_all_events()
        self.city.tracker.track_deaths() # track
        yield self.env.timeout(np.inf)

    def set_tracing_method(self, tracing_method):
        """
        Sets tracing method and initializes recommendation levels.
        """
        self.intervention = tracing_method
        # (delete) remove intervention_start
        self.update_recommendations_level(intervention_start=True)

    def run(self):
        """
        Transitions `self` from one `Activity` to other
        Note: use -O command line option to avoid checking for assertions

        Yields:
            simpy.events.Event:
        """

        previous_activity, next_activity = None, self.mobility_planner.get_next_activity()
        while True:

            #
            if next_activity.human_dies:
                yield self.env.process(self.expire())

            #
            if next_activity.location is not None:

                # (debug) to print the schedule for someone
                # if self.name == "human:7":
                #       print("A\t", self.env.timestamp, self, next_activity)

                # /!\ TODO - P - check for the capacity at a location; it requires adjustment of timestamps
                # with next_activity.location.request() as request:
                #     yield request
                #     yield self.env.process(self.transition_to(next_activity, previous_activity))

                assert abs(self.env.timestamp - next_activity.start_time).seconds == 0, "start times do not align..."
                yield self.env.process(self.transition_to(next_activity, previous_activity))
                assert abs(self.env.timestamp - next_activity.end_time).seconds == 0, " end times do not align..."

                previous_activity, next_activity = next_activity, self.mobility_planner.get_next_activity()

            else:
                # supervised or invitation type of activities (prepend_name) will require to refresh their location
                # because this activity depends on parent_activity, we need to give a full 1 second to guarantee
                # that parent activity will have its location confirmed. This creates a discrepancy in env.timestamp and
                # next_activity.start_time.
                next_activity.refresh_location()

                # (debug) to print the schedule for someone
                # if self.name == "human:110":
                #       print("A\t", self.env.timestamp, self, next_activity)

                yield self.env.timeout(1)

                # realign the activities and keep the previous_activity as it is if this next_activity can't be scheduled
                next_activity.adjust_time(seconds=1, start=True)
                if next_activity.duration <= 0:
                    next_activity = self.mobility_planner.get_next_activity()
                    continue
                previous_activity.adjust_time(seconds=1, start=False)

    def increment_healthy_day(self):
        if not self.state[2]: # not infectious
            self.healthy_days += 1

    ############################## MOBILITY ##################################
    @property
    def lat(self):
        return self.location.lat if self.location else self.household.lat

    @property
    def lon(self):
        return self.location.lon if self.location else self.household.lon

    @property
    def obs_lat(self):
        if self.conf.get("LOCATION_TECH") == 'bluetooth':
            return round(self.lat + self.rng.normal(0, 2))
        else:
            return round(self.lat + self.rng.normal(0, 10))

    @property
    def obs_lon(self):
        if self.conf.get("LOCATION_TECH") == 'bluetooth':
            return round(self.lon + self.rng.normal(0, 2))
        else:
            return round(self.lon + self.rng.normal(0, 10))

    def transition_to(self, next_activity, previous_activity):
        """
        Enter/Exit human to/from a `location` for some `duration`.
        Once human is at a location, encounters are sampled.
        During the stay, human is likely to be infected either by a contagion or
        through environmental transmission.
        Cold/Flu/Allergy onset also takes place in this function.

        Args:
            next_activity (covid19sim.utils.mobility_planner.Acitvity): next activity to do
            previous_activity (covid19sim.utils.mobility_planner.Acitvity): previous activity where human was

        Yields:
            (simpy.events.Timeout)
        """
        # print("before", self.env.timestamp, self, location, duration)
        location = next_activity.location
        duration = next_activity.duration
        type_of_activity = next_activity.name

        # track transitions & locations visited
        self.city.tracker.track_mobility(previous_activity, next_activity, self)

        # add human to the location
        self.location = location
        location.add_human(self)

        self.location_start_time = self.env.now
        self.location_leaving_time = self.location_start_time + duration

        # check if human needs a test if it's a hospital
        self.check_if_needs_covid_test(at_hospital=isinstance(location, (Hospital, ICU)))

        yield self.env.timeout(duration)
        # print("after", self.env.timestamp, self, location, duration)

        # only sample interactions if there is a possibility of infection or message exchanges
        if duration >= min(self.conf['MIN_MESSAGE_PASSING_DURATION'], self.conf['INFECTION_DURATION']):
            # sample interactions with other humans at this location
            # unknown are the ones that self is not aware of e.g. person sitting next to self in a cafe
            # sleep is an inactive stage so we sample only unknown interactions
            known_interactions, unknown_interactions = location.sample_interactions(self, unknown_only = type_of_activity == "sleep")
            self.interact_with(known_interactions, type="known")
            self.interact_with(unknown_interactions, type="unknown")

        # environmental transmission
        location.check_environmental_infection(self)

        # remove human from this location
        location.remove_human(self)

    def interact_with(self, interaction_profile, type):
        """
        Implements the exchange of bluetooth messages and contagion at the time of encounter.

        Args:
            interaction_profile: each element is expected as follows -
                human (covid19sim.human.Human): other human with whom to interact
                distance_profile (covid19sim.locations.location.DistanceProfile): distance from which this encounter took place (cms)
                duration or t_near (float): duration for which this encounter took place (seconds)
            type (string): type of interaction to sample. expects "known", "unknown"
        """
        for other_human, distance_profile, t_near in interaction_profile:

            # keeping known connections help in bringing two people together resulting in repeated contacts of known ones
            if type  == "known":
                self.known_connections.add(other_human)
                other_human.known_connections.add(self)

            # compute detected bluetooth distance and exchange bluetooth messages if conditions are satisfied
            h1_msg, h2_msg = self._exchange_app_messages(other_human, distance_profile.distance, t_near)

            contact_condition = (
                distance_profile.distance <= self.conf.get("INFECTION_RADIUS")
                and t_near >= self.conf.get("INFECTION_DURATION")
            )

            # used for matching "mobility" between methods
            scale_factor_passed = contact_condition and self.rng.random() < self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")

            self.city.tracker.track_mixing(human1=self, human2=other_human, duration=t_near,
                            distance_profile=distance_profile, timestamp=self.env.timestamp, location=self.location,
                            interaction_type=type, contact_condition=contact_condition, global_mbility_factor=scale_factor_passed)

            # Conditions met for possible infection (https://www.cdc.gov/coronavirus/2019-ncov/hcp/guidance-risk-assesment-hcp.html)
            if contact_condition:

                # increment effective contacts
                if (
                    self.conf['RISK_MODEL'] == ""
                    or  (
                        self.conf['INTERVENTION_START_TIME'] is not None
                        and self.env.timestamp >= self.conf['INTERVENTION_START_TIME']
                    )
                ):
                    self._increment_effective_contacts(other_human)
                    other_human._increment_effective_contacts(self)

                # infection
                infector, infectee, p_infection = None, None, 0
                if scale_factor_passed:
                    infector, infectee, p_infection = self.check_covid_contagion(other_human, t_near, h1_msg, h2_msg)

                # determine if cold and flu contagion occured
                self.check_cold_and_flu_contagion(other_human)

                # logging
                Event.log_encounter(
                    self.conf['COLLECT_LOGS'],
                    self,
                    other_human,
                    location=self.location,
                    duration=t_near,
                    distance=distance_profile.distance,
                    infectee=None if not infectee else infectee.name,
                    p_infection=p_infection,
                    time=self.env.timestamp
                )

    def _increment_effective_contacts(self, other_human):
        """
        Increments attributs related to count effective contacts of `self`.

        Args:
            other_human (covid19sim.human.Human): `human` with whom contact just happened
        """
        self.num_contacts += 1
        self.effective_contacts += self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")
        # if not other_human.state.index(1) in [1,2]:
        if not self.state[2]: # if not infectious, then you are "healthy"
            self.healthy_effective_contacts += self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")

    def exposure_array(self, date):
        """
        [summary]

        Args:
            date ([type]): [description]

        Returns:
            [type]: [description]
        """
        warnings.warn("Deprecated in favor of inference.helper.exposure_array()", DeprecationWarning)
        # dont change the logic in here, it needs to remain FROZEN
        exposed = False
        exposure_day = None
        if self.infection_timestamp:
            exposure_day = (date - self.infection_timestamp).days
            if exposure_day >= 0 and exposure_day < 14:
                exposed = True
            else:
                exposure_day = None
        return exposed, exposure_day

    def recovered_array(self, date):
        """
        [summary]

        Args:
            date ([type]): [description]

        Returns:
            [type]: [description]
        """
        warnings.warn("Deprecated in favor of inference.helper.recovered_array()", DeprecationWarning)
        is_recovered = False
        recovery_day = (date - self.recovered_timestamp).days
        if recovery_day >= 0 and recovery_day < 14:
            is_recovered = True
        else:
            recovery_day = None
        return is_recovered, recovery_day

    @property
    def infectiousnesses(self):
        """Returns a list of this human's infectiousnesses over `TRACING_N_DAYS_HISTORY` days."""
        expected_history_len = self.conf.get("TRACING_N_DAYS_HISTORY")
        if not self.infectiousness_history_map:
            return [0] * expected_history_len
        latest_day = max(list(self.infectiousness_history_map.keys()))
        oldest_day = latest_day - expected_history_len + 1
        result = [self.infectiousness_history_map[oldest_day]
                  if oldest_day in self.infectiousness_history_map else 0.0]
        for day_idx in range(oldest_day + 1, latest_day + 1):
            if day_idx in self.infectiousness_history_map:
                result.append(self.infectiousness_history_map[day_idx])
            else:
                result.append(result[-1])
        assert len(result) == expected_history_len
        return result[::-1]  # index 0 = latest day

    def fill_infectiousness_history_map(self, current_day):
        """
        Populates past infectiousness values in the map.

        Args:
            current_day (int): day for which it infectiousness value need to be updated.
        """
        if (
            self.conf['RISK_MODEL'] == "transformer"
            or self.conf['COLLECT_TRAINING_DATA']
        ):
            # /!\ Only used for oracle and transformer
            if current_day not in self.infectiousness_history_map:
                # contrarily to risk, infectiousness only changes once a day (human behavior has no impact)
                self.infectiousness_history_map[current_day] = calculate_average_infectiousness(self)

    ############################## RISK PREDICTION #################################
    def _exchange_app_messages(self, other_human, distance, duration):
        """
        Implements the exchange of encounter messages between `self` and `other_human`.

        Args:
            other_human (covid19sim.human.Human) other human that self is communiciating via bluetooth
            distance (float): actual distance of encounter with other_human (cms)
            duration (float): time duration of encounter (seconds)

        Returns:
            h1_msg ():
            h2_msg ():
        """
        if not other_human.has_app or not self.has_app:
            return None, None

        t_near = duration
        # phone_bluetooth_noise is a value selected between 0 and 2 meters to approximate the noise in the manufacturers bluetooth chip
        # distance is the "average" distance of the encounter
        # self.rng.random() - 0.5 gives a uniform random variable centered at 0
        # we scale by the distance s.t. if the true distance of the encounter is 2m you could think it is 0m or 4m,
        # whereas an encounter of 1m has a possible distance of 0.5 and 1.5m
        # a longer discussion is contained in docs/bluetooth.md
        approximated_bluetooth_distance = distance + distance * (self.rng.rand() - 0.5) * np.mean([self.phone_bluetooth_noise, other_human.phone_bluetooth_noise])
        assert approximated_bluetooth_distance <= 2*distance

        h1_msg, h2_msg = None, None
        # The maximum distance of a message which we would consider to be "high risk" and therefore meriting an
        # encounter message is under 2 meters for at least 5 minutes.
        if approximated_bluetooth_distance < self.conf.get("MAX_MESSAGE_PASSING_DISTANCE") and \
                t_near > self.conf.get("MIN_MESSAGE_PASSING_DURATION") and \
                self.has_app and \
                other_human.has_app:

            remaining_time_in_contact = t_near
            encounter_time_granularity = self.conf.get("MIN_MESSAGE_PASSING_DURATION")
            exchanged = False
            while remaining_time_in_contact > encounter_time_granularity:
                exchanged = True
                # note: every loop we will overwrite the messages but it doesn't matter since
                # they're recorded in the contact books and we only need one for exposure flagging
                h1_msg, h2_msg = exchange_encounter_messages(
                    h1=self,
                    h2=other_human,
                    # TODO: could adjust real timestamps in encounter messages based on remaining time?
                    # env_timestamp=self.env.timestamp - datetime.timedelta(minutes=remaining_time_in_contact),
                    # the above change might break clustering asserts if we somehow jump across timeslots/days
                    env_timestamp=self.env.timestamp,
                    initial_timestamp=self.env.initial_timestamp,
                    use_gaen_key=self.conf.get("USE_GAEN"),
                )
                remaining_time_in_contact -= encounter_time_granularity

            if exchanged:
                self.city.tracker.track_bluetooth_communications(human1=self, human2=other_human, location=self.location, timestamp=self.env.timestamp)

            Event.log_encounter_messages(
                self.conf['COLLECT_LOGS'],
                self,
                other_human,
                location=self.location,
                duration=duration,
                distance=distance,
                time=self.env.timestamp
            )

        return h1_msg, h2_msg

    @property
    def baseline_risk(self):
        if self.is_removed:
            return 0.0
        elif self.reported_test_result == "positive":
            return 1.0
        elif self.reported_test_result == "negative":
            return 0.2
        else:
            return self.conf.get("BASELINE_RISK_VALUE")

    @property
    def risk(self):
        if self.risk_history_map:
            cur_day = int(self.env.now - self.env.ts_initial) // SECONDS_PER_DAY
            if cur_day in self.risk_history_map:
                return self.risk_history_map[cur_day]
            else:
                last_day = max(self.risk_history_map.keys())
                return self.risk_history_map[last_day]
        else:
            return self.baseline_risk

    @property
    def risk_level(self):
        return min(self.proba_to_risk_level_map(self.risk), 15)

    def update_recommendations_level(self, intervention_start=False):
        if not self.has_app:
            self._rec_level = -1
            return

        self._rec_level = self.intervention.get_recommendations_level(
            self,
            self.conf.get("REC_LEVEL_THRESHOLDS"),
            self.conf.get("MAX_RISK_LEVEL"),
            intervention_start=intervention_start,
        )

        if self.conf.get("RISK_MODEL") == "digital":
            assert self._rec_level == 0 or self._rec_level == 3

        self.intervened_behavior.trigger_intervention(reason="risk-level-update", human=self)

    @property
    def rec_level(self):
        return self._rec_level

    def __repr__(self):
        return f"H:{self.name} age:{self.age}, SEIR:{int(self.is_susceptible)}{int(self.is_exposed)}{int(self.is_infectious)}{int(self.is_removed)}"
