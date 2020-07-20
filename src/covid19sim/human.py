"""
Contains the `Human` class that defines the behavior of human.
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
from covid19sim.interventions.behaviors import Behavior
from covid19sim.interventions.recommendation_manager import NonMLRiskComputer
from covid19sim.utils.utils import compute_distance, proba_to_risk_fn
from covid19sim.locations.city import PersonalMailboxType
from covid19sim.locations.hospital import Hospital, ICU
from covid19sim.log.event import Event
from collections import deque

from covid19sim.utils.utils import _normalize_scores, draw_random_discrete_gaussian, filter_open, filter_queue_max
from covid19sim.epidemiology.human_properties import may_develop_severe_illness, _get_inflammatory_disease_level,\
    _get_preexisting_conditions, _get_random_sex, get_carefulness, get_age_bin
from covid19sim.epidemiology.viral_load import compute_covid_properties, viral_load_for_day
from covid19sim.epidemiology.symptoms import _get_cold_progression, _get_flu_progression,\
    _get_allergy_progression
from covid19sim.epidemiology.p_infection import get_human_human_p_transmission, infectiousness_delta
from covid19sim.utils.constants import SECONDS_PER_MINUTE, SECONDS_PER_HOUR, SECONDS_PER_DAY
from covid19sim.inference.message_utils import ContactBook, exchange_encounter_messages, RealUserIDType
from covid19sim.utils.visits import Visits

class Human(object):
    """
    [summary]
    """

    def __init__(self, env, city, name, age, rng, has_app, infection_timestamp, household, workplace, profession, rho=0.3, gamma=0.21, conf={}):
        """
        [summary]

        Args:
            env ([type]): [description]
            city ([type]): [description]
            name ([type]): [description]
            age ([type]): [description]
            rng ([type]): [description]
            infection_timestamp ([type]): [description]
            household ([type]): [description]
            workplace ([type]): [description]
            profession ([type]): [description]
            rho (float, optional): [description]. Defaults to 0.3.
            gamma (float, optional): [description]. Defaults to 0.21.
            conf (dict): yaml experiment configuration
        """

        """
        Biological Properties
        Covid-19
        App-related
        Interventions
        Risk prediction
        Mobility
        """

        # Utility References
        self.conf = conf  # Simulation-level Configurations
        self.env = env  # Simpy Environment (primarily used for timing / syncronization)
        self.city = city  # Manages lots of things inc. initializing humans and locations, tracking/updating humans

        # SEIR Tracking
        self.recovered_timestamp = datetime.datetime.min  # Time of recovery from covid -- min if not yet recovered
        self._infection_timestamp = None  # private time of infection with covid - implemented this way to ensure only infected 1 time
        self.infection_timestamp = infection_timestamp  # time of infection with covid
        self.n_infectious_contacts = 0  # number of high-risk (infected someone) contacts with an infected individual.
        self.exposure_source = None  # source that exposed this human to covid (and infected them). None if not infected.

        # Human-related properties
        self.name: RealUserIDType = f"human:{name}"  # Name of this human
        self.rng = np.random.RandomState(rng.randint(2 ** 16))  # RNG for this particular human
        self.profession = profession  # The job this human has (e.g. healthcare worker, retired, school, etc)
        self.is_healthcare_worker = True if profession == "healthcare" else False  # convenience boolean to check if is healthcare worker
        self.known_connections = set() # keeps track of all otehr humans that this human knows of
        self._workplace = (None,) # initialized this way to be consistent with the final deque assignment
        self.does_not_work = False # to identify those who weren't assigned any workplace from the beginning
        self.work_start_time, self.work_end_time, self.working_days = None, None, []

        # Logging / Tracking
        self.track_this_human = False  # tracks transition of human everytime there is a change in it's location. see `self.track_me`
        self.my_history = []  # if `track_this_human` is True, records of transition is stored in this list
        self.r0 = []  # TODO: @PRATEEK plz comment this
        self._events = []  # TODO: @PRATEEK plz comment this


        """ Biological Properties """
        # Individual Characteristics
        self.sex = _get_random_sex(self.rng, self.conf)  # The sex of this person conforming with Canadian statistics
        self.age = age  # The age of this person, conforming with Canadian statistics
        _age_bin = get_age_bin(age, width=10)  # Age bins of width 10 are required for Oxford-like COVID-19 infection model and social mixing tracker
        self.normalized_susceptibility = self.conf['NORMALIZED_SUSCEPTIBILITY_BY_AGE'][_age_bin.bin]  # Susceptibility to Covid-19 by age
        self.mean_daily_interaction_age_group = self.conf['MEAN_DAILY_INTERACTION_FOR_AGE_GROUP'][_age_bin.bin]  # Social mixing is determined by age
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
        self.has_allergies = self.rng.rand() < self.conf.get("P_ALLERGIES")  # determines whether this person has allergies
        len_allergies = self.rng.normal(1/self.carefulness, 1)   # determines the number of symptoms this persons allergies would present with (if they start experiencing symptoms)
        self.len_allergies = 7 if len_allergies > 7 else np.ceil(len_allergies)
        self.allergy_progression = _get_allergy_progression(self.rng)  # if this human starts having allergy symptoms, then there is a progression of symptoms over one or multiple days


        """ Covid-19 """
        # Covid-19 properties
        self.viral_load_plateau_height, self.viral_load_plateau_start, self.viral_load_plateau_end = None, None, None  # Determines aspects of the piece-wise linear viral load curve for this human
        self.incubation_days = None  # number of days the virus takes to incubate before the person becomes infectious
        self.recovery_days = None  # number of recovery days post viral load plateau
        self.infectiousness_onset_days = None  # number of days after exposure that this person becomes infectious
        self.can_get_really_sick = may_develop_severe_illness(self.age, self.sex, self.rng)  # boolean representing whether this person may need to go to the hospital
        self.can_get_extremely_sick = self.can_get_really_sick and self.rng.random() >= 0.7  # &severe; 30% of severe cases need ICU
        self.never_recovers = self.rng.random() <= self.conf.get("P_NEVER_RECOVERS")[min(math.floor(self.age/10), 8)]  # boolean representing that this person will die if they are infected with Covid-19
        self.initial_viral_load = self.rng.rand() if infection_timestamp is not None else 0  # starting value for Covid-19 viral load if this person is one of the initially exposed people
        self.is_immune = False  # whether this person is immune to Covid-19 (happens after recovery)
        if self.infection_timestamp is not None:  # if this is an initially Covid-19 sick person
            compute_covid_properties(self)  # then we pre-calculate the course of their disease
        self.last_state = self.state  # And we set their SEIR state (starts as either Susceptible or Exposed)

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
        self.has_app = has_app  # Does this prson have the app
        time_slot = rng.randint(0, 24)  # Assign this person to some timeslot
        self.time_slots = [
            int((time_slot + i * 24 / self.conf.get('UPDATES_PER_DAY')) % 24)
            for i in range(self.conf.get('UPDATES_PER_DAY'))
        ]  # If people update their risk predictions 4 times per day (every 6 hours) then this code assigns the specific times _this_ person will update
        self.phone_bluetooth_noise = self.rng.rand()  # Error in distance estimation using Bluetooth with a specific type of phone is sampled from a uniform distribution between 0 and 1

        # Observed attributes; whether people enter stuff in the app
        self.has_logged_info = self.has_app and self.rng.rand() < self.carefulness  # Determines whether this person writes their demographic data into the app
        self.obs_is_healthcare_worker = True if self.is_healthcare_worker and self.rng.random()<0.9 else False  # 90% of the time, healthcare workers will declare it
        self.obs_age = self.age if self.has_app and self.has_logged_info else None  # The age of this human reported to the app
        self.obs_sex = self.sex if self.has_app and self.has_logged_info else None  # The sex of this human reported to the app
        self.obs_preexisting_conditions = self.preexisting_conditions if self.has_app and self.has_logged_info else []  # the preexisting conditions of this human reported to the app
        self.obs_hospitalized = False  # Whether this person was hospitalized (as reported to the app)
        self.obs_in_icu = False  # Whether this person was put in the ICU (as reported to the app)


        """ Interventions """
        self.tracing = False  # A reference to the NonMLRiskComputer logic engine which implements tracing methods
        self.WEAR_MASK = False  # A boolean value determining whether this person will try to wear a mask during encounters
        self.wearing_mask = False  # A bolean value that represents whether this person is currently wearing a mask
        self.mask_efficacy = 0.  # A float value representing how good this person is at wearing a mask (i.e. healthcare workers are better than others)
        self.notified = False  # Value indicating whether this person has been "notified" to start following some interventions
        self.tracing_method = None  # Type of contact tracing to do, e.g. Transformer or binary contact tracing or heuristic
        self._maintain_extra_distance = deque((0,))  # Represents the extra distance this person could take as a result of an intervention
        self._follows_recommendations_today = None  # Whether this person will follow app recommendations today
        self._rec_level = -1  # Recommendation level used for Heuristic / ML methods
        self._intervention_level = -1  # Intervention level (level of behaviour modification to apply), for logging purposes
        self.recommendations_to_follow = OrderedSet()  # which recommendations this person will follow now
        self._time_encounter_reduction_factor = deque((1.0,))  # how much does this person try to reduce the amount of time they are in contact with others
        self.hygiene = 0  # start everyone with a baseline hygiene. Only increase it once the intervention is introduced.
        self._test_recommended = False  # does the app recommend that this person should get a covid-19 test
        self.effective_contacts = 0  # A scaled number of the high-risk contacts (under 2m for over 15 minutes) that this person had
        self.num_contacts = 0  # unscaled number of high-risk contacts


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
        self.assign_household(household)  # assigns this person to the specified household
        self.rho = rho  # controls mobility (how often this person goes out and visits new places)
        self.gamma = gamma  # controls mobility (how often this person goes out and visits new places)
        self.rest_at_home = False  # determines whether people rest at home due to feeling sick (used to track mobility due to symptoms)
        self.visits = Visits()  # used to help implement mobility
        self.travelled_recently = self.rng.rand() > self.conf.get("P_TRAVELLED_INTERNATIONALLY_RECENTLY")
        self.last_date = defaultdict(lambda : self.env.initial_timestamp.date())  # used to track the last time this person did various things (like record smptoms)
        self.mobility_planner = MobilityPlanner(self, self.env, self.conf)

        self.location_leaving_time = self.env.ts_initial + SECONDS_PER_HOUR
        self.location_start_time = self.env.ts_initial

    @property
    def follows_recommendations_today(self):
        last_date = self.last_date["follow_recommendations"]
        current_date = self.env.timestamp.date()
        if last_date is None or (current_date - last_date).days > 0:
            proba = self.conf.get("DROPOUT_RATE")
            self.last_date["follow_recommendations"] = self.env.timestamp.date()
            self._follows_recommendations_today = self.rng.rand() < (1 - proba)
        return self._follows_recommendations_today

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
        self._workplace = deque((workplace,))  # Created as a list because we sometimes modify human's workplace to WFH if in quarantine, then go back to work when released

    @property
    def workplace(self):
        return self._workplace[0]

    def set_temporary_workplace(self, new_workplace):
        self._workplace.appendleft(new_workplace)

    def revert_workplace(self):
        workplace = self._workplace[-1]
        self._workplace.clear()
        self._workplace.appendleft(workplace)

    @property
    def maintain_extra_distance(self):
        return self._maintain_extra_distance[0]

    def set_temporary_maintain_extra_distance(self, new_maintain_extra_distance):
        self._maintain_extra_distance.appendleft(new_maintain_extra_distance)

    def revert_maintain_extra_distance(self):
        maintain_extra_distance = self._maintain_extra_distance[-1]
        self._maintain_extra_distance.clear()
        self._maintain_extra_distance.appendleft(maintain_extra_distance)

    @property
    def time_encounter_reduction_factor(self):
        return self._time_encounter_reduction_factor[0]

    def set_temporary_time_encounter_reduction_factor(self, new_time_encounter_reduction_factor):
        self._time_encounter_reduction_factor.appendleft(new_time_encounter_reduction_factor)

    def revert_time_encounter_reduction_factor(self):
        time_encounter_reduction_factor = self._time_encounter_reduction_factor[-1]
        self._time_encounter_reduction_factor.clear()
        self._time_encounter_reduction_factor.appendleft(time_encounter_reduction_factor)

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
    def infection_timestamp(self):
        """
        Returns the timestamp when the human was infected by COVID.
        Returns None if human is not exposed or infectious.
        """
        return self._infection_timestamp

    @infection_timestamp.setter
    def infection_timestamp(self, val):
        """
        Sets the infection_timestamp to val.
        Raises AssertError at an attempt to overwrite infection_timestamp.
        """
        if self.infection_timestamp is not None:
            assert val is None, f"{self}: attempt to overwrite infection_timestamp"

        assert val is None or isinstance(val, datetime.datetime), f"{self}: improper type {type(val)} being assigned to infection_timestamp"
        self._infection_timestamp = val

    @property
    def is_susceptible(self):
        """
        Returns True if human is susceptible to being infected by Covid-19

        Returns:
            bool: if human is susceptible, False if not
        """
        return not self.is_exposed and not self.is_infectious and not self.is_removed

    @property
    def is_exposed(self):
        """
        Returns True if human has been exposed to Covid-19 but cannot yet infect anyone else

        Returns:
            bool: if human is exposed, False if not
        """
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp < datetime.timedelta(days=self.infectiousness_onset_days)

    @property
    def is_infectious(self):
        """
        Returns True if human is infectious i.e. is able to infect others

        Returns:
            bool: if human is infectious, False if not
        """
        return not self.is_removed and self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.infectiousness_onset_days)

    @property
    def is_removed(self):
        """
        Returns True if human is either dead or has recovered from COVID and can't be reinfected i.e is immune

        Returns:
            bool: True if human is immune or dead, False if not
        """
        return self.is_immune or self.is_dead

    @property
    def is_dead(self):
        """
        Returns True if the human is dead, otherwise False.

        Returns:
            bool: True if dead, False if not.
        """
        return self.recovered_timestamp == datetime.datetime.max

    @property
    def is_incubated(self):
        """
        Returns True if the human has spent sufficient time to culture the virus, otherwise False.

        Returns:
            bool: True if Covid-19 incubated, False if not.
        """
        return (not self.is_asymptomatic and self.infection_timestamp is not None and
                self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.incubation_days))

    @property
    def state(self):
        """
        The state (SEIR) that this person is in (True if in state, False otherwise)

        Returns:
            bool: True for the state this person is in (Susceptible, Exposed, Infectious, Removed)
        """
        return [int(self.is_susceptible), int(self.is_exposed), int(self.is_infectious), int(self.is_removed)]

    @property
    def has_cold(self):
        return self.cold_timestamp is not None

    @property
    def has_flu(self):
        return self.flu_timestamp is not None

    @property
    def has_allergy_symptoms(self):
        return self.allergy_timestamp is not None

    @property
    def days_since_covid(self):
        """
        The number of days since infection with Covid-19 for this person

        Returns:
            Int or None: Returns an Integer representing the number of days since infection, or None if not infected.
        """
        if self.infection_timestamp is None:
            return
        return (self.env.timestamp-self.infection_timestamp).days

    @property
    def days_since_cold(self):
        """
        The number of days since infection with cold for this person

        Returns:
            Int or None: Returns an Integer representing the number of days since infection, or None if not infected.
        """
        if self.cold_timestamp is None:
            return
        return (self.env.timestamp-self.cold_timestamp).days

    @property
    def days_since_flu(self):
        """
        The number of days since infection with flu for this person

        Returns:
            Int or None: Returns an Integer representing the number of days since infection, or None if not infected.
        """
        if self.flu_timestamp is None:
            return
        return (self.env.timestamp-self.flu_timestamp).days

    @property
    def days_since_allergies(self):
        """
        The number of days since allergies began for this person

        Returns:
            Int or None: Returns an Integer representing the number of days since allgergies started, or None if no allergies.
        """
        if self.allergy_timestamp is None:
            return
        return (self.env.timestamp-self.allergy_timestamp).days

    @property
    def is_really_sick(self):
        return self.can_get_really_sick and 'severe' in self.symptoms

    @property
    def is_extremely_sick(self):
        return self.can_get_extremely_sick and 'severe' in self.symptoms

    @property
    def viral_load(self):
        """
        Calculates the elapsed time since infection, returning this person's current viral load

        Returns:
            Float: Returns a real valued number between 0. and 1. indicating amount of viral load (proportional to infectiousness)
        """
        return viral_load_for_day(self, self.env.timestamp)

    def get_infectiousness_for_day(self, timestamp, is_infectious):
        """
        Scales the infectiousness value using pre-existing conditions

        Returns:
            [type]: [description]
        """
        infectiousness = 0.
        severity_multiplier = 1
        if is_infectious:
            if 'immuno-compromised' in self.preexisting_conditions:
              severity_multiplier += self.conf['IMMUNOCOMPROMISED_SEVERITY_MULTIPLIER_ADDITION']
            if 'cough' in self.symptoms:
              severity_multiplier += self.conf['COUGH_SEVERITY_MULTIPLIER_ADDITION']
            infectiousness = (viral_load_for_day(self, timestamp) * severity_multiplier * self.infection_ratio)/self.conf['INFECTIOUSNESS_NORMALIZATION_CONST']
        return infectiousness

    @property
    def infectiousness(self):
        return self.get_infectiousness_for_day(self.env.timestamp, self.is_infectious)

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
        if self.last_date['symptoms'] == self.env.timestamp.date():
            return

        self.last_date['symptoms'] = self.env.timestamp.date()

        if self.cold_timestamp is not None:
            t = self.days_since_cold
            if t < len(self.cold_progression):
                self.cold_symptoms = self.cold_progression[t]
            else:
                self.cold_symptoms = []

        if self.flu_timestamp is not None:
            t = self.days_since_flu
            if t < len(self.flu_progression):
                self.flu_symptoms = self.flu_progression[t]
            else:
                self.flu_symptoms = []

        if self.infection_timestamp is not None and not self.is_asymptomatic:
            t = self.days_since_covid
            if self.is_removed or t >= len(self.covid_progression):
                self.covid_symptoms = []
            else:
                self.covid_symptoms = self.covid_progression[t]

        if self.allergy_timestamp is not None:
            self.allergy_symptoms = self.allergy_progression[0]

        all_symptoms = set(self.flu_symptoms + self.cold_symptoms + self.allergy_symptoms + self.covid_symptoms)
        # self.new_symptoms = list(all_symptoms - set(self.all_symptoms))
        # TODO: remove self.all_symptoms in favor of self.rolling_all_symptoms[0]
        self.all_symptoms = list(all_symptoms)
        self.rolling_all_symptoms.appendleft(self.all_symptoms)
        self.city.tracker.track_symptoms(self)

    def update_reported_symptoms(self):
        """
        [summary]
        """
        self.update_symptoms()

        if self.last_date['reported_symptoms'] == self.env.timestamp.date():
            return

        self.last_date['reported_symptoms'] = self.env.timestamp.date()

        reported_symptoms = [s for s in self.rolling_all_symptoms[0] if self.rng.random() < self.carefulness]
        self.rolling_all_reported_symptoms.appendleft(reported_symptoms)

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
            time_time (str): time of testing
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
        self._will_report_test_result = self.rng.random() < self.carefulness
        if isinstance(self.location, (Hospital, ICU)):
            self.time_to_test_result = self.conf['TEST_TYPES'][test_type]['time_to_result']['in-patient']
        else:
            self.time_to_test_result = self.conf['TEST_TYPES'][test_type]['time_to_result']['out-patient']
        self.test_result_validated = self.test_type == "lab"
        Event.log_test(self.conf.get('COLLECT_LOGS'), self, self.test_time)
        self._test_results.appendleft((
            self.hidden_test_result,
            self._will_report_test_result,
            self.env.timestamp,  # for result availability checking later
            self.time_to_test_result,  # in days
        ))
        self.city.tracker.track_tested_results(self)

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
            if "severe" in self.symptoms:
                should_get_test = self.rng.rand() < self.conf['P_TEST_SEVERE']

            elif "moderate" in self.symptoms:
                should_get_test = self.rng.rand() < self.conf['P_TEST_MODERATE']

            elif "mild" in self.symptoms:
                should_get_test = self.rng.rand() < self.conf['P_TEST_MILD']

            # has been recommended the test by an intervention
            if not should_get_test and self._test_recommended:
                should_get_test = self.rng.random() < self.follows_recommendations_today

            if not should_get_test:
                # Has symptoms that a careful person would fear to be covid
                SUSPICIOUS_SYMPTOMS = set(self.conf['GET_TESTED_SYMPTOMS_CHECKED_BY_SELF'])
                if set(self.symptoms) & SUSPICIOUS_SYMPTOMS:
                    should_get_test = self.rng.rand() < self.carefulness

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
        if self.is_infectious and self.days_since_covid >= self.recovery_days:
            self.city.tracker.track_recovery(self.n_infectious_contacts, self.recovery_days)

            # TO DISCUSS: Should the test result be reset here? We don't know in reality
            # when the person has recovered; currently not reset
            # self.reset_test_result()
            self.infection_timestamp = None
            self.all_symptoms, self.covid_symptoms = [], []

            if self.never_recovers:
                yield self.env.process(self.expire())
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
        if not (self.is_infectious ^ other_human.is_infectious):
            return None, None, 0

        infector, infectee, p_infection = None, None, None
        if self.is_infectious:
            infector, infectee = self, other_human
            infectee_msg = h2_msg
        else:
            assert other_human.is_infectious
            infector, infectee = other_human, self
            infectee_msg = h1_msg

        p_infection = get_human_human_p_transmission(infector,
                                      infectiousness_delta(infector, t_near),
                                      infectee,
                                      self.location.social_contact_factor,
                                      self.conf['CONTAGION_KNOB'],
                                      self.conf['MASK_EFFICACY_FACTOR'],
                                      self.conf['HYGIENE_EFFICACY_FACTOR'],
                                      self,
                                      other_human)

        x_human = infector.rng.random() < p_infection
        self.city.tracker.track_p_infection(x_human, p_infection, infector.viral_load)
        if x_human and infectee.is_susceptible:
            infector.n_infectious_contacts += 1
            infectee.infection_timestamp = self.env.timestamp
            compute_covid_properties(infectee)
            if infectee_msg is not None:  # could be None if we are not currently tracing
                infectee_msg._exposition_event = True

            # logging & tracker
            Event.log_exposed(self.conf.get('COLLECT_LOGS'), infectee, infector, p_infection, self.env.timestamp)
            self.city.tracker.track_infection('human', from_human=infector, to_human=infectee, location=self.location, timestamp=self.env.timestamp)
        else:
            infector, infectee = None, None
        return infector, infectee, p_infection

    def move_to_hospital_if_required(self):
        """
        decision to move `self` to the hospital is made here.
        """
        # Behavioral imperatives
        if self.is_extremely_sick:
            if self.age < 80 or (self.denied_icu is None and self.rng.rand() < 0.5): # oxf study: 80+ 50% no ICU
                self.city.tracker.track_hospitalization(self, "icu")
                if self.age >= 80:
                    self.denied_icu = False
                yield self.env.process(self.excursion(self.city, "hospital-icu"))
            else:
                if self.denied_icu:
                    time_since_denial = (self.env.timestamp.date() - self.last_date["denied_icu"]).days
                    if time_since_denial >= self.denied_icu_days:
                        yield self.env.process(self.expire())
                else:
                    self.last_date["denied_icu"] = self.env.timestamp.date()
                    self.denied_icu = True
                    self.denied_icu_days = int(scipy.stats.gamma.rvs(1, loc=2.5))

        elif self.is_really_sick:
            self.city.tracker.track_hospitalization(self)
            yield self.env.process(self.excursion(city, "hospital"))


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
        if self.is_dead or\
                not isinstance(self.tracing_method, NonMLRiskComputer) or\
                self.tracing_method.risk_model == "transformer":
            return

        # All tracing methods that are _not ML_ (heuristic, bdt1, bdt2, etc) will compute new risks here
        risks = self.tracing_method.compute_risk(self, personal_mailbox, self.city.hd)
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
        assert self.tracing_method.risk_model == "transformer"
        assert len(risk_history) == self.contact_book.tracing_n_days_history, \
            "unexpected transformer history coverage; what's going on?"
        for day_offset_idx in range(len(risk_history)):  # note: idx:0 == today
            self.risk_history_map[current_day_idx - day_offset_idx] = risk_history[day_offset_idx]

    def wear_mask(self):
        """
        Determines whether this human wears a mask given their carefulness and how good at masks they are (mask_efficacy)
        """
        # if you don't wear a mask, then it is not effective
        if not self.WEAR_MASK:
            self.wearing_mask, self.mask_efficacy = False, 0
            return

        # people do not wear masks at home
        self.wearing_mask = True
        if self.location == self.household:
            self.wearing_mask = False

        # if they go to a store, they are more likely to wear a mask
        if self.location.location_type == 'store':
            if self.carefulness > 0.6:
                self.wearing_mask = True
            elif self.rng.rand() < self.carefulness * self.conf.get("BASELINE_P_MASK"):
                self.wearing_mask = True
        elif self.rng.rand() < self.carefulness * self.conf.get("BASELINE_P_MASK"):
            self.wearing_mask = True

        # efficacy - people do not wear it properly
        if self.wearing_mask:
            if self.workplace.location_type == 'hospital':
              self.mask_efficacy = self.conf.get("MASK_EFFICACY_HEALTHWORKER")
            else:
              self.mask_efficacy = self.conf.get("MASK_EFFICACY_NORMIE")
        else:
            self.mask_efficacy = 0

    def recover_health(self):
        """
        [summary]
        """
        if (self.cold_timestamp is not None and
            self.days_since_cold >= len(self.cold_progression)):
            self.cold_timestamp = None
            self.cold_symptoms = []

        if (self.flu_timestamp is not None and
            self.days_since_flu >= len(self.flu_progression)):
            self.flu_timestamp = None
            self.flu_symptoms = []

        if (self.allergy_timestamp is not None and
            self.days_since_allergies >= len(self.allergy_progression)):
            self.allergy_timestamp = None
            self.allergy_symptoms = []

    def catch_other_disease_at_random(self):
        # # assumption: no other infection if already infected with covid
        # if self.infection_timestamp is not None:
        #     return

        # Catch a random cold
        if self.cold_timestamp is None and self.rng.random() < self.conf["P_COLD_TODAY"]:
            self.cold_timestamp  = self.env.timestamp
            # print("caught cold")
            return

        # Catch a random flu (TODO: model seasonality through P_FLU_TODAY)
        if self.flu_timestamp is None and self.rng.random() < self.conf["P_FLU_TODAY"]:
            self.flu_timestamp = self.env.timestamp
            return

        # Have random allergy symptoms
        if self.has_allergies and self.rng.random() < self.conf["P_HAS_ALLERGIES_TODAY"]:
            self.allergy_timestamp = self.env.timestamp
            # print("caught allergy")
            return

    def how_am_I_feeling(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        current_symptoms = self.symptoms
        if current_symptoms == []:
            return 1.0

        if getattr(self, "_quarantine", None) and self.follows_recommendations_today:
            return 0.1

        if sum(x in current_symptoms for x in ["severe", "extremely_severe"]) > 0:
            return 0.2

        elif self.test_result == "positive":
            return 0.1

        elif sum(x in current_symptoms for x in ["trouble_breathing"]) > 0:
            return 0.3

        elif sum(x in current_symptoms for x in ["moderate", "fever"]) > 0:
            return 0.5

        elif sum(x in current_symptoms for x in ["cough", "fatigue", "gastro", "aches", "mild"]) > 0:
            return 0.6

        return 1.0

    def expire(self):
        """
        This function (generator) will cause the human to expire, after which self.is_dead==True.
        Yields self.env.timeout(np.inf), which when passed to env.procces will inactivate self
        for the remainder of the simulation.

        Yields:
            generator
        """
        self.recovered_timestamp = datetime.datetime.max
        self.all_symptoms, self.covid_symptoms = [], []
        Event.log_recovery(self.conf.get('COLLECT_LOGS'), self, self.env.timestamp, death=True)
        if self in self.location.humans:
            self.location.remove_human(self)
        self.tracker.track_deaths(self)
        yield self.env.timeout(np.inf)

    def assert_state_changes(self):
        """
        [summary]
        """
        next_state = {0:[1], 1:[2], 2:[0, 3], 3:[3]}
        assert sum(self.state) == 1, f"invalid compartment for {self.name}: {self.state}"
        if self.last_state != self.state:
            # can skip the compartment if hospitalized in exposed
            # can also skip the compartment if incubation days is very small
            if not self.obs_hospitalized:
                if not self.state.index(1) in next_state[self.last_state.index(1)]:
                    warnings.warn(f"invalid compartment transition for {self.name}: {self.last_state} to {self.state}"
                        f"incubation days:{self.incubation_days:3.3f} infectiousness onset days {self.infectiousness_onset_days}"
                        f"recovery days {self.recovery_days: 3.3f}", RuntimeWarning)

            self.last_state = self.state

    def notify(self, intervention=None):
        """
        This function is called once on the intervention day to notify `Human`.
        `self.notified` is a flag that is used debugging interventions.
        If the interevention is of type `Tracing`, everyone is initalized from 0 recommendation level
        and associated behavior modifications. Subsequent changes in behavior are dependent on recommendation level
        changes during the course of simulation.

        All other interventions modify behavior only once on intervention day.
        NOTE: DROPOUT_RATE might affect the behavior modifications.

        Args:
            intervention (BehaviorIntervention, optional): intervention that `Human` should follow. Defaults to None.
        """
        if (
            intervention is not None
            and not self.notified
        ):
            self.tracing = False
            if isinstance(intervention, NonMLRiskComputer):
                self.tracing = True
                self.tracing_method = intervention
            self.update_recommendations_level(intervention_start=True)
            recommendations = intervention.get_recommendations(self)
            self.apply_intervention(recommendations)
            self.notified = True

    def apply_intervention(self, new_recommendations: list):
        """ We provide a list of recommendations and a human, we revert the human's existing recommendations
            and apply the new ones"""
        for old_rec in self.recommendations_to_follow:
            old_rec.revert(self)
        self.recommendations_to_follow = OrderedSet()

        for new_rec in new_recommendations:
            assert isinstance(new_rec, Behavior)
            if self.follows_recommendations_today:
                new_rec.modify(self)
                self.recommendations_to_follow.add(new_rec)

    def run_mobility_reduction_check(self):
        # self.how_am_I_feeling = 1.0 (great) will make rest_at_home = False
        if not self.rest_at_home:
            i_feel = self.how_am_I_feeling()
            if self.rng.random() > i_feel:
                self.rest_at_home = True
        elif self.rest_at_home and self.how_am_I_feeling() == 1.0 and self.is_removed:
            self.rest_at_home = False


    def run_2(self, city):
        """
        """
        previous_activity, next_activity = None, self.mobility_planner.get_next_activity()
        while True:
            self.run_mobility_reduction_check()
            self.move_to_hospital_if_required()
            if next_activity.location is not None:
                # Note: use -O command line option to avoid checking for assertions
                # print("A\t", self.env.timestamp, self, next_activity)
                assert abs(self.env.timestamp - next_activity.start_time).seconds == 0, "start times do not align..."
                yield self.env.process(self.transition_to(next_activity, previous_activity))
                # print("B\t", self.env.timestamp, self, next_activity)
                assert abs(self.env.timestamp - next_activity.end_time).seconds == 0, " end times do not align..."
                previous_activity, next_activity = next_activity, self.mobility_planner.get_next_activity()
            else:
                next_activity.refresh_location()
                # print("C\t", self.env.timestamp, self, next_activity, "depends on", next_activity.parent_activity_pointer)
                # because this activity depends on parent_activity, we need to give a full 1 second to guarantee
                # that parent activity will have its location confirmed. This creates a discrepancy in env.timestamp and
                # next_activity.start_time.
                yield self.env.timeout(1)

                # realign the activities
                next_activity.start_in_seconds += 1
                next_activity.start_time += datetime.timedelta(seconds=1)
                next_activity.duration -= 1
                previous_activity.end_in_seconds += 1
                previous_activity.end_time += datetime.timedelta(seconds=1)
                previous_activity.duration += 1

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

    def excursion(self, city, location_type):
        """
        Redirects `self` to a relevant location corresponding to `location_type`.

        Args:
            city (covid19sim.locations.city): `City` object in which `self` resides
            location_type (str): type of location `self` will be moved to.

        Raises:
            ValueError: Can't redirect `self` to an unknown location type

        Yields:
            simpy.events.Process
        """
        if location_type == "grocery":
            grocery_store = self._select_location(activity="grocery", city=city)
            if grocery_store is None:
                # Either grocery stores are not open, or all queues are too long, so return
                return
            t = draw_random_discrete_gaussian(self.avg_shopping_time, self.scale_shopping_time, self.rng)
            with grocery_store.request() as request:
                yield request
                # If we make it here, it counts as a visit to the shop
                self.count_shop+=1
                yield self.env.process(self.at(grocery_store, city, t))

        elif location_type == "exercise":
            park = self._select_location(activity="exercise", city=city)
            if park is None:
                # No parks are open, so return
                return
            self.count_exercise+=1
            t = draw_random_discrete_gaussian(self.avg_exercise_time, self.scale_exercise_time, self.rng)
            yield self.env.process(self.at(park, city, t))

        elif location_type == "work":
            t = draw_random_discrete_gaussian(self.avg_working_minutes, self.scale_working_minutes, self.rng)
            if (not self.does_not_work
                and self.workplace.is_open_for_business):
                yield self.env.process(self.at(self.workplace, city, t))
            else:
                # work from home
                yield self.env.process(self.at(self.household, city, t))

        elif location_type == "hospital":
            hospital = self._select_location(location_type=location_type, city=city)
            if hospital is None: # no more hospitals
                # The patient dies
                yield self.env.process(self.expire())

            self.obs_hospitalized = True
            if self.infection_timestamp is not None:
                t = self.recovery_days - (self.env.timestamp - self.infection_timestamp).total_seconds() / 86400 # DAYS
                t = max(t * 24 * 60,0)
            else:
                t = len(self.symptoms)/10 * 60 # FIXME: better model
            yield self.env.process(self.at(hospital, city, t))

        elif location_type == "hospital-icu":
            icu = self._select_location(location_type=location_type, city=city)
            if icu is None:
                # The patient dies
                yield self.env.process(self.expire())

            if len(self.preexisting_conditions) < 2:
                extra_time = self.rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            else:
                extra_time = self.rng.choice([1, 2, 3], p=[0.2, 0.3, 0.5]) # DAYS
            t = self.viral_load_plateau_end - self.viral_load_plateau_start + extra_time

            yield self.env.process(self.at(icu, city, t * 24 * 60))

        elif location_type == "socialize":
            S = 0
            p_exp = 1.0
            while self.count_misc <= self.max_misc_per_week:
                if self.rng.random() > p_exp:  # return home
                    yield self.env.process(self.at(self.household, city, 60))
                    break

                loc = self._select_location(activity='socialize', city=city)
                if loc is None:
                    return # No leisure spots are open, or without long queues, so return

                S += 1
                p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
                with loc.request() as request:
                    yield request
                    self.count_misc += 1 # If we make it here, it counts as a leisure visit
                    t = draw_random_discrete_gaussian(self.avg_misc_time, self.scale_misc_time, self.rng)
                    yield self.env.process(self.at(loc, city, t))

        else:
            raise ValueError(f'Unknown excursion type:{location_type}')

    def track_me(self, new_location):
        row = {
            # basic info
            'time':self.env.timestamp,
            'hour':self.env.hour_of_day(),
            'day':self.env.day_of_week(),
            'state':self.state,
            'has_app':self.has_app,
            # change of location
            'location':str(self.location),
            'new_location':str(new_location)    ,
            # health
            'cold':self.has_cold,
            'flu':self.has_flu,
            'allergies':self.has_allergies,
            # app-related
            'rec_level': self.rec_level,
            'risk':self.risk,
            'risk_level':self.risk_level,
            # test
            'test_result':self.test_result,
            'hidden_test_results': self.hidden_test_result,
            'test_time': self.test_time,
        }
        self.my_history.append(row)

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
        if self.track_this_human:
            self.track_me(location)

        # add human to the location
        self.location = location
        location.add_human(self)
        self.location_start_time = self.env.now
        self.location_leaving_time = self.location_start_time + duration

        # do regular checks on whether to wear a mask
        # check if human needs a test if it's a hospital
        self.check_if_needs_covid_test(at_hospital=isinstance(location, (Hospital, ICU)))
        self.wear_mask()

        yield self.env.timeout(duration)
        # print("after", self.env.timestamp, self, location, duration)

        # only sample interactions if there is a possibility of infection or message exchanges
        # sleep is an inactive stage; phone is also assumed to be in background mode.
        if (duration > min(self.conf['MIN_MESSAGE_PASSING_DURATION'], self.conf['INFECTION_DURATION'])
            and "sleep" not in type_of_activity):
            # sample interactions with other humans at this location
            # unknown are the ones that self is not aware of e.g. person sitting next to self in a cafe
            known_interactions, unknown_interactions = location.sample_interactions(self)
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
                and t_near > self.conf.get("INFECTION_DURATION")
            )
            self.city.tracker.track_mixing(human1=self, human2=other_human, duration=t_near,
                            distance_profile=distance_profile, timestamp=self.env.timestamp, location=self.location,
                            interaction_type=type, contact_condition=contact_condition)

            # Conditions met for possible infection
            # https://www.cdc.gov/coronavirus/2019-ncov/hcp/guidance-risk-assesment-hcp.html
            if contact_condition:

                # used for matching "mobility" between methods
                scale_factor_passed = self.rng.random() < self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")
                cur_day = (self.env.timestamp - self.env.initial_timestamp).days
                if cur_day > self.conf.get("INTERVENTION_DAY"):
                    self.num_contacts += 1
                    self.effective_contacts += self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")

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

    def exposure_array(self, date):
        """
        [summary]

        Args:
            date ([type]): [description]

        Returns:
            [type]: [description]
        """
        warnings.warn("Deprecated in favor of frozen.helper.exposure_array()", DeprecationWarning)
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
        warnings.warn("Deprecated in favor of frozen.helper.recovered_array()", DeprecationWarning)
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

        t_near_in_minutes = duration / SECONDS_PER_MINUTE
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
                t_near_in_minutes > self.conf.get("MIN_MESSAGE_PASSING_DURATION") and \
                self.tracing and \
                self.has_app and \
                h.has_app:

            remaining_time_in_contact = t_near_in_minutes
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
                self.city.tracker.track_bluetooth_communications(human1=self, human2=h, timestamp=self.env.timestamp)

            Event.log_encounter_messages(
                self.conf['COLLECT_LOGS'],
                self,
                other_human,
                location=location,
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
        """
        [summary]

        Returns:
            [type]: [description]
        """
        if self.risk_history_map:
            cur_day = (self.env.timestamp - self.env.initial_timestamp).days
            if cur_day in self.risk_history_map:
                return self.risk_history_map[cur_day]
            else:
                last_day = max(self.risk_history_map.keys())
                return self.risk_history_map[last_day]
        else:
            return self.baseline_risk

    @property
    def risk_level(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return min(self.proba_to_risk_level_map(self.risk), 15)

    def update_recommendations_level(self, intervention_start=False):
        if not self.has_app or not isinstance(self.tracing_method, NonMLRiskComputer):
            self._rec_level = -1
        else:
            # FIXME: maybe merge Quarantine in RiskBasedRecommendationGetter with 2 levels
            if self.tracing_method.risk_model in ["manual", "digital"]:
                if self.risk == 1.0:
                    self._rec_level = 3
                else:
                    self._rec_level = 0
            else:
                self._rec_level = self.tracing_method.recommendation_getter.get_recommendations_level(
                    self,
                    self.conf.get("REC_LEVEL_THRESHOLDS"),
                    self.conf.get("MAX_RISK_LEVEL"),
                    intervention_start=intervention_start,
                )

    @property
    def rec_level(self):
        return self._rec_level

    def __repr__(self):
        return f"H:{self.name} age:{self.age}, SEIR:{int(self.is_susceptible)}{int(self.is_exposed)}{int(self.is_infectious)}{int(self.is_removed)}"
