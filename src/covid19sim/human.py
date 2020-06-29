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
from covid19sim.epidemiology.p_infection import get_p_infection, infectiousness_delta
from covid19sim.utils.constants import SECONDS_PER_MINUTE, SECONDS_PER_HOUR, SECONDS_PER_DAY
from covid19sim.inference.message_utils import ContactBook, exchange_encounter_messages, RealUserIDType
from covid19sim.utils.visits import Visits
from covid19sim.native._native import BaseHuman

class Human(BaseHuman):
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

        super().__init__(env)

        # Utility References
        self.conf = conf  # Simulation-level Configurations
        self.env = env  # Simpy Environment (primarily used for timing / syncronization)
        self.city = city  # Manages lots of things inc. initializing humans and locations, tracking/updating humans

        # SEIR Tracking
        self.recovered_timestamp = datetime.datetime.min  # Time of recovery from covid -- min if not yet recovered
        self._infection_timestamp = None  # private time of infection with covid - implemented this way to ensure only infected 1 time
        self.infection_timestamp = infection_timestamp  # time of infection with covid
        self.n_infectious_contacts = 0  # number of high-risk contacts with an infected individual.
        self.exposure_source = None  # source that exposed this human to covid (and infected them). None if not infected.

        # Human-related properties
        self.name: RealUserIDType = f"human:{name}"  # Name of this human
        self.rng = np.random.RandomState(rng.randint(2 ** 16))  # RNG for this particular human
        self.profession = profession  # The job this human has (e.g. healthcare worker, retired, school, etc)
        self.is_healthcare_worker = True if profession == "healthcare" else False  # convenience boolean to check if is healthcare worker
        self._workplace = deque((workplace,))  # Created as a list because we sometimes modify human's workplace to WFH if in quarantine, then go back to work when released

        # Logging / Tracking
        self.track_this_human = False  # TODO: @PRATEEK plz comment this
        self.my_history = []  # TODO: @PRATEEK plz comment this
        self.r0 = []  # TODO: @PRATEEK plz comment this
        self._events = []  # TODO: @PRATEEK plz comment this


        """ Biological Properties """
        # Individual Characteristics
        self.sex = _get_random_sex(self.rng, self.conf)  # The sex of this person conforming with Canadian statistics
        self.age = age  # The age of this person, conforming with Canadian statistics
        self.age_bin = get_age_bin(age, conf)  # Age bins required for Oxford-like COVID-19 infection model and social mixing tracker
        self.normalized_susceptibility = self.conf['NORMALIZED_SUSCEPTIBILITY_BY_AGE'][self.age_bin]  # Susceptibility to Covid-19 by age
        self.mean_daily_interaction_age_group = self.conf['MEAN_DAILY_INTERACTION_FOR_AGE_GROUP'][self.age_bin]  # Social mixing is determined by age
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
        self.last_location = self.location  # tracks the last place this person was
        self.last_duration = 0  # tracks how long this person was somewhere

        # Habits
        self.avg_shopping_time = draw_random_discrete_gaussian(
            self.conf.get("AVG_SHOP_TIME_MINUTES"),
            self.conf.get("SCALE_SHOP_TIME_MINUTES"),
            self.rng
        )
        self.scale_shopping_time = draw_random_discrete_gaussian(
            self.conf.get("AVG_SCALE_SHOP_TIME_MINUTES"),
            self.conf.get("SCALE_SCALE_SHOP_TIME_MINUTES"),
            self.rng
        )

        self.avg_exercise_time = draw_random_discrete_gaussian(
            self.conf.get("AVG_EXERCISE_MINUTES"),
            self.conf.get("SCALE_EXERCISE_MINUTES"),
            self.rng
        )
        self.scale_exercise_time = draw_random_discrete_gaussian(
            self.conf.get("AVG_SCALE_EXERCISE_MINUTES"),
            self.conf.get("SCALE_SCALE_EXERCISE_MINUTES"),
            self.rng
        )

        self.avg_working_minutes = draw_random_discrete_gaussian(
            self.conf.get("AVG_WORKING_MINUTES"),
            self.conf.get("SCALE_WORKING_MINUTES"),
            self.rng
        )
        self.scale_working_minutes = draw_random_discrete_gaussian(
            self.conf.get("AVG_SCALE_WORKING_MINUTES"),
            self.conf.get("SCALE_SCALE_WORKING_MINUTES"),
            self.rng
        )

        self.avg_misc_time = draw_random_discrete_gaussian(
            self.conf.get("AVG_MISC_MINUTES"),
            self.conf.get("SCALE_MISC_MINUTES"),
            self.rng
        )
        self.scale_misc_time = draw_random_discrete_gaussian(
            self.conf.get("AVG_SCALE_MISC_MINUTES"),
            self.conf.get("SCALE_SCALE_MISC_MINUTES"),
            self.rng
        )

        #getting the number of shopping days and hours from a distribution
        self.number_of_shopping_days = draw_random_discrete_gaussian(
            self.conf.get("AVG_NUM_SHOPPING_DAYS"),
            self.conf.get("SCALE_NUM_SHOPPING_DAYS"),
            self.rng
        )
        self.number_of_shopping_hours = draw_random_discrete_gaussian(
            self.conf.get("AVG_NUM_SHOPPING_HOURS"),
            self.conf.get("SCALE_NUM_SHOPPING_HOURS"),
            self.rng
        )

        #getting the number of exercise days and hours from a distribution
        self.number_of_exercise_days = draw_random_discrete_gaussian(
            self.conf.get("AVG_NUM_EXERCISE_DAYS"),
            self.conf.get("SCALE_NUM_EXERCISE_DAYS"),
            self.rng
        )
        self.number_of_exercise_hours = draw_random_discrete_gaussian(
            self.conf.get("AVG_NUM_EXERCISE_HOURS"),
            self.conf.get("SCALE_NUM_EXERCISE_HOURS"),
            self.rng
        )

        #getting the number of misc hours from a distribution
        self.number_of_misc_hours = draw_random_discrete_gaussian(
            self.conf.get("AVG_NUM_MISC_HOURS", 5),
            self.conf.get("SCALE_NUM_MISC_HOURS", 1),
            self.rng
        )

        #Multiple shopping days and hours
        self.shopping_days = self.rng.choice(range(7), self.number_of_shopping_days)
        self.shopping_hours = self.rng.choice(range(7, 20), self.number_of_shopping_hours)

        #Multiple exercise days and hours
        self.exercise_days = self.rng.choice(range(7), self.number_of_exercise_days)
        self.exercise_hours = self.rng.choice(range(7, 20), self.number_of_exercise_hours)

        #Limiting the number of hours spent shopping per week
        self.max_misc_per_week = draw_random_discrete_gaussian(
            self.conf.get("AVG_MAX_NUM_MISC_PER_WEEK"),
            self.conf.get("SCALE_MAX_NUM_MISC_PER_WEEK"),
            self.rng
        )
        self.count_misc = 0

        # Limiting the number of hours spent exercising per week
        self.max_exercise_per_week = draw_random_discrete_gaussian(
            self.conf.get("AVG_MAX_NUM_EXERCISE_PER_WEEK"),
            self.conf.get("SCALE_MAX_NUM_EXERCISE_PER_WEEK"),
            self.rng
        )
        self.count_exercise = 0

        #Limiting the number of hours spent shopping per week
        self.max_shop_per_week = draw_random_discrete_gaussian(
            self.conf.get("AVG_MAX_NUM_SHOP_PER_WEEK"),
            self.conf.get("SCALE_MAX_NUM_SHOP_PER_WEEK"),
            self.rng
        )
        self.count_shop = 0

        # leisure hours on weekends
        self.misc_hours = self.rng.choice(range(7, 24), self.number_of_misc_hours)
        self.work_start_hour = self.rng.choice(range(7, 17), 3)
        self.location_leaving_time = self.env.ts_initial + SECONDS_PER_HOUR
        self.location_start_time = self.env.ts_initial


    @property
    def follows_recommendations_today(self):
        last_date = self.last_date["follow_recommendations"]
        current_date = self.env.timestamp.date()
        if last_date is None or (current_date - last_date).days > 0:
            proba = self.conf.get("DROPOUT_RATE")
            self.last_date["follow_recommendations"] = current_date
            self._follows_recommendations_today = self.rng.rand() < (1 - proba)
        return self._follows_recommendations_today

    def assign_household(self, location):
        """
        [summary]

        Args:
            location ([type]): [description]
        """
        self.household = location
        self.location = location
        if self.profession == "retired":
            self._workplace[-1] = location

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
    def state(self):
        """
        The state (SEIR) that this person is in (True if in state, False otherwise)

        Returns:
            bool: True for the state this person is in (Susceptible, Exposed, Infectious, Removed)
        """
        return [int(self.is_susceptible), int(self.is_exposed), int(self.is_infectious), int(self.is_removed)]

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
        if 'cough' in self.symptoms:
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
        current_date = self.env.timestamp.date()
        if self.last_date['reported_symptoms'] == current_date:
            return

        self.last_date['reported_symptoms'] = current_date

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

    def run(self, city):
        """
        [summary]

        Args:
            city (City): [description]

        Yields:
            [type]: [description]
        """
        self.household.humans.add(self)
        while True:
            hour, day = self.env.hour_of_day, self.env.day_of_week
            # BUG: Every time this loop is re-entered on Monday, the following
            #      code will execute.
            if day==0:
                self.count_exercise = 0
                self.count_shop = 0
                self.count_misc = 0

            # TODO - P - ideally check this every hour in base.py
            # used for tracking serial interval
            # person needs to show symptoms in order for this to be true.
            # is_incubated checks for asymptomaticity and whether the days since exposure is
            # greater than incubation_days.
            # Note: it doesn't count symptom start time from environmental infection or asymptomatic/presymptomatic infections
            # reference is in city.tracker.track_serial_interval.__doc__
            if self.is_incubated and self.covid_symptom_start_time is None and any(self.symptoms):
                self.covid_symptom_start_time = self.env.now
                city.tracker.track_serial_interval(self.name)

            # TODO - P - ideally check this every hour in base.py
            # recover
            if self.is_infectious and self.days_since_covid >= self.recovery_days:
                city.tracker.track_recovery(self.n_infectious_contacts, self.recovery_days)

                # TO DISCUSS: Should the test result be reset here? We don't know in reality
                # when the person has recovered; currently not reset
                # self.reset_test_result()
                self.ts_covid19_infection = float('inf')
                self.all_symptoms, self.covid_symptoms = [], []

                if self.never_recovers:
                    yield self.env.process(self.expire())
                else:
                    self.ts_covid19_recovery = self.env.now
                    self.is_immune = not self.conf.get("REINFECTION_POSSIBLE")

                    # "resample" the chance probability of never recovering again (experimental)
                    if not self.is_immune:
                        self.never_recovers = self.rng.random() < self.conf.get("P_NEVER_RECOVERS")[
                            min(math.floor(self.age / 10), 8)]

                    Event.log_recovery(self.conf.get('COLLECT_LOGS'), self, self.env.timestamp, death=False)

            self.assert_state_changes()

            # Mobility
            # self.how_am_I_feeling = 1.0 (great) --> rest_at_home = False
            if not self.rest_at_home:
                # set it once for the rest of the disease path
                i_feel = self.how_am_I_feeling()
                if self.rng.random() > i_feel:
                    self.rest_at_home = True
                    # print(f"{self} C:{self.has_cold} A:{self.has_allergy_symptoms} staying home because I feel {i_feel} {self.symptoms}")

            # happens when recovered
            elif self.rest_at_home and self.how_am_I_feeling() == 1.0 and self.is_removed:
                self.rest_at_home = False
                # print(f"{self} C:{self.has_cold} A:{self.has_allergy_symptoms} going out because I recovered {self.symptoms}")


            # Behavioral imperatives
            if self.is_extremely_sick:
                if self.age < 80 or (self.denied_icu is None and self.rng.rand() < 0.5): # oxf study: 80+ 50% no ICU
                    city.tracker.track_hospitalization(self, "icu")
                    if self.age >= 80:
                        self.denied_icu = False
                    yield self.env.process(self.excursion(city, "hospital-icu"))
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
                city.tracker.track_hospitalization(self)
                yield self.env.process(self.excursion(city, "hospital"))

            # Work is a partial imperitive
            if (not self.profession=="retired" and
                not self.env.is_weekend and
                hour in self.work_start_hour and
                not self.rest_at_home):
                yield self.env.process(self.excursion(city, "work"))

            # TODO (EM) These optional and erratic behaviours should be more probabalistic,
            # with probs depending on state of lockdown of city
            # Lockdown should also close a fraction of the shops

            elif ( hour in self.shopping_hours and
                   day in self.shopping_days and
                   self.count_shop<=self.max_shop_per_week and
                   not self.rest_at_home):
                yield self.env.process(self.excursion(city, "shopping"))

            elif ( hour in self.exercise_hours and
                    day in self.exercise_days and
                    self.count_exercise<=self.max_exercise_per_week and
                    not self.rest_at_home):
                yield  self.env.process(self.excursion(city, "exercise"))

            elif ( self.env.is_weekend and
                    self.rng.random() < 0.5 and
                    not self.rest_at_home and
                    hour in self.misc_hours and
                    self.count_misc < self.max_misc_per_week):
                yield  self.env.process(self.excursion(city, "leisure"))

            yield self.env.process(self.at(self.household, city, 60))

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
        [summary]

        Args:
            city ([type]): [description]
            type ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]

        Yields:
            [type]: [description]
        """
        if location_type == "shopping":
            grocery_store = self._select_location(location_type="stores", city=city)
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
            park = self._select_location(location_type="park", city=city)
            if park is None:
                # No parks are open, so return
                return
            self.count_exercise+=1
            t = draw_random_discrete_gaussian(self.avg_exercise_time, self.scale_exercise_time, self.rng)
            yield self.env.process(self.at(park, city, t))

        elif location_type == "work":
            t = draw_random_discrete_gaussian(self.avg_working_minutes, self.scale_working_minutes, self.rng)
            if self.workplace.is_open_for_business:
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
                t = self.recovery_days - (self.env.now - self.ts_covid19_infection) / SECONDS_PER_DAY # DAYS
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

        elif location_type == "leisure":
            S = 0
            p_exp = 1.0
            leisure_count = 0
            while True:
                if self.rng.random() > p_exp:  # return home
                    yield self.env.process(self.at(self.household, city, 60))
                    break

                loc = self._select_location(location_type='miscs', city=city)
                if loc is None:
                    # No leisure spots are open, or without long queues, so return
                    return
                S += 1
                p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
                with loc.request() as request:
                    yield request
                    leisure_count=1 # If we make it here, it counts as a leisure visit
                    t = draw_random_discrete_gaussian(self.avg_misc_time, self.scale_misc_time, self.rng)
                    yield self.env.process(self.at(loc, city, t))
            self.count_misc+=leisure_count
        else:
            raise ValueError(f'Unknown excursion type:{location_type}')

    def track_me(self, new_location):
        row = {
            # basic info
            'time':self.env.timestamp,
            'hour':self.env.hour_of_day,
            'day':self.env.day_of_week,
            'state':self.state,
            'has_app':self.has_app,
            # change of location
            'location':str(self.location),
            'new_location':str(new_location)    ,
            # health
            'cold':self.has_cold,
            'flu':self.has_flu,
            'allergies':"allergies" in self.preexisting_conditions,
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

    def at(self, location, city, duration):
        """
        Enter/Exit human to/from a `location` for some `duration`.
        Once human is at a location, encounters are sampled.
        During the stay, human is likely to be infected either by a contagion or
        through environmental transmission.
        Cold/Flu/Allergy onset also takes place in this function.

        Args:
            location (Location): next location to enter
            city (City): city in which human resides
            duration (float): time duration for which human stays at this location (minutes)

        Yields:
            [type]: [description]
        """
        city.tracker.track_trip(from_location=self.location.location_type, to_location=location.location_type, age=self.age, hour=self.env.hour_of_day)
        if self.track_this_human:
            self.track_me(location)

        # add the human to the location
        self.location = location
        location.add_human(self)
        self.wear_mask()

        self.location_start_time = self.env.now
        self.location_leaving_time = self.location_start_time + duration*SECONDS_PER_MINUTE
        area = self.location.area

        self.check_if_needs_covid_test(at_hospital=isinstance(location, (Hospital, ICU)))

        # accumulate time at household
        if location == self.household:
            if self.last_location != self.household:
                self.last_duration = duration
                self.last_location = location
            else:
                self.last_duration += duration
        else:
            if self.last_location == self.household:
                city.tracker.track_social_mixing(location=self.household, duration=self.last_duration)

            self.last_location = location
            city.tracker.track_social_mixing(location=location, duration=self.last_duration)

        # Report all the encounters (epi transmission)
        env_timestamp = self.env.timestamp
        for h in location.humans:
            if h == self:
                continue

            # age mixing #FIXME: find a better way
            # at places other than the household, you mix with everyone
            if location != self.household and not self.rng.random() < (0.1 * abs(self.age - h.age) + 1) ** -1:
                continue

            # first term is packing metric for the location in cm
            packing_term = 100 * math.sqrt(area/len(self.location.humans)) # cms
            encounter_term = self.rng.uniform(self.conf.get("MIN_DIST_ENCOUNTER"), self.conf.get("MAX_DIST_ENCOUNTER"))
            social_distancing_term = 0.5*(self.maintain_extra_distance + h.maintain_extra_distance) #* self.rng.rand()
            # if you're in a space, you cannot be more than packing term apart
            distance = min(max(encounter_term + social_distancing_term, 0), packing_term)

            if distance == packing_term:
                city.tracker.track_encounter_distance(
                    "A\t1", packing_term, encounter_term,
                    social_distancing_term, distance, location)
            else:
                city.tracker.track_encounter_distance(
                    "A\t0", packing_term, encounter_term,
                    social_distancing_term, distance, location)

            t_overlap = (min(self.location_leaving_time, h.location_leaving_time) -
                         max(self.location_start_time,   h.location_start_time)) / SECONDS_PER_MINUTE
            t_near = self.rng.random() * t_overlap * max(self.time_encounter_reduction_factor, h.time_encounter_reduction_factor)

            # phone_bluetooth_noise is a value selected between 0 and 2 meters to approximate the noise in the manufacturers bluetooth chip
            # distance is the "average" distance of the encounter
            # self.rng.random() - 0.5 gives a uniform random variable centered at 0
            # we scale by the distance s.t. if the true distance of the encounter is 2m you could think it is 0m or 4m,
            # whereas an encounter of 1m has a possible distance of 0.5 and 1.5m
            # a longer discussion is contained in docs/bluetooth.md
            approximated_bluetooth_distance = distance + distance * (self.rng.rand() - 0.5) * 0.5*(self.phone_bluetooth_noise + h.phone_bluetooth_noise)
            assert approximated_bluetooth_distance <= 2*distance

            h1_msg, h2_msg = None, None

            # The maximum distance of a message which we would consider to be "high risk" and therefore meriting an
            # encounter message is under 2 meters for at least 5 minutes.
            if approximated_bluetooth_distance < self.conf.get("MAX_MESSAGE_PASSING_DISTANCE") and \
                    t_near > self.conf.get("MIN_MESSAGE_PASSING_DURATION") and \
                    self.tracing and \
                    self.has_app and \
                    h.has_app:
                remaining_time_in_contact = t_near
                encounter_time_granularity = self.conf.get("MIN_MESSAGE_PASSING_DURATION")
                exchanged = False
                while remaining_time_in_contact > encounter_time_granularity:
                    exchanged = True
                    # note: every loop we will overwrite the messages but it doesn't matter since
                    # they're recorded in the contact books and we only need one for exposure flagging
                    h1_msg, h2_msg = exchange_encounter_messages(
                        h1=self,
                        h2=h,
                        # TODO: could adjust real timestamps in encounter messages based on remaining time?
                        # env_timestamp=self.env.timestamp - datetime.timedelta(minutes=remaining_time_in_contact),
                        # the above change might break clustering asserts if we somehow jump across timeslots/days
                        env_timestamp=env_timestamp,
                        initial_timestamp=self.env.initial_timestamp,
                        use_gaen_key=self.conf.get("USE_GAEN"),
                    )
                    remaining_time_in_contact -= encounter_time_granularity

                if exchanged:
                    city.tracker.track_bluetooth_communications(human1=self, human2=h, timestamp = env_timestamp)

                Event.log_encounter_messages(
                    self.conf['COLLECT_LOGS'],
                    self,
                    h,
                    location=location,
                    duration=t_near,
                    distance=distance,
                    time=env_timestamp
                )

            contact_condition = (
                distance <= self.conf.get("INFECTION_RADIUS")
                and t_near > self.conf.get("INFECTION_DURATION")
            )

            # Conditions met for possible infection
            # https://www.cdc.gov/coronavirus/2019-ncov/hcp/guidance-risk-assesment-hcp.html
            if contact_condition:
                city.tracker.track_social_mixing(human1=self, human2=h, duration=t_near, timestamp = env_timestamp)
                city.tracker.track_encounter_events(human1=self, human2=h, location=location, distance=distance, duration=t_near)
                city.tracker.track_encounter_distance("B\t0", packing_term, encounter_term, social_distancing_term, distance, location=None)

                # used for matching "mobility" between methods
                scale_factor_passed = self.rng.random() < self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")
                cur_day = int(self.env.now - self.env.ts_initial) // SECONDS_PER_DAY
                if cur_day > self.conf.get("INTERVENTION_DAY"):
                    self.num_contacts += 1
                    self.effective_contacts += self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")

                infector, infectee, p_infection = None, None, None
                if (self.is_infectious ^ h.is_infectious) and scale_factor_passed:
                    if self.is_infectious:
                        infector, infectee = self, h
                        infectee_msg = h2_msg
                    else:
                        assert h.is_infectious
                        infector, infectee = h, self
                        infectee_msg = h1_msg

                    p_infection = get_p_infection(infector,
                                                  infectiousness_delta(infector, t_near),
                                                  infectee,
                                                  location.social_contact_factor,
                                                  self.conf['CONTAGION_KNOB'],
                                                  self.conf['MASK_EFFICACY_FACTOR'],
                                                  self.conf['HYGIENE_EFFICACY_FACTOR'],
                                                  self,
                                                  h)

                    x_human = infector.rng.random() < p_infection
                    city.tracker.track_p_infection(x_human, p_infection, infector.viral_load)
                    if x_human and infectee.is_susceptible:
                        infectee.ts_covid19_infection = self.env.now
                        infectee.initial_viral_load = infector.rng.random()
                        compute_covid_properties(infectee)

                        infector.n_infectious_contacts += 1

                        Event.log_exposed(self.conf.get('COLLECT_LOGS'), infectee, infector, p_infection, env_timestamp)

                        if infectee_msg is not None:  # could be None if we are not currently tracing
                            infectee_msg._exposition_event = True
                        city.tracker.track_infection('human', from_human=infector, to_human=infectee, location=location, timestamp=env_timestamp)
                    else:
                        infector, infectee = None, None

                # cold transmission
                if self.has_cold ^ h.has_cold:
                    cold_infector, cold_infectee = h, self
                    if self.has_cold:
                        cold_infector, cold_infectee = self, h

                    # assumed no overloading of covid
                    if not cold_infectee.has_covid:
                        if self.rng.random() < self.conf.get("COLD_CONTAGIOUSNESS"):
                            cold_infectee.ts_cold_symptomatic = self.env.now
                            # print("cold transmission occured")

                # flu tansmission
                if self.has_flu ^ h.has_flu:
                    flu_infector, flu_infectee = h, self
                    if self.has_flu:
                        flu_infector, flu_infectee = self, h

                    # assumed no overloading of covid
                    if flu_infectee.has_covid: # BUG: Either this or cold is wrong
                        if self.rng.random() < self.conf.get("FLU_CONTAGIOUSNESS"):
                            flu_infectee.ts_flu_symptomatic = self.env.now

                Event.log_encounter(
                    self.conf['COLLECT_LOGS'],
                    self,
                    h,
                    location=location,
                    duration=t_near,
                    distance=distance,
                    infectee=None if not infectee else infectee.name,
                    p_infection=p_infection,
                    time=env_timestamp
                )

        assert env_timestamp == self.env.timestamp
        yield self.env.timeout(duration * SECONDS_PER_MINUTE)

        # environmental transmission
        p_infection = self.conf.get("ENVIRONMENTAL_INFECTION_KNOB") * location.contamination_probability * (1 - self.mask_efficacy)
        x_environment = location.contamination_probability > 0 and self.rng.random() < p_infection
        if x_environment and self.is_susceptible:
            self.ts_covid19_infection = self.env.now
            self.initial_viral_load = self.rng.random()
            compute_covid_properties(self)
            city.tracker.track_infection('env', from_human=None, to_human=self, location=location, timestamp=self.env.timestamp)
            Event.log_exposed(self.conf.get('COLLECT_LOGS'), self, location, p_infection, self.env.timestamp)

        location.remove_human(self)

    def _select_location(self, location_type, city):
        """
        Preferential exploration treatment to visit places
        rho, gamma are treated in the paper for normal trips
        Here gamma is multiplied by a factor to supress exploration for parks, stores.

        Args:
            location_type ([type]): [description]
            city ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if location_type == "park":
            S = self.visits.n_parks
            self.adjust_gamma = 1.0
            pool_pref = self.parks_preferences
            locs = filter_open(city.parks)
            visited_locs = self.visits.parks

        elif location_type == "stores":
            S = self.visits.n_stores
            self.adjust_gamma = 1.0
            pool_pref = self.stores_preferences
            # Only consider locations open for business and not too long queues
            locs = filter_queue_max(filter_open(city.stores), self.conf.get("MAX_STORE_QUEUE_LENGTH"))
            visited_locs = self.visits.stores

        elif location_type == "hospital":
            for hospital in sorted(filter_open(city.hospitals), key=lambda x:compute_distance(self.location, x)):
                if len(hospital.humans) < hospital.capacity:
                    return hospital
            return None

        elif location_type == "hospital-icu":
            for hospital in sorted(filter_open(city.hospitals), key=lambda x:compute_distance(self.location, x)):
                if len(hospital.icu.humans) < hospital.icu.capacity:
                    return hospital.icu
            return None

        elif location_type == "miscs":
            S = self.visits.n_miscs
            self.adjust_gamma = 1.0
            pool_pref = [(compute_distance(self.location, m) + 1e-1) ** -1 for m in city.miscs if
                         m != self.location]
            # Only consider locations open for business and not too long queues
            locs = filter_queue_max(filter_open(city.miscs), self.conf.get("MAX_MISC_QUEUE_LENGTH"))
            visited_locs = self.visits.miscs

        else:
            raise ValueError(f'Unknown location_type:{location_type}')

        if S == 0:
            p_exp = 1.0
        else:
            p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)

        if self.rng.random() < p_exp and S != len(locs):
            # explore
            cands = [i for i in locs if i not in visited_locs]
            cands = [(loc, pool_pref[i]) for i, loc in enumerate(cands)]
        else:
            # exploit, but can only return to locs that are open
            cands = [
                (i, count)
                for i, count in visited_locs.items()
                if i.is_open_for_business
                and len(i.queue)<=self.conf.get("MAX_STORE_QUEUE_LENGTH")
            ]

        if cands:
            cands, scores = zip(*cands)
            loc = self.rng.choice(cands, p=_normalize_scores(scores))
            visited_locs[loc] += 1
            return loc
        else:
            return None

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
