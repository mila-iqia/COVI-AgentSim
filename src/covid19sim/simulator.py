"""
Contains the `Human` class that defines the behavior of human.
"""

import math
import datetime
import numpy as np
import typing
import warnings
from collections import defaultdict
from orderedset import OrderedSet

from covid19sim.interventions import Tracing
from covid19sim.utils import compute_distance, proba_to_risk_fn
from covid19sim.base import Event, PersonalMailboxType, Hospital, ICU
from collections import deque

from covid19sim.utils import _normalize_scores, _get_random_sex, _get_covid_progression, \
    _get_preexisting_conditions, _draw_random_discreet_gaussian, _sample_viral_load_piecewise, \
    _get_cold_progression, _get_flu_progression, _get_allergy_progression, _get_get_really_sick, \
    filter_open, filter_queue_max, calculate_average_infectiousness, _get_inflammatory_disease_level, _get_disease_days
from covid19sim.constants import SECONDS_PER_MINUTE, SECONDS_PER_HOUR
from covid19sim.frozen.message_utils import ContactBook, exchange_encounter_messages, RealUserIDType


class Visits(object):
    """
    [summary]
    """

    def __init__(self):
        """
        [summary]
        """
        self.parks = defaultdict(int)
        self.stores = defaultdict(int)
        self.hospitals = defaultdict(int)
        self.miscs = defaultdict(int)

    @property
    def n_parks(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return len(self.parks)

    @property
    def n_stores(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return len(self.stores)

    @property
    def n_hospitals(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return len(self.hospitals)

    @property
    def n_miscs(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return len(self.miscs)


class Human(object):
    """
    [summary]
    """

    def __init__(self, env, city, name, age, rng, has_app, infection_timestamp, household, workplace, profession, rho=0.3, gamma=0.21, symptoms=[], test_results=None, conf={}):
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
            symptoms (list, optional): [description]. Defaults to [].
            test_results ([type], optional): [description]. Defaults to None.
            conf (dict): yaml experiment configuration
        """
        self.conf = conf
        self.env = env
        self.city = city
        self._events = []
        self.name: RealUserIDType = f"human:{name}"
        self.rng = rng
        self.has_app = has_app
        self.profession = profession
        self.is_healthcare_worker = True if profession == "healthcare" else False
        self.assign_household(household)
        self.workplace = workplace
        self.rho = rho
        self.gamma = gamma

        self.age = age
        self.sex = _get_random_sex(self.rng)
        self.preexisting_conditions = _get_preexisting_conditions(self.age, self.sex, self.rng)
        self.inflammatory_disease_level = _get_inflammatory_disease_level(self.rng, self.preexisting_conditions, self.conf.get("INFLAMMATORY_CONDITIONS"))

        age_modifier = 2 if self.age > 40 or self.age < 12 else 2
        # &carefulness
        if self.rng.rand() < self.conf.get("P_CAREFUL_PERSON"):
            self.carefulness = (round(self.rng.normal(55, 10)) + self.age/2) / 100
        else:
            self.carefulness = (round(self.rng.normal(25, 10)) + self.age/2) / 100

        # allergies
        self.has_allergies = self.rng.rand() < self.conf.get("P_ALLERGIES")
        len_allergies = self.rng.normal(1/self.carefulness, 1)
        self.len_allergies = 7 if len_allergies > 7 else np.ceil(len_allergies)
        self.allergy_progression = _get_allergy_progression(self.rng)

        # logged info can be quite different
        self.has_logged_info = self.has_app and self.rng.rand() < self.carefulness
        self.obs_is_healthcare_worker = True if self.is_healthcare_worker and self.rng.random()<0.9 else False # 90% of the time, healthcare workers will declare it
        self.obs_age = self.age if self.has_app and self.has_logged_info else None
        self.obs_sex = self.sex if self.has_app and self.has_logged_info else None
        self.obs_preexisting_conditions = self.preexisting_conditions if self.has_app and self.has_logged_info else []

        self.rest_at_home = False # to track mobility due to symptoms
        self.visits = Visits()
        self.travelled_recently = self.rng.rand() > self.conf.get("P_TRAVELLED_INTERNATIONALLY_RECENTLY")

        # &symptoms, &viral-load
        # probability of being asymptomatic is basically 50%, but a bit less if you're older and a bit more if you're younger
        self.is_asymptomatic = self.rng.rand() < self.conf.get("BASELINE_P_ASYMPTOMATIC") - (self.age - 50) * 0.5 / 100 # e.g. 70: baseline-0.1, 20: baseline+0.15
        self.infection_ratio = None

        # Indicates whether this person will show severe signs of illness.
        self.cold_timestamp = None
        self.flu_timestamp = None
        self.allergy_timestamp = None
        self.can_get_really_sick = _get_get_really_sick(self.age, self.sex, self.rng)
        self.can_get_extremely_sick = self.can_get_really_sick and self.rng.random() >= 0.7 # &severe; 30% of severe cases need ICU
        self.never_recovers = self.rng.random() <= self.conf.get("P_NEVER_RECOVERS")[min(math.floor(self.age/10),8)]
        self.obs_hospitalized = False
        self.obs_in_icu = False

        # possibly initialized as infected
        self.recovered_timestamp = datetime.datetime.min
        self.is_immune = False # different from asymptomatic
        self.viral_load_plateau_height, self.viral_load_plateau_start, self.viral_load_plateau_end = None,None,None
        self.infectiousness_onset_days = None # 1 + self.rng.normal(loc=self.conf.get("INFECTIOUSNESS_ONSET_DAYS_AVG"), scale=self.conf.get("INFECTIOUSNESS_ONSET_DAYS_STD"))
        self.incubation_days = None # self.infectiousness_onset_days + self.viral_load_plateau_start + self.rng.normal(loc=self.conf.get("SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_AVG"), scale=self.conf.get("SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_STD")
        self.recovery_days = None # self.infectiousness_onset_days + self.viral_load_recovered
        self.test_time, self.test_type = None, None
        self.got_new_test_results = False
        self.test_results = deque(((None, datetime.datetime.max),))
        self.test_result_validated = None
        self.reported_test_result = None
        self.reported_test_type = None
        self._infection_timestamp = None
        self.infection_timestamp = infection_timestamp
        self.initial_viral_load = self.rng.rand() if infection_timestamp is not None else 0
        if self.infection_timestamp is not None:
            self.compute_covid_properties()


        # counters and memory
        self.r0 = []
        self.has_logged_symptoms = False
        self.last_state = self.state
        self.n_infectious_contacts = 0
        self.last_date = defaultdict(lambda : self.env.initial_timestamp.date())
        self.last_location = self.location
        self.last_duration = 0

        # interventions & risk prediction
        self.tracing = False
        self.WEAR_MASK =  False
        self.notified = False
        self.tracing_method = None
        self.maintain_extra_distance = 0
        self.how_much_I_follow_recommendations = self.conf.get('PERCENT_FOLLOW')
        self.recommendations_to_follow = OrderedSet()
        self.time_encounter_reduction_factor = 1.0
        self.hygiene = 0 # start everyone with a baseline hygiene. Only increase it once the intervention is introduced.
        self.test_recommended = False
        self.effective_contacts = 0
        self.num_contacts = 0

        # risk prediction
        self.contact_book = ContactBook(
            tracing_n_days_history=self.conf.get("TRACING_N_DAYS_HISTORY"),
        )
        self.infectiousness_history_map = dict()
        self.risk_history_map = dict()  # updated inside the human's (current) timeslot
        self.prev_risk_history_map = dict()  # used to check how the risk changed since the last timeslot
        self.history_map_last_message = dict()
        self.last_sent_update_gaen = 0
        self.last_message_risk_history_map = {}
        self.proba_to_risk_level_map = proba_to_risk_fn(np.array(self.conf.get('RISK_MAPPING')))

        # Message Passing and Risk Prediction
        self.clusters = None
        self.exposure_source = None
        # create 24 timeslots to do your updating
        time_slot = rng.randint(0, 24)
        self.time_slots = [
            int((time_slot + i * 24 / self.conf.get('UPDATES_PER_DAY')) % 24)
            for i in range(self.conf.get('UPDATES_PER_DAY'))
        ]

        # symptoms
        self.symptom_start_time = None
        self.cold_progression = _get_cold_progression(self.age, self.rng, self.carefulness, self.preexisting_conditions, self.can_get_really_sick, self.can_get_extremely_sick)
        self.flu_progression = _get_flu_progression(
            self.age, self.rng, self.carefulness, self.preexisting_conditions,
            self.can_get_really_sick, self.can_get_extremely_sick, self.conf.get("AVG_FLU_DURATION")
        )
        self.all_symptoms, self.cold_symptoms, self.flu_symptoms, self.covid_symptoms, self.allergy_symptoms = [], [], [], [], []
        # Padding the array
        self.rolling_all_symptoms = deque(
            [tuple()] * self.conf.get('TRACING_N_DAYS_HISTORY'),
            maxlen=self.conf.get('TRACING_N_DAYS_HISTORY')
        )
        self.rolling_all_reported_symptoms = deque(
            [tuple()] * self.conf.get('TRACING_N_DAYS_HISTORY'),
            maxlen=self.conf.get('TRACING_N_DAYS_HISTORY')
        )

        # habits
        self.avg_shopping_time = _draw_random_discreet_gaussian(
            self.conf.get("AVG_SHOP_TIME_MINUTES"),
            self.conf.get("SCALE_SHOP_TIME_MINUTES"),
            self.rng
        )
        self.scale_shopping_time = _draw_random_discreet_gaussian(
            self.conf.get("AVG_SCALE_SHOP_TIME_MINUTES"),
            self.conf.get("SCALE_SCALE_SHOP_TIME_MINUTES"),
            self.rng
        )

        self.avg_exercise_time = _draw_random_discreet_gaussian(
            self.conf.get("AVG_EXERCISE_MINUTES"),
            self.conf.get("SCALE_EXERCISE_MINUTES"),
            self.rng
        )
        self.scale_exercise_time = _draw_random_discreet_gaussian(
            self.conf.get("AVG_SCALE_EXERCISE_MINUTES"),
            self.conf.get("SCALE_SCALE_EXERCISE_MINUTES"),
            self.rng
        )

        self.avg_working_minutes = _draw_random_discreet_gaussian(
            self.conf.get("AVG_WORKING_MINUTES"),
            self.conf.get("SCALE_WORKING_MINUTES"),
            self.rng
        )
        self.scale_working_minutes = _draw_random_discreet_gaussian(
            self.conf.get("AVG_SCALE_WORKING_MINUTES"),
            self.conf.get("SCALE_SCALE_WORKING_MINUTES"),
            self.rng
        )

        self.avg_misc_time = _draw_random_discreet_gaussian(
            self.conf.get("AVG_MISC_MINUTES"),
            self.conf.get("SCALE_MISC_MINUTES"),
            self.rng
        )
        self.scale_misc_time = _draw_random_discreet_gaussian(
            self.conf.get("AVG_SCALE_MISC_MINUTES"),
            self.conf.get("SCALE_SCALE_MISC_MINUTES"),
            self.rng
        )

        #getting the number of shopping days and hours from a distribution
        self.number_of_shopping_days = _draw_random_discreet_gaussian(
            self.conf.get("AVG_NUM_SHOPPING_DAYS"),
            self.conf.get("SCALE_NUM_SHOPPING_DAYS"),
            self.rng
        )
        self.number_of_shopping_hours = _draw_random_discreet_gaussian(
            self.conf.get("AVG_NUM_SHOPPING_HOURS"),
            self.conf.get("SCALE_NUM_SHOPPING_HOURS"),
            self.rng
        )

        #getting the number of exercise days and hours from a distribution
        self.number_of_exercise_days = _draw_random_discreet_gaussian(
            self.conf.get("AVG_NUM_EXERCISE_DAYS"),
            self.conf.get("SCALE_NUM_EXERCISE_DAYS"),
            self.rng
        )
        self.number_of_exercise_hours = _draw_random_discreet_gaussian(
            self.conf.get("AVG_NUM_EXERCISE_HOURS"),
            self.conf.get("SCALE_NUM_EXERCISE_HOURS"),
            self.rng
        )

        #Multiple shopping days and hours
        self.shopping_days = self.rng.choice(range(7), self.number_of_shopping_days)
        self.shopping_hours = self.rng.choice(range(7, 20), self.number_of_shopping_hours)

        #Multiple exercise days and hours
        self.exercise_days = self.rng.choice(range(7), self.number_of_exercise_days)
        self.exercise_hours = self.rng.choice(range(7, 20), self.number_of_exercise_hours)

        #Limiting the number of hours spent shopping per week
        self.max_misc_per_week = _draw_random_discreet_gaussian(
            self.conf.get("AVG_MAX_NUM_MISC_PER_WEEK"),
            self.conf.get("SCALE_MAX_NUM_MISC_PER_WEEK"),
            self.rng
        )
        self.count_misc=0

        # Limiting the number of hours spent exercising per week
        self.max_exercise_per_week = _draw_random_discreet_gaussian(
            self.conf.get("AVG_MAX_NUM_EXERCISE_PER_WEEK"),
            self.conf.get("SCALE_MAX_NUM_EXERCISE_PER_WEEK"),
            self.rng
        )
        self.count_exercise=0

        #Limiting the number of hours spent shopping per week
        self.max_shop_per_week = _draw_random_discreet_gaussian(
            self.conf.get("AVG_MAX_NUM_SHOP_PER_WEEK"),
            self.conf.get("SCALE_MAX_NUM_SHOP_PER_WEEK"),
            self.rng
        )
        self.count_shop=0

        self.work_start_hour = self.rng.choice(range(7, 17), 3)


    def assign_household(self, location):
        """
        [summary]

        Args:
            location ([type]): [description]
        """
        self.household = location
        self.location = location
        if self.profession == "retired":
            self.workplace = location

    def __repr__(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return f"H:{self.name} age:{self.age}, SEIR:{int(self.is_susceptible)}{int(self.is_exposed)}{int(self.is_infectious)}{int(self.is_removed)}"

    ########### MEMORY OPTIMIZATION ###########
    @property
    def events(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self._events

    def events_slice(self, begin, end):
        """
        [summary]

        Args:
            begin ([type]): [description]
            end ([type]): [description]

        Returns:
            [type]: [description]
        """
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
        """
        [summary]

        Args:
            end ([type]): [description]

        Returns:
            [type]: [description]
        """
        end_i = len(self._events)
        for i, event in enumerate(self._events):
            if event['time'] >= end:
                end_i = i
                break

        events_slice, self._events = self._events[:end_i], self._events[end_i:]

        return events_slice

    ########### EPI ###########

    @property
    def tracing(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self._tracing

    @tracing.setter
    def tracing(self, value):
        """
        [summary]

        Args:
            value ([type]): [description]
        """
        self._tracing = value

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
        [summary]

        Returns:
            [type]: [description]
        """
        return not self.is_exposed and not self.is_infectious and not self.is_removed

    @property
    def is_exposed(self):
        """
        [summary]

        Returns:
            [type]: [description]
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
        [summary]

        Returns:
            [type]: [description]
        """
        return (not self.is_asymptomatic and self.infection_timestamp is not None and
                self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.incubation_days))

    @property
    def state(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return [int(self.is_susceptible), int(self.is_exposed), int(self.is_infectious), int(self.is_removed)]

    @property
    def has_cold(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self.cold_timestamp is not None

    @property
    def has_flu(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self.flu_timestamp is not None

    @property
    def has_allergy_symptoms(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self.allergy_timestamp is not None

    @property
    def days_since_covid(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        if self.infection_timestamp is None:
            return
        return (self.env.timestamp-self.infection_timestamp).days

    @property
    def days_since_cold(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        if self.cold_timestamp is None:
            return
        return (self.env.timestamp-self.cold_timestamp).days

    @property
    def days_since_flu(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        if self.flu_timestamp is None:
            return
        return (self.env.timestamp-self.flu_timestamp).days

    @property
    def days_since_allergies(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        if self.allergy_timestamp is None:
            return
        return (self.env.timestamp-self.allergy_timestamp).days

    @property
    def is_really_sick(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self.can_get_really_sick and 'severe' in self.symptoms

    @property
    def is_extremely_sick(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self.can_get_extremely_sick and 'severe' in self.symptoms

    def compute_covid_properties(self):
        """
        Pre-computes viral load curve.
        Specifically, characteristics of viral load plateau curve, i.e., height, start/end of plateau,
        start of infectiousness and when the symptom will show up.
        """
        # NOTE: all the days returned here are with respect to exposure day
        self.infectiousness_onset_days, self.viral_load_peak_start, \
            self.incubation_days, self.viral_load_plateau_start, \
                self.viral_load_plateau_end, self.recovery_days, \
                    self.viral_load_peak_height, self.viral_load_plateau_height = _get_disease_days(self.rng, self.conf, self.age, self.inflammatory_disease_level)

        # for ease of calculation, make viral load parameters relative to infectiousness onset
        self.viral_load_peak_start -= self.infectiousness_onset_days
        self.viral_load_plateau_start -= self.infectiousness_onset_days
        self.viral_load_plateau_end -= self.infectiousness_onset_days

        # precompute peak-plateau slope
        denominator = (self.viral_load_plateau_start - self.viral_load_peak_start)
        numerator =  self.viral_load_peak_height - self.viral_load_plateau_height
        self.peak_plateau_slope = numerator / denominator
        assert self.peak_plateau_slope >= 0 , f"viral load should decrease after peak. peak:{self.viral_load_peak_height} plateau height:{self.viral_load_plateau_height}"

        # percomupte plateau-end - recovery slope (should be negative because it is decreasing)
        numerator = self.viral_load_plateau_height
        denominator = self.recovery_days - self.viral_load_plateau_end
        self.plateau_end_recovery_slope = numerator / denominator
        assert self.plateau_end_recovery_slope >= 0, f"slopes are assumed to be positive for ease of calculation"

        self.covid_progression = []
        if not self.is_asymptomatic:
            self.covid_progression = _get_covid_progression(self.initial_viral_load, self.viral_load_plateau_start, self.viral_load_plateau_end,
                                            self.recovery_days, age=self.age, incubation_days=self.incubation_days,
                                            really_sick=self.can_get_really_sick, extremely_sick=self.can_get_extremely_sick,
                                            rng=self.rng, preexisting_conditions=self.preexisting_conditions, carefulness=self.carefulness)

        all_symptoms = set(symptom for symptoms_per_day in self.covid_progression for symptom in symptoms_per_day)
        # infection ratios
        if self.is_asymptomatic:
            self.infection_ratio = self.conf['ASYMPTOMATIC_INFECTION_RATIO']

        elif sum(x in all_symptoms for x in ['moderate', 'severe', 'extremely-severe']) > 0:
            self.infection_ratio = 1.0

        else:
            self.infection_ratio = self.conf['MILD_INFECTION_RATIO']

    def viral_load_for_day(self, timestamp):
        """ Calculates the elapsed time since infection, returning this person's current viral load"""

        if self.infection_timestamp is None:
            return 0.

        # calculates the time since infection in days
        days_infectious = (timestamp - self.infection_timestamp).total_seconds() / 86400 - self.infectiousness_onset_days

        if days_infectious < 0:
            return 0.

        # Rising to peak
        if days_infectious < self.viral_load_peak_start:
            cur_viral_load = self.viral_load_peak_height * days_infectious / (self.viral_load_peak_start)

        # Descending to plateau from peak
        elif days_infectious < self.viral_load_plateau_start:
            days_since_peak = days_infectious - self.viral_load_peak_start
            cur_viral_load = self.viral_load_peak_height - self.peak_plateau_slope * days_since_peak

        # plateau duration
        elif days_infectious < self.viral_load_plateau_end:
            cur_viral_load = self.viral_load_plateau_height

        # during recovery
        else:
            days_since_plateau_end = days_infectious - self.viral_load_plateau_end
            cur_viral_load = self.viral_load_plateau_height - self.plateau_end_recovery_slope * days_since_plateau_end

        assert 0 <= cur_viral_load <= 1, f"effective viral load out of bounds. viral load:{cur_viral_load}"

        return cur_viral_load

    @property
    def viral_load(self):
        """
        Calculates the elapsed time since infection, returning this person's current viral load

        Returns:
            [type]: [description]
        """
        return self.viral_load_for_day(self.env.timestamp)

    def get_infectiousness_for_day(self, timestamp, is_infectious):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        severity_multiplier = 1
        if is_infectious:
            if self.can_get_really_sick:
              severity_multiplier = 1
            if self.is_extremely_sick:
              severity_multiplier = 1
            if 'immuno-compromised' in self.preexisting_conditions:
              severity_multiplier += 0.2
            if 'cough' in self.symptoms:
              severity_multiplier += 0.25

        # max multiplier = 1 + 0.2 + 0.25 + 1 = 2.45
        # re-normalize [0-1]
        infectiousness = (self.viral_load_for_day(timestamp) * severity_multiplier)/2.45
        return infectiousness

    @property
    def infectiousness(self):
        return self.get_infectiousness_for_day(self.env.timestamp, self.is_infectious)

    @property
    def obs_symptoms(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        warnings.warn("Deprecated in favor of Human.all_reported_symptoms()", DeprecationWarning)
        return self.all_reported_symptoms

    @property
    def symptoms(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        # TODO: symptoms should not be updated here.
        #  Explicit call to Human.update_symptoms() should be required
        self.update_symptoms()
        return self.rolling_all_symptoms[0]

    @property
    def all_reported_symptoms(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        if not self.has_app:
            return []

        # TODO: symptoms should not be updated here.
        # Explicit call to Human.update_reported_symptoms() should be required
        self.update_reported_symptoms()
        return self.rolling_all_reported_symptoms[0]

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
        # Keep reported symptoms in sync
        # TODO: Inconsistency could come from Human.symptoms being accessed instead of Human.all_symptoms
        self.update_reported_symptoms()
        self.check_covid_testing_needs()
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

    @test_result.setter
    def test_result(self, val):
        if val is None:
            self.test_type = None
            self.test_time = None
            self.hidden_test_result = None
            self.time_to_test_result = None
            self.test_result_validated = None
            self.reported_test_result = None
            self.reported_test_type = None

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
        if isinstance(self.location, (Hospital, ICU)):
            self.time_to_test_result = self.conf['TEST_TYPES'][test_type]['time_to_result']['in-patient']
        else:
            self.time_to_test_result = self.conf['TEST_TYPES'][test_type]['time_to_result']['out-patient']

        # test reporting behavior
        if self.test_type == "lab":
            self.test_result_validated = True
        else:
            self.test_result_validated = False

        if self.has_app and self.rng.random() < self.carefulness:
            self.reported_test_result = self.test_result
            self.reported_test_type = self.test_type
        else:
            self.reported_test_result = None
            self.reported_test_type = None

        if self.conf.get('COLLECT_LOGS'):
            Event.log_test(self, self.test_time)

        self.test_results.appendleft((self.hidden_test_result, self.env.timestamp))
        self.city.tracker.track_tested_results(self)
        self.got_new_test_results = True

    def check_covid_testing_needs(self, at_hospital=False):
        """
        Checks whether self needs a test or not. Note: this only adds self to the test queue (not administer a test yet) of City.
        It is called every time symptoms are updated. It is also called from GetTested intervention.

        It depends upon the following factors -
            1. if `Human` is at a hospital, TEST_SYMPTOMS_FOR_HOSPITAL are checked for
            2. elsewhere there is a proability related to whether symptoms are "severe", "moderate", or "mild"
            3. if test_recommended is true (set by app recommendations)
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
            if not should_get_test and self.test_recommended:
                should_get_test = self.rng.random() < self.how_much_I_follow_recommendations

            if not should_get_test:
                # Has symptoms that a careful person would fear to be covid
                SUSPICIOUS_SYMPTOMS = set(self.conf['GET_TESTED_SYMPTOMS_CHECKED_BY_SELF'])
                if set(self.symptoms) and SUSPICIOUS_SYMPTOMS:
                    should_get_test = self.rng.rand() < self.carefulness

        if should_get_test:
            self.city.add_to_test_queue(self)

    def initialize_daily_risk(self, current_day_idx: int):
        """Initializes the risk history map with a new/copied risk value for the given day, if needed.

        Will also drop old unnecessary entries in the current & previous risk history maps.
        """
        if not self.risk_history_map:  # if we're on day zero, add a baseline risk value in
            self.risk_history_map[current_day_idx] = self.conf.get("BASELINE_RISK_VALUE")
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
        self.last_date["symptoms_updated"] = current_timestamp.date()
        self.update_symptoms()  # FIXME: duplicate call in 'update_reported_symptoms'?
        self.update_reported_symptoms()
        if self.got_new_test_results:
            if self.test_result == "positive":
                self.risk_history_map[current_day_idx] = 1.0
            elif self.test_result == "negative":
                self.risk_history_map[current_day_idx] = .2
        self.contact_book.cleanup_contacts(init_timestamp, current_timestamp)

        update_messages = []
        if self.tracing_method is not None:
            update_reason = "risk_update"  # default update reason (if we don't get more specific)

            if isinstance(self.tracing_method, Tracing) and self.tracing_method.risk_model != "transformer":
                # if not running transformer, we're using basic tracing --- do it now, it won't be batched later
                if not self.is_removed and not self.got_new_test_results:
                    # note: we check to make sure we don't have test results here not to overwrite the risk value
                    self.risk_history_map[current_day_idx] = \
                        self.tracing_method.compute_risk(self, personal_mailbox, self.city.hd)
                    update_reason = "tracing"

            if self.symptoms and self.tracing_method.propagate_symptoms:
                target_symptoms = ["severe", "trouble_breathing"]
                if any([s in target_symptoms for s in self.symptoms]) and not self.has_logged_symptoms:
                    assert self.tracing_method.risk_model != "transformer"
                    self.risk_history_map[current_day_idx] = max(0.8, self.risk)
                    self.has_logged_symptoms = True  # FIXME: this is never turned back off? but we can get reinfected?
                    update_reason = "symptoms"

            update_messages = [
                # if we had any encounters for which we have not sent an initial message, do it now
                *self.contact_book.generate_initial_updates(
                    current_day_idx=current_day_idx,
                    current_timestamp=current_timestamp,
                    risk_history_map=self.risk_history_map,
                    proba_to_risk_level_map=self.proba_to_risk_level_map,
                    tracing_method=self.tracing_method,
                ),
                # then, generate risk level update messages for all other encounters (if needed)
                *self.contact_book.generate_updates(
                    current_day_idx=current_day_idx,
                    current_timestamp=current_timestamp,
                    prev_risk_history_map=self.prev_risk_history_map,
                    curr_risk_history_map=self.risk_history_map,
                    proba_to_risk_level_map=self.proba_to_risk_level_map,
                    update_reason=update_reason,
                    tracing_method=self.tracing_method,
                ),
            ]
            if self.check_if_latest_risk_level_changed():
                self.tracing_method.modify_behavior(self)

        if current_day_idx not in self.infectiousness_history_map:
            # contrarily to risk, infectiousness only changes once a day (human behavior has no impact)
            self.infectiousness_history_map[current_day_idx] = calculate_average_infectiousness(self)
        # update the 'prev risk history' since we just generated the necessary updates
        self.prev_risk_history_map[current_day_idx] = self.risk_history_map[current_day_idx]
        self.got_new_test_results = False  # if we did have new results, we caught them above
        return update_messages

    def apply_transformer_risk_updates(
            self,
            current_day_idx: int,
            current_timestamp: datetime.datetime,
            risk_history: typing.List[float],
    ):
        """Applies a vector of risk values predicted by the transformer to this human, generating
        update messages for past encounters & modifying behavior if necessary.

        Args:
            current_day_idx: the current day index inside the simulation.
            current_timestamp: the current timestamp of the simulation.
            risk_history: the risk history vector predicted by the transformer.

        Returns:
            The update messages to send to contacts through the server.
        """
        assert self.tracing_method.risk_model == "transformer"
        assert len(risk_history) == self.contact_book.tracing_n_days_history, \
            "unexpected transformer history coverage; what's going on?"
        for day_offset_idx in range(len(risk_history)):  # note: idx:0 == today
            self.risk_history_map[current_day_idx - day_offset_idx] = risk_history[day_offset_idx]
        update_messages = self.contact_book.generate_updates(
            current_day_idx=current_day_idx,
            current_timestamp=current_timestamp,
            prev_risk_history_map=self.prev_risk_history_map,
            curr_risk_history_map=self.risk_history_map,
            proba_to_risk_level_map=self.proba_to_risk_level_map,
            update_reason="transformer",
            tracing_method=self.tracing_method,
        )
        if self.check_if_latest_risk_level_changed():
            self.tracing_method.modify_behavior(self)
        for i in range(len(risk_history)):
            self.prev_risk_history_map[current_day_idx - i] = risk_history[i]
        return update_messages

    def wear_mask(self, put_on=False):
        """
        [summary]

        Args:
            put_on (bool, optional): [description]. Defaults to False.
        """
        if not self.WEAR_MASK:
            self.wearing_mask, self.mask_efficacy = False, 0
            return

        self.wearing_mask = True
        if self.location == self.household:
            self.wearing_mask = False

        # if self.location.location_type == 'store':
        #     if self.carefulness > 0.6:
        #         self.wearing_mask = True
        #     elif self.rng.rand() < self.carefulness * self.conf.get("BASELINE_P_MASK"):
        #         self.wearing_mask = True
        # elif self.rng.rand() < self.carefulness * self.conf.get("BASELINE_P_MASK"):
        #     self.wearing_mask = True

        # efficacy - people do not wear it properly
        if self.wearing_mask:
            if  self.workplace.location_type == 'hospital':
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
        # assumption: no other infection if already infected with covid
        if self.infection_timestamp is not None:
            return

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

        if getattr(self, "_quarantine", None) and self.rng.random() < self.how_much_I_follow_recommendations:
            return 0.1

        if sum(x in current_symptoms for x in ["severe", "extremely_severe"]) > 0:
            return 0.0

        elif self.test_result == "positive":
            return 0.1

        elif sum(x in current_symptoms for x in ["trouble_breathing"]) > 0:
            return 0.3

        elif sum(x in current_symptoms for x in ["moderate", "fever"]) > 0:
            return 0.3

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
        self.risk = 0  # assume bodies are harmless, no more contacts can occur
        self.all_symptoms, self.covid_symptoms = [], []
        if self.conf.get('COLLECT_LOGS'):
            Event.log_recovery(self, self.env.timestamp, death=True)
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
        [summary]

        Args:
            intervention ([type], optional): [description]. Defaults to None.
            collect_training_data (bool, optional): [description]. Defaults to False.
        """
        # FIXME: PERCENT_FOLLOW < 1 will throw an error because ot self.notified somewhere
        if (
            intervention is not None
            and not self.notified
            and self.rng.random() < self.conf.get('PERCENT_FOLLOW')
        ):
            self.tracing = False
            if isinstance(intervention, Tracing):
                self.tracing = True
                self.tracing_method = intervention
                self.risk = 0  # FIXME: may affect contacts before next timeslot
                # initiate with basic recommendations
                # FIXME: Check isinstance of the RiskBasedRecommendations class
                if intervention.risk_model not in ['manual', 'digital']:
                    intervention.modify_behavior(self)
            else:
                intervention.modify_behavior(self)
            self.notified = True

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
            hour, day = self.env.hour_of_day(), self.env.day_of_week()
            if day==0:
                self.count_exercise=0
                self.count_shop=0

            # recover from cold/flu/allergies if it's time
            self.recover_health()

            # track symptoms
            if self.is_incubated and self.symptom_start_time is None:
                self.symptom_start_time = self.env.timestamp
                # it doesn't count environmental infection or primary case or asymptomatic/presymptomatic infections; refer the definition
                city.tracker.track_generation_times(self.name)

            # recover
            if self.is_infectious and self.days_since_covid >= self.recovery_days:
                city.tracker.track_recovery(self.n_infectious_contacts, self.recovery_days)

                # TO DISCUSS: Should the test result be reset here? We don't know in reality
                # when the person has recovered; currently not reset
                # self.test_result = None
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

                    if self.conf.get('COLLECT_LOGS'):
                        Event.log_recovery(self, self.env.timestamp, death=False)

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
            elif self.rest_at_home and self.how_am_I_feeling() == 1.0:
                self.rest_at_home = False
                # print(f"{self} C:{self.has_cold} A:{self.has_allergy_symptoms} going out because I recovered {self.symptoms}")


            # Behavioral imperatives
            if self.is_extremely_sick:
                city.tracker.track_hospitalization(self, "icu")
                yield self.env.process(self.excursion(city, "hospital-icu"))

            elif self.is_really_sick:
                city.tracker.track_hospitalization(self)
                yield self.env.process(self.excursion(city, "hospital"))

            # Work is a partial imperitive
            if (not self.profession=="retired" and
                not self.env.is_weekend() and
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

            elif ( self.env.is_weekend() and
                    self.rng.random() < 0.5 and
                    not self.rest_at_home and
                    not self.count_misc<=self.max_misc_per_week):
                yield  self.env.process(self.excursion(city, "leisure"))

            yield self.env.process(self.at(self.household, city, 60))

    ############################## MOBILITY ##################################
    @property
    def lat(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self.location.lat if self.location else self.household.lat

    @property
    def lon(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self.location.lon if self.location else self.household.lon

    @property
    def obs_lat(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        if self.conf.get("LOCATION_TECH") == 'bluetooth':
            return round(self.lat + self.rng.normal(0, 2))
        else:
            return round(self.lat + self.rng.normal(0, 10))

    @property
    def obs_lon(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
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
            t = _draw_random_discreet_gaussian(self.avg_shopping_time, self.scale_shopping_time, self.rng)
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
            t = _draw_random_discreet_gaussian(self.avg_exercise_time, self.scale_exercise_time, self.rng)
            yield self.env.process(self.at(park, city, t))

        elif location_type == "work":
            t = _draw_random_discreet_gaussian(self.avg_working_minutes, self.scale_working_minutes, self.rng)
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
                    t = _draw_random_discreet_gaussian(self.avg_misc_time, self.scale_misc_time, self.rng)
                    yield self.env.process(self.at(loc, city, t))
            self.count_misc+=leisure_count
        else:
            raise ValueError(f'Unknown excursion type:{location_type}')

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
            duration (float): time duration for which human stays at this location

        Yields:
            [type]: [description]
        """
        city.tracker.track_trip(from_location=self.location.location_type, to_location=location.location_type, age=self.age, hour=self.env.hour_of_day())

        # add the human to the location
        self.location = location
        location.add_human(self)
        self.wear_mask()

        self.start_time   = self.env.now
        self.leaving_time = self.start_time + duration*SECONDS_PER_MINUTE
        area = self.location.area
        initial_viral_load = 0

        self.check_covid_testing_needs(at_hospital=isinstance(location, (Hospital, ICU)))

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
        for h in location.humans:
            if h == self:
                continue

            # age mixing #FIXME: find a better way
            # at places other than the household, you mix with everyone
            if location != self.household and not self.rng.random() < (0.1 * abs(self.age - h.age) + 1) ** -1:
                continue

            # first term is packing metric for the location in cm
            packing_term = 100 * np.sqrt(area/len(self.location.humans)) # cms
            encounter_term = self.rng.uniform(self.conf.get("MIN_DIST_ENCOUNTER"), self.conf.get("MAX_DIST_ENCOUNTER"))
            social_distancing_term = np.mean([self.maintain_extra_distance, h.maintain_extra_distance]) * self.rng.rand()
            # if you're in a space, you cannot be more than packing term apart
            distance = np.clip(encounter_term + social_distancing_term, a_min=0, a_max=packing_term)
            # TODO: how often do we clip here and for which location type?
            # TODO: what is the distribution of distances here (and each type)?
            # if distance == packing_term:
                # print(f"packing:{packing_term:5.2f} encounter:{encounter_term} social_distancing:{social_distancing_term} distance clipped:{distance == packing_term} distance:{distance} {location}")

            if distance == packing_term:
                city.tracker.track_encounter_distance(
                    "A\t1", packing_term, encounter_term,
                    social_distancing_term, distance, location)
            else:
                city.tracker.track_encounter_distance(
                    "A\t0", packing_term, encounter_term,
                    social_distancing_term, distance, location)

            # risk model
            h1_msg, h2_msg = None, None
            if (
                self.conf.get("MIN_MESSAGE_PASSING_DISTANCE") < distance < self.conf.get("MAX_MESSAGE_PASSING_DISTANCE")
            ):
                if self.tracing and self.has_app and h.has_app:
                    h1_msg, h2_msg = exchange_encounter_messages(
                        h1=self,
                        h2=h,
                        env_timestamp=self.env.timestamp,
                        initial_timestamp=self.env.initial_timestamp,
                        # note: the granularity here does not really matter, it's only used to keep map sizes small
                        # in the clustering algorithm --- in reality, only the encounter day matters
                        minutes_granularity=self.conf.get("ENCOUNTER_TIME_GRANULARITY_MINS", 60 * 12),
                    )

            t_overlap = (min(self.leaving_time, getattr(h, "leaving_time", self.env.ts_initial+SECONDS_PER_HOUR)) -
                         max(self.start_time,   getattr(h, "start_time",   self.env.ts_initial+SECONDS_PER_HOUR))) / SECONDS_PER_MINUTE
            t_near = self.rng.random() * t_overlap * self.time_encounter_reduction_factor

            city.tracker.track_social_mixing(human1=self, human2=h, duration=t_near, timestamp = self.env.timestamp)
            contact_condition = (
                distance <= self.conf.get("INFECTION_RADIUS")
                and t_near > self.conf.get("INFECTION_DURATION")
            )

            # Conditions met for possible infection #https://www.cdc.gov/coronavirus/2019-ncov/hcp/guidance-risk-assesment-hcp.html
            if contact_condition:
                city.tracker.track_encounter_distance("B\t0", packing_term, encounter_term, social_distancing_term, distance, location=None)
                # TODO: whats the distribution of distances that get here (each distance type)?

                proximity_factor = 1
                if self.conf.get("INFECTION_DISTANCE_FACTOR") or self.conf.get("INFECTION_DURATION_FACTOR"):
                    # currently unused
                    proximity_factor = (
                        self.conf.get("INFECTION_DISTANCE_FACTOR") * (1 - distance / self.conf.get("INFECTION_RADIUS"))
                        + self.conf.get("INFECTION_DURATION_FACTOR") * min((t_near - self.conf.get("INFECTION_DURATION")) / self.conf.get("INFECTION_DURATION"), 1)
                    )

                # used for matching "mobility" between methods
                scale_factor_passed = self.rng.random() < self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")
                cur_day = (self.env.timestamp - self.env.initial_timestamp).days
                if cur_day > self.conf.get("INTERVENTION_DAY"):
                    self.num_contacts += 1
                    self.effective_contacts += self.conf.get("GLOBAL_MOBILITY_SCALING_FACTOR")

                infector, infectee = None, None
                if (self.is_infectious ^ h.is_infectious) and scale_factor_passed:
                    # TODO: whats the distribution of distances that get here (each distance type)?
                    if self.is_infectious:
                        infector, infectee = self, h
                        infectee_msg = h2_msg
                    else:
                        assert h.is_infectious
                        infector, infectee = h, self
                        infectee_msg = h1_msg

                    ratio = infector.infection_ratio
                    p_infection = infector.infectiousness * ratio * proximity_factor

                    # factors that can reduce probability of transmission (no-source)
                    mask_efficacy = (self.mask_efficacy + h.mask_efficacy)*self.conf['MASK_EFFICACY_FACTOR']
                    hygiene_efficacy = self.hygiene + h.hygiene
                    reduction_factor = self.conf["CONTAGION_KNOB"] +  mask_efficacy + hygiene_efficacy
                    p_infection *= np.exp(-reduction_factor * infector.n_infectious_contacts)

                    x_human = infector.rng.random() < p_infection
                    if x_human and infectee.is_susceptible:
                        infectee.infection_timestamp = self.env.timestamp
                        infectee.initial_viral_load = infector.rng.random()
                        infectee.compute_covid_properties()

                        infector.n_infectious_contacts += 1

                        if self.conf.get('COLLECT_LOGS'):
                            Event.log_exposed(infectee, infector, self.env.timestamp)

                        if infectee_msg is not None:  # could be None if we are not currently tracing
                            infectee_msg._exposition_event = True
                        city.tracker.track_infection('human', from_human=infector, to_human=infectee, location=location, timestamp=self.env.timestamp)
                        city.tracker.track_covid_properties(infectee)
                    else:
                        infector, infectee = None, None

                # cold transmission
                if self.has_cold ^ h.has_cold:
                    cold_infector, cold_infectee = h, self
                    if self.cold_timestamp is not None:
                        cold_infector, cold_infectee = self, h

                    # assumed no overloading of covid
                    if cold_infectee.infection_timestamp is None:
                        if self.rng.random() < self.conf.get("COLD_CONTAGIOUSNESS"):
                            cold_infectee.cold_timestamp = self.env.timestamp
                            # print("cold transmission occured")

                # flu tansmission
                if self.has_flu ^ h.has_flu:
                    flu_infector, flu_infectee = h, self
                    if self.flu_timestamp is not None:
                        flu_infector, flu_infectee = self, h

                    # assumed no overloading of covid
                    if flu_infectee.infection_timestamp is not None:
                        if self.rng.random() < self.conf.get("FLU_CONTAGIOUSNESS"):
                            flu_infectee.flu_timestamp = self.env.timestamp

                city.tracker.track_encounter_events(human1=self, human2=h, location=location, distance=distance, duration=t_near)

                if self.conf.get('COLLECT_LOGS'):
                    Event.log_encounter(
                        self,
                        h,
                        location=location,
                        duration=t_near,
                        distance=distance,
                        infectee=None if not infectee else infectee.name,
                        time=self.env.timestamp
                    )

        yield self.env.timeout(duration * SECONDS_PER_MINUTE)

        # environmental transmission
        p_infection = self.conf.get("ENVIRONMENTAL_INFECTION_KNOB") * location.contamination_probability * (1 - self.mask_efficacy) # &prob_infection
        # initial_viral_load += p_infection
        x_environment = location.contamination_probability > 0 and self.rng.random() < p_infection
        if x_environment and self.is_susceptible:
            self.infection_timestamp = self.env.timestamp
            self.initial_viral_load = self.rng.random()
            self.compute_covid_properties()

            if self.conf.get('COLLECT_LOGS'):
                Event.log_exposed(self, location, self.env.timestamp)

            city.tracker.track_infection('env', from_human=None, to_human=self, location=location, timestamp=self.env.timestamp)
            city.tracker.track_covid_properties(self)

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
            hospital = None
            for hospital in sorted(filter_open(city.hospitals), key=lambda x:compute_distance(self.location, x)):
                if len(hospital.humans) < hospital.capacity:
                    return hospital
            return None

        elif location_type == "hospital-icu":
            icu = None
            for hospital in sorted(filter_open(city.hospitals), key=lambda x:compute_distance(self.location, x)):
                if len(hospital.icu.humans) < hospital.icu.capacity:
                    return hospital.icu
            return None

        elif location_type == "miscs":
            S = self.visits.n_miscs
            self.adjust_gamma = 1.0
            pool_pref = [(compute_distance(self.location, m) + 1e-1) ** -1 for m in city.miscs if
                         m != self.location]
            pool_locs = [m for m in city.miscs if m != self.location]
            # Only consider locations open for business and not too long queues
            locs = filter_queue_max(filter_open(city.miscs))
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

    def symptoms_at_time(self, now, symptoms):
        """
        [summary]

        Args:
            now ([type]): [description]
            symptoms ([type]): [description]

        Returns:
            [type]: [description]
        """
        warnings.warn("Deprecated", DeprecationWarning)
        if not symptoms:
            return []
        if not self.symptom_start_time:
            return []
        sickness_day = (now - self.symptom_start_time).days
        if not sickness_day:
            return []
        if sickness_day > 14:
            rolling_all_symptoms_till_day = symptoms[sickness_day-14: sickness_day]
        else:
            rolling_all_symptoms_till_day = symptoms[:sickness_day]
        return rolling_all_symptoms_till_day

    def get_test_result_array(self, date):
        """
        [summary]

        Args:
            date ([type]): [description]

        Returns:
            [type]: [description]
        """
        warnings.warn("Deprecated in favor of frozen.helper.get_test_result_array()", DeprecationWarning)
        # dont change the logic in here, it needs to remain FROZEN
        results = np.zeros(14)
        result_day = (date - self.test_time).days
        if result_day >= 0 and result_day < 14:
            results[result_day] = 1
        return results

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
        # dont change the logic in here, it needs to remain FROZEN
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
    def risk(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        if not self.risk_history_map:
            return self.conf.get("BASELINE_RISK_VALUE")

        cur_day = (self.env.timestamp - self.env.initial_timestamp).days
        if self.is_removed:
            self.risk_history_map[cur_day] = 0.0
        elif self.test_result == "positive":
            self.risk_history_map[cur_day] = 1.0
        elif self.test_result == "negative":
            # TODO:  Risk because of negaitve results should not go down to 0.20. It should be max(o.2, current_risk). It should also depend on the test_type
            self.risk_history_map[cur_day] = 0.2

        if cur_day in self.risk_history_map:
            return self.risk_history_map[cur_day]
        else:
            last_day = max(self.risk_history_map.keys())
            return self.risk_history_map[last_day]

    @property
    def risk_level(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        _proba_to_risk_level = proba_to_risk_fn(np.array(self.conf.get('RISK_MAPPING')))
        return min(_proba_to_risk_level(self.risk), 15)

    @risk.setter
    def risk(self, val):
        """
        [summary]

        Args:
            val ([type]): [description]
        """
        cur_day = (self.env.timestamp - self.env.initial_timestamp).days
        self.risk_history_map[cur_day] = val

    @property
    def rec_level(self):
        if isinstance(self.tracing_method, Tracing):
            # FIXME: maybe merge Quarantine in RiskBasedRecommendations with 2 levels
            if self.tracing_method.risk_model in ["manual", "digital"]:
                if self.risk == 1.0:
                    return 3
                else:
                    return 0

            return self.tracing_method.intervention.get_recommendations_level(
                self.risk_level,
                self.conf.get("REC_LEVEL_THRESHOLDS"),
                self.conf.get("MAX_RISK_LEVEL")
            )
        return -1
