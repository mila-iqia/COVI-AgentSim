from collections import deque
import os

from covid19sim.frozen.clusters import Clusters
from covid19sim.frozen.utils import create_new_uid, Message, UpdateMessage, encode_message, encode_update_message
from covid19sim.utils import _normalize_scores, _get_random_sex, _get_covid_progression, \
     _get_preexisting_conditions, _draw_random_discreet_gaussian, _sample_viral_load_piecewise, \
     _get_cold_progression, _get_flu_progression, _get_allergy_progression, proba_to_risk_fn, _get_get_really_sick
from covid19sim.base import *
from covid19sim.config import LOG_RISK_MAPPING

if COLLECT_LOGS is False:
    Event = DummyEvent


_proba_to_risk_level = proba_to_risk_fn(np.exp(np.array(LOG_RISK_MAPPING)))


class Visits(object):

    def __init__(self):
        self.parks = defaultdict(int)
        self.stores = defaultdict(int)
        self.hospitals = defaultdict(int)
        self.miscs = defaultdict(int)

    @property
    def n_parks(self):
        return len(self.parks)

    @property
    def n_stores(self):
        return len(self.stores)

    @property
    def n_hospitals(self):
        return len(self.hospitals)

    @property
    def n_miscs(self):
        return len(self.miscs)


class Human(object):

    def __init__(self, env, city, name, age, rng, infection_timestamp, household, workplace, profession, rho=0.3, gamma=0.21, symptoms=[],
                 test_results=None):
        self.env = env
        self.city = city
        self._events = []
        self.name = f"human:{name}"
        self.rng = rng
        self.profession = profession
        self.is_healthcare_worker = True if profession == "healthcare" else False
        self.assign_household(household)
        self.workplace = workplace
        self.rho = rho
        self.gamma = gamma

        self.age = age
        self.sex = _get_random_sex(self.rng)
        self.dead = False
        self.preexisting_conditions = _get_preexisting_conditions(self.age, self.sex, self.rng)

        age_modifier = 2 if self.age > 40 or self.age < 12 else 2
        # &carefulness
        if self.rng.rand() < P_CAREFUL_PERSON:
            self.carefulness = (round(self.rng.normal(55, 10)) + self.age/2) / 100
        else:
            self.carefulness = (round(self.rng.normal(25, 10)) + self.age/2) / 100

        self.has_app = self.rng.rand() < (P_HAS_APP / age_modifier) + (self.carefulness / 2)

        # allergies
        self.has_allergies = self.rng.rand() < P_ALLERGIES
        len_allergies = self.rng.normal(1/self.carefulness, 1)
        self.len_allergies = 7 if len_allergies > 7 else np.ceil(len_allergies)
        self.allergy_progression = _get_allergy_progression(self.rng)

        # logged info can be quite different
        self.has_logged_info = self.has_app and self.rng.rand() < self.carefulness
        self.obs_is_healthcare_worker = True if self.is_healthcare_worker and rng.random()<0.9 else False # 90% of the time, healthcare workers will declare it
        self.obs_age = self.age if self.has_app and self.has_logged_info else None
        self.obs_sex = self.sex if self.has_app and self.has_logged_info else None
        self.obs_preexisting_conditions = self.preexisting_conditions if self.has_app and self.has_logged_info else []

        self.rest_at_home = False # to track mobility due to symptoms
        self.visits = Visits()
        self.travelled_recently = self.rng.rand() > P_TRAVELLED_INTERNATIONALLY_RECENTLY

        # &symptoms, &viral-load
        # probability of being asymptomatic is basically 50%, but a bit less if you're older and a bit more if you're younger
        self.is_asymptomatic = self.rng.rand() < BASELINE_P_ASYMPTOMATIC - (self.age - 50) * 0.5 / 100 # e.g. 70: baseline-0.1, 20: baseline+0.15
        self.asymptomatic_infection_ratio = ASYMPTOMATIC_INFECTION_RATIO if self.is_asymptomatic else 0.0 # draw a beta with the distribution in documents

        # Indicates whether this person will show severe signs of illness.
        self.cold_timestamp = self.env.timestamp if self.rng.random() < P_COLD else None
        self.flu_timestamp = self.env.timestamp if self.rng.random() < P_FLU else None # different from asymptomatic
        self.allergy_timestamp = self.env.timestamp if self.rng.random() < P_HAS_ALLERGIES_TODAY else None
        self.can_get_really_sick = _get_get_really_sick(self.age, self.sex, self.rng)
        self.can_get_extremely_sick = self.can_get_really_sick and self.rng.random() >= 0.7 # &severe; 30% of severe cases need ICU
        self.never_recovers = self.rng.random() <= P_NEVER_RECOVERS[min(math.floor(self.age/10),8)]
        self.obs_hospitalized = False
        self.obs_in_icu = False

        # possibly initialized as infected
        self.recovered_timestamp = datetime.datetime.min
        self.is_immune = False # different from asymptomatic
        self.viral_load_plateau_height, self.viral_load_plateau_start, self.viral_load_plateau_end, self.viral_load_recovered = None,None,None,None
        self.infectiousness_onset_days = None # 1 + self.rng.normal(loc=INFECTIOUSNESS_ONSET_DAYS_AVG, scale=INFECTIOUSNESS_ONSET_DAYS_STD)
        self.incubation_days = None # self.infectiousness_onset_days + self.viral_load_plateau_start + self.rng.normal(loc=SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_AVG, scale=SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_STD)
        self.recovery_days = None # self.infectiousness_onset_days + self.viral_load_recovered
        self.test_result, self.test_type = None, None
        self.infection_timestamp = infection_timestamp
        self.initial_viral_load = self.rng.rand() if infection_timestamp is not None else 0
        if self.infection_timestamp is not None:
            self.compute_covid_properties()
            print(f"{self} is infected")

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
        self.how_much_I_follow_recommendations = PERCENT_FOLLOW
        self.recommendations_to_follow = OrderedSet()
        self.time_encounter_reduction_factor = 1.0
        self.hygiene = self.carefulness
        self.test_recommended = False

        # risk prediction
        self.risk = BASELINE_RISK_VALUE
        self.start_risk = BASELINE_RISK_VALUE
        self.risk_level = _proba_to_risk_level(self.risk)
        self.rec_level = -1 # risk-based recommendations
        self.past_N_days_contacts = [OrderedSet()]
        self.n_contacts_tested_positive = defaultdict(int)
        self.contact_book = Contacts(self.has_app)
        self.message_info = { 'traced': False, \
                'receipt':datetime.datetime.max, \
                'delay':BIG_NUMBER, 'n_contacts_tested_positive': defaultdict(lambda :[0]),
                "n_contacts_symptoms":defaultdict(lambda :[0]), "n_contacts_risk_updates":defaultdict(lambda :[0]),
                "n_risk_decreased": defaultdict(lambda :[0]), "n_risk_increased":defaultdict(lambda :[0]),
                "n_risk_mag_increased":defaultdict(lambda :[0]), "n_risk_mag_decreased":defaultdict(lambda :[0])
                }
        self.risk_history = np.repeat(BASELINE_RISK_VALUE, 14)


        # Message Passing and Risk Prediction
        self.sent_messages = {}
        self.messages = []
        self.update_messages = []
        self.clusters = Clusters()
        self.tested_positive_contact_count = 0
        self.infectiousnesses = deque([0] * 14, maxlen=14)
        self.uid = create_new_uid(rng)
        self.exposure_message = None
        self.exposure_source = None
        self.test_time = datetime.datetime.max

        # symptoms
        self.symptom_start_time = None
        self.cold_progression = _get_cold_progression(self.age, self.rng, self.carefulness, self.preexisting_conditions, self.can_get_really_sick, self.can_get_extremely_sick)
        self.flu_progression = _get_flu_progression(self.age, self.rng, self.carefulness, self.preexisting_conditions, self.can_get_really_sick, self.can_get_extremely_sick)
        self.all_symptoms, self.cold_symptoms, self.flu_symptoms, self.covid_symptoms, self.allergy_symptoms = [], [], [], [], []

        # habits
        self.avg_shopping_time = _draw_random_discreet_gaussian(AVG_SHOP_TIME_MINUTES, SCALE_SHOP_TIME_MINUTES, self.rng)
        self.scale_shopping_time = _draw_random_discreet_gaussian(AVG_SCALE_SHOP_TIME_MINUTES,
                                                                  SCALE_SCALE_SHOP_TIME_MINUTES, self.rng)

        self.avg_exercise_time = _draw_random_discreet_gaussian(AVG_EXERCISE_MINUTES, SCALE_EXERCISE_MINUTES, self.rng)
        self.scale_exercise_time = _draw_random_discreet_gaussian(AVG_SCALE_EXERCISE_MINUTES,
                                                                  SCALE_SCALE_EXERCISE_MINUTES, self.rng)

        self.avg_working_minutes = _draw_random_discreet_gaussian(AVG_WORKING_MINUTES, SCALE_WORKING_MINUTES, self.rng)
        self.scale_working_minutes = _draw_random_discreet_gaussian(AVG_SCALE_WORKING_MINUTES, SCALE_SCALE_WORKING_MINUTES, self.rng)

        self.avg_misc_time = _draw_random_discreet_gaussian(AVG_MISC_MINUTES, SCALE_MISC_MINUTES, self.rng)
        self.scale_misc_time = _draw_random_discreet_gaussian(AVG_SCALE_MISC_MINUTES, SCALE_SCALE_MISC_MINUTES, self.rng)

        #getting the number of shopping days and hours from a distribution
        self.number_of_shopping_days = _draw_random_discreet_gaussian(AVG_NUM_SHOPPING_DAYS, SCALE_NUM_SHOPPING_DAYS, self.rng)
        self.number_of_shopping_hours = _draw_random_discreet_gaussian(AVG_NUM_SHOPPING_HOURS, SCALE_NUM_SHOPPING_HOURS, self.rng)

        #getting the number of exercise days and hours from a distribution
        self.number_of_exercise_days = _draw_random_discreet_gaussian(AVG_NUM_EXERCISE_DAYS, SCALE_NUM_EXERCISE_DAYS, self.rng)
        self.number_of_exercise_hours = _draw_random_discreet_gaussian(AVG_NUM_EXERCISE_HOURS, SCALE_NUM_EXERCISE_HOURS, self.rng)

        #Multiple shopping days and hours
        self.shopping_days = self.rng.choice(range(7), self.number_of_shopping_days)
        self.shopping_hours = self.rng.choice(range(7, 20), self.number_of_shopping_hours)

        #Multiple exercise days and hours
        self.exercise_days = self.rng.choice(range(7), self.number_of_exercise_days)
        self.exercise_hours = self.rng.choice(range(7, 20), self.number_of_exercise_hours)

        #Limiting the number of hours spent shopping per week
        self.max_misc_per_week = _draw_random_discreet_gaussian(AVG_MAX_NUM_MISC_PER_WEEK, SCALE_MAX_NUM_MISC_PER_WEEK, self.rng)
        self.count_misc=0

        # Limiting the number of hours spent exercising per week
        self.max_exercise_per_week = _draw_random_discreet_gaussian(AVG_MAX_NUM_EXERCISE_PER_WEEK, SCALE_MAX_NUM_EXERCISE_PER_WEEK, self.rng)
        self.count_exercise=0

        #Limiting the number of hours spent shopping per week
        self.max_shop_per_week = _draw_random_discreet_gaussian(AVG_MAX_NUM_SHOP_PER_WEEK, SCALE_MAX_NUM_SHOP_PER_WEEK, self.rng)
        self.count_shop=0

        self.work_start_hour = self.rng.choice(range(7, 17), 3)



    def assign_household(self, location):
        self.household = location
        self.location = location
        if self.profession == "retired":
            self.workplace = location

    def __repr__(self):
        return f"H:{self.name} age:{self.age}, SEIR:{int(self.is_susceptible)}{int(self.is_exposed)}{int(self.is_infectious)}{int(self.is_removed)}"

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
    def tracing(self):
        return self._tracing

    @tracing.setter
    def tracing(self, value):
        self._tracing = value

    @property
    def is_susceptible(self):
        return not self.is_exposed and not self.is_infectious and not self.is_removed and not self.is_immune

    @property
    def is_exposed(self):
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp < datetime.timedelta(days=self.infectiousness_onset_days)

    @property
    def is_infectious(self):
        return self.infection_timestamp is not None and self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.infectiousness_onset_days)

    @property
    def is_removed(self):
        return self.recovered_timestamp == datetime.datetime.max

    @property
    def is_incubated(self):
        return (not self.is_asymptomatic and self.infection_timestamp is not None and
                self.env.timestamp - self.infection_timestamp >= datetime.timedelta(days=self.incubation_days))

    @property
    def state(self):
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
        if self.infection_timestamp is None:
            return
        return (self.env.timestamp-self.infection_timestamp).days

    @property
    def days_since_cold(self):
        if self.cold_timestamp is None:
            return
        return (self.env.timestamp-self.cold_timestamp).days

    @property
    def days_since_flu(self):
        if self.flu_timestamp is None:
            return
        return (self.env.timestamp-self.flu_timestamp).days

    @property
    def days_since_allergies(self):
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
        """ Calculates the elapsed time since infection, returning this person's current viral load"""
        if not self.infection_timestamp:
            return 0.
        # calculates the time since infection in days
        days_infectious = (self.env.timestamp - self.infection_timestamp).total_seconds() / 86400 - self.infectiousness_onset_days

        if days_infectious < 0:
            return 0.

        # implements the piecewise linear function
        if days_infectious < self.viral_load_plateau_start:
            cur_viral_load = self.viral_load_plateau_height * days_infectious / self.viral_load_plateau_start
        elif days_infectious < self.viral_load_plateau_end:
            cur_viral_load = self.viral_load_plateau_height
        else:
            cur_viral_load = self.viral_load_plateau_height - self.viral_load_plateau_height * (days_infectious - self.viral_load_plateau_end) / (self.viral_load_recovered - self.viral_load_plateau_end)

        # the viral load cannot be negative
        if cur_viral_load < 0:
            cur_viral_load = 0.

        return cur_viral_load

    @property
    def infectiousness(self):
        severity_multiplier = 1
        if self.is_infectious:
            if self.can_get_really_sick:
              severity_multiplier = 1
            if self.is_extremely_sick:
              severity_multiplier = 1
            if 'immuno-compromised' in self.preexisting_conditions:
              severity_multiplier += 0.2
            if 'cough' in self.symptoms:
              severity_multiplier += 0.25
            severity_multiplier += (1-self.hygiene)
        return self.viral_load * severity_multiplier

    @property
    def obs_symptoms(self):
        if not self.has_app:
            return []
        reported_symptoms = []
        for symptom in self.all_symptoms:
            if self.rng.random() < self.carefulness:
                reported_symptoms.append(symptom)
        return reported_symptoms

    @property
    def symptoms(self):
        if self.last_date['symptoms'] != self.env.timestamp.date():
            self.last_date['symptoms'] = self.env.timestamp.date()
            self.update_symptoms()
        return self.all_symptoms

    @property
    def all_reported_symptoms(self):
        if not self.has_app:
            return []

        reported_symptoms = []
        for symptom in self.all_symptoms:
            if self.rng.random() < self.carefulness:
                reported_symptoms.append(symptom)
        return reported_symptoms

    def update_symptoms(self):
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
        self.all_symptoms = list(all_symptoms)

    def compute_covid_properties(self):
        self.viral_load_plateau_height, \
          self.viral_load_plateau_start, \
            self.viral_load_plateau_end, \
              self.viral_load_recovered = _sample_viral_load_piecewise(
                                             rng=self.rng, age=self.age,
                                             initial_viral_load=self.initial_viral_load)
        self.infectiousness_onset_days = 1 + self.rng.normal(loc=INFECTIOUSNESS_ONSET_DAYS_AVG, scale=INFECTIOUSNESS_ONSET_DAYS_STD)
        # FIXME : Make incubation_days as lognormal
        self.incubation_days = self.infectiousness_onset_days + self.viral_load_plateau_start + self.rng.normal(loc=SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_AVG, scale=SYMPTOM_ONSET_WRT_VIRAL_LOAD_PEAK_STD)
        self.recovery_days = self.infectiousness_onset_days + self.viral_load_recovered

        self.covid_progression = _get_covid_progression(self.initial_viral_load, self.viral_load_plateau_start, self.viral_load_plateau_end,
                                        self.viral_load_recovered, age=self.age, incubation_days=self.incubation_days,
                                        really_sick=self.can_get_really_sick, extremely_sick=self.can_get_extremely_sick,
                                        rng=self.rng, preexisting_conditions=self.preexisting_conditions, carefulness=self.carefulness)

    def get_tested(self, city, source="illness"):
        if not city.tests_available:
            return False

        # TODO: get observed data on testing / who gets tested when??
        if any(self.symptoms) and self.rng.rand() > P_TEST:
            self.test_type = city.get_available_test()
            if self.rng.rand() > TEST_TYPES[self.test_type]['P_FALSE_NEGATIVE']:
                self.test_result =  'negative'
            else:
                self.test_result =  'positive'

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

            return True

        return False

    def wear_mask(self, put_on=False):
        if not self.WEAR_MASK:
            self.wearing_mask, self.mask_efficacy = False, 0
            return

        self.wearing_mask = True
        if self.location == self.household:
            self.wearing_mask = False

        # if self.location.location_type == 'store':
        #     if self.carefulness > 0.6:
        #         self.wearing_mask = True
        #     elif self.rng.rand() < self.carefulness * BASELINE_P_MASK:
        #         self.wearing_mask = True
        # elif self.rng.rand() < self.carefulness * BASELINE_P_MASK :
        #     self.wearing_mask = True

        # efficacy - people do not wear it properly
        if self.wearing_mask:
            if  self.workplace.location_type == 'hospital':
              self.mask_efficacy = MASK_EFFICACY_HEALTHWORKER
            else:
              self.mask_efficacy = MASK_EFFICACY_NORMIE
        else:
            self.mask_efficacy = 0

    def recover_health(self):
        if (self.cold_timestamp is not None and
            self.days_since_cold >= len(self.cold_progression)):
            self.cold_timestamp = None
            self.cold_symptoms = []

        if (self.flu_timestamp is not None and
            self.days_since_flu >= len(self.flu_progression)):
            self.flu_timestamp = None
            self.flu_symptoms = []

        if (self.allergy_timestamp is not None and
            self.days_since_allergies >= self.len_allergies):
            self.allergy_timestamp = None
            self.allergy_symptoms = []

    def how_am_I_feeling(self):
        current_symptoms = self.symptoms
        if current_symptoms == []:
            return 1.0

        if getattr(self, "_quarantine", None) and self.rng.random() < self.how_much_I_follow_recommendations:
            return 0.10

        if sum(x in current_symptoms for x in ["severe", "extremely_severe"]) > 0:
            return 0.0

        elif self.test_result == "positive":
            return 0.1

        elif sum(x in current_symptoms for x in ["trouble_breathing"]) > 0:
            return 0.3 * (1 + self.carefulness)

        elif sum(x in current_symptoms for x in ["moderate", "mild", "fever"]) > 0:
            return 0.2

        elif sum(x in current_symptoms for x in ["cough", "fatigue", "gastro", "aches"]) > 0:
            return 0.2

        elif sum(x in current_symptoms for x in ["runny_nose", "loss_of_taste"]) > 0:
            return 0.3

        return 1.0

    def assert_state_changes(self):
        next_state = {0:[1], 1:[2], 2:[0, 3], 3:[3]}
        assert sum(self.state) == 1, f"invalid compartment for {self.name}: {self.state}"
        if self.last_state != self.state:
            # can skip the compartment if hospitalized in exposed
            if not self.obs_hospitalized:
                assert self.state.index(1) in next_state[self.last_state.index(1)], f"invalid compartment transition for {self.name}: {self.last_state} to {self.state}"
            self.last_state = self.state

    def notify(self, intervention=None, collect_training_data=False):
        if collect_training_data:
            self.tracing = True
            self.tracing_method = Tracing(risk_model="naive", max_depth=1, symptoms=False, risk=False, should_modify_behavior=False)
            return

        # FIXME: PERCENT_FOLLOW < 1 will throw an error because ot self.notified somewhere
        if intervention is not None and not self.notified and self.rng.random() < PERCENT_FOLLOW:
            self.tracing = False
            if isinstance(intervention, Tracing):
                self.tracing = True
                self.tracing_method = intervention
                self.rec_level = 0
                # initiate with basic recommendations
                # FIXME: Check isinstance of the RiskBasedRecommendations class
                if intervention.risk_model not in ['manual', 'digital']:
                    intervention.modify_behavior(self)
            else:
                intervention.modify_behavior(self)
            self.notified = True

    def run(self, city):
        """
           1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
           State  h h h h h h h h h sh sh h  h  h  ac h  h  h  h  h  h  h  h  h
        """
        self.household.humans.add(self)
        while True:
            hour, day = self.env.hour_of_day(), self.env.day_of_week()
            if day==0:
                self.count_exercise=0
                self.count_shop=0

            if self.last_date['run'] != self.env.timestamp.date():
                self.last_date['run'] = self.env.timestamp.date()
                self.update_symptoms()
                self.update_risk(symptoms=self.symptoms)
                self.infectiousnesses.appendleft(self.infectiousness)
                Event.log_daily(self, self.env.timestamp)
                city.tracker.track_symptoms(self)

                # keep only past N_DAYS contacts
                if self.tracing:
                    for type_contacts in ['n_contacts_tested_positive', 'n_contacts_symptoms', \
                    'n_risk_increased', 'n_risk_decreased', "n_risk_mag_decreased", "n_risk_mag_increased"]:
                        for order in self.message_info[type_contacts]:
                            if len(self.message_info[type_contacts][order]) > TRACING_N_DAYS_HISTORY:
                                self.message_info[type_contacts][order] = self.message_info[type_contacts][order][1:]
                            self.message_info[type_contacts][order].append(0)

                # if self.tracing and self.message_info['traced']:
                #     if (self.env.timestamp - self.message_info['receipt']).days >= self.message_info['delay']:
                #         # print(f"{self.tracing_method}: Traced {self}")
                #         self.update_risk(value=True)

            # recover from cold/flu/allergies if it's time
            self.recover_health()

            # track symptoms
            if self.is_incubated and self.symptom_start_time is None:
                self.symptom_start_time = self.env.timestamp
                city.tracker.track_generation_times(self.name) # it doesn't count environmental infection or primary case or asymptomatic/presymptomatic infections; refer the definition

            # log test
            # TODO: needs better conditions; log test based on some condition on symptoms
            if self.test_recommended or  \
                (self.is_incubated and
                self.test_result != "positive" and
                self.env.timestamp - self.symptom_start_time >= datetime.timedelta(days=TEST_DAYS)):
                # make testing a function of age/hospitalization/travel
                if self.get_tested(city):
                    Event.log_test(self, self.env.timestamp)
                    self.test_time = self.env.timestamp
                    city.tracker.track_tested_results(self, self.test_result, self.test_type)
                    self.update_risk(test_results=True)

            # recover
            if self.is_infectious and self.days_since_covid >= self.recovery_days:
                city.tracker.track_recovery(self.n_infectious_contacts, self.recovery_days)
                self.infection_timestamp = None # indicates they are no longer infected
                if self.never_recovers:
                    self.recovered_timestamp = datetime.datetime.max
                    self.dead = True
                else:
                    if not REINFECTION_POSSIBLE:
                        self.recovered_timestamp = datetime.datetime.max
                        self.is_immune = not REINFECTION_POSSIBLE
                    else:
                        self.recovered_timestamp = self.env.timestamp
                        self.test_result, self.test_type = None, None
                    self.never_recovers = self.rng.random() <= P_NEVER_RECOVERS[min(math.floor(self.age/10),8)]
                    self.dead = False

                self.update_risk(recovery=True)
                self.infection_timestamp = None # indicates they are no longer infected
                self.all_symptoms, self.covid_symptoms = [], []
                Event.log_recovery(self, self.env.timestamp, self.dead)
                if self.dead:
                    yield self.env.timeout(np.inf)

            self.assert_state_changes()

            # Mobility
            # self.how_am_I_feeling = 1.0 (great) --> rest_at_home = False
            if not self.rest_at_home:
                # set it once for the rest of the disease path
                if self.rng.random() > self.how_am_I_feeling():
                    self.rest_at_home = True

            # happens when recovered
            elif self.rest_at_home and self.how_am_I_feeling() == 1.0:
                self.rest_at_home = False

            # if self.name == "human:69":print(f"{self} rest_at_home: {self.rest_at_home} S:{len(self.symptoms)} flu:{self.has_flu} cold:{self.has_cold}")

            if self.is_extremely_sick:
                city.tracker.track_hospitalization(self, "icu")
                yield self.env.process(self.excursion(city, "hospital-icu"))

            elif self.is_really_sick:
                city.tracker.track_hospitalization(self)
                yield self.env.process(self.excursion(city, "hospital"))

            if (not self.env.is_weekend() and
                hour in self.work_start_hour and
                not self.rest_at_home):
                yield self.env.process(self.excursion(city, "work"))

            elif ( hour in self.shopping_hours and
                   day in self.shopping_days and
                   self.count_shop<=self.max_shop_per_week and
                   not self.rest_at_home):
                self.count_shop+=1
                yield self.env.process(self.excursion(city, "shopping"))


            elif ( hour in self.exercise_hours and
                    day in self.exercise_days and
                    self.count_exercise<=self.max_exercise_per_week and
                    not self.rest_at_home):
                self.count_exercise+=1
                yield  self.env.process(self.excursion(city, "exercise"))

            elif (self.env.is_weekend() and
                    self.rng.random() < 0.5 and
                    not self.rest_at_home and
                    not self.count_misc<=self.max_misc_per_week):
                self.count_misc+=1
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
        if LOCATION_TECH == 'bluetooth':
            return round(self.lat + self.rng.normal(0, 2))
        else:
            return round(self.lat + self.rng.normal(0, 10))

    @property
    def obs_lon(self):
        if LOCATION_TECH == 'bluetooth':
            return round(self.lon + self.rng.normal(0, 2))
        else:
            return round(self.lon + self.rng.normal(0, 10))

    def excursion(self, city, type):

        if type == "shopping":
            grocery_store = self._select_location(location_type="stores", city=city)
            t = _draw_random_discreet_gaussian(self.avg_shopping_time, self.scale_shopping_time, self.rng)
            with grocery_store.request() as request:
                yield request
                yield self.env.process(self.at(grocery_store, city, t))

        elif type == "exercise":
            park = self._select_location(location_type="park", city=city)
            t = _draw_random_discreet_gaussian(self.avg_exercise_time, self.scale_exercise_time, self.rng)
            yield self.env.process(self.at(park, city, t))

        elif type == "work":
            t = _draw_random_discreet_gaussian(self.avg_working_minutes, self.scale_working_minutes, self.rng)
            yield self.env.process(self.at(self.workplace, city, t))

        elif type == "hospital":
            # print(f"{self} got hospitalized")
            hospital = self._select_location(location_type=type, city=city)
            if hospital is None: # no more hospitals
                self.dead = True
                self.recovered_timestamp = datetime.datetime.max
                yield self.env.timeout(np.inf)

            self.obs_hospitalized = True
            if self.infection_timestamp is not None:
                t = self.recovery_days - (self.env.timestamp - self.infection_timestamp).total_seconds() / 86400 # DAYS
                t = max(t * 24 * 60,0)
            else:
                t = len(self.symptoms)/10 * 60 # FIXME: better model
            yield self.env.process(self.at(hospital, city, t))

        elif type == "hospital-icu":
            # print(f"{self} got icu-ed")
            icu = self._select_location(location_type=type, city=city)
            if icu is None:
                self.dead = True
                self.recovered_timestamp = datetime.datetime.max
                yield self.env.timeout(np.inf)

            if len(self.preexisting_conditions) < 2:
                extra_time = self.rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            else:
                extra_time = self.rng.choice([1, 2, 3], p=[0.2, 0.3, 0.5]) # DAYS
            t = self.viral_load_plateau_end - self.viral_load_plateau_start + extra_time

            yield self.env.process(self.at(icu, city, t * 24 * 60))

        elif type == "leisure":
            S = 0
            p_exp = 1.0
            while True:
                if self.rng.random() > p_exp:  # return home
                    yield self.env.process(self.at(self.household, city, 60))
                    break

                loc = self._select_location(location_type='miscs', city=city)
                S += 1
                p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
                with loc.request() as request:
                    yield request
                    t = _draw_random_discreet_gaussian(self.avg_misc_time, self.scale_misc_time, self.rng)
                    yield self.env.process(self.at(loc, city, t))
        else:
            raise ValueError(f'Unknown excursion type:{type}')

    def at(self, location, city, duration):
        city.tracker.track_trip(from_location=self.location.location_type, to_location=location.location_type, age=self.age, hour=self.env.hour_of_day())

        # add the human to the location
        self.location = location
        location.add_human(self)
        self.wear_mask()

        self.leaving_time = duration + self.env.now
        self.start_time = self.env.now
        area = self.location.area
        initial_viral_load = 0

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
            # places other than the household, you mix with everyone
            if location != self.household and not self.rng.random() < (0.1 * abs(self.age - h.age) + 1) ** -1:
                continue

            distance =  np.sqrt(int(area/len(self.location.humans))) + \
                            self.rng.randint(MIN_DIST_ENCOUNTER, MAX_DIST_ENCOUNTER) + \
                            self.maintain_extra_distance
            # risk model
            # TODO: Add GPS measurements as conditions; refer JF's docs
            if MIN_MESSAGE_PASSING_DISTANCE < distance <  MAX_MESSAGE_PASSING_DISTANCE:
                if self.tracing:
                    self.contact_book.add(human=h, timestamp=self.env.timestamp, self_human=self)
                    h.contact_book.add(human=self, timestamp=self.env.timestamp, self_human=h)
                    cur_day = (self.env.timestamp - self.env.initial_timestamp).days
                    if self.has_app and h.has_app and (cur_day >= INTERVENTION_DAY):
                        self.contact_book.messages.append(h.cur_message(cur_day))
                        h.contact_book.messages.append(self.cur_message(cur_day))
                        self.contact_book.messages_by_day[cur_day].append(h.cur_message(cur_day))
                        h.contact_book.messages_by_day[cur_day].append(self.cur_message(cur_day))

                        h.contact_book.sent_messages_by_day[cur_day].append(h.cur_message(cur_day))
                        self.contact_book.sent_messages_by_day[cur_day].append(self.cur_message(cur_day))


                # FIXME: ideally encounter should be here. this will generate a lot of encounters

            t_overlap = min(self.leaving_time, getattr(h, "leaving_time", 60)) - max(self.start_time, getattr(h, "start_time", 60))
            t_near = self.rng.random() * t_overlap * self.time_encounter_reduction_factor

            city.tracker.track_social_mixing(human1=self, human2=h, duration=t_near, timestamp = self.env.timestamp)
            contact_condition = distance <= INFECTION_RADIUS and t_near > INFECTION_DURATION

            # Conditions met for possible infection
            if contact_condition:
                proximity_factor = 1
                if INFECTION_DISTANCE_FACTOR or INFECTION_DURATION_FACTOR:
                    proximity_factor = INFECTION_DISTANCE_FACTOR * (1 - distance/INFECTION_RADIUS) + INFECTION_DURATION_FACTOR * min((t_near - INFECTION_DURATION)/INFECTION_DURATION, 1)
                mask_efficacy = (self.mask_efficacy + h.mask_efficacy)*2

                # TODO: merge the two clauses into one (follow cold and flu)
                infectee = None
                if self.is_infectious:
                    ratio = self.asymptomatic_infection_ratio  if self.is_asymptomatic else 1.0
                    p_infection = self.infectiousness * ratio * proximity_factor
                    # FIXME: remove hygiene from severity multiplier; init hygiene = 0; use sum here instead
                    reduction_factor = CONTAGION_KNOB + sum(getattr(x, "_hygiene", 0) for x in [self, h]) + mask_efficacy
                    p_infection *= np.exp(-reduction_factor * self.n_infectious_contacts)

                    x_human = self.rng.random() < p_infection

                    if x_human and h.is_susceptible:
                        h.infection_timestamp = self.env.timestamp
                        h.initial_viral_load = h.rng.random()
                        h.compute_covid_properties()
                        infectee = h.name

                        self.n_infectious_contacts+=1
                        Event.log_exposed(h, self, self.env.timestamp)
                        h.exposure_message = encode_message(self.cur_message((self.env.timestamp - self.env.initial_timestamp).days))
                        city.tracker.track_infection('human', from_human=self, to_human=h, location=location, timestamp=self.env.timestamp)
                        city.tracker.track_covid_properties(h)
                        # print(f"{self.name} infected {h.name} at {location}")

                elif h.is_infectious:
                    ratio = h.asymptomatic_infection_ratio  if h.is_asymptomatic else 1.0
                    p_infection = h.infectiousness * ratio * proximity_factor # &prob_infectious
                    # FIXME: remove hygiene from severity multiplier; init hygiene = 0; use sum here instead
                    reduction_factor = CONTAGION_KNOB + sum(getattr(x, "_hygiene", 0) for x in [self, h]) + mask_efficacy
                    p_infection *= np.exp(-reduction_factor * h.n_infectious_contacts) # hack to control R0
                    x_human = self.rng.random() < p_infection

                    if x_human and self.is_susceptible:
                        self.infection_timestamp = self.env.timestamp
                        self.initial_viral_load = self.rng.random()
                        self.compute_covid_properties()
                        infectee = self.name

                        h.n_infectious_contacts+=1
                        Event.log_exposed(self, h, self.env.timestamp)
                        city.tracker.track_infection('human', from_human=h, to_human=self, location=location, timestamp=self.env.timestamp)
                        city.tracker.track_covid_properties(self)
                        # print(f"{h.name} infected {self.name} at {location}")

                # other transmissions
                if self.cold_timestamp is not None or h.cold_timestamp is not None:
                    cold_infector, cold_infectee = h, self
                    if self.cold_timestamp is not None:
                        cold_infector, cold_infectee = self, h

                    if self.rng.random() < COLD_CONTAGIOUSNESS:
                        cold_infectee.cold_timestamp = self.env.timestamp

                if self.flu_timestamp is not None or h.flu_timestamp is not None:
                    flu_infector, flu_infectee = h, self
                    if self.cold_timestamp is not None:
                        flu_infector, flu_infectee = self, h

                    if self.rng.random() < FLU_CONTAGIOUSNESS:
                        flu_infectee.flu_timestamp = self.env.timestamp

                city.tracker.track_encounter_events(human1=self, human2=h, location=location, distance=distance, duration=t_near)
                Event.log_encounter(self, h,
                                    location=location,
                                    duration=t_near,
                                    distance=distance,
                                    infectee=infectee,
                                    time=self.env.timestamp
                                    )

        yield self.env.timeout(duration / TICK_MINUTE)

        # environmental transmission
        p_infection = ENVIRONMENTAL_INFECTION_KNOB * location.contamination_probability * (1-self.mask_efficacy) # &prob_infection
        # initial_viral_load += p_infection
        x_environment = location.contamination_probability > 0 and self.rng.random() < p_infection
        if x_environment and self.is_susceptible:
            self.infection_timestamp = self.env.timestamp
            self.initial_viral_load = self.rng.random()
            self.compute_covid_properties()
            Event.log_exposed(self, location,  self.env.timestamp)
            city.tracker.track_infection('env', from_human=None, to_human=self, location=location, timestamp=self.env.timestamp)
            city.tracker.track_covid_properties(self)
            # print(f"{self.name} is infected at {location}")

        # Catch a random cold
        if self.cold_timestamp is None and self.rng.random() < P_COLD:
            self.cold_timestamp  = self.env.timestamp

        # Catch a random flu
        if self.flu_timestamp is None and self.rng.random() < P_FLU:
            self.flu_timestamp = self.env.timestamp

        # Have random allergy symptoms
        if self.has_allergies and self.rng.random() < P_HAS_ALLERGIES_TODAY:
            self.allergy_timestamp = self.env.timestamp

        location.remove_human(self)

    def _select_location(self, location_type, city):
        """
        Preferential exploration treatment to visit places
        rho, gamma are treated in the paper for normal trips
        Here gamma is multiplied by a factor to supress exploration for parks, stores.
        """
        if location_type == "park":
            S = self.visits.n_parks
            self.adjust_gamma = 1.0
            pool_pref = self.parks_preferences
            locs = city.parks
            visited_locs = self.visits.parks

        elif location_type == "stores":
            S = self.visits.n_stores
            self.adjust_gamma = 1.0
            pool_pref = self.stores_preferences
            locs = city.stores
            visited_locs = self.visits.stores

        elif location_type == "hospital":
            hospital = None
            for hospital in sorted(city.hospitals, key=lambda x:compute_distance(self.location, x)):
                if len(hospital.humans) < hospital.capacity:
                    return hospital
            return None

        elif location_type == "hospital-icu":
            icu = None
            for hospital in sorted(city.hospitals, key=lambda x:compute_distance(self.location, x)):
                if len(hospital.icu.humans) < hospital.icu.capacity:
                    return hospital.icu
            return None

        elif location_type == "miscs":
            S = self.visits.n_miscs
            self.adjust_gamma = 1.0
            pool_pref = [(compute_distance(self.location, m) + 1e-1) ** -1 for m in city.miscs if
                         m != self.location]
            pool_locs = [m for m in city.miscs if m != self.location]
            locs = city.miscs
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
            # exploit
            cands = [(i, count) for i, count in visited_locs.items()]

        cands, scores = zip(*cands)
        loc = self.rng.choice(cands, p=_normalize_scores(scores))
        visited_locs[loc] += 1
        return loc

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if state.get("env"):
            del state['env']
            del state['_events']
            del state['visits']
            del state['household']
            del state['location']
            del state['workplace']
            del state['exercise_hours']
            del state['exercise_days']
            del state['shopping_days']
            del state['shopping_hours']
            del state['work_start_hour']
            del state['profession']
            del state['rho']
            del state['gamma']
            del state['rest_at_home']
            del state['never_recovers']
            del state['last_state']
            del state['avg_shopping_time']
            del state['city']
            del state['count_shop']
            del state['last_date']
            del state['message_info']
            state['messages'] = [encode_message(message) for message in state['contact_book'].messages if message.day == state['contact_book'].messages[-1].day]
            state['update_messages'] = state['contact_book'].update_messages
            del state['contact_book']
            del state['last_location']
            del state['recommendations_to_follow']
            del state['tracing_method']
            if state.get('_workplace'):
                del state['_workplace']

        # add a stand-in for property
        state["all_reported_symptoms"] = self.all_reported_symptoms
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)


    def cur_message(self, day):
        """creates the current message for this user"""
        message = Message(self.uid, self.risk_level, day, self.name)
        return message

    def cur_message_risk_update(self, day, old_uid, old_risk, sent_at):
        return UpdateMessage(old_uid, self.risk_level, old_risk, day, sent_at, self.name)

    def symptoms_at_time(self, now, symptoms):
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
        # dont change the logic in here, it needs to remain FROZEN
        results = np.zeros(14)
        result_day = (date - self.test_time).days
        if result_day >= 0 and result_day < 14:
            results[result_day] = 1
        return results

    def exposure_array(self, date):
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
        # dont change the logic in here, it needs to remain FROZEN
        is_recovered = False
        recovery_day = (date - self.recovered_timestamp).days
        if recovery_day >= 0 and recovery_day < 14:
            is_recovered = True
        else:
            recovery_day = None
        return is_recovered, recovery_day


    ############################## RISK PREDICTION #################################

    def update_risk_level(self):
        if not self.is_removed and self.tracing_method.risk_model == "transformer":
            assert(self.risk_history is not None)
            cur_day = (self.env.timestamp - self.env.initial_timestamp).days
            for day in range(cur_day, TRACING_N_DAYS_HISTORY + cur_day -1):
                old_risk_level_on_day = _proba_to_risk_level(self.prev_risk_history[day-cur_day])
                new_risk_level_on_day = _proba_to_risk_level(self.risk_history[day-cur_day+1])
                if old_risk_level_on_day != new_risk_level_on_day:
                    self.risk = self.risk_history[day-cur_day+1]
                    self.risk_level = min(new_risk_level_on_day, 15)
                    for message in self.contact_book.messages_by_day[day-1]:
                        my_old_message = self.contact_book.sent_messages_by_day[day-1][0]
                        sent_at = int(my_old_message.unobs_id[6:])
                        self.city.hd[message.unobs_id].contact_book.update_messages.append(
                            encode_update_message(self.cur_message_risk_update(my_old_message.day, my_old_message.uid, old_risk_level_on_day, sent_at)))

            self.risk_level = min(_proba_to_risk_level(self.risk_history[0]), 15)
            self.risk = self.risk_history[0]
            new_risk_level = min(_proba_to_risk_level(self.risk), 15)
            if new_risk_level != self.risk_level:
                self.risk_level = new_risk_level
                self.tracing_method.modify_behavior(self)
        else:
            new_risk_level = _proba_to_risk_level(self.risk)
            if new_risk_level != self.risk_level:
                if self.tracing_method.propagate_risk:
                    payload = {'change': new_risk_level > self.risk_level, 'magnitude': abs(new_risk_level - self.risk_level) }
                    self.contact_book.send_message(self, self.tracing_method, order=1, reason="risk_update", payload=payload)
                self.risk_level = new_risk_level

                self.tracing_method.modify_behavior(self)

    def update_risk(self, recovery=False, test_results=False, update_messages=None, symptoms=None):
        if not self.tracing:
            return

        if self.tracing:
            if recovery:
                if self.is_removed:
                    self.risk = 0.0
                else:
                    self.risk = BASELINE_RISK_VALUE
                self.update_risk_level()

            if test_results:
                if self.test_result == "positive":
                    self.risk = 1.0
                    self.contact_book.send_message(self, self.tracing_method, order=1, reason="test")
                elif self.test_result == "negative":
                    self.risk = 0.20
                self.update_risk_level()

        if self.tracing_method.risk_model != "transformer":
            if symptoms and self.tracing_method.propagate_symptoms:
                if sum(x in symptoms for x in ['severe', 'trouble_breathing']) > 0 and not self.has_logged_symptoms:
                    self.risk = max(0.8, self.risk)
                    self.update_risk_level()
                    self.contact_book.send_message(self, self.tracing_method, order=1, reason="symptoms")
                    self.has_logged_symptoms = True

            # update risk level because of update messages in run() to avoid redundant updates and cascading of message passing
            if (update_messages and
                not self.is_removed and
                self.test_result != "positive"):
                # if update_messages['reason'] == "risk_update":
                    # print(f"{self} traced")
                self.message_info['traced'] = True
                self.message_info['receipt'] = min(self.env.timestamp, self.message_info['receipt'])
                self.message_info['delay'] = min(update_messages['delay'], self.message_info['delay'])
                order = update_messages['order']
                propagate_further = order < self.tracing_method.max_depth

                if update_messages['reason'] == "test":
                    self.message_info['n_contacts_tested_positive'][order][-1] += update_messages['n']

                elif update_messages['reason'] == "symptoms":
                    self.message_info['n_contacts_symptoms'][order][-1] += update_messages['n']

                elif update_messages['reason'] == "risk_update":
                    self.message_info['n_contacts_risk_updates'][order][-1] += update_messages['n']
                    propagate_further = order < self.tracing_method.propage_risk_max_depth

                if update_messages['payload']:
                    if update_messages['payload']['change']:
                        self.message_info['n_risk_increased'][order][-1] += 1
                        self.message_info['n_risk_mag_increased'][order][-1] += update_messages['payload']["magnitude"]
                    else:
                        self.message_info['n_risk_decreased'][order][-1] += 1
                        self.message_info['n_risk_mag_decreased'][order][-1] += update_messages['payload']["magnitude"]

                if propagate_further:
                    self.contact_book.send_message(self, self.tracing_method, order=order+1, reason=update_messages['reason'])
