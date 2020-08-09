"""
Scheduler to introduce intervention in City.
"""
from collections import deque
from covid19sim.utils.constants import SECONDS_PER_DAY
from covid19sim.interventions.utils import get_tracing_method, get_intervention_conf, get_intervention_string
from covid19sim.utils.utils import log

class InterventionScheduler(object):
    """
    """
    def __init__(self, city, humans, conf):
        self.city = city
        self.humans = humans
        self.conf = conf

        self.sequence = conf['INTERVENTION_SEQUENCE']
        self.trigger_key = conf['INTERVENTION_TRIGGER']
        self.intervention_queue = deque(self.sequence)
        self.tracing_method = None
        self.current_intervention = {}
        self.intervention_count = 0

    def check_and_apply_intervention_if_applicable(self):
        """
        """
        if not self.trigger_now():
            return

        next_intervention_name = self.intervention_queue.popleft()[1]
        intervention_conf = get_intervention_conf(self.conf, next_intervention_name)

        # correctness of configuration file
        assert intervention_conf['N_BEHAVIOR_LEVELS'] >= 2, "At least 2 behavior levels are required to model behavior changes"

        log(f"\n *** ****** *** ****** *** INITIATING PHASE # {self.intervention_count} *** *** ****** *** ******\n", self.city.logfile)

        intervention_string = get_intervention_string(intervention_conf)
        intervention_conf['INTERVENTION'] = intervention_string
        log(intervention_string, self.city.logfile)

        # download app
        app_is_required = intervention_conf['APP_REQUIRED']
        if app_is_required:
            assert self.tracing_method is None, "NotImplementedError: Attempting to download apps twice."
            self.city.have_some_humans_download_the_app()
            self.tracing_method = get_tracing_method(risk_model=intervention_conf['RISK_MODEL'], conf=intervention_conf)
        else:
            self.tracing_method = None

        # initialize everyone from the baseline behavior
        for human in self.humans:
            # this should come first so that update_recommendations_level can access intervention related parameters
            human.intervened_behavior.update(intervention_conf)
            if self.tracing_method is not None:
                human.set_tracing_method(self.tracing_method)

        # log reduction levels
        log("\nCONTACT REDUCTION LEVELS (first one is not used) -", self.city.logfile)
        for location_type, value in human.intervened_behavior.reduction_levels.items():
            log(f"{location_type}: {value} ", self.city.logfile)

        # start tracking risk attributes
        if self.tracing_method is not None:
            self.city.tracker.track_daily_recommendation_levels(set_tracing_started_true=True)

        # modify knobs because now people are more aware
        if intervention_conf['ASSUME_NO_ENVIRONMENTAL_INFECTION_AFTER_INTERVENTION_START']:
            self.conf['_ENVIRONMENTAL_INFECTION_KNOB'] = 0.0
        log(f"Using ENVIRONMENTAL_INFECTION_KNOB: {self.conf['_ENVIRONMENTAL_INFECTION_KNOB']}", self.city.logfile)

        if intervention_conf['ASSUME_NO_UNKNOWN_INTERACTIONS_AFTER_INTERVENTION_START']:
            self.conf['_MEAN_DAILY_UNKNOWN_CONTACTS'] = 0.0
        log(f"Using MEAN_DAILY_UNKNOWN_CONTACTS: {self.conf['_MEAN_DAILY_UNKNOWN_CONTACTS']}", self.city.logfile)

        if intervention_conf['ASSUME_SAFE_HOSPITAL_DAILY_INTERACTIONS_AFTER_INTERVENTION_START']:
            log(f"SAMPLING 0 KNOWN INTERACTIONS IN HOSPITALS", self.city.logfile)

        intervention_conf['INTERVENTION_START_TIME'] = self.city.env.timestamp
        self.current_intervention = intervention_conf
        log("\n*** *** ****** *** ****** *** ****** *** ****** *** ****** *** ****** *** ****** *** ***\n", self.city.logfile)

        self.conf[f'{self.intervention_count}-INTERVENTION'] = intervention_conf
        self.intervention_count += 1

    def trigger_now(self):
        """
        Determines if next intervention should be applied now.

        Returns:
            (bool): True if its time for next intervention.
        """
        if len(self.intervention_queue) == 0:
            return False

        trigger_threshold = self.intervention_queue[0][0]
        if self.trigger_key == "PERCENT_INFECTED":
            n_infected = sum(self.city.tracker.cases_per_day)
            n_people = self.city.tracker.n_people
            percent_infected = 100 * n_infected / n_people

            if percent_infected >= trigger_threshold:
                return True

        elif self.trigger_key == "DAYS_SINCE_OUTBREAK":
            n_days = (self.city.env.timestamp - self.city.env.initial_timestamp).total_seconds()

            if n_days >= trigger_threshold * SECONDS_PER_DAY:
                return True

        else:
            raise ValueError(f"Unknown trigger key: {self.trigger_key}")

        return False
