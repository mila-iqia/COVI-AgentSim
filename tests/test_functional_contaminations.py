import datetime
import logging
import unittest
import warnings
from collections import namedtuple

import numpy as np

from tests.utils import get_test_conf

TEST_CONF_NAME = "contaminations.yaml"


InterventionProps = namedtuple('InterventionProps',
                               ['name', 'risk_model', 'tracing_order'])


def _init_infector(human):
    from covid19sim.epidemiology.symptoms import _get_covid_progression

    # Force some properties on the infector
    human.is_asymptomatic = False
    human.asymptomatic_infection_ratio = 0.0

    human.can_get_really_sick = True
    human.can_get_extremely_sick = True

    # Set the average viral loads
    human.initial_viral_load = 0.5
    human.viral_load_peak_height = \
        human.conf['MIN_VIRAL_LOAD_PEAK_HEIGHT'] + \
        (human.conf['MAX_VIRAL_LOAD_PEAK_HEIGHT'] -
         human.conf['MIN_VIRAL_LOAD_PEAK_HEIGHT']) / 2
    human.viral_load_plateau_height = \
        human.viral_load_peak_height * \
        (human.conf['MIN_MULTIPLIER_PLATEAU_HEIGHT'] +
         (human.conf['MAX_MULTIPLIER_PLATEAU_HEIGHT'] -
          human.conf['MIN_MULTIPLIER_PLATEAU_HEIGHT']) / 2)

    #
    # Code copied from Human.compute_covid_properties() - begin
    #
    # precompute peak-plateau slope
    denominator = (human.viral_load_plateau_start - human.viral_load_peak_start)
    numerator = human.viral_load_peak_height - human.viral_load_plateau_height
    human.peak_plateau_slope = numerator / denominator
    assert human.peak_plateau_slope >= 0, f"viral load should decrease after peak. peak:{human.viral_load_peak_height} plateau height:{human.viral_load_plateau_height}"

    # percomupte plateau-end - recovery slope (should be negative because it is decreasing)
    numerator = human.viral_load_plateau_height
    denominator = human.recovery_days - (human.viral_load_plateau_end + human.infectiousness_onset_days)
    human.plateau_end_recovery_slope = numerator / denominator
    assert human.plateau_end_recovery_slope >= 0, f"slopes are assumed to be positive for ease of calculation"

    human.covid_progression = []
    if not human.is_asymptomatic:
        human.covid_progression = _get_covid_progression(
            human.initial_viral_load, human.viral_load_plateau_start,
            human.viral_load_plateau_end,
            human.recovery_days, age=human.age,
            incubation_days=human.incubation_days,
            infectiousness_onset_days=human.infectiousness_onset_days,
            really_sick=human.can_get_really_sick,
            extremely_sick=human.can_get_extremely_sick,
            rng=human.rng,
            preexisting_conditions=human.preexisting_conditions,
            carefulness=human.carefulness)

    all_symptoms = set(symptom for symptoms_per_day in human.covid_progression for symptom in symptoms_per_day)
    # infection ratios
    if sum(x in all_symptoms for x in ['moderate', 'severe', 'extremely-severe']) > 0:
        human.infection_ratio = 1.0
    else:
        human.infection_ratio = human.conf['MILD_INFECTION_RATIO']
    #
    # Code copied from Human.compute_covid_properties() - end
    #

    # Make the human at his most infectious (Peak viral load day)
    human._infection_timestamp = \
        human.env.timestamp + \
        datetime.timedelta(days=-(human.incubation_days))


class ContaminationTestInterface(unittest.TestCase):
    from covid19sim.human import Human

    human_at = Human.at

    class EnvMock:
        def __init__(self, timestamp):
            self.timestamp = timestamp

        @property
        def now(self):
            return 0

        @property
        def ts_initial(self):
            return 0

        @property
        def initial_timestamp(self):
            return self.timestamp

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

        def timeout(self, _):
            pass

    test_seed = 42
    n_people = 0
    init_percent_sick = 0
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    simulation_days = 0
    intervention_day = 0

    @staticmethod
    def human_at_mock(self, *args, **kwargs):
        # Note: this skips the environmental transmission and the removal of
        # the human from the location
        return next(ContaminationTestInterface.human_at(self, *args, **kwargs))

    def setUp(self):
        from covid19sim.locations.city import City

        ContaminationTestInterface.Human.at = ContaminationTestInterface.human_at_mock

        self.config = get_test_conf(TEST_CONF_NAME)

        self.config['COLLECT_LOGS'] = False
        self.config['INTERVENTION_DAY'] = self.intervention_day
        self.config['APP_UPTAKE'] = -1
        self.config['TRANSFORMER_EXP_PATH'] = "https://drive.google.com/file/d/1QhiZehbxNOhA-7n37h6XEHTORIXweXc6"
        self.config['LOGGING_LEVEL'] = "WARNING"
        # Prevent testing to ease analysis of 1st and 2 order tracing
        self.config['TEST_TYPES']['lab']['capacity'] = 0
        self.config['BASELINE_RISK_VALUE'] = 0.0

        self.config['INTERVENTION'] = 'Tracing'
        self.config['RISK_MODEL'] = 'transformer'
        self.config['TRACING_ORDER'] = None

        logging.basicConfig(level=getattr(logging, self.config["LOGGING_LEVEL"].upper()))

        city_x_range = (0, 1000)
        city_y_range = (0, 1000)

        self.env = self.EnvMock(self.start_time)
        self.rng = np.random.RandomState(self.test_seed)

        self.city = City(
            self.EnvMock(self.start_time),
            self.n_people,
            self.init_percent_sick,
            self.rng,
            city_x_range,
            city_y_range,
            ContaminationTestInterface.Human,
            self.config,
        )

        self.infector = None
        self.healthy_humans = [human for human in self.city.humans
                               if human.infection_timestamp is None]
        self.meeting_store = self.city.stores[0]

        for human in self.city.humans:
            if human.infection_timestamp:
                self.infector = human
                break

        _init_infector(self.infector)

        # Reset RNG
        self.rng = np.random.RandomState(self.test_seed)

    def tearDown(self):
        ContaminationTestInterface.Human.at = ContaminationTestInterface.human_at


class HealthyPopulationContaminationTest(ContaminationTestInterface):
    test_seed = 42
    n_people = 10
    init_percent_sick = 0.1  # Get 1 human sick
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    simulation_days = 5
    intervention_day = 0

    def test(self):
        self.assertIsNot(self.infector, None)
        self.assertEqual(len(self.healthy_humans), len(self.city.humans) - 1)

        # Healthy human should never infect others
        for _ in range(1000):
            for human in self.healthy_humans:
                human.at(self.meeting_store, self.city, 30)
                self.assertIs(human.infection_timestamp, None)
            self.env.timestamp += datetime.timedelta(minutes=30)


class ContaminationTestAt30min(ContaminationTestInterface):
    test_seed = 42
    n_people = 1000
    init_percent_sick = 0.001  # Get 1 human sick
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    simulation_days = 5
    intervention_day = 0
    at_duration_per_visit = 30
    at_duration_total = 48 * 60

    def test(self):
        self.assertIsNot(self.infector, None)
        self.assertEqual(len(self.healthy_humans), len(self.city.humans) - 1)

        # We want to test with a bad case scenario
        self.assertFalse(self.infector.is_asymptomatic)
        self.assertTrue(self.infector.can_get_really_sick)
        self.assertTrue(self.infector.can_get_extremely_sick)
        # self.assertTrue(self.infector.is_exposed)
        self.assertTrue(self.infector.is_infectious)
        self.assertEqual(self.infector.infection_ratio, 1.0)

        minutes_to_contamination = [0] * len(self.healthy_humans)

        for h_i, human in enumerate(self.healthy_humans):
            with self.subTest(human=human.name):
                # Reset env.timestamp
                self.env.timestamp = self.start_time
                for _ in range(self.at_duration_total // self.at_duration_per_visit):
                    self.infector.at(self.meeting_store, self.city, self.at_duration_per_visit)
                    human.at(self.meeting_store, self.city, self.at_duration_per_visit)
                    minutes_to_contamination[h_i] += self.at_duration_per_visit
                    self.meeting_store.remove_human(self.infector)
                    self.meeting_store.remove_human(human)
                    self.env.timestamp += datetime.timedelta(minutes=self.at_duration_per_visit)
                    if human.infection_timestamp is not None:
                        break
                # Return human to his household
                human.at(human.household, self.city, 0)
                try:
                    self.assertIsNot(human.infection_timestamp, None)
                except AssertionError as error:
                    warnings.warn(f"{human.name} spent {self.at_duration_total} "
                                  f"min at location {self.meeting_store} with "
                                  f"infector {self.infector} without getting sick: "
                                  f"{str(error)}",
                                  RuntimeWarning)

        try:
            # 1 day of continuous contact should make the vast majority of the humans sick
            # TODO: Put some values that make sense here
            self.assertLessEqual(sum(minutes_to_contamination) / len(minutes_to_contamination),
                                 24 * 60)
        except AssertionError as error:
            warnings.warn(f"Average minutes spend in location {self.meeting_store} "
                          f"to have a human get infected by {self.infector} is "
                          f"{sum(minutes_to_contamination) / len(minutes_to_contamination)}: "
                          f"{str(error)}",
                          RuntimeWarning)


class ContaminationTestAt60min(ContaminationTestAt30min):
    at_duration_per_visit = 60


class ContaminationTestAt120min(ContaminationTestAt30min):
    at_duration_per_visit = 120
