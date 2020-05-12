"""
Derived City classes to implement specific control simulation environments
"""
import simpy
import math
import datetime
import itertools
from collections import defaultdict

from covid19sim.configs.config import *
from covid19sim.utils import compute_distance, _get_random_area, _draw_random_discreet_gaussian, get_intervention, calculate_average_infectiousness
from covid19sim.track import Tracker
from covid19sim.interventions import *
from covid19sim.frozen.utils import update_uid
from covid19sim.configs.constants import TICK_MINUTE
from covid19sim.configs.exp_config import ExpConfig
from covid19sim.base import City, Env

class ExternalBuildCity(City):
    """
    A City environment the user can build with externally defined code
    Useful for controlled scenarios and functional testing
    """

    def __init__(self, env, rng, x_range, y_range, start_time):
        """

        Args:
            env (simpy.Environment): [description]
            rng (np.random.RandomState): Random number generator
            x_range (tuple): (min_x, max_x)
            y_range (tuple): (min_y, max_y)
            start_time (datetime.datetime): City's initial datetime
        """
        self.env = env
        self.rng = rng
        self.x_range = x_range
        self.y_range = y_range
        self.total_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        self.n_people = 0
        self.start_time = start_time
        self.last_date_to_check_tests = self.env.timestamp.date()
        self.test_count_today = defaultdict(int)

        # Get the test type with the lowest preference?
        # TODO - EM: Should this rather sort on 'preference' in descending order? 
        self.test_type_preference = list(zip(*sorted(TEST_TYPES.items(), key=lambda x:x[1]['preference'])))[0]
    
        self.humans = []
        self.households = OrderedSet()
        self.stores = []
        self.senior_residencys = []
        self.hospitals = []
        self.miscs = []
        self.parks = []
        self.schools = []
        self.workplaces = []

    def initWorld(self):
        self.log_static_info()
        print("Computing preferences")
        self._compute_preferences()
        self.tracker = Tracker(self.env, self)
        # self.tracker.track_initialized_covid_params(self.humans)
        self.intervention = None
