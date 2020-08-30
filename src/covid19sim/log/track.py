"""
Contains a class to track several simulation metrics.
It is initialized as an attribute of the city and called at several places in `Human`.
"""
import os
import datetime
import math
import typing
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from runstats import Statistics

import dill
import networkx as nx
import numpy as np
import pandas as pd

from covid19sim.plotting.plot_rt import PlotRt
from covid19sim.utils.utils import log, _get_seconds_since_midnight
from covid19sim.utils.constants import AGE_BIN_WIDTH_5, ALL_LOCATIONS, SECONDS_PER_DAY, SECONDS_PER_HOUR
LOCATION_TYPES_TO_TRACK_MIXING = ["house", "work", "school", "other", "all"]
from covid19sim.epidemiology.symptoms import MILD, MODERATE, SEVERE, EXTREMELY_SEVERE
from covid19sim.inference.server_utils import DataCollectionServer, DataCollectionClient, \
    default_datacollect_frontend_address
from covid19sim.utils.utils import log, copy_obj_array_except_env
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_MINUTE
from covid19sim.interventions.tracing import Heuristic
if typing.TYPE_CHECKING:
    from covid19sim.human import Human

# used by - next_generation_matrix,
SNAPSHOT_PERCENT_INFECTED_THRESHOLD = 2 # take a snapshot every time percent infected of population increases by this amount


def check_if_tracking(f):
    def wrapper(*args, **kwargs):
        if args[0].start_tracking:
            return f(*args, **kwargs)
    return wrapper

def _compute_ngm(next_generation_matrix):
    """
    Computes Next Generation Matrix.
    """
    unique_n_humans = [len(next_generation_matrix['infectious_people'][idx]) for idx in range(len(AGE_BIN_WIDTH_5))]
    unique_n_humans = np.expand_dims(unique_n_humans, axis=1)
    m = np.zeros_like(next_generation_matrix["total"])
    np.divide(next_generation_matrix["total"], unique_n_humans, where=unique_n_humans!=0, out=m)
    return m

def print_dict(title, dic, is_sorted=None, top_k=None, logfile=None):
    if is_sorted is None:
        items = dic.items()
    else:
        items = sorted(dic.items(), key=lambda x: x[1])[:top_k]
        if is_sorted == "desc":
            items = reversed(items)
    ml = max([len(str(k)) for k in dic.keys()] + [0]) + 2
    aligned = "{:" + str(ml) + "}"
    log(
        "{}:\n   ".format(title) +
        "\n    ".join((aligned + ": {:5.4f}").format(str(k), v) for k, v in items),
        logfile
    )

def normalize_counter(counter, normalizer=None):
    if normalizer:
        total = normalizer
    else:
        total = float(sum(counter.values()))

    for key in counter:
        counter[key] /= total
    return counter

def _get_location_type_to_track_mixing(human, location):
    """
    Maps a location to a corresponding label in POLYMOD study.

    Args:
        human (covid19sim.human.Human): `human` who is at a `location`
        location (covid19sim.locations.location.Location): `location` where an interaction took place
    Returns
        (str): a corresponding label for POLYMOD study.
    """
    if location == human.household:
        return "house"

    # if location.location_type == "WORKPLACE":
        # return "work"

    # doing this first matters because kids are assigned their workplace as schools
    if location.location_type == "SCHOOL":
        return "school"

    if location == human.workplace:
        return "work"


    return "other"

class Tracker(object):
    """
    Keeps track of several aspects of the simulation. It is called from various locations in the entire codebase
    to keep track of relevant metrics.
    """
    def __init__(self, env, city, conf, logfile):
        """
        Args:
            env (simpy.Environment): Keeps track of events and their schedule
            city ([type]): [description]
            conf (dict): yaml configuration of the experiment
            logfile (str): filepath where the console output and final tracked metrics will be logged.
        """
        self.fully_initialized = False
        self.env = env
        self.city = city
        self.conf = conf
        self.logfile = logfile
        today = self.env.timestamp.date()
        self.start_tracking = False # flag to indicate if infections have been seeded in the population
        self.init_infected = []

        self.last_day = {
                'track_recovery': today,
                "track_infection": today,
                'social_mixing': today,
                'bluetooth_communications': today
            }

        # all about contacts
        contact_matrices_fmt = defaultdict(lambda: {
                    'avg': (0, np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))),
                    'total': np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5))),
                    'n_people': defaultdict(lambda : set()),
                    'avg_daily': (0, np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))),
                    "unique_avg_daily": (0, np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))),
                })
        self.contact_matrices = {
                            "known": deepcopy(contact_matrices_fmt),
                            "all": deepcopy(contact_matrices_fmt),
                            "within_contact_condition": deepcopy(contact_matrices_fmt),
                            }

        contact_duration_matrices_fmt = defaultdict(lambda: {
                    'avg': (0, np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))),
                    'total': np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5))),
                    'avg_daily': (0, np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))),
                })
        self.contact_duration_matrices = {
                            "known": deepcopy(contact_duration_matrices_fmt),
                            "all": deepcopy(contact_duration_matrices_fmt),
                            "within_contact_condition": deepcopy(contact_duration_matrices_fmt),
                            }

        self.bluetooth_contact_matrices = defaultdict(lambda: {
                    'avg_daily': (0, np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))),
                    'total': np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5))),
                })

        self.mean_daily_contacts_per_agegroup = {
            'weekday': defaultdict(lambda: (0, np.zeros(len(AGE_BIN_WIDTH_5)))),
            'weekend': defaultdict(lambda: (0, np.zeros(len(AGE_BIN_WIDTH_5)))),
            "all_days": defaultdict(lambda : (0,0))
        }

        self.mean_daily_contact_duration_per_agegroup = {
            'weekday': defaultdict(lambda: (0, np.zeros(len(AGE_BIN_WIDTH_5)))),
            'weekend': defaultdict(lambda: (0, np.zeros(len(AGE_BIN_WIDTH_5)))),
            "all_days": defaultdict(lambda : (0,0))
        }

        self.mean_daily_contacts = {
            "weekend": defaultdict(lambda: (0,0)),
            "weekday": defaultdict(lambda: (0,0)),
        }
        self.mean_daily_contact_duration = {
            "weekend": defaultdict(lambda: (0,0)),
            "weekday": defaultdict(lambda: (0,0)),
        }

        self.contact_distance_profile = {
            "known": defaultdict(dict),
            "all": defaultdict(dict),
            "within_contact_condition": defaultdict(dict)
        }

        self.contact_duration_profile = {
            "known": defaultdict(dict),
            "all": defaultdict(dict),
            "within_contact_condition": defaultdict(dict)
        }

        self.contact_attributes = {
            "SEIR": np.zeros((4, 4)),
            "SEIR_snapshots": [],
            "SI": {
                "p_infection": 0,
                "viral_load": 0,
                "infectiousness": 0,
                "infection": 0,
                "total": 0
            }
        }

        # infection related contacts
        infection_matrix_fmt = {
            "caused_infection": np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5))),
            "risky_contact": np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))
        }
        self.human_human_infection_matrix = defaultdict(lambda : deepcopy(infection_matrix_fmt))

        self.next_generation_matrix = {
            "total": np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5))),
            'infectious_people': defaultdict(lambda : set()),
            "last_snapshot_percent_infected": 0.0,
            "snapshots": [] # stores NGMs at different times during the simulation
        }

        infection_histogram_fmt = {
            "caused_infection": np.zeros(len(AGE_BIN_WIDTH_5)),
            "risky_contact": np.zeros(len(AGE_BIN_WIDTH_5))
        }
        self.environment_human_infection_histogram = defaultdict(lambda : deepcopy(infection_histogram_fmt))

        self.average_infectious_contacts = defaultdict(lambda : {'infection_count':0, 'humans':set()})
        self.p_infection_at_contact = {
            "human": [],
            "environment": []
        }

        # mobility
        self.transition_probability = {
                                    "weekday": defaultdict(lambda : defaultdict(int)),
                                    "weekend": defaultdict(lambda : defaultdict(int))
                                    }

        self.day_fraction_spent_activity = {
                                        "weekday": defaultdict(lambda :(0,0)),
                                        "weekend": defaultdict(lambda :(0,0))
                                        }
        self.activity_attributes = {
            "end_time": defaultdict(lambda: Statistics()),
            "duration": defaultdict(lambda: Statistics())
        }

        self.socialize_activity_data = {
            "group_size": Statistics(),
            "location_frequency": defaultdict(int),
            "start_time": Statistics()
        }
        self.outside_daily_contacts = []
        self.n_outside_daily_contacts = 0

        # infection stats
        self.n_infected_init = 0 # to be initialized in `initialize`
        self.cases_per_day = []
        self.cumulative_incidence = []
        self.r = []

        # tracing related stats
        self.recommended_levels_daily = []
        self.mobility = []
        self.expected_mobility = []

        # infection
        self.humans_quarantined_state = defaultdict(list)
        self.humans_state = defaultdict(list)
        self.humans_rec_level = defaultdict(list)
        self.humans_intervention_level = defaultdict(list)

        # epi related
        self.serial_intervals = []
        self.serial_interval_book_to = defaultdict(dict)
        self.serial_interval_book_from = defaultdict(dict)
        self.recovery_stats = {
            'n_recovered': 0,
            'timestamps': []
        }
        # keeps running average
        self.covid_properties = {
            'incubation_days': [0, 0],
            'recovery_days': [0, 0],
            'infectiousness_onset_days': [0, 0]
        }

        # testing & hospitalization
        self.hospitalization_per_day = [0]
        self.critical_per_day = [0]
        self.deaths_per_day = [0]
        self.test_results_per_day = defaultdict(lambda :{'positive':0, 'negative':0})
        self.tested_per_day = [0]

        # demographics
        self.age_bins = [(x[0], x[1]) for x in sorted(self.conf.get("P_AGE_REGION"), key = lambda x:x[0])]
        self.n_people = self.n_humans = self.city.n_people
        self.adoption_rate = 0.0
        self.human_has_app = set()

        # symptoms
        self.symptoms = {'covid': defaultdict(int), 'all':defaultdict(int)}
        self.symptoms_set = {'covid': defaultdict(set), 'all': defaultdict(set)}

        # risk model
        self.risk_values = []
        self.avg_infectiousness_per_day = []
        self.risk_attributes = []
        self.tracing_started = False

        # behavior
        self.daily_quarantine = {
            "app_users": [],
            "all": [],
            "false_app_users": [],
            "false_all": []
        }
        self.quarantine_monitor = []

        # monitors
        self.human_monitor = {}
        self.infection_monitor = []
        self.test_monitor = []

        # update messages
        self.infector_infectee_update_messages = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda :{'unknown':{}, 'contact':{}})))
        self.to_human_max_msg_per_day = defaultdict(lambda : defaultdict(lambda :-1))
        self.infection_graph = set()

        # (debug) track all humans all the time
        self.keep_full_human_copies = city.conf.get("KEEP_FULL_OBJ_COPIES", False)
        self.collection_server, self.collection_client = None, None
        if self.keep_full_human_copies:
            assert not city.conf.get("COLLECT_TRAINING_DATA", False), \
                "cannot collect human object copies & training data simultanously, both use same server"
            assert os.path.isdir(city.conf["outdir"])
            self.collection_server = DataCollectionServer(
                data_output_path=os.path.join(city.conf["outdir"], "human_backups.hdf5"),
                human_count=city.conf["n_people"],
                simulation_days=city.conf["simulation_days"],
                config_backup=city.conf,
                encode_deltas=True,
            )
            self.collection_server.start()
            self.collection_client = DataCollectionClient(
                server_address=city.conf.get("data_collection_server_address", default_datacollect_frontend_address),
            )

    def initialize(self):
        self.summarize_population()
        self.fully_initialized = True

    def summarize_population(self):
        """
        Logs statistics related to demographics.
        """
        # warn by (*) string if something is off in percentage from census by this much amount
        WARN_RELATIVE_PERCENTAGE_THRESHOLD = 25
        WARN_SIGNAL = " (**#@#**) "

        log("\n######## SIMULATOR KNOBS #########", self.logfile)
        log(f"HOUSEHOLD_ASSORTATIVITY_STRENGTH: {self.conf['HOUSEHOLD_ASSORTATIVITY_STRENGTH']}", self.logfile)
        log(f"WORKPLACE_ASSORTATIVITY_STRENGTH: {self.conf['WORKPLACE_ASSORTATIVITY_STRENGTH']}", self.logfile)
        log(f"P_INVITATION_ACCEPTANCE: {self.conf['P_INVITATION_ACCEPTANCE']}", self.logfile)
        log(f"BEGIN_PREFERENTIAL_ATTACHMENT_FACTOR: {self.conf['BEGIN_PREFERENTIAL_ATTACHMENT_FACTOR']}", self.logfile)
        log(f"END_PREFERENTIAL_ATTACHMENT_FACTOR: {self.conf['END_PREFERENTIAL_ATTACHMENT_FACTOR']}", self.logfile)
        log(f"P_HOUSE_OVER_MISC_FOR_SOCIALS: {self.conf['P_HOUSE_OVER_MISC_FOR_SOCIALS']}", self.logfile)
        log(f"CONTAGION_KNOB: {self.conf['CONTAGION_KNOB']}", self.logfile)
        log(f"ENVIRONMENTAL_INFECTION_KNOB: {self.conf['ENVIRONMENTAL_INFECTION_KNOB']}", self.logfile)
        log(f"TIME_SPENT_SCALE_FACTOR_FOR_SHORT_ACTIVITIES: {self.conf['TIME_SPENT_SCALE_FACTOR_FOR_SHORT_ACTIVITIES']}", self.logfile)
        log(f"TIME_SPENT_SCALE_FACTOR_FOR_WORK: {self.conf['TIME_SPENT_SCALE_FACTOR_FOR_WORK']}", self.logfile)
        log(f"TIME_SPENT_SCALE_FACTOR_SLEEP_AWAKE: {self.conf['TIME_SPENT_SCALE_FACTOR_SLEEP_AWAKE']}", self.logfile)


        log("\n######## DEMOGRAPHICS / SYNTHETIC POPULATION #########", self.logfile)
        log(f"NB: (i) census numbers are in brackets. (ii) {WARN_SIGNAL} marks a {WARN_RELATIVE_PERCENTAGE_THRESHOLD} % realtive deviation from census\n", self.logfile)
        # age distribution
        x = np.array([h.age for h in self.city.humans])
        cns_avg = self.conf['AVERAGE_AGE_REGION']
        cns_median = self.conf['MEDIAN_AGE_REGION']
        log(f"Age (census) - mean: {x.mean():3.3f} ({cns_avg}), median: {np.median(x):3.0f} ({cns_median}), std: {x.std():3.3f}", self.logfile)

        # gender distribution
        str_to_print = "Gender: "
        x = np.array([h.sex for h in self.city.humans])
        for z in np.unique(x):
            p = 100 * x[x==z].shape[0]/self.n_people
            str_to_print += f"{z}: {p:2.3f} % | "
        log(str_to_print, self.logfile)

        ###### house initialization
        log("\n*** House allocation *** ", self.logfile)
        # senior residencies
        self.n_senior_residency_residents = sum(len(sr.residents) for sr in self.city.senior_residences)
        p = 100*self.n_senior_residency_residents / self.n_people
        cns = 100*self.conf['P_COLLECTIVE']
        warn = WARN_SIGNAL if 100*abs(p-cns)/cns > WARN_RELATIVE_PERCENTAGE_THRESHOLD else ""
        log(f"{warn}Total ( %) number of residents in senior residencies (census): {self.n_senior_residency_residents} ({p:2.2f} %) ({cns:2.2f})", self.logfile)

        # house allocation
        n_houses = len(self.city.households)
        sizes = np.zeros(n_houses)
        multigenerationals, only_adults = np.zeros(n_houses), np.zeros(n_houses)
        solo_ages = []
        for i, house in enumerate(self.city.households):
            sizes[i] = len(house.residents)
            multigenerationals[i] = house.allocation_type.multigenerational
            only_adults[i] = min(h.age for h in house.residents) > self.conf['MAX_AGE_CHILDREN']
            # solo dwellers
            if len(house.residents) == 1:
                solo_ages.append(house.residents[0].age)

        ## counts
        log(f"Total houses: {n_houses}", self.logfile)
        p = sizes.mean()
        cns = self.conf['AVG_HOUSEHOLD_SIZE']
        warn = WARN_SIGNAL if 100*abs(p-cns)/cns > WARN_RELATIVE_PERCENTAGE_THRESHOLD else ""
        log(f"{warn}Average house size - {p: 2.3f} ({cns: 2.3f})", self.logfile)
        ## size
        census = self.conf['P_HOUSEHOLD_SIZE']
        str_to_print = "Household size - simulation% (census): "
        for i,z in enumerate(sorted(np.unique(sizes))):
            p = 100 * (sizes == z).sum() / n_houses
            cns = 100*census[i]
            warn = WARN_SIGNAL if 100*abs(p-cns)/cns > WARN_RELATIVE_PERCENTAGE_THRESHOLD else ""
            str_to_print += f"{warn} {z}: {p:2.2f} % ({cns: 3.2f}) | "
        log(str_to_print, self.logfile)

        # solo dwellers
        estimated_solo_dwellers_mean_age = sum([x[2] * (x[0] + x[1]) / 2 for x in self.conf['P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1']])
        simulated_solo_dwellers_age_given_housesize1 = [[x[0], x[1], 0] for x in self.conf['P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1']]
        n_solo_houses = len(solo_ages) + 1e-6 # to avoid ZeroDivisionError
        for age in solo_ages:
            for i,x in enumerate(self.conf['P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1']):
                if x[0] <= age <= x[1]:
                    simulated_solo_dwellers_age_given_housesize1[i][2] += 1
        simulated_solo_dwellers_age_given_housesize1 = [[x[0], x[1], x[2]/n_solo_houses] for x in simulated_solo_dwellers_age_given_housesize1]
        simulated_solo_dwellers_mean_age = sum([x[2] * (x[0] + x[1]) / 2 for x in simulated_solo_dwellers_age_given_housesize1])
        warn = WARN_SIGNAL if 100*abs(estimated_solo_dwellers_mean_age-simulated_solo_dwellers_mean_age)/estimated_solo_dwellers_mean_age > WARN_RELATIVE_PERCENTAGE_THRESHOLD else ""
        str_to_print = f"{warn}Solo dwellers : Average age absolute: {np.mean(solo_ages): 2.2f} (Average with mid point of age groups - simulated:{simulated_solo_dwellers_mean_age: 2.2f} census:{estimated_solo_dwellers_mean_age: 2.2f}) | "
        # str_to_print += f"Median age: {np.median(solo_ages)}"
        log(str_to_print, self.logfile)

        ## type
        str_to_print = "Household type: "
        p = 100 * multigenerationals.mean()
        cns = 100*self.conf['P_MULTIGENERATIONAL_FAMILY']
        warn = WARN_SIGNAL if 100*abs(p-cns)/cns > WARN_RELATIVE_PERCENTAGE_THRESHOLD else ""
        str_to_print += f"{warn}multi-generation: {p:2.2f} % ({cns:2.2f}) | "
        p = 100 * only_adults.mean()
        str_to_print += f"Only adults: {p:2.2f} % | "
        log(str_to_print, self.logfile)

        # allocation types
        allocation_types = [res.allocation_type for res in self.city.households]
        living_arrangements = [res.basestr for res in allocation_types]
        census = np.array([x.probability for x in allocation_types])
        allocation_types = np.array(living_arrangements)
        str_to_print = "Allocation types: "
        for atype in np.unique(allocation_types):
            p = 100*(allocation_types == atype).mean()
            cns = 100*census[allocation_types == atype][0].item()
            warn = WARN_SIGNAL if 100*abs(p-cns)/cns > WARN_RELATIVE_PERCENTAGE_THRESHOLD else ""
            str_to_print += f"{warn}{atype}: {p:2.3f} %  ({cns: 2.2f})| "
        log(str_to_print, self.logfile)

        ###### location initialization
        log("\n *** Locations *** ", self.logfile)

        # counts
        str_to_print = "Counts: "
        for location_type in ALL_LOCATIONS:
            n_locations = len(getattr(self.city, f"{location_type.lower()}s"))
            str_to_print += f"{location_type}: {n_locations} | "
        log(str_to_print, self.logfile)

        ####### work force
        log("\n *** Workforce *** ", self.logfile)

        # workplaces, stores, miscs
        all_workplaces = [self.city.workplaces, self.city.stores, self.city.miscs]
        for workplaces in all_workplaces:
            subnames = defaultdict(lambda : {'count': 0, 'n_workers':0})
            n_workers = np.zeros_like(workplaces)
            avg_workers_age = np.zeros_like(workplaces)
            for i, workplace in enumerate(workplaces):
                n_workers[i] = workplace.n_workers
                avg_workers_age[i] = np.mean([worker.age for worker in workplace.workers])
                if workplace.location_type == "WORKPLACE":
                    subnames[workplace.name.split(":")[0]]['count'] += 1
                    subnames[workplace.name.split(":")[0]]['n_workers'] += workplace.n_workers
            if len(workplaces) > 0:
                name = workplaces[0].location_type
                log(f"{name} - Total workforce: {n_workers.sum()} | Average number of workers: {n_workers.mean(): 2.2f} | Average age of workers: {avg_workers_age.mean(): 2.2f}", self.logfile)
            if subnames:
                for workplace_type, val in subnames.items():
                    log(f"\tNumber of {workplace_type} - {val['count']}. Total number of workers - {val['n_workers']}", self.logfile)

        # hospitals
        n_nurses = np.zeros_like(self.city.hospitals)
        n_doctors = np.zeros_like(self.city.hospitals)
        for i,hospital in enumerate(self.city.hospitals):
            n_nurses[i] = hospital.n_nurses
            n_doctors[i] = hospital.n_doctors

        n_nurses_senior_residences = np.zeros_like(self.city.senior_residences)
        for i,senior_residence in enumerate(self.city.senior_residences):
            n_nurses_senior_residences[i] = senior_residence.n_nurses

        total_workforce = n_nurses.sum() + n_doctors.sum() + n_nurses_senior_residences.sum()
        str_to_print = f"HOSPITALS - Total workforce: {total_workforce} "
        str_to_print += f"| Number of doctors: {n_doctors.sum()} "
        str_to_print += f"| Number of nurses: {n_nurses.sum()} "
        str_to_print += f"| Number of nurses at SENIOR_RESIDENCES: {n_nurses_senior_residences.sum()}"
        log(str_to_print, self.logfile)

        # schools
        n_teachers = np.zeros_like(self.city.schools)
        n_students = np.zeros_like(self.city.schools)
        subnames = defaultdict(lambda : {'count':0, 'n_teachers':0, 'n_students':0})
        for i, school in enumerate(self.city.schools):
            n_teachers[i] = school.n_teachers
            n_students[i] = school.n_students
            subname = school.name.split(":")[0]
            subnames[subname]['count'] += 1
            subnames[subname]['n_teachers'] += school.n_teachers
            subnames[subname]['n_students'] += school.n_students

        str_to_print = f"SCHOOL - Number of teachers: {n_teachers.sum()} "
        str_to_print += f"| Number of students: {n_students.sum()}"
        str_to_print += f"| Average number of teachers: {n_teachers.mean(): 2.2f}"
        str_to_print += f"| Average number of students: {n_students.mean(): 2.2f}"
        log(str_to_print, self.logfile)

        for subname, val in subnames.items():
            log(f"\tNumber of {subname} - {val['count']}. Number of students: {val['n_students']}. Number of teachers: {val['n_teachers']}", self.logfile)

        log("\n *** Disease related initialization stats *** ", self.logfile)
        # disease related
        self.frac_asymptomatic = sum(h.is_asymptomatic for h in self.city.humans)/self.n_people
        log(f"Percentage of population that is asymptomatic {100*self.frac_asymptomatic: 2.3f}", self.logfile)

    def log_seed_infections(self):
        """
        Logs who is seeded as infected.
        """
        log("\n *** ****** *** ****** *** COVID infection seeded *** *** ****** *** ******\n", self.logfile)

        self.n_infected_init = self.city.n_init_infected
        log(f"Total number of infected humans {self.n_infected_init}", self.logfile)
        for human in self.city.humans:
            if human.is_exposed:
                self.init_infected.append(human)
                log(f"\t{human} @ {human.household} living with {len(human.household.residents) - 1} other residents", self.logfile)

        log(f"\nPREFERENTIAL_ATTACHMENT_FACTOR: {self.conf['END_PREFERENTIAL_ATTACHMENT_FACTOR']}", self.logfile)
        log("\n*** *** ****** *** ****** *** ****** *** ****** *** ****** *** ****** *** ****** *** ***\n", self.logfile)

        self.start_tracking = True

        #
        self.cases_per_day = [self.n_infected_init]
        self.s_per_day = [self.n_people - self.n_infected_init]
        self.e_per_day = [self.n_infected_init]
        self.i_per_day = [0]
        self.r_per_day = [0]
        self.ei_per_day = [self.n_infected_init]

        # risk model
        self.risk_precision_daily = [self.compute_risk_precision()]

    def compute_generation_time(self):
        """
        Generation time is the time from exposure day until an infection occurs.

        Returns:
            (float): mean generation time
        """
        times = []
        for x in self.infection_monitor:
            if x['from']:
                times.append((x['infection_timestamp'] - x['from_infection_timestamp']).total_seconds() / SECONDS_PER_DAY)

        return np.mean(times).item()

    def compute_serial_interval(self):
        """
        Computes mean of all serial intervals.
        For description of serial interval, refer self.track_serial_interval

        Returns:
            (float): serial interval
        """
        return np.mean(self.serial_intervals)

    @check_if_tracking
    def track_serial_interval(self, human_name):
        """
        tracks serial interval ("time duration between a primary case-patient (infector) having symptom onset and a secondary case-patient (infectee) having symptom onset")
        reference: https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article

        `self.serial_interval_book` maps infectee.name to covid_symptom_start_time of infector who infected this infectee.

        Args:
            human_name (str): name of `Human` who just experienced some symptoms
        """

        def register_serial_interval(infector, infectee):
            serial_interval = (infectee.covid_symptom_start_time - infector.covid_symptom_start_time).days # DAYS
            self.serial_intervals.append(serial_interval)

        # Pending intervals which manifested symptoms?
        # With human_name as infectee?
        remove = []
        for from_name, (to_human, from_human) in self.serial_interval_book_to[human_name].items():
            if from_human.covid_symptom_start_time is not None:
                # We know to_human.covid_symptom_start_time is not None because it happened before calling this func
                # Therefore, it is time to ...
                register_serial_interval(from_human, to_human)
                remove.append(from_name)
        # Remove pending intervals which were completed
        for from_name in remove:
            self.serial_interval_book_to[human_name].pop(from_name)
            self.serial_interval_book_from[from_name].pop(human_name)

        # With human_name as infector?
        remove = []
        for to_name, (to_human, from_human) in self.serial_interval_book_from[human_name].items():
            if to_human.covid_symptom_start_time is not None:
                # We know from_human.covid_symptom_start_time is not None because it happened before calling this func
                # Therefore, it is time to ...
                register_serial_interval(from_human, to_human)
                remove.append(to_name)
        # Remove pending intervals which were completed
        for to_name in remove:
            self.serial_interval_book_from[human_name].pop(to_name)
            self.serial_interval_book_to[to_name].pop(human_name)

    @check_if_tracking
    def increment_day(self):
        """
        [summary]
        """
        # cumulative incidence (Note: susceptible of prev day is needed here)
        if self.s_per_day[-1]:
            self.cumulative_incidence += [self.cases_per_day[-1] / self.s_per_day[-1]]
        else:
            self.cumulative_incidence.append(0)

        self.cases_per_day.append(0)

        self.s_per_day.append(sum(h.is_susceptible for h in self.city.humans))
        self.e_per_day.append(sum(h.is_exposed for h in self.city.humans))
        self.i_per_day.append(sum(h.is_infectious for h in self.city.humans))
        self.r_per_day.append(sum(h.is_removed for h in self.city.humans))
        self.ei_per_day.append(self.e_per_day[-1] + self.i_per_day[-1])

        for human in self.city.humans:
            if human.is_susceptible:
                state = 'S'
            elif human.is_exposed:
                state = 'E'
            elif human.is_infectious:
                state = 'I'
            elif human.is_removed:
                state = 'R'
            else:
                raise ValueError(f"{human} is not in any of SEIR states")

            self.humans_quarantined_state[human.name].append(human.intervened_behavior.is_under_quarantine)
            self.humans_state[human.name].append(state)
            self.humans_rec_level[human.name].append(human.rec_level)
            self.humans_intervention_level[human.name].append(human._intervention_level)

        # test_per_day
        self.tested_per_day.append(0)
        self.hospitalization_per_day.append(0)
        self.critical_per_day.append(0)
        self.deaths_per_day.append(0)

        # recommendation levels related statistics
        self.track_daily_recommendation_levels()

        # risk model
        prec, lift, recall = self.compute_risk_precision(daily=True)
        self.risk_precision_daily.append((prec, lift, recall))
        self.risk_values.append([(h.risk, h.is_exposed or h.is_infectious, h.test_result, len(h.symptoms) == 0) for h in self.city.humans])
        row = []
        for h in self.city.humans:
            row.append({
                "infection_timestamp": h.infection_timestamp,
                "n_infectious_contacts": h.n_infectious_contacts,
                "risk": h.risk,
                "risk_level": h.risk_level,
                "rec_level": h.rec_level,
                "state": h.state.index(1),
                "test_result": h.test_result,
                "n_symptoms": len(h.symptoms),
                "symptom_severity": self.compute_severity(h.symptoms),
                "reported_symptom_severity": self.compute_severity(h.reported_symptoms),
                "name": h.name,
                "dead": h.is_dead,
                "reported_test_result": h.reported_test_result,
                "n_reported_symptoms": len(h.reported_symptoms),
            })

        self.human_monitor[self.env.timestamp.date()-datetime.timedelta(days=1)] = row

        # epi
        self.avg_infectiousness_per_day.append(np.mean([h.infectiousness for h in self.city.humans]))

        # behavior
        # /!\ `intervened_behavior.is_quarantined()` has dropout
        x = np.array([(human.has_app, human.intervened_behavior.is_under_quarantine, human.is_susceptible or human.is_removed) for human in self.city.humans])
        n_quarantined_app_users = (x[:, 0] * x[:, 1]).sum()
        n_quarantined = x[:, 1].sum()
        n_false_quarantined = (x[:, 1] * x[:, 2]).sum()
        n_false_quarantined_app_users = (x[:, 0] * x[:, 1] * x[:, 2]).sum()
        self.daily_quarantine['app_users'].append(n_quarantined_app_users)
        self.daily_quarantine['all'].append(n_quarantined)
        self.daily_quarantine['false_app_users'].append(n_false_quarantined_app_users)
        self.daily_quarantine['false_all'].append(n_false_quarantined)

    def compute_severity(self, symptoms):
        severity = 0
        for s in symptoms:
            if "extremely-severe" == s:
                severity = 4
            elif "severe" == s and severity < 4:
                severity = 3
            elif "moderate" == s and severity < 3:
                severity = 2
            elif "mild" == s and severity < 2:
                severity = 1
        return severity

    @check_if_tracking
    def track_daily_recommendation_levels(self, set_tracing_started_true=False):
        """
        Tracks aggregate recommendation levels of humans in the city.
        """
        if set_tracing_started_true:
            self.tracing_started = True

        G, B, O, R, M, EM = 0,0,0,0,0,0
        if self.tracing_started:
            rec_levels = np.array([h.rec_level for h in self.city.humans if not h.is_dead])
            G = sum(rec_levels == 0)
            B = sum(rec_levels == 1)
            O = sum(rec_levels == 2)
            R = sum(rec_levels == 3)
            no_app = sum(rec_levels == -1)

            M = 1.0 * G + 0.8 * B + 0.20 * O + 0.05 * R + 1 * no_app
            EM = np.mean([1-h.risk for h in self.city.humans if not h.is_dead])

        #
        self.recommended_levels_daily.append([G, B, O, R])
        self.mobility.append(M)
        self.expected_mobility.append(EM)

    def compute_risk_precision(self, daily=True, until_days=None):
        """
        [summary]

        Args:
            daily (bool, optional): [description]. Defaults to True.
            threshold (float, optional): [description]. Defaults to 0.5.
            until_days ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if daily:
            all = [(h.risk, h.is_exposed or h.is_infectious) for h in self.city.humans]
            no_test = [(h.risk, h.is_exposed or h.is_infectious) for h in self.city.humans if h.test_result != "positive"]
            no_test_symptoms = [(h.risk, h.is_exposed or h.is_infectious) for h in self.city.humans if h.test_result != "positive" and len(h.symptoms) == 0]
        else:
            all = [(x[0],x[1]) for daily_risk_values in self.risk_values[:until_days] for x in daily_risk_values]
            no_test = [(x[0], x[1]) for daily_risk_values in self.risk_values[:until_days] for x in daily_risk_values if not x[2]]
            no_test_symptoms = [(x[0], x[1]) for daily_risk_values in self.risk_values[:until_days] for x in daily_risk_values if not x[2] and x[3]]

        top_k = [0.01, 0.03, 0.05, 0.10]
        total_infected = 1.0*sum(1 for x,y in all if y)
        all = sorted(all, key=lambda y:-y[0])
        no_test = sorted(no_test, key=lambda y:-y[0])
        no_test_symptoms = sorted(no_test_symptoms, key = lambda y:-y[0])

        lift = [[], [], []]
        top_k_prec = [[],[],[]]
        recall =[]
        idx = 0
        for type in [all, no_test, no_test_symptoms]:
            for k in top_k:
                xy = type[:math.ceil(k * len(type))]
                pred = 1.0*sum(1 for x,y in xy if y)

                # precision
                if len(xy):
                    top_k_prec[idx].append(pred/len(xy))
                else:
                    # happens when the population size is too small.
                    warnings.warn(f"population size {len(all)} too small to compute top-{k} precision for {len(xy)} people.", RuntimeWarning)
                    top_k_prec[idx].append(-1)
                # lift
                if total_infected:
                    lift[idx].append(pred/(k*total_infected))
                else:
                    lift[idx].append(0) # FIXME: it might not be correct definition for Lift

            z = sum(1 for x,y in type if y)
            recall.append(0)
            if z:
                recall[-1] = 1.0*sum(1 for x,y in type if y)/z
            idx += 1
        return top_k_prec, lift, recall

    @check_if_tracking
    def track_humans(self, hd: typing.Dict, current_timestamp: datetime.datetime):
        """
        Keeps record of humans and their attributes at each hour (called hourly from city)

        Args:
            hd (dict):
            current_timestamp: (datetime.datetime):
        """
        if (
            self.conf['RISK_MODEL'] == ""
            or not (self.conf['track_all'] or self.conf['track_humans'])
        ):
            return

        for name, h in hd.items():
            order_1_contacts = h.contact_book.get_contacts(hd)

            self.risk_attributes.append({
                "has_app": h.has_app,
                "risk": h.risk,
                "risk_level": h.risk_level,
                "reason": h.heuristic_reasons,
                "rec_level": h.rec_level,
                "exposed": h.is_exposed,
                "infectious": h.is_infectious,
                "symptoms": len(h.symptoms),
                "symptom_names": h.reported_symptoms,
                "clusters": h.intervention.extract_clusters(h) if type(h.intervention) == Heuristic else [],
                "current_prevalence": h.city.prevalence,
                "test": h.test_result,
                "recovered": h.is_removed,
                "timestamp": self.env.timestamp,
                "test_recommended": h._test_recommended,
                "name": h.name,
                "order_1_is_exposed": any([c.is_exposed for c in order_1_contacts]),
                "order_1_is_infectious": any([c.is_infectious for c in order_1_contacts]),
                "order_1_is_presymptomatic": any([c.is_infectious and
                                                  len(c.symptoms) == 0 for c in order_1_contacts]),
                "order_1_is_symptomatic": any([c.is_infectious and
                                               len(c.symptoms) > 0 for c in order_1_contacts]),
                "order_1_is_tested": any([c.test_result == "positive" for c in order_1_contacts]),
            })
        if self.keep_full_human_copies:
            assert self.collection_client is not None
            human_backups = copy_obj_array_except_env(hd)
            for name, human in human_backups.items():
                human_id = int(name.split(":")[-1]) - 1
                current_day = (current_timestamp - self.city.start_time).days
                self.collection_client.write(current_day, current_timestamp.hour, human_id, human)
            # @@@@@ TODO: do something with location backups
            # location_backups = copy_obj_array_except_env(self.city.get_all_locations())

    def track_app_adoption(self):
        """
        Stores app adoption rate and humans who have the app.
        """
        self.adoption_rate = sum(h.has_app for h in self.city.humans) / self.n_people
        log(f"adoption rate: {100*self.adoption_rate:3.2f} %\n", self.logfile)
        self.human_has_app = set([h.name for h in self.city.humans if h.has_app])

    @check_if_tracking
    def track_covid_properties(self, human):
        """
        Keeps a running average of various covid related properties.

        Args:
            human (covid19sim.human.Human): `human` who got infected
        """
        n, avg = self.covid_properties['incubation_days']
        self.covid_properties['incubation_days'] = (n+1, (avg*n + human.incubation_days)/(n+1))

        n, avg = self.covid_properties['recovery_days']
        self.covid_properties['recovery_days'] = (n+1, (avg*n + human.recovery_days)/(n+1))

        n, avg = self.covid_properties['infectiousness_onset_days']
        self.covid_properties['infectiousness_onset_days'] = (n+1, (n*avg +human.infectiousness_onset_days)/(n+1))

    @check_if_tracking
    def track_hospitalization(self, human, type=None):
        """
        [summary]

        Args:
            human ([type]): [description]
            type ([type], optional): [description]. Defaults to None.
        """
        self.hospitalization_per_day[-1] += 1
        if type == "icu":
            self.critical_per_day[-1] += 1

    @check_if_tracking
    def track_deaths(self):
        """
        Keeps count of deaths per day.
        Args:
            human (covid19sim.human.Human): `human` who died`
        """
        self.deaths_per_day[-1] += 1

    @check_if_tracking
    def track_quarantine(self, human, unquarantine=False):
        """
        Keeps record of quarantined and unquarantined individuals.

        Args:
            human (covid19sim.human.Human): human who is about to quarantine or release from one.
            unquarantine (bool): if `human` is about to be released from quarantine
        """
        # /!\ redundant information being logged on unquarantine
        self.quarantine_monitor.append({
                "name": human.name,
                "infection_timestamp": human.infection_timestamp,
                "infectiousness_onset_days": human.infectiousness_onset_days,
                "incubation_days": human.incubation_days,
                "reason": human.intervened_behavior.quarantine_reason,
                "duration": human.intervened_behavior.quarantine_duration,
                "timestamp": self.env.timestamp,
                "unquarantine": unquarantine
            })


    @check_if_tracking
    def track_infection(self, source, from_human, to_human, location, timestamp, p_infection, success, viral_load=-1, infectiousness=-1):
        """
        Called every time someone is infected either by other `Human` or through envrionmental contamination.
        It tracks the following -
            1. (list) cases_per_day - Number of cases per day
            2. (list) infection_monitor - Several attributes of infector and infectee at the time of infection
            3. (list) p_infection_at_contact - Various attributes of viral_load and p_infection at the time of contact
            4. (np.array) infection_matrix - number of risky contacts / infectious contacts between different age groups
            5. (set) serial_interval_book_to/_from - stores who infected whom for later computation of serial intervals
            6. (dict) average_infectious_contacts - stores number of infections and unique infectors to calculate empirical Ro

        Args:
            type (str): Type of transmissions, i.e., "human" or "environmental".
            from_human (Human): `Human` who infected to_human. None if its environmental infection.
            to_human (Human): `Human` who got infected
            location (Location): `Location` where the event took place.
            timestamp (datetime.datetime): time at which this event took place.
            p_infection: the probability of infection computed at the time of infection .
            success (bool): whether it was successful to infect the infectee
            viral_load (float, optional): viral load of `from_human` who infected `to_human`. -1 if its environmental infection.
            infectiousness (float, optional): infectiousness of `from_human` who infected `to_human`. -1 if its environmental infection.
        """
        assert source in ["human", "environment"], f"Unknown infection type: {type}"

        if success:
            self.cases_per_day[-1] += 1
            self.infection_monitor.append({
                "from": None if not source =="human" else from_human.name,
                "from_risk":  None if not source=="human" else from_human.risk,
                "from_risk_level": None if not source=="human" else from_human.risk_level,
                "from_rec_level": None if not source=="human" else from_human.rec_level,
                "from_infection_timestamp": None if not source=="human" else from_human.infection_timestamp,
                "from_is_asymptomatic": None if not source=="human" else from_human.is_asymptomatic,
                "from_has_app": None if not source == "human" else from_human.has_app,
                "to": to_human.name,
                "to_risk": to_human.risk,
                "to_risk_level": to_human.risk_level,
                "to_rec_level": to_human.rec_level,
                "infection_date": timestamp.date(),
                "infection_timestamp":timestamp,
                "to_is_asymptomatic": to_human.is_asymptomatic,
                "to_has_app": to_human.has_app,
                "location_type": location.location_type,
                "location": location.name,
                "p_infection": p_infection,
                "to_human_infectiousness_onset_days": to_human.infectiousness_onset_days
            })

            # bookkeeping needed for track_update_messages
            if from_human is not None:
                self.infection_graph.add((from_human.name, to_human.name))

        #
        type_of_location = location.location_type
        y = to_human.age_bin_width_5.index
        if from_human is not None:
            type_of_location = _get_location_type_to_track_mixing(from_human, location)
            x = from_human.age_bin_width_5.index

        # self.p_infection.append([infection, p_infection, viral_load])
        self.p_infection_at_contact[source].append((success, p_infection, viral_load, infectiousness))

        if source == "human":

            self.human_human_infection_matrix[type_of_location]["risky_contact"][x, y] += 1
            self.human_human_infection_matrix["all"]["risky_contact"][x, y] += 1

            infection_attrs = [success, p_infection, viral_load, infectiousness]
            self.track_contact_attributes(from_human, to_human, infection_attrs=infection_attrs)

            if success:
                # for transmission matrix
                self.human_human_infection_matrix[type_of_location]["caused_infection"][x, y] += 1
                self.human_human_infection_matrix["all"]["caused_infection"][x, y] += 1

                # for next generation matrix
                self.next_generation_matrix["total"][x, y] += 1
                self.next_generation_matrix['infectious_people'][x].add(from_human)
                # snapshot
                last_percent_infected = 0.0
                if len(self.next_generation_matrix["snapshots"]) > 0:
                    last_percent_infected = self.next_generation_matrix["snapshots"][-1][0]

                percent_infected = 100 * sum(self.cases_per_day) / self.n_people
                if percent_infected - last_percent_infected > SNAPSHOT_PERCENT_INFECTED_THRESHOLD:
                    ngm = _compute_ngm(self.next_generation_matrix)
                    self.next_generation_matrix['snapshots'].append((percent_infected, ngm))
                    # R = np.linalg.eigvals(ngm).max().real
                    # print(f"% infected: {percent_infected: 2.2f} R:{R: 2.2f}")
                # for serial interval
                # Keep records of the infection so that serial intervals can be registered when symptoms appear
                # Note: We need a bidirectional record (to/from), because we can't anticipate which (to or from) will manifest symptoms first
                self.serial_interval_book_to[to_human.name][from_human.name] = (to_human, from_human)
                self.serial_interval_book_from[from_human.name][to_human.name] = (to_human, from_human)

                # for transmission broken down by symptomaticity and location
                if from_human.is_asymptomatic:
                    symptomaticity_key = "asymptomatic"
                elif not from_human.is_asymptomatic and not from_human.is_incubated:
                    symptomaticity_key = "presymptomatic"
                else:
                    symptomaticity_key = "symptomatic"

                for key in [symptomaticity_key, location.location_type, "all"]:
                    self.average_infectious_contacts[key]['infection_count'] += 1
                    self.average_infectious_contacts[key]['humans'].add(from_human.name)
                return

        elif source == "environment":

            self.environment_human_infection_histogram["all"]["risky_contact"][y] += 1
            self.environment_human_infection_histogram[type_of_location]["risky_contact"][y] += 1

            if success:
                # environmental transmission
                self.environment_human_infection_histogram["all"]["caused_infection"][y] += 1
                self.environment_human_infection_histogram[type_of_location]["caused_infection"][y] += 1
            return

    @check_if_tracking
    def track_update_messages(self, from_human, to_human, payload):
        """
        Track which update messages are sent and when (used for plotting)
        """
        if (from_human.name, to_human.name) in self.infection_graph:
            reason = payload['reason']
            assert reason in ['unknown', 'contact'], "improper reason for sending a message"
            model = self.city.conf.get("RISK_MODEL")
            count = self.infector_infectee_update_messages[from_human.name][to_human.name][self.env.timestamp][reason].get('count', 0)
            x = {'method':model, 'new_risk_level':payload['new_risk_level'], 'count':count+1}
            self.infector_infectee_update_messages[from_human.name][to_human.name][self.env.timestamp][reason] = x
        else:
            old_max_risk_level = self.to_human_max_msg_per_day[to_human.name][self.env.timestamp.date()]
            self.to_human_max_msg_per_day[to_human.name][self.env.timestamp.date()] = max(old_max_risk_level, payload['new_risk_level'])

    @check_if_tracking
    def track_symptoms(self, human=None, count_all=False):
        """
        Keeps a set of symptoms experienced by `Human` until it stops experiencing them.
        It is called from `self.update_symptoms` from `Human`.
        When the symptoms are empty, it adds them to the counter.

        attributes used:
            self.symptoms_set (dict (dict (set))):
                Maps reason of symptoms, i.e., covid, cold, flu, etc. or all  to a set for each human.
                When the symptoms are over, human is deleted from the this dictionary.
                Upon deletion, symptoms count are noted in `self.symptoms`

            self.symptoms (dict(dict(int))):
                It keeps the frequency of symptoms once the human's symptoms are finished.

        Args:
            human (Human, optional): `Human` object. Defaults to None.
            count_all (bool, optional): clears up self.symptoms_set to add the count of symptoms to self.symptoms.
                                        It is used at the end of simulation to aggregate information.
                                        destroys self.symptoms_set to avoid any errors.
        """
        def _aggregate_symptoms_count(symptoms_set, symptoms):
            symptoms['n'] += 1
            for s in symptoms_set:
                symptoms[s.name] += 1

            return symptoms

        # clear up the current count of symptoms
        if count_all:
            for key in ["all", "covid"]:
                remaining_humans = list(self.symptoms_set[key].keys())
                for human in remaining_humans:
                    self.symptoms[key] = _aggregate_symptoms_count(self.symptoms_set[key][human], self.symptoms[key])
                    self.symptoms_set[key].pop(human)

            return self.symptoms

        # track COVID symptoms to estimate P(symptom = x | COVID = 1)
        # if infected with COVID, keep accumulating symptoms until recovery
        if  human.infection_timestamp is not None:
            self.symptoms_set['covid'][human.name].update(human.covid_symptoms)
        else:
            # once human has recovered, count the symptoms into `self.symptoms` and remove this human from symptoms_set
            if human.name in self.symptoms_set['covid']:
                self.symptoms["covid"] = _aggregate_symptoms_count(self.symptoms_set['covid'][human.name], self.symptoms['covid'])
                self.symptoms_set['covid'].pop(human.name)

        # track reported symptoms to estimate P(symptom = x)
        # Note 1: `human` can be experiencing COVID, cold, flu or allergy.
        # Note 2: `human` can experience it multiple times in the simulation.
        # Accumulate symptoms everytime human starts experiencing them, and aggregate them at the end. Restart if `human` experiences them again.
        if len(human.all_reported_symptoms) > 0:
            self.symptoms_set['all'][human.name].update(human.all_reported_symptoms)
        else:
            if human.name in self.symptoms_set['all']:
                self.symptoms["all"] = _aggregate_symptoms_count(self.symptoms_set['all'][human.name], self.symptoms['all'])
                self.symptoms_set['all'].pop(human.name)

    def compute_symptom_prevalence(self):
        """
        Aggregates symptom statistics.
        """
        self.symptoms = self.track_symptoms(count_all = True)
        symptom_raw_counts, symptom_prevalence = {}, {}
        for key in ["all", "covid"]:
            total_incidence = self.symptoms[key]['n'] + 1e-6  # to avoid ZeroDivisionError
            symptom_raw_counts[key] = {'incidence':total_incidence}
            symptom_prevalence[key] = {}
            for s,v in self.symptoms[key].items():
                if s == 'n':
                    continue
                symptom_raw_counts[key][s] = v
                symptom_prevalence[key][s] = v / total_incidence

        return {
            "counts": symptom_raw_counts,
            "symptom_prevalence": symptom_prevalence
        }

    def get_estimated_covid_prevalence(self):
        """
        Uses observable statistic to compute prevalence of COVID.
        """
        past_14_days_hospital_cases = sum(self.hospitalization_per_day[:-14])
        past_14_days_tests = sum(self.tested_per_day[:-14])
        past_14_days_positive_test_results = sum(result['positive'] for date, result in self.test_results_per_day.items() if date <= self.env.timestamp.date())
        actual_cases = sum(h.is_exposed or h.is_infectious for h in self.city.humans)

        return {
            "cases": actual_cases / self.n_people,
            "estimation_by_hospitalization": past_14_days_hospital_cases / self.n_people,
            "estimation_by_test": past_14_days_positive_test_results / self.n_people,
        }

    def track_tested_results(self, human):
        """
        Keeps count of tests on a particular day. It is called every time someone is tested.
        NOTE: it is assumed to be called at the time of test, and due to the delay in time to result
        the function increments the count of tests on future dates.

        attributes used : self.test_monitor, self.test_results_per_day

        Args:
            human (Human): `Human` who got tested
            test_result (str): "positive" or "negative"
            test_type (str): type of test administered to `Human`
        """
        test_result_arrival_time = human.test_time + datetime.timedelta(days=human.time_to_test_result)
        test_result_arrival_date = test_result_arrival_time.date()
        self.test_results_per_day[test_result_arrival_date][human.hidden_test_result] += 1
        self.tested_per_day[-1] += 1
        self.test_monitor.append({
            "name": human.name,
            "symptoms": list(human.symptoms),
            "test_time": human.test_time,
            "result_time": test_result_arrival_time,
            "test_type": human.test_type,
            "test_result": human.hidden_test_result,
            "infection_timestamp": human.infection_timestamp,
            "infectiousness_onset_days": human.infectiousness_onset_days,
            "symptom_start_time": human.covid_symptom_start_time,
            "cold_timestamp": human.cold_timestamp,
            "flu_timestamp": human.flu_timestamp,
            "allegy_symptom_onset": human.allergy_timestamp
        })

    def compute_test_statistics(self, logfile=False):
        """

        """
        tests_per_human = Counter([m["name"] for m in self.test_monitor])
        max_tests_per_human = max(tests_per_human.values(), default=0)

        # percent of population tested
        n_tests = len(self.test_monitor)
        n_people = len(self.city.humans)
        percent_tested = 1.0 * n_tests/n_people
        daily_percent_test_results = [sum(x.values())/n_people for x in self.test_results_per_day.values()]

        # positivity rate
        n_positives = sum(x["positive"] for x in self.test_results_per_day.values())
        n_negatives = sum(x["negative"] for x in self.test_results_per_day.values())
        positivity_rate = n_positives/(n_positives + n_negatives + 1e-6)

        # symptoms | tests
        # count of humans who has symptom x given a test was administered
        test_given_symptoms_statistics = Counter([s for tm in self.test_monitor for s in tm["symptoms"]])
        positive_test_given_symptoms = Counter([s for tm in self.test_monitor for s in tm["symptoms"] if tm['test_result'] == "positive"])
        negative_test_given_symptoms = Counter([s for tm in self.test_monitor for s in tm["symptoms"] if tm['test_result'] == "negative"])

        # new infected - tests (per day)
        infected_minus_tests_per_day = [x - y for x,y in zip(self.e_per_day, self.tested_per_day)]

        log("\n######## COVID Testing Statistics #########", logfile)
        log(f"Positivity rate: {100*positivity_rate: 2.3f} %", logfile)
        log(f"Total Tests: {n_positives + n_negatives} Total positive tests: {n_positives} Total negative tests: {n_negatives}", logfile)
        log(f"Maximum tests given to an individual: {max_tests_per_human}", logfile)
        log(f"Proportion of population tested until end: {100 * percent_tested: 4.3f} %", logfile)
        log(f"Proportion of population tested daily Avg: {100 * np.mean(daily_percent_test_results): 4.3f} %", logfile)
        log(f"Proportion of population tested daily Max: {100 * max(daily_percent_test_results, default=0): 4.3f} %", logfile)
        log(f"Proportion of population tested daily Min: {100 * min(daily_percent_test_results, default=0): 4.3f} %", logfile)
        # log(f"infected - tests daily Avg: {np.mean(infected_minus_tests_per_day): 4.3f}", logfile)

        log(f"P(tested | symptoms = x), where x is ", logfile)
        for x in [EXTREMELY_SEVERE, SEVERE, MODERATE, MILD]:
            # total number of humans who has symptom x
            n_humans_who_experienced_symptom_x = self.symptoms["all"][x]
            if n_humans_who_experienced_symptom_x:
                p = min(test_given_symptoms_statistics[x]/n_humans_who_experienced_symptom_x, 1)
                log(f"    {x} {p: 3.3f}", logfile)

        test_given_symptoms_statistics = normalize_counter(test_given_symptoms_statistics, normalizer=n_tests)
        print_dict("P(symptoms = x | tested), where x is", test_given_symptoms_statistics, is_sorted="desc", logfile=logfile, top_k=10)
        # positive_test_given_symptoms = normalize_counter(positive_test_given_symptoms, normalizer=n_tests)
        # print_dict("P(symptoms = x | test is +), where x is", positive_test_given_symptoms, is_sorted="desc", logfile=logfile)

    @check_if_tracking
    def track_recovery(self, human):
        """
        Keeps record of attributes related to recovery like recovery timeperiod and infectious contacts during the disease.

        Args:
            human (covid19sim.human.Human): `human` who just recovered
        """
        self.recovery_stats['n_recovered'] += 1
        self.recovery_stats['timestamps'].append((self.env.timestamp, human.n_infectious_contacts))

    @check_if_tracking
    def track_mobility(self, current_activity, next_activity, human):
        """
        Aggregates information about mobility pattern of humans. Following information is being aggregated -
            1. transition_probabilities from one location type to another on weekends and weekdays
            2. fraction of day spent doing a certain activity on weekends or weekdays
            3. statistics on group size of "socialize" activities
            4. fraction of time "socialize" activity took place at a location_type

        Args:
            current_activity (covid19sim.utils.mobility_planner.Activity): [description]
            next_activity (covid19sim.utils.mobility_planner.Activity): [description]
            human (covid19sim.human.Human): human for which sleep schedule needs to be added.
        """
        if not (self.conf['track_all'] or self.conf['track_mobility']) or current_activity is None:
            return

        # economic model
        # if (
        #     current_activity.owner.age > 25
        #     and curernt_activity.name == "work"
        #     and current_activity.location.location_type == "WORKPLACE"
        # ):
        #     pass

        # forms a transition probability on weekdays and weekends
        type_of_day = ['weekday', 'weekend'][self.env.is_weekend]
        from_location = current_activity.location.location_type
        to_location = current_activity.location.location_type
        self.transition_probability[type_of_day][from_location][to_location] += 1

        # proportion of day spend in activity.name broken down by age groups
        name = current_activity.name if "supervised" not in current_activity.prepend_name else f"supervised-{current_activity.name}"
        n, avg = self.day_fraction_spent_activity[type_of_day][name]
        m = current_activity.duration / SECONDS_PER_DAY
        self.day_fraction_spent_activity[type_of_day][name] = (n+1, (n*avg + m)/(n+1))

        # histogram of number of people with whom socialization happens
        if next_activity.name == "socialize":
            self.socialize_activity_data['group_size'].push(len(next_activity.rsvp))
            self.socialize_activity_data["location_frequency"][next_activity.location.location_type] += 1
            self.socialize_activity_data["start_time"].push(_get_seconds_since_midnight(next_activity.start_time))

        # if "-cancel-" not in next_activity.append_name and next_activity.prepend_name == "":
        #     self.start_time_activity[next_activity.name].push(_get_seconds_since_midnight(next_activity.start_time))

        self.activity_attributes["end_time"][next_activity.name].push(_get_seconds_since_midnight(next_activity.end_time))
        self.activity_attributes["duration"][next_activity.name].push(next_activity.duration)

    @check_if_tracking
    def track_mixing(self, human1, human2, duration, distance_profile, timestamp, location, interaction_type, contact_condition, global_mbility_factor):
        """
        Stores counts and statistics to generate various aspects of contacts and social mixing.
        Following are being tracked -
            1. (symmetric matrix) Mean daily contacts between two age groups broken down by location and type of interaction ("known", "all", "within_contact_condition")
            2. (asymmetric matrix) Mean daily contacts per person in two age groups (asymmetric) broken down by location and type of interaction ("known", "all", "within_contact_condition")
            3. (symmetric matrix) Mean duration of contacts between two age groups (symmetric) broken down by location and type of interaction ("known", "all", "within_contact_condition")
            4. (asymmetric matrix) Mean duration of contacts per person in two age groups (asymmetric) broken down by location and type of interaction ("known", "all", "within_contact_condition")
            5. (1D array) Mean daily contacts per age group on weekdays and weekends for each location type (only for "known" contacts)
            6. (1D array) Mean daily contact duration per age group on weekdays and weekends for each location type (only for "known" contacts)
            7. (scalar) Mean daily contact per person per day for each location type (only for "known" contacts)
            8. (scalar) Mean daily contact duration per person per day for each location type(only for "known" contacts)
            9. (histogram) counts of encounter distance in bins of 10 cms for each location type and interaction type ("known", "all", "within_contact_condition")
            10.(histogram) counts of encounter duration in bins of 1 min for each location type and interaction type ("known", "all", "within_contact_condition")

        NOTE: These values are aggregated every simulation day. At the end of the simulation one needs to call this function with all arguments as None to perform updates.

        Args:
            human1 (covid19sim.human.Human): one of the two `human`s involved in the encounter.
            human2 (covid19sim.human.Human): other of the two `human`s involved in the encounter
            duration (float): time duration got which this encounter took place (seconds)
            distance_profile (covid19sim.locations.location.DistanceProfile): distance from which these two humans met (cms)
            timestamp (datetime.datetime): timestamp at which this event took place
            location (Location): `Location` where this encounter took place
            interaction_type (string): type of interaction i.e. "known" or "unknown". We keep track of "known" contacts and "all" contacts (that combines "known" and "unknown")
            contact_condition (bool): whether the encounter was within the contact conditions for infection to happen
            global_mobility_factor (bool): A globally influenced factor that along with contact_condition determines infection when there is a chance of one.
        """
        if not (self.conf['track_all'] or self.conf['track_mixing']):
            return

        update_only = False
        if human1 is None or human2 is None:
            update_only = True

        day = timestamp.date()
        last_day = self.last_day['social_mixing']
        # compute averages and reset the values
        if  last_day != day or update_only:

            # everything related to contact matrices
            for interaction_type in self.contact_matrices.keys():
                for location_type in LOCATION_TYPES_TO_TRACK_MIXING:
                    C = self.contact_matrices[interaction_type][location_type]
                    D = self.contact_duration_matrices[interaction_type][location_type]

                    # number of contacts per age group
                    # mean daily contacts per age group (symmetric matrix)
                    n, M = C['avg_daily']
                    C['avg_daily'] = (n+1, (n*M + C['total'])/(n+1))

                    # /!\ Storing number of people per age group might lead to memory problems. It might not be sutiable for larger simulations.
                    # mean daily contacts per person in an age group (similar to survey matrices)
                    n, M = C['unique_avg_daily']
                    n_people = np.zeros_like(M)
                    for i, j in C['n_people'].keys():
                        n_people[i, j] = len(C['n_people'][(i,j)])

                    # number of unique people age group i met in a day = n_people[i, :].sum()
                    # number of unique people in age group i = n_people[:, i].sum()
                    # number of unique people age group i met in an age group j = n_people[i, j] (NOTE: the difference wrt C['n_people'])
                    # we follow the same convention of i,j==> j reporting about i so we take transpose here.
                    C['unique_avg_daily'] = (n+1, (n*M + n_people.transpose()) / (n+1))

                    n, M = C['avg']
                    m = np.zeros_like(C['total'])
                    np.divide(C['total'], n_people, where=(n_people!=0), out=m)
                    C['avg'] = (n+1, (n*M + m)/(n+1))

                    # mean duration per contact (minutes) (symmetric matrix)
                    n, M = D['avg_daily']
                    D['avg_daily'] = (n+1, (n*M + D['total'])/(n+1))

                    # mean duration per contact per person (minutes) (similar to survey matrices)
                    n, M = D['avg']
                    m = np.zeros_like(D['total'])
                    np.divide(D['total'], n_people, where=(n_people!=0), out=m)
                    D['avg'] = (n+1, (n*M + m)/(n+1))

                    # mean dailies per age group or otherwise for "known" contacts only. This is what is available via surveys.
                    if interaction_type == "known":
                        # mean weekday and weekend contact/contact duration per age group is only recorded for "known" contacts.
                        type_of_day = ['weekday', 'weekend'][last_day.weekday() >= 5]
                        n, Da = self.mean_daily_contact_duration_per_agegroup[type_of_day][location_type]
                        n, Ca = self.mean_daily_contacts_per_agegroup[type_of_day][location_type]

                        ## total for yesterday's contacts per age group
                        contacts_per_agegroup = C["total"].sum(axis=0)
                        contact_duration_per_agegroup = D["total"].sum(axis=0)
                        n_unique_people_per_agegroup = n_people.sum(axis=0)

                        # update averages
                        m = np.zeros_like(contacts_per_agegroup)
                        np.divide(contacts_per_agegroup, n_unique_people_per_agegroup, where=(n_unique_people_per_agegroup!=0), out=m)
                        self.mean_daily_contacts_per_agegroup[type_of_day][location_type] = (n+1, (n * Ca + m)/(n+1))

                        m = np.zeros_like(contact_duration_per_agegroup)
                        np.divide(contact_duration_per_agegroup, n_unique_people_per_agegroup, where=(n_unique_people_per_agegroup!=0), out=m)
                        self.mean_daily_contact_duration_per_agegroup[type_of_day][location_type] = (n+1, (n * Da + m)/(n+1))

                        # mean daily contacts
                        n, d = self.mean_daily_contact_duration[type_of_day][location_type]
                        n, c = self.mean_daily_contacts[type_of_day][location_type]
                        total_contacts = C["total"].sum() / 2
                        total_contact_duration = D["total"].sum() / 2
                        total_unique_people = n_people.sum() / 2

                        m = total_contact_duration / total_unique_people if total_unique_people else 0.0
                        self.mean_daily_contact_duration[type_of_day][location_type] = (n+1, (d*n + m) / (n+1))

                        m = total_contacts / total_unique_people if total_unique_people else 0.0
                        self.mean_daily_contacts[type_of_day][location_type] = (n+1, (c*n + m) / (n+1))

                    # reset the matrices for counting the next day's events
                    C['total'] = np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))
                    C['n_people'] = defaultdict(lambda : set())
                    D['total'] = np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))

            self.outside_daily_contacts.append(1.0 * self.n_outside_daily_contacts/len(self.city.humans))
            self.n_outside_daily_contacts = 0
            self.last_day['social_mixing'] = day

        if update_only:
            return

        # record the values
        human1_type_of_place = _get_location_type_to_track_mixing(human1, location)
        human2_type_of_place = _get_location_type_to_track_mixing(human2, location)

        i = human1.age_bin_width_5.index
        j = human2.age_bin_width_5.index

        # everything related to the contact matrices is recorded here
        interaction_types = ["all"]
        if interaction_type == "known":
            interaction_types.append("known")
        if contact_condition:
            interaction_types.append("within_contact_condition")
            self.track_contact_attributes(human1, human2)

            # outside daily contacts
            self.n_outside_daily_contacts += 0.5 if human1_type_of_place != "HOUSEHOLD" else 0
            self.n_outside_daily_contacts += 0.5 if human2_type_of_place != "HOUSEHOLD" else 0


        for interaction_type in interaction_types:
            for location_type in ['all', human1_type_of_place, human2_type_of_place]:
                C = self.contact_matrices[interaction_type][location_type]
                D = self.contact_duration_matrices[interaction_type][location_type]

                C['total'][i, j] += 1
                C['total'][j, i] += 1
                # in surveys, ij ==> j is the participant and i is the reported contact
                C['n_people'][(j,i)].add(human1.name) # i reports about j; len of this set is number of unique people in i that met j
                C['n_people'][(i,j)].add(human2.name) # j reports about i
                D['total'][i,j] += duration / SECONDS_PER_MINUTE
                D['total'][j,i] += duration / SECONDS_PER_MINUTE

                # Note that the use of math.ceil makes the upper limit inclusive.
                # record distance profile/frequency counts (per 10 cms). It keeps counts for (distance_category - 10,  distance_category].
                distance_category = 10 * math.ceil(distance_profile.distance / 10)
                Cp = self.contact_distance_profile[interaction_type][location_type]
                Cp[distance_category] = Cp.get(distance_category, 0) + 1

                # record duration profile/frequency counts (per 1 min)
                duration_category = math.ceil(duration / SECONDS_PER_MINUTE)
                Dp = self.contact_duration_profile[interaction_type][location_type]
                Dp[duration_category] = Dp.get(duration_category, 0) + 1

    @check_if_tracking
    def track_contact_attributes(self, human1, human2, infection_attrs=[]):
        """
        Keeps record of contacts among susceptibles, exposed, infectious, and recovered.

        Args:
            human1 (covid19sim.human.Human): one of the two humans involved in the contact
            human2 (covid19sim.human.Human): other human
            infection_attrs (list): elements are attributes of infection
        """
        assert human1 is not None and human2 is not None, f"not a contact since one of {human1} or {human2} is None"

        state1 = human1.state.index(1)
        state2 = human2.state.index(1)
        self.contact_attributes["SEIR"][state1, state2] += 1
        self.contact_attributes["SEIR"][state2, state1] += 1

        # snapshot
        # TODO
        if len(infection_attrs) > 0:
            success, p_infection, viral_load, infectiousness = infection_attrs
            self.contact_attributes['SI']['p_infection'] += p_infection
            self.contact_attributes['SI']['viral_load'] += viral_load
            self.contact_attributes['SI']['infectiousness'] += infectiousness
            self.contact_attributes['SI']['infection'] += success
            self.contact_attributes['SI']['total'] += 1

    @check_if_tracking
    def track_bluetooth_communications(self, human1, human2, location, timestamp):
        """
        Keeps track of mean daily unique bluetooth encounters between two age groups.
        It is used to visualize the subset of contacts that are captured by bluetooth communication.
        Compare it with `self.contact_matrices["all"][location_type]["avg_daily"]`.

        Args:
            human1 (covid19sim.human.Human): one of the two `human`s involved in an bluetooth exchange
            human2 (covid19sim.human.Human): other of the two `human`s involved in an bluetooth exchange
            timestamp (datetime.datetime): timestamp when this exchange took place
        """
        if not (self.conf['track_all'] or self.conf['track_bluetooth_communications']):
            return

        assert human1.has_app and human2.has_app, "tracking bluetooth communications for human who doesn't have an app"

        update_only = False
        if human1 is None or human2 is None:
            update_only = True

        day = timestamp.date()
        # compute average and reset
        if self.last_day['bluetooth_communications']  != day or update_only:
            for location_type in LOCATION_TYPES_TO_TRACK_MIXING:
                B = self.bluetooth_contact_matrices[location_type]
                n, M = B['avg_daily']
                B['avg_daily'] = (n+1, (n*M + B['total'])/(n+1))

                # reset values
                B['total'] = np.zeros((len(AGE_BIN_WIDTH_5), len(AGE_BIN_WIDTH_5)))
            self.last_day['bluetooth_communications'] = day

        if update_only:
            return

        # record the new values
        type_of_place = _get_location_type_to_track_mixing(human1, location)
        i = human1.age_bin_width_5.index
        j = human2.age_bin_width_5.index
        for location_type in ['all', type_of_place]:
            B = self.bluetooth_contact_matrices[location_type]
            B['total'][i, j] += 1
            B['total'][j, i] += 1

    def get_contact_data(self):
        """
        Removes references to `Human` objects from all the contact related dictionaries.

        Returns:
            (dict): keys are names of the metrics, and values are the cleaned metrics
        """
        # contact matrices
        cm = {}
        for key0, value0 in self.contact_matrices.items():
            cm[key0] = {}
            for key1, value1 in value0.items():
                value1.pop("n_people")
                value1.pop("total")
                cm[key0][key1] = value1

        # contact duration matrices
        cdm = {}
        for key0, value0 in self.contact_duration_matrices.items():
            cdm[key0] = {}
            for key1, value1 in value0.items():
                value1.pop("total")
                cdm[key0][key1] = value1

        # bluetooth contact matrices
        bcm = {}
        for key0, value0 in self.bluetooth_contact_matrices.items():
            value0.pop("total")
            bcm[key0] = value0['avg_daily'][1]

        return {
            "contact_matrices": cm,
            "contact_duration_matrices": cdm,
            "bluetooth_contact_matrices": bcm,
            "mean_daily_contacts_per_agegroup":self.mean_daily_contacts_per_agegroup,
            "mean_daily_contact_duration_per_agegroup": self.mean_daily_contact_duration_per_agegroup,
            "mean_daily_contacts": self.mean_daily_contacts,
            "mean_daily_contact_duration": self.mean_daily_contact_duration,
            "contact_distance_profile": self.contact_distance_profile,
            "contact_duration_profile": self.contact_duration_profile,
            "contact_attributes": self.contact_attributes
        }

    def get_infectious_contact_data(self):
        """
        Removes weakref from data collected in `track_infection`.

        Returns:
            (dict): keys are names of metrics and values are tracked/computed metrics
        """
        aic = {}
        for key, val in self.average_infectious_contacts.items():
            if key not in aic:
                aic[key] = {}
            aic[key]['infection_count'] = val['infection_count']
            aic[key]['unique_humans'] = len(val['humans'])

        # add a final NGM
        percent_infected = 100 * sum(self.cases_per_day) / self.n_people
        ngm = _compute_ngm(self.next_generation_matrix)
        self.next_generation_matrix['snapshots'].append((percent_infected, ngm))

        return {
            "human_human_infection_matrix": self.human_human_infection_matrix,
            "environment_human_infection_histogram": self.environment_human_infection_histogram,
            "average_infectious_contacts": aic,
            "p_infection_at_contact_human": self.p_infection_at_contact["human"],
            "p_infection_at_contact_environment": self.p_infection_at_contact["environment"],
            "next_generation_matrix_snapshots": self.next_generation_matrix['snapshots']
        }

    def compute_probability_of_transmission(self):
        """
        If X interactions qualify as close contact and Y of them resulted in an infection,
        probability of transmission is defined as Y/X

        Returns:
            (float): probability of transmission
        """
        all_human_contacts = self.human_human_infection_matrix["all"]["risky_contact"].sum()
        all_infectious_contacts = self.human_human_infection_matrix["all"]["caused_infection"].sum()
        return all_infectious_contacts/ (all_human_contacts + 1e-6) # to avoid ZeroDivisionError

    def compute_effective_contacts(self, since_intervention=True):
        """
        Effective contacts are those that qualify to be within contact_condition in `Human.at`.
        These are major candidates for infectious contacts.
        """
        all_effective_contacts = 0
        all_healthy_effective_contacts = 0
        all_healthy_days = 0
        all_contacts = 0
        for human in self.city.humans:
            all_effective_contacts += human.effective_contacts
            all_healthy_effective_contacts += human.healthy_effective_contacts
            all_healthy_days += human.healthy_days
            all_contacts += human.num_contacts

        days = self.conf['simulation_days']
        if since_intervention and self.conf['INTERVENTION_DAY'] >= 0 :
            days = self.conf['simulation_days'] - self.conf['INTERVENTION_DAY']

        # recover GLOBAL_MOBILITY_SCALING_FACTOR
        scale_factor = 0 if all_contacts == 0 else all_effective_contacts / all_contacts

        return all_effective_contacts / (days * self.n_people), all_healthy_effective_contacts / all_healthy_days, scale_factor

    def write_metrics(self):
        """
        Writes various metrics to `self.logfile`. Prints them if `self.logfile` is None.
        """

        SIMULATION_GENERATION_TIME = self.compute_generation_time()
        SIMULATION_SERIAL_INTERVAL = self.compute_serial_interval()

        log("\n######## COVID PROPERTIES #########", self.logfile)
        log(f"Avg. incubation days {self.covid_properties['incubation_days'][1]: 5.2f}", self.logfile)
        log(f"Avg. recovery days {self.covid_properties['recovery_days'][1]: 5.2f}", self.logfile)
        log(f"Avg. infectiousnes onset days {self.covid_properties['infectiousness_onset_days'][1]: 5.2f}", self.logfile)

        log("\n######## COVID SPREAD #########", self.logfile)
        all_human_transmissions = self.human_human_infection_matrix["all"]["caused_infection"].sum()
        all_environmental_transmissions = self.environment_human_infection_histogram["all"]["caused_infection"].sum()
        total_transmissions = all_human_transmissions + all_environmental_transmissions + 1e-6 # to prevent ZeroDivisionError
        percentage_environmental_transmission = 100 * all_environmental_transmissions / total_transmissions
        R0 = self.average_infectious_contacts["all"]['infection_count'] / (len(self.average_infectious_contacts["all"]['humans']) + 1e-6)

        log(f"# human-human transmissions {all_human_transmissions}", self.logfile )
        log(f"# environment-human transmissions {all_environmental_transmissions}", self.logfile )
        log(f"environmental transmission ratio {percentage_environmental_transmission:2.3f} %", self.logfile )
        log(f"Average generation time {SIMULATION_GENERATION_TIME} ", self.logfile)
        log(f"Average serial interval {SIMULATION_SERIAL_INTERVAL} ", self.logfile)
        log(f"Empirical Ro {R0: 2.3f} (WARNING: It is an underestimate because it doesn't consider all infectious contacts during the recovery period of infected humans towards the end of the simulation) ", self.logfile)

        log("\n******** Symptomaticity and Disease Spread *********", self.logfile)
        total = sum(self.average_infectious_contacts[x]['infection_count'] for x in ['symptomatic','presymptomatic', 'asymptomatic'])
        assert total == all_human_transmissions, "human-human transmission do not match with transmissions broken down by symptomaticity"
        total += 1e-6 # to avoid ZeroDivisionError

        log("\nR0 ( % Transmission ) of all human-human transmission", self.logfile)
        for key in ["asymptomatic", "presymptomatic", "symptomatic"]:
            count = self.average_infectious_contacts[key]["infection_count"]
            n_humans = len(self.average_infectious_contacts[key]["humans"]) + 1e-6 # to avoid ZeroDivisionError
            log(f"* {key} R0 {count/n_humans: 2.3f} ({100 * count / total: 2.3f} %)", self.logfile)

        log("\n******** Locations and Disease Spread *********", self.logfile)

        log("\nR0 ( % Transmission ) of all human-human transmission", self.logfile)
        all_locations = [x for x in self.average_infectious_contacts.keys() if x not in ['asymptomatic', 'presymptomatic', 'symptomatic', "all"]]
        for key in all_locations:
            count = self.average_infectious_contacts[key]["infection_count"]
            n_humans = len(self.average_infectious_contacts[key]["humans"]) + 1e-6 # to avoid ZeroDivisionError
            log(f"* {key} R0 {count/n_humans: 2.3f} ({100 * count / total: 2.3f} %)", self.logfile)

        log("\n% Transmission of all environmental transmissions", self.logfile)
        total = all_environmental_transmissions + 1e-6 # to avoid ZeroDivisionError
        all_locations = self.environment_human_infection_histogram.keys()
        for key in all_locations:
            x = self.environment_human_infection_histogram[key]['caused_infection'].sum()
            log(f"* % {key} transmission {100 * x / total :2.3f} %", self.logfile)

        log("\n######## SYMPTOMS #########", self.logfile)
        x = self.compute_symptom_prevalence()['symptom_prevalence']

        TITLE = "P(symptoms = x | covid patient), where x is"
        print_dict(TITLE, x['covid'], is_sorted="desc", top_k=10, logfile=self.logfile)

        TITLE = "P(symptoms = x | human had some sickness e.g. cold, flu, allergies, covid), where x is"
        print_dict(TITLE, x['all'], is_sorted="desc", top_k=10, logfile=self.logfile)

        log("\n######## CONTACT PATTERNS #########", self.logfile)
        if not (self.conf['track_all'] or self.conf['track_mixing']):
            log(f"CAUTION: NOT TRACKED", self.logfile)

        str_to_print = "weekday - "
        for location_type in LOCATION_TYPES_TO_TRACK_MIXING:
            x = self.mean_daily_contacts["weekday"][location_type][1]
            str_to_print += f"| {location_type}: {x:2.3f}"
        log(str_to_print, self.logfile)

        str_to_print = "weekend - "
        for location_type in LOCATION_TYPES_TO_TRACK_MIXING:
            x = self.mean_daily_contacts["weekend"][location_type][1]
            str_to_print += f"| {location_type}: {x:2.3f}"
        log(str_to_print, self.logfile)

        log("\n######## MOBILITY STATISTICS #########", self.logfile)
        if not (self.conf['track_all'] or self.conf['track_mobility']):
            log(f"CAUTION: NOT TRACKED", self.logfile)

        activities = ["work", "socialize", "grocery", "exercise", "idle", "sleep"]

        log("Proportion of day spent in activities - ", self.logfile)
        # unsupervised
        log("\nUnsupervised activities - ", self.logfile)
        for type_of_day in ["weekday", "weekend"]:
            str_to_print = f"{type_of_day} - "
            for activity in activities:
                x = self.day_fraction_spent_activity[type_of_day][activity][1]
                str_to_print += f"| {activity}: {x:2.3f}"
            log(str_to_print, self.logfile)

        # supervised
        log(f"\nSupervised activities - ", self.logfile)
        for type_of_day in ["weekday", "weekend"]:
            str_to_print = f"{type_of_day} - "
            for activity in activities:
                x = self.day_fraction_spent_activity[type_of_day][f"supervised-{activity}"][1]
                str_to_print += f"| {activity}: {x:2.3f}"
            log(str_to_print, self.logfile)

        log("\nSocial groups -", self.logfile)
        str_to_print = "size - "
        group_sizes = self.socialize_activity_data['group_size']
        str_to_print += f"mean: {group_sizes.mean():2.2f} | "
        str_to_print += f"std: {group_sizes.stddev(): 2.2f} | "

        min_size = 0 if len(group_sizes) == 0 else group_sizes.minimum()
        str_to_print += f"min: {min_size: 2.2f} | "

        max_size = 0 if len(group_sizes) == 0 else group_sizes.maximum()
        str_to_print += f"max: {max_size: 2.2f} | "
        log(str_to_print, self.logfile)

        str_to_print = "location - "
        locations = self.socialize_activity_data["location_frequency"].keys()
        total = sum(self.socialize_activity_data["location_frequency"].values()) + 1e-6 # to avoid ZeroDivisionError
        str_to_print += f"total visits {total} | "
        for location in locations:
            m = self.socialize_activity_data["location_frequency"][location]
            str_to_print += f"{location}: {m} {100*m/total:2.2f} % | "
        log(str_to_print, self.logfile)

        str_to_print = "Social network properties (degree statistics) - "
        degrees = np.array([len(h.known_connections) for h in self.city.humans])
        str_to_print += f"mean {np.mean(degrees): 2.2f} | "
        str_to_print += f"std. {np.std(degrees): 2.2f} | "
        str_to_print += f"min {min(degrees): 2.2f} | "
        str_to_print += f"max {max(degrees): 2.2f} | "
        str_to_print += f"median {np.median(degrees): 2.2f}"
        log(str_to_print, self.logfile)

        # start time of activities
        for attr in ["end_time", "duration"]:
            str_to_print = f"\n{attr} - "
            log(str_to_print, self.logfile)
            for type_of_activty, metrics in self.activity_attributes[attr].items():
                str_to_print = f"{type_of_activty} - "
                str_to_print += f"mean: {metrics.mean()/SECONDS_PER_HOUR: 2.2f} | "
                str_to_print += f"std: {metrics.stddev()/SECONDS_PER_HOUR: 2.2f} | "

                min_value = 0 if len(group_sizes) == 0 else metrics.minimum()
                str_to_print += f"min: {min_value / SECONDS_PER_HOUR : 2.2f} | "

                max_value = 0 if len(group_sizes) == 0 else metrics.maximum()
                str_to_print += f"max: {max_value / SECONDS_PER_HOUR : 2.2f} | "

                log(str_to_print, self.logfile)

        self.compute_test_statistics(self.logfile)

        log("\n######## Effective Contacts & % infected #########", self.logfile)
        p_infected = 100 * sum(self.cases_per_day) / len(self.city.humans)
        effective_contacts, healthy_effective_contacts, scale_factor = self.compute_effective_contacts()
        p_transmission = self.compute_probability_of_transmission()
        log(f"Eff. contacts: {effective_contacts:5.3f} \t Healthy Eff. Contacts {healthy_effective_contacts:5.3f} \th % infected: {p_infected: 2.3f}%", self.logfile)
        if scale_factor:
            log(f"effective contacts per contacts (GLOBAL_MOBILITY_SCALING_FACTOR): {scale_factor}", self.logfile)
        log(f"Probability of transmission: {p_transmission:2.3f}", self.logfile)
        log(f"Serial interval: {SIMULATION_SERIAL_INTERVAL: 5.3f}", self.logfile)

        log("\n######## Bayesian Estimates of Rt #########", self.logfile)
        cases_per_day = self.cases_per_day
        serial_interval = SIMULATION_GENERATION_TIME # generation time is used in simulations
        if serial_interval == 0:
            serial_interval = 7.0
            log("WARNING: serial_interval is 0", self.logfile)
        log(f"using serial interval :{serial_interval}", self.logfile)

        plotrt = PlotRt(R_T_MAX=4, sigma=0.25, GAMMA=1.0 / serial_interval)
        most_likely, _ = plotrt.compute(cases_per_day, r0_estimate=2.5)
        log(f"Rt: {most_likely[:20]}", self.logfile)

    def write_for_training(self, humans, outfile, conf):
        """ Writes some data out for the ML predictor """
        data = dict()
        data['hospitalization_per_day'] = self.hospitalization_per_day

        # parse test results
        data['positive_test_results_per_day'] = []
        data['negative_test_results_per_day'] = []
        for d in self.test_results_per_day.values():
            data['positive_test_results_per_day'].append(d['positive'])
            data['negative_test_results_per_day'].append(d['negative'])

        data['tested_per_day'] = self.tested_per_day
        data['i_per_day'] = self.i_per_day
        data['adoption_rate'] = self.adoption_rate
        data['lab_test_capacity'] = conf['TEST_TYPES']['lab']['capacity']
        data['n_people'] = conf['n_people']

        data['humans'] = {}
        for human in humans:
            humans_data = {}
            humans_data['viral_load_plateau_start'] = human.viral_load_plateau_start
            humans_data['viral_load_plateau_height'] = human.viral_load_plateau_height
            humans_data['viral_load_plateau_end'] = human.viral_load_plateau_end
            humans_data['viral_load_peak_start'] = human.viral_load_peak_start
            humans_data['viral_load_peak_height'] = human.viral_load_peak_height
            humans_data['viral_load_plateau_end'] = human.viral_load_plateau_end
            humans_data['incubation_days'] = human.incubation_days
            humans_data['recovery_days'] = human.recovery_days
            data['humans'][human.name] = humans_data
        with open(outfile, 'wb') as f:
            dill.dump(data, f)
