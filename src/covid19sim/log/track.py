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
from covid19sim.utils.constants import SECONDS_PER_DAY
if typing.TYPE_CHECKING:
    from covid19sim.human import Human


def print_dict(title, dic, is_sorted=None, top_k=None, logfile=None):
    if not is_sorted:
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

def get_nested_dict(nesting):
    """
    [summary]

    Args:
        nesting ([type]): [description]

    Returns:
        [type]: [description]
    """
    if nesting == 1:
        return defaultdict(int)
    elif nesting == 2:
        return defaultdict(lambda : defaultdict(int))
    elif nesting == 3:
        return defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    elif nesting == 4:
        return defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(int))))


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

        # filename to store intermediate results; useful for bigger simulations;
        timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if city.conf.get("INTERVENTION_DAY") == -1:
            name = "unmitigated"
        else:
            name = city.conf.get("RISK_MODEL")
        self.filename = f"tracker_data_n_{city.n_people}_{timenow}_{name}.pkl"
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

        self.contacts = {
                'human_infection': np.zeros((150,150)),
                'env_infection':get_nested_dict(1),
                'location_env_infection': get_nested_dict(2),
                'location_human_infection': defaultdict(lambda: np.zeros((150,150))),
                }

        # mobility
        self.n_outside_daily_contacts = 0
        self.transition_probability = {
                                    "weekday":get_nested_dict(2),
                                    "weekend":get_nested_dict(2)
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

        # infection
        self.p_infection = []
        self.infection_graph = nx.DiGraph()
        self.humans_state = defaultdict(list)
        self.humans_rec_level = defaultdict(list)
        self.humans_intervention_level = defaultdict(list)

        # R0 and Generation times
        self.avg_infectious_duration = 0
        self.n_recovery = 0
        self.n_infectious_contacts = 0
        self.n_contacts = 0
        self.serial_intervals = []
        self.serial_interval_book_to = defaultdict(dict)
        self.serial_interval_book_from = defaultdict(dict)
        self.n_env_infection = 0
        self.recovered_stats = []
        self.covid_properties = defaultdict(lambda : [0,0])

        # cumulative incidence
        self.cumulative_incidence = []
        self.cases_per_day = [0]
        self.r_0 = defaultdict(lambda : {'infection_count':0, 'humans':set()})
        self.r = []

        # testing & hospitalization
        self.hospitalization_per_day = [0]
        self.critical_per_day = [0]
        self.deaths_per_day = [0]
        self.test_results_per_day = defaultdict(lambda :{'positive':0, 'negative':0})
        self.tested_per_day = [0]

        # demographics
        self.age_bins = [(x[0], x[1]) for x in sorted(self.conf.get("P_AGE_REGION"), key = lambda x:x[0])]
        self.n_people = self.n_humans = self.city.n_people
        self.human_has_app = None
        self.adoption_rate = 0.0

        # symptoms
        self.symptoms = {'covid': defaultdict(int), 'all':defaultdict(int)}
        self.symptoms_set = {'covid': defaultdict(set), 'all': defaultdict(set)}


        # risk model
        self.ei_per_day = []
        self.risk_values = []
        self.avg_infectiousness_per_day = []
        self.risk_attributes = []

        # monitors
        self.human_monitor = {}
        self.infection_monitor = []
        self.test_monitor = []

        # update messages
        self.infector_infectee_update_messages = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda :{'unknown':{}, 'contact':{}})))
        self.to_human_max_msg_per_day = defaultdict(lambda : defaultdict(lambda :-1))
        self.human_has_app = set()

    def initialize(self):
        self.s_per_day = [sum(h.is_susceptible for h in self.city.humans)]
        self.e_per_day = [sum(h.is_exposed for h in self.city.humans)]
        self.i_per_day = [sum(h.is_infectious for h in self.city.humans)]
        self.r_per_day = [sum(h.is_removed for h in self.city.humans)]

        M, G, B, O, R, EM = self.compute_mobility()
        self.recommended_levels_daily = [[G, B, O, R]]
        self.mobility = [M]
        self.expected_mobility = [EM]
        self.summarize_population()

        # risk model
        self.risk_precision_daily = [self.compute_risk_precision()]
        self.init_infected = [human for human in self.city.humans if human.is_exposed]
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
        log(f"PREFERENTIAL_ATTACHMENT_FACTOR: {self.conf['PREFERENTIAL_ATTACHMENT_FACTOR']}", self.logfile)
        log(f"P_HOUSE_OVER_MISC_FOR_SOCIALS: {self.conf['P_HOUSE_OVER_MISC_FOR_SOCIALS']}", self.logfile)
        log(f"CONTAGION_KNOB: {self.conf['CONTAGION_KNOB']}", self.logfile)
        log(f"ENVIRONMENTAL_INFECTION_KNOB: {self.conf['ENVIRONMENTAL_INFECTION_KNOB']}", self.logfile)
        log(f"TIME_SPENT_SCALE_FACTOR_FOR_SHORT_ACTIVITIES: {self.conf['TIME_SPENT_SCALE_FACTOR_FOR_SHORT_ACTIVITIES']}", self.logfile)
        log(f"TIME_SPENT_SCALE_FACTOR_FOR_WORK: {self.conf['TIME_SPENT_SCALE_FACTOR_FOR_WORK']}", self.logfile)
        log(f"TIME_SPENT_SCALE_FACTOR_SLEEP_AWAKE: {self.conf['TIME_SPENT_SCALE_FACTOR_SLEEP_AWAKE']}", self.logfile)


        log("\n######## DEMOGRAPHICS / SYNTHETIC POPULATION #########", self.logfile)
        log(f"NB: (i) census numbers are in brackets. (ii) {WARN_SIGNAL} marks a {WARN_RELATIVE_PERCENTAGE_THRESHOLD}% realtive deviation from census\n", self.logfile)
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
            str_to_print += f"{z}: {p:2.3f}% | "
        log(str_to_print, self.logfile)

        ###### house initialization
        log("\n*** House allocation *** ", self.logfile)
        # senior residencies
        self.n_senior_residency_residents = sum(len(sr.residents) for sr in self.city.senior_residences)
        p = 100*self.n_senior_residency_residents / self.n_people
        cns = 100*self.conf['P_COLLECTIVE']
        warn = WARN_SIGNAL if 100*abs(p-cns)/cns > WARN_RELATIVE_PERCENTAGE_THRESHOLD else ""
        log(f"{warn}Total (%) number of residents in senior residencies (census): {self.n_senior_residency_residents} ({p:2.2f}%) ({cns:2.2f})", self.logfile)

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
            str_to_print += f"{warn} {z}: {p:2.2f}% ({cns: 3.2f}) | "
        log(str_to_print, self.logfile)

        # solo dwellers
        estimated_solo_dwellers_mean_age = sum([x[2] * (x[0] + x[1]) / 2 for x in self.conf['P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1']])
        simulated_solo_dwellers_age_given_housesize1 = [[x[0], x[1], 0] for x in self.conf['P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1']]
        n_solo_houses = len(solo_ages)
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
        str_to_print += f"{warn}multi-generation: {p:2.2f}% ({cns:2.2f}) | "
        p = 100 * only_adults.mean()
        str_to_print += f"Only adults: {p:2.2f}% | "
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
            str_to_print += f"{warn}{atype}: {p:2.3f}%  ({cns: 2.2f})| "
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

        self.n_infected_init = self.city.n_init_infected
        log(f"Total number of infected humans {self.n_infected_init}", self.logfile)
        for human in self.city.humans:
            if human.is_exposed:
                log(f"\t{human} @ {human.household} living with {len(human.household.residents) - 1} other residents", self.logfile)

    def get_R(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        # https://web.stanford.edu/~jhj1/teachingdocs/Jones-on-R0.pdf; vlaid over a long time horizon
        # average infectious contacts (transmission) * average number of contacts * average duration of infection
        time_since_start =  (self.env.now - self.env.ts_initial) / SECONDS_PER_DAY # DAYS
        if time_since_start == 0:
            return -1

        if time_since_start > 365:
            # tau= self.n_infectious_contacts / self.n_contacts
            # c_bar = self.n_contacts / time_since_start
            tau_times_c_bar = self.n_infectious_contacts / time_since_start
            d = self.avg_infectious_duration
            return tau_times_c_bar * d
        else:
            # x = [h.n_infectious_contacts for h in self.city.humans if h.n_infectious_contacts > 0]
            if self.recovered_stats:
                n, total = zip(*self.recovered_stats)
            else:
                n, total = [0], [0]

            if sum(n):
                return 1.0 * sum(total)/sum(n)
            return 0

    def get_R0(self, logfile=None):
        """
        [summary]

        Args:
            logfile ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if len(self.r) > 0:
            for idx,x in enumerate(self.r):
                if x >0:
                    return np.mean(self.r[idx:idx+5])
        else:
            log("not enough data points to estimate r0. Falling back to average")
            return self.get_R()

    def get_generation_time(self):
        """
        Generation time is the time from exposure day until an infection occurs.
        """
        times = []
        for x in self.infection_monitor:
            if x['from']:
                times.append((x['infection_timestamp'] - x['from_infection_timestamp']).total_seconds() / SECONDS_PER_DAY)

        return np.mean(times)

    def get_serial_interval(self):
        """
        Returns serial interval.
        For description of serial interval, refer self.track_serial_interval

        Returns:
            float: serial interval
        """
        return np.mean(self.serial_intervals)

    def track_serial_interval(self, human_name):
        """
        tracks serial interval ("time duration between a primary case-patient (infector) having symptom onset and a secondary case-patient (infectee) having symptom onset")
        reference: https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article

        `self.serial_interval_book` maps infectee.name to covid_symptom_start_time of infector who infected this infectee.

        Args:
            human_name (str): name of `Human` who just experienced some symptoms
        """

        def register_serial_interval(infector, infectee):
            serial_interval = (infectee.covid_symptom_start_time - infector.covid_symptom_start_time) / SECONDS_PER_DAY # DAYS
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
                state = 'N/A'
            self.humans_state[human.name].append(state)
            self.humans_rec_level[human.name].append(human.rec_level)
            self.humans_intervention_level[human.name].append(human._intervention_level)

        # Rt
        self.r.append(self.get_R())

        # recovery stats
        self.recovered_stats.append([0,0])
        if len(self.recovered_stats) > self.city.conf.get("EFFECTIVE_R_WINDOW"):
            self.recovered_stats = self.recovered_stats[1:]

        # test_per_day
        self.tested_per_day.append(0)
        self.hospitalization_per_day.append(0)
        self.critical_per_day.append(0)
        self.deaths_per_day.append(0)

        # mobility
        M, G, B, O, R, EM = self.compute_mobility()
        self.mobility.append(M)
        self.expected_mobility.append(EM)

        # risk model
        prec, lift, recall = self.compute_risk_precision(daily=True)
        self.risk_precision_daily.append((prec, lift, recall))
        self.recommended_levels_daily.append([G, B, O, R])
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
                "name": h.name,
                "dead": h.is_dead,
                "reported_test_result": h.reported_test_result,
                "n_reported_symptoms": len(h.reported_symptoms),
            })

        self.human_monitor[self.env.timestamp.date()-datetime.timedelta(days=1)] = row

        #
        self.avg_infectiousness_per_day.append(np.mean([h.infectiousness for h in self.city.humans]))

    def compute_mobility(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        EM, M, G, B, O, R= 0, 0, 0, 0, 0, 0
        for h in self.city.humans:
            G += h.rec_level == 0
            B += h.rec_level == 1
            O += h.rec_level == 2
            R += h.rec_level == 3
            M +=  1.0 * (h.rec_level == 0) + 0.8 * (h.rec_level == 1) + \
                    0.20 * (h.rec_level == 2) + 0.05 * (h.rec_level == 3) + 1*(h.rec_level==-1)

            EM += (1-h.risk) # proxy for mobility
        return M, G, B, O, R, EM/len(self.city.humans)

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

    def track_humans(self, hd: typing.Dict, current_timestamp: datetime.datetime):
        for name, h in hd.items():
            order_1_contacts = h.contact_book.get_contacts(hd)
            self.risk_attributes.append({
                "has_app": h.has_app,
                "risk": h.risk,
                "risk_level": h.risk_level,
                "rec_level": h.rec_level,
                "exposed": h.is_exposed,
                "infectious": h.is_infectious,
                "symptoms": len(h.symptoms),
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
        self.adoption_rate = sum(h.has_app for h in self.city.humans) / self.n_people
        print(f"adoption rate: {100*self.adoption_rate:3.2f}%")
        self.human_has_app = set([h.name for h in self.city.humans if h.has_app])

    def track_covid_properties(self, human):
        """
        [summary]

        Args:
            human ([type]): [description]
        """
        n, avg = self.covid_properties['incubation_days']
        self.covid_properties['incubation_days'] = (n+1, (avg*n + human.incubation_days)/(n+1))

        n, avg = self.covid_properties['recovery_days']
        self.covid_properties['recovery_days'] = (n+1, (avg*n + human.recovery_days)/(n+1))

        n, avg = self.covid_properties['infectiousness_onset_days']
        self.covid_properties['infectiousness_onset_days'] = (n+1, (n*avg +human.infectiousness_onset_days)/(n+1))

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

    def track_deaths(self):
        """
        Keeps count of deaths per day.
        Args:
            human (covid19sim.human.Human): `human` who died`
        """
        self.deaths_per_day[-1] += 1

    def track_infection(self, type, from_human, to_human, location, timestamp, p_infection):
        """
        Called every time someone is infected either by other `Human` or through envrionmental contamination.

        Args:
            type (str): Type of transmissions, i.e., human or environmental.
            from_human (Human): `Human` who infected to_human
            to_human (Human): `Human` who got infected
            location (Location): `Location` where the even took place.
            timestamp (datetime.datetime): time at which this event took place.
            p_infection: the probability of infection threshold that passed.
        """
        for i, (l,u) in enumerate(self.age_bins):
            if from_human and l <= from_human.age <= u:
                from_bin = i
            if l <= to_human.age <= u:
                to_bin = i

        self.cases_per_day[-1] += 1
        self.infection_monitor.append({
            "from": None if not type=="human" else from_human.name,
            "from_risk":  None if not type=="human" else from_human.risk,
            "from_risk_level": None if not type=="human" else from_human.risk_level,
            "from_rec_level": None if not type=="human" else from_human.rec_level,
            "from_infection_timestamp": None if not type=="human" else from_human.infection_timestamp,
            "from_is_asymptomatic": None if not type=="human" else from_human.is_asymptomatic,
            "from_has_app": None if not type == "human" else from_human.has_app,
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
        })

        if type == "human":
            self.contacts["human_infection"][from_human.age, to_human.age] += 1
            self.contacts["location_human_infection"][location.location_type][from_human.age, to_human.age] += 1

            delta = timestamp - from_human.infection_timestamp
            self.infection_graph.add_node(from_human.name, bin=from_bin, time=from_human.infection_timestamp)
            self.infection_graph.add_node(to_human.name, bin=to_bin, time=timestamp)
            self.infection_graph.add_edge(from_human.name, to_human.name,  timedelta=delta)

            # Keep records of the infection so that serial intervals
            # can be registered when symptoms appear
            # Note: We need a bidirectional record (to/from), because we can't
            # anticipate which (to or from) will manifest symptoms first
            self.serial_interval_book_to[to_human.name][from_human.name] = (to_human, from_human)
            self.serial_interval_book_from[from_human.name][to_human.name] = (to_human, from_human)

            if from_human.is_asymptomatic:
                self.r_0['asymptomatic']['infection_count'] += 1
                self.r_0['asymptomatic']['humans'].add(from_human.name)
            elif not from_human.is_asymptomatic and not from_human.is_incubated:
                self.r_0['presymptomatic']['infection_count'] += 1
                self.r_0['presymptomatic']['humans'].add(from_human.name)
            else:
                self.r_0['symptomatic']['infection_count'] += 1
                self.r_0['symptomatic']['humans'].add(from_human.name)

            self.r_0[location.location_type]['infection_count'] += 1
            self.r_0[location.location_type]['humans'].add(from_human.name)

        else:
            self.n_env_infection += 1
            self.contacts["env_infection"][to_bin] += 1
            self.contacts["location_env_infection"][location.location_type][to_bin] += 1
            self.infection_graph.add_node(to_human.name, bin=to_bin, time=timestamp)
            self.infection_graph.add_edge(-1, to_human.name,  timedelta="")

    def track_update_messages(self, from_human, to_human, payload):
        """ Track which update messages are sent and when (used for plotting) """
        if self.infection_graph.has_edge(from_human.name, to_human.name):
            reason = payload['reason']
            assert reason in ['unknown', 'contact'], "improper reason for sending a message"
            model = self.city.conf.get("RISK_MODEL")
            count = self.infector_infectee_update_messages[from_human.name][to_human.name][self.env.timestamp][reason].get('count', 0)
            x = {'method':model, 'new_risk_level':payload['new_risk_level'], 'count':count+1}
            self.infector_infectee_update_messages[from_human.name][to_human.name][self.env.timestamp][reason] = x
        else:
            old_max_risk_level = self.to_human_max_msg_per_day[to_human.name][self.env.timestamp.date()]
            self.to_human_max_msg_per_day[to_human.name][self.env.timestamp.date()] = max(old_max_risk_level, payload['new_risk_level'])

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
        if count_all:
            for human in self.symptoms_set['covid']:
                self.symptoms['covid']['n'] += 1
                for s in self.symptoms_set['covid'][human]:
                    self.symptoms['covid'][s] += 1

            for human in self.symptoms_set['all']:
                self.symptoms['all']['n'] += 1
                for s in self.symptoms_set['all'][human]:
                    self.symptoms['all'][s] += 1

            delattr(self, "symptoms_set")
            return

        if human.covid_symptoms:
            self.symptoms_set['covid'][human.name].update(human.covid_symptoms)
        else:
            if human.name in self.symptoms_set['covid']:
                self.symptoms['covid']['n'] += 1
                for s in self.symptoms_set['covid'][human.name]:
                    self.symptoms['covid'][s] += 1
                self.symptoms_set['covid'].pop(human.name)

        if human.all_symptoms:
            self.symptoms_set['all'][human.name].update(human.all_symptoms)
        else:
            if human.name in self.symptoms_set['all']:
                self.symptoms['all']['n'] += 1
                for s in self.symptoms_set['all'][human.name]:
                    self.symptoms['all'][s] += 1
                self.symptoms_set['all'].pop(human.name)


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
        proportion_infected = sum(not h.is_susceptible for h in self.city.humans)/n_people

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
        log(f"Proportion infected : {100*proportion_infected: 2.3f}%", logfile)
        log(f"Positivity rate: {100*positivity_rate: 2.3f}%", logfile)
        log(f"Total Tests: {n_positives + n_negatives} Total positive tests: {n_positives} Total negative tests: {n_negatives}", logfile)
        log(f"Maximum tests given to an individual: {max_tests_per_human}", logfile)
        log(f"Proportion of population tested until end: {100 * percent_tested: 4.3f}%", logfile)
        log(f"Proportion of population tested daily Avg: {100 * np.mean(daily_percent_test_results): 4.3f}%", logfile)
        log(f"Proportion of population tested daily Max: {100 * max(daily_percent_test_results, default=0): 4.3f}%", logfile)
        log(f"Proportion of population tested daily Min: {100 * min(daily_percent_test_results, default=0): 4.3f}%", logfile)
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

    def track_recovery(self, n_infectious_contacts, duration):
        """
        [summary]

        Args:
            n_infectious_contacts ([type]): [description]
            duration ([type]): [description]
        """
        self.n_infectious_contacts += n_infectious_contacts
        self.avg_infectious_duration = (self.n_recovery * self.avg_infectious_duration + duration) / (self.n_recovery + 1)
        self.n_recovery += 1

        n, total = self.recovered_stats[-1]
        self.recovered_stats[-1] = [n+1, total + n_infectious_contacts]

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

        # forms a transition probability on weekdays and weekends
        type_of_day = ['weekday', 'weekend'][self.env.is_weekend()]
        from_location = current_activity.location.location_type
        to_location = current_activity.location.location_type
        self.transition_probability[type_of_day][from_location][to_location] += 1

        # proportion of day spend in activity.name broken down by age groups
        n, avg = self.day_fraction_spent_activity[type_of_day][current_activity.name]
        m = current_activity.duration / SECONDS_PER_DAY
        self.day_fraction_spent_activity[type_of_day][current_activity.name] = (n+1, (n*avg + m)/(n+1))

        # histogram of number of people with whom socialization happens
        if next_activity.name == "socialize":
            self.socialize_activity_data['group_size'].push(len(next_activity.rsvp))
            self.socialize_activity_data["location_frequency"][next_activity.location.location_type] += 1
            self.socialize_activity_data["start_time"].push(_get_seconds_since_midnight(next_activity.start_time))

        # if "-cancel-" not in next_activity.append_name and next_activity.prepend_name == "":
        #     self.start_time_activity[next_activity.name].push(_get_seconds_since_midnight(next_activity.start_time))

        self.activity_attributes["end_time"][next_activity.name].push(_get_seconds_since_midnight(next_activity.end_time))
        self.activity_attributes["duration"][next_activity.name].push(next_activity.duration)

    def track_mixing(self, human1, human2, duration, distance_profile, timestamp, location, interaction_type, contact_condition):
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
            11. TODO: Clean Up. (list of strings) Records distance terms related to encounters i.e. packing term, social distancing distance, and total distance

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

        for interaction_type in interaction_types:
            for location_type in ['all', human1_type_of_place, human2_type_of_place]:
                C = self.contact_matrices[interaction_type][location_type]
                D = self.contact_duration_matrices[interaction_type][location_type]

                C['total'][i, j] += 1
                C['total'][j, i] += 1
                # in surveys, ij ==> j is the participant and i is the reported contact
                C['n_people'][(j,i)].add(human1.name) # i reports about j; len of this set is number of unique people in i that met j
                C['n_people'][(i,j)].add(human2.name) # j reports about i
                D['total'][i,j] += duration
                D['total'][j,i] += duration

                # Note that the use of math.ceil makes the upper limit inclusive.
                # record distance profile/frequency counts (per 10 cms). It keeps counts for (distance_category - 10,  distance_category].
                distance_category = 10 * math.ceil(distance_profile.distance / 10)
                Cp = self.contact_distance_profile[interaction_type][location_type]
                Cp[distance_category] = Cp.get(distance_category, 0) + 1

                # record duration profile/frequency counts (per 1 min)
                duration_category = math.ceil(duration)
                Dp = self.contact_duration_profile[interaction_type][location_type]
                Dp[duration_category] = Dp.get(duration_category, 0) + 1

            # replacement of track_encounter_distance.
            # TODO: Clean Up. How exactly is this being used?
            packing_term = distance_profile.packing_term
            encounter_term = distance_profile.encounter_term
            social_distancing_term = distance_profile.social_distancing_term
            distance = distance_profile.distance
            if contact_condition:
                str = 'B\t{}\t{}\t{}\t{}'.format(packing_term, encounter_term, social_distancing_term, distance)
            else:
                type = "A\t1" if distance == packing_term else "A\t0"
                str = '{}\t{}\t{}\t{}\t{}\t{}'.format(type, location, packing_term, encounter_term, social_distancing_term, distance)
            self.encounter_distances.append(str)

            if human1.location != human1.household:
                self.n_outside_daily_contacts += 1

    def track_bluetooth_communications(self, human1, human2, timestamp):
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
        type_of_place = _get_location_type_to_track_mixing(location)
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
            bcm[key0] = {}
            for key1, value1 in value0.items():
                value1.pop("total")
                bcm[key0][key1] = value1

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
        }

    def track_p_infection(self, infection, p_infection, viral_load):
        """
        Keeps track of attributes related to infection to be used for calculating
        probability of transmission.
        """
        self.p_infection.append([infection, p_infection, viral_load])

    def compute_probability_of_transmission(self):
        """
        If X interactions qualify as close contact and Y of them resulted in an infection,
        probability of transmission is defined as Y/X
        """
        total_infections = sum(x[0] for x in self.p_infection)
        total_contacts = len(self.p_infection)
        return 0 if total_contacts == 0 else total_infections / total_contacts

    def compute_effective_contacts(self, since_intervention=True):
        """
        Effective contacts are those that qualify to be within contact_condition in `Human.at`.
        These are major candidates for infectious contacts.
        """
        all_effective_contacts = 0
        all_contacts = 0
        for human in self.city.humans:
            all_effective_contacts += human.effective_contacts
            all_contacts += human.num_contacts

        days = self.conf['simulation_days']
        if since_intervention and self.conf['INTERVENTION_DAY'] > 0 :
            days = self.conf['simulation_days'] - self.conf['INTERVENTION_DAY']

        # recover GLOBAL_MOBILITY_SCALING_FACTOR
        scale_factor = 0 if all_contacts == 0 else all_effective_contacts / all_contacts

        return all_effective_contacts / (days * self.n_people), scale_factor

    def write_metrics(self):
        """
        Writes various metrics to `self.logfile`. Prints them if `self.logfile` is None.
        """

        log("\n######## COVID PROPERTIES #########", self.logfile)
        log(f"Avg. incubation days {self.covid_properties['incubation_days'][1]: 5.2f}", self.logfile)
        log(f"Avg. recovery days {self.covid_properties['recovery_days'][1]: 5.2f}", self.logfile)
        log(f"Avg. infectiousnes onset days {self.covid_properties['infectiousness_onset_days'][1]: 5.2f}", self.logfile)

        log("\n######## COVID SPREAD #########", self.logfile)
        x = 0
        if len(self.infection_monitor) > 0:
            x = 1.0*self.n_env_infection/len(self.infection_monitor)
        log(f"human-human transmissions {len(self.infection_monitor)}", self.logfile )
        log(f"environment-human transmissions {self.n_env_infection}", self.logfile )
        log(f"environmental transmission ratio {x:5.3f}", self.logfile )
        log(f"Generation times {self.get_generation_time()} ", self.logfile)


        log("******** R0 *********", self.logfile)
        if self.r_0['asymptomatic']['infection_count'] > 0:
            x = 1.0 * self.r_0['asymptomatic']['infection_count']/len(self.r_0['asymptomatic']['humans'])
        else:
            x = 0.0
        log(f"Asymptomatic R0 {x}", self.logfile)

        if self.r_0['presymptomatic']['infection_count'] > 0:
            x = 1.0 * self.r_0['presymptomatic']['infection_count']/len(self.r_0['presymptomatic']['humans'])
        else:
            x = 0.0
        log(f"Presymptomatic R0 {x}", self.logfile)

        if self.r_0['symptomatic']['infection_count'] > 0 :
            x = 1.0 * self.r_0['symptomatic']['infection_count']/len(self.r_0['symptomatic']['humans'])
        else:
            x = 0.0
        log(f"Symptomatic R0 {x}", self.logfile )

        log("******** Transmission Ratios *********", self.logfile)
        total = sum(self.r_0[x]['infection_count'] for x in ['symptomatic','presymptomatic', 'asymptomatic'])
        total += self.n_env_infection

        total += 1e-6 # to avoid ZeroDivisionError
        x = self.r_0['asymptomatic']['infection_count']
        log(f"% asymptomatic transmission {100*x/total :5.2f}%", self.logfile)

        x = self.r_0['presymptomatic']['infection_count']
        log(f"% presymptomatic transmission {100*x/total :5.2f}%", self.logfile)

        x = self.r_0['symptomatic']['infection_count']
        log(f"% symptomatic transmission {100*x/total :5.2f}%", self.logfile)

        log("******** R0 LOCATIONS *********", self.logfile)
        for loc_type, v in self.r_0.items():
            if loc_type in ['asymptomatic', 'presymptomatic', 'symptomatic']:
                continue
            if v['infection_count']  > 0:
                x = 1.0 * v['infection_count']/len(v['humans'])
                log(f"{loc_type} R0 {x}", self.logfile)

        log("\n######## SYMPTOMS #########", self.logfile)
        self.track_symptoms(count_all=True)
        total = self.symptoms['covid']['n']
        tmp_s = {}
        for s,v in self.symptoms['covid'].items():
            if s == 'n':
                continue
            tmp_s[s] = v/total
        print_dict("P(symptoms = x | covid patient), where x is", tmp_s, is_sorted="desc", top_k=10, logfile=self.logfile)

        total = self.symptoms['all']['n']
        tmp_s = {}
        for s,v in self.symptoms['covid'].items():
            if s == 'n':
                continue
            tmp_s[s] = v/total
        print_dict("P(symptoms = x | human had some sickness e.g. cold, flu, allergies, covid), where x is", tmp_s, is_sorted="desc", top_k=10, logfile=self.logfile)


        log("\n######## CONTACT PATTERNS #########", self.logfile)
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
        str_to_print += f"min: {group_sizes.minimum(): 2.2f} | "
        str_to_print += f"max: {group_sizes.maximum(): 2.2f} | "
        log(str_to_print, self.logfile)

        str_to_print = "location - "
        locations = self.socialize_activity_data["location_frequency"].keys()
        total = sum(self.socialize_activity_data["location_frequency"].values())
        str_to_print += f"total visits {total} | "
        for location in locations:
            m = self.socialize_activity_data["location_frequency"][location]
            str_to_print += f"{location}: {m} {100*m/total:2.2f}% | "
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
                str_to_print += f"min: {metrics.minimum()/SECONDS_PER_HOUR: 2.2f} | "
                str_to_print += f"max: {metrics.maximum()/SECONDS_PER_HOUR: 2.2f} | "
                log(str_to_print, self.logfile)

        self.compute_test_statistics(self.logfile)

        log("\n######## Effective Contacts & % infected #########", self.logfile)
        p_infected = 100 * sum(self.cases_per_day) / len(self.city.humans)
        effective_contacts, scale_factor = self.compute_effective_contacts()
        p_transmission = self.compute_probability_of_transmission()
        infectiousness_duration = self.covid_properties['recovery_days'][1] - self.covid_properties['infectiousness_onset_days'][1]
        # R0 = Susceptible population x Duration of infectiousness x p_transmission
        # https://www.youtube.com/watch?v=wZabMDS0CeA
        # valid only when small portion of population is infected
        r0 = p_transmission * effective_contacts * infectiousness_duration

        log(f"Eff. contacts: {effective_contacts:5.3f} \t % infected: {p_infected: 2.3f}%", self.logfile)
        if scale_factor:
            log(f"effective contacts per contacts (GLOBAL_MOBILITY_SCALING_FACTOR): {scale_factor}", self.logfile)
        # log(f"Probability of transmission: {p_transmission:2.3f}", self.logfile)
        # log(f"Ro (valid only when small proportion of population is infected): {r0: 2.3f}", self.logfile)
        # log("definition of small might be blurry for a population size of less than 1000  ", self.logfile)
        # log(f"Serial interval: {self.get_serial_interval(): 5.3f}", self.logfile)

        log("\n######## Rt #########", self.logfile)
        cases_per_day = self.cases_per_day
        serial_interval = self.get_generation_time()
        if serial_interval == 0:
            serial_interval = 7.0
            log("WARNING: serial_interval is 0", self.logfile)
        log(f"using serial interval :{serial_interval}", self.logfile)

        plotrt = PlotRt(R_T_MAX=4, sigma=0.25, GAMMA=1.0 / serial_interval)
        most_likely, _ = plotrt.compute(cases_per_day, r0_estimate=2.5)
        log(f"Rt: {most_likely[:20]}", self.logfile)

    def plot_metrics(self, dirname):
        """
        [summary]

        Args:
            dirname ([type]): [description]
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import seaborn as sns
        import os

        x = pd.DataFrame.from_dict(self.contacts['all'])
        x = x[sorted(x.columns)]
        fig = x.iplot(kind='heatmap', asFigure=True, title="all_contacts")
        fig.savefig(f"{dirname}/all_contacts.png")

        x = self.contacts['env_infection']
        g = self.infection_graph
        nx.nx_pydot.write_dot(g,'DiGraph.dot')
        pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='dot')
        nx.draw_networkx(g, pos, with_labels=True)
        plt.savefig(f"{dirname}/infection_graph.png")

        os.makedirs(f"{dirname}/contact_stats", exist_ok=True)
        types = sorted(self.city.conf.get("LOCATION_DISTRIBUTION").keys())
        ages = sorted(self.city.conf.get("HUMAN_DISTRIBUTION").keys(), key = lambda x:x[0])
        for hour, v1 in self.transition_probability.items():
            images = []
            fig,ax =  plt.subplots(3,2, figsize=(18,12), sharex=True, sharey=False)
            pos = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)}

            for age_bin in range(len(ages)):
                v2 = v1[age_bin]
                x = pd.DataFrame.from_dict(v2, orient='index')
                x = x.reindex(index=types, columns=types)
                x = x.div(x.sum(1), axis=0)
                g = sns.heatmap(x, ax=ax[pos[age_bin][0]][pos[age_bin][1]],
                    linewidth=0.5, linecolor='black', annot=True, vmin=0.0, vmax=1.0, cmap=sns.cm.rocket_r)
                g.set_title(f"{ages[age_bin][0]} <= age < {ages[age_bin][1]}")

            fig.suptitle(f"Hour {hour}", fontsize=16)
            fig.savefig(f"{dirname}/contact_stats/hour_{hour}.png")

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
