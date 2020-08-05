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

import dill
import networkx as nx
import numpy as np
import pandas as pd

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
    [summary]
    """
    def __init__(self, env, city):
        """
        [summary]

        Args:
            object ([type]): [description]
            env ([type]): [description]
            city ([type]): [description]
        """
        self.fully_initialized = False
        self.env = env
        self.city = city
        self.adoption_rate = 0
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

        # infection & contacts
        self.contacts = {
                'all_encounters':np.zeros((150,150)),
                'location_all_encounters': defaultdict(lambda: np.zeros((150,150))),
                'human_infection': np.zeros((150,150)),
                'env_infection':get_nested_dict(1),
                'location_env_infection': get_nested_dict(2),
                'location_human_infection': defaultdict(lambda: np.zeros((150,150))),
                'duration': {'avg': (0, np.zeros((150,150))), 'total': np.zeros((150,150)), 'n': np.zeros((150,150))},
                'histogram_duration': [0],
                'location_duration':defaultdict(lambda : [0]),
                'n_contacts': {
                        'avg': (0, np.zeros((150,150))),
                        'total': np.zeros((150,150)),
                        'n_people': defaultdict(lambda : set())
                        },
                'n_bluetooth_contacts': {
                        'avg': (0, np.zeros((150,150))),
                        'total': np.zeros((150,150)),
                        'n_people': defaultdict(lambda : set())
                        },
                }
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
        day = self.env.timestamp.strftime("%d %b")
        self.last_day = {'track_recovery':day, "track_infection":day, 'social_mixing':day, 'bluetooth_communications': day}
        self.cumulative_incidence = []
        self.cases_per_day = [0]
        self.r_0 = defaultdict(lambda : {'infection_count':0, 'humans':set()})
        self.r = []

        # testing & hospitalization
        self.hospitalization_per_day = [0]
        self.critical_per_day = [0]
        self.test_results_per_day = defaultdict(lambda :{'positive':0, 'negative':0})
        self.tested_per_day = [0]

        # demographics
        self.age_bins = sorted(self.city.conf.get("HUMAN_DISTRIBUTION").keys(), key = lambda x:x[0])
        self.n_people = self.n_humans = self.city.n_people
        self.human_has_app = None
        self.adoption_rate = 0.0

        # track encounters
        self.last_encounter_day = self.env.day_of_week
        self.last_encounter_hour = self.env.hour_of_day
        self.day_encounters = defaultdict(lambda : [0.,0.,0.])
        self.hour_encounters = defaultdict(lambda : [0.,0.,0.])
        self.daily_age_group_encounters = defaultdict(lambda :[0.,0.,0.])

        self.dist_encounters = defaultdict(int)
        self.time_encounters = defaultdict(int)
        self.encounter_distances = []

        # symptoms
        self.symptoms = {'covid': defaultdict(int), 'all':defaultdict(int)}
        self.symptoms_set = {'covid': defaultdict(set), 'all': defaultdict(set)}

        # mobility
        self.n_outside_daily_contacts = 0
        self.transition_probability = get_nested_dict(4)
        self.rec_feelings = []
        self.outside_daily_contacts = []

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

        M, G, B, O, R, EM, F = self.compute_mobility()
        self.recommended_levels_daily = [[G, B, O, R]]
        self.mobility = [M]
        self.expected_mobility = [EM]
        self.summarize_population()
        self.feelings = [F]

        # risk model
        self.risk_precision_daily = [self.compute_risk_precision()]
        self.init_infected = [human for human in self.city.humans if human.is_exposed]
        self.fully_initialized = True

    def summarize_population(self):
        """
        [summary]
        """
        self.n_infected_init = sum([h.is_exposed for h in self.city.humans])
        print(f"initial infection {self.n_infected_init}")

        self.age_distribution = pd.DataFrame([h.age for h in self.city.humans])
        print("age distribution\n", self.age_distribution.describe())

        self.sex_distribution = pd.DataFrame([h.sex for h in self.city.humans])
        # print("gender distribution\n", self.gender_distribution.describe())

        if self.city.households:
            self.house_age = pd.DataFrame([np.mean([h.age for h in house.residents]) for house in self.city.households])
            self.house_size = pd.DataFrame([len(house.residents) for house in self.city.households])
            print("house age distribution\n", self.house_age.describe())
            print("house size distribution\n", self.house_size.describe())

        self.frac_asymptomatic = sum(h.is_asymptomatic for h in self.city.humans)/len(self.city.humans)
        print("asymptomatic fraction", self.frac_asymptomatic)

        self.n_seniors = sum(1 for h in self.city.humans if h.household.location_type == "senior_residency")
        print("n_seniors", self.n_seniors)

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

        # mobility
        M, G, B, O, R, EM, F = self.compute_mobility()
        self.mobility.append(M)
        self.expected_mobility.append(EM)
        self.feelings.append(F)
        self.rec_feelings.extend([(h.rec_level, h.how_am_I_feeling()) for h in self.city.humans])

        # risk model
        prec, lift, recall = self.compute_risk_precision(daily=True)
        self.risk_precision_daily.append((prec,lift, recall))
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
                "symptom_severity": self.compute_severity(h.reported_symptoms),
                "name": h.name,
                "dead": h.is_dead,
                "reported_test_result": h.reported_test_result,
                "n_reported_symptoms": len(h.reported_symptoms),
            })

        self.human_monitor[self.env.timestamp.date()-datetime.timedelta(days=1)] = row

        #
        self.avg_infectiousness_per_day.append(np.mean([h.infectiousness for h in self.city.humans]))

        # if len(self.city.humans) > 5000:
            # self.dump_metrics()

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

    def compute_mobility(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        EM, M, G, B, O, R, F = 0, 0, 0, 0, 0, 0, 0
        for h in self.city.humans:
            G += h.rec_level == 0
            B += h.rec_level == 1
            O += h.rec_level == 2
            R += h.rec_level == 3
            M +=  1.0 * (h.rec_level == 0) + 0.8 * (h.rec_level == 1) + \
                    0.20 * (h.rec_level == 2) + 0.05 * (h.rec_level == 3) + 1*(h.rec_level==-1)
            F += h.how_am_I_feeling()

            EM += (1-h.risk) # proxy for mobility
        return M, G, B, O, R, EM/len(self.city.humans), F/len(self.city.humans)

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
        max_tests_per_human = max(tests_per_human.values())

        # percent of population tested
        n_tests = len(self.test_monitor)
        n_people = len(self.city.humans)
        percent_tested = 1.0 * n_tests/n_people
        daily_percent_test_results = [sum(x.values())/n_people for x in self.test_results_per_day.values()]
        proportion_infected = sum(not h.is_susceptible for h in self.city.humans)/n_people

        # positivity rate
        n_positives = sum(x["positive"] for x in self.test_results_per_day.values())
        n_negatives = sum(x["negative"] for x in self.test_results_per_day.values())
        positivity_rate = n_positives/(n_positives + n_negatives)

        # symptoms | tests
        # count of humans who has symptom x given a test was administered
        test_given_symptoms_statistics = Counter([s for tm in self.test_monitor for s in tm["symptoms"]])
        positive_test_given_symptoms = Counter([s for tm in self.test_monitor for s in tm["symptoms"] if tm['test_result'] == "positive"])
        negative_test_given_symptoms = Counter([s for tm in self.test_monitor for s in tm["symptoms"] if tm['test_result'] == "negative"])

        # new infected - tests (per day)
        infected_minus_tests_per_day = [x - y for x,y in zip(self.e_per_day, self.tested_per_day)]

        log("######## COVID Testing Statistics #########", logfile)
        log(f"Proportion infected : {100*proportion_infected: 2.3f}%", logfile)
        log(f"Positivity rate: {100*positivity_rate: 2.3f}%", logfile)
        log(f"Total Tests: {n_positives + n_negatives} Total positive tests: {n_positives} Total negative tests: {n_negatives}", logfile)
        log(f"Maximum tests given to an individual: {max_tests_per_human}", logfile)
        log(f"Proportion of population tested until end: {100 * percent_tested: 4.3f}%", logfile)
        log(f"Proportion of population tested daily Avg: {100 * np.mean(daily_percent_test_results): 4.3f}%", logfile)
        log(f"Proportion of population tested daily Max: {100 * max(daily_percent_test_results): 4.3f}%", logfile)
        log(f"Proportion of population tested daily Min: {100 * min(daily_percent_test_results): 4.3f}%", logfile)
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

    def track_trip(self, from_location, to_location, age, hour):
        """
        [summary]

        Args:
            from_location ([type]): [description]
            to_location ([type]): [description]
            age ([type]): [description]
            hour ([type]): [description]
        """
        bin = None
        for i, (l,u) in enumerate(self.age_bins):
            if l <= age < u:
                bin = i

        self.transition_probability[hour][bin][from_location][to_location] += 1

    def track_social_mixing(self, **kwargs):
        """
        Keeps count of average daily encounters between different ages.
        """
        duration = kwargs.get('duration')
        bin = math.floor(duration/15)
        location = kwargs.get('location', None)

        if location is None:
            x = len(self.contacts['histogram_duration'])
            if bin >= x:
                self.contacts['histogram_duration'].extend([0 for _ in range(bin - x + 1)])
            self.contacts['histogram_duration'][bin] += 1

            timestamp = kwargs.get('timestamp')
            day = timestamp.strftime("%d %b")

            if self.last_day['social_mixing'] != day:
                # average duration per contact across age groups (minutes)
                n, M = self.contacts['duration']['avg']
                where = self.contacts['duration']['n'] != 0
                m = np.divide(self.contacts['duration']['total'], self.contacts['duration']['n'], where=where)
                self.contacts['duration']['avg'] = (n+1, (n*M + m)/(n+1))

                self.contacts['duration']['total'] = np.zeros((150,150))
                self.contacts['duration']['n'] = np.zeros((150,150))

                # number of contacts across age groups
                n, M = self.contacts['n_contacts']['avg']
                m = self.contacts['n_contacts']['total']
                any_contact = np.zeros_like(m)
                for age1, age2 in self.contacts['n_contacts']['n_people'].keys():
                    x = len(self.contacts['n_contacts']['n_people'][age1, age2])
                    m[age1, age2] /=  x
                    any_contact[age1, age2] = 1.0

                # update values
                self.contacts['n_contacts']['avg'] = (n+1, (n*M + m)/(n+1))
                self.contacts['n_contacts']['n_people'] = defaultdict(lambda : set())
                self.contacts['n_contacts']['total'] = np.zeros((150,150))

                self.outside_daily_contacts.append(1.0 * self.n_outside_daily_contacts/len(self.city.humans))
                self.n_outside_daily_contacts = 0
                self.last_day['social_mixing'] = day

            else:
                human1 = kwargs.get('human1', None)
                human2 = kwargs.get('human2', None)
                if human1 is not None and human2 is not None:
                    self.contacts['duration']['total'][human1.age, human2.age] += duration
                    self.contacts['duration']['total'][human2.age, human1.age] += duration
                    self.contacts['duration']['n'][human1.age, human2.age] += 1
                    self.contacts['duration']['n'][human2.age, human1.age] += 1

                    self.contacts['n_contacts']['total'][human1.age, human2.age] += 1
                    self.contacts['n_contacts']['total'][human2.age, human1.age] += 1
                    self.contacts['n_contacts']['n_people'][(human1.age, human2.age)].add(human1.name)
                    self.contacts['n_contacts']['n_people'][(human2.age, human1.age)].add(human2.name)

                    if human1.location != human1.household:
                        self.n_outside_daily_contacts += 1

        if location is not None:
            x = len(self.contacts['location_duration'][location.location_type])
            if bin >= x:
                self.contacts['location_duration'][location.location_type].extend([0 for _ in range(bin - x + 1)])
            self.contacts['location_duration'][location.location_type][bin] += 1

    def track_bluetooth_communications(self, human1, human2, timestamp):
        day = timestamp.strftime("%d %b")
        if self.last_day['bluetooth_communications']  != day:
            n, M = self.contacts['n_bluetooth_contacts']['avg']
            m = self.contacts['n_bluetooth_contacts']['total']
            any_contact = np.zeros_like(m)
            for age1, age2 in self.contacts['n_bluetooth_contacts']['n_people'].keys():
                x = len(self.contacts['n_bluetooth_contacts']['n_people'][age1, age2])
                m[age1, age2] /=  x
                any_contact[age1, age2] = 1.0

            # update values
            self.contacts['n_bluetooth_contacts']['avg'] = (n+1, (n*M + m)/(n+1))
            self.contacts['n_bluetooth_contacts']['n_people'] = defaultdict(lambda : set())
            self.contacts['n_bluetooth_contacts']['total'] = np.zeros((150,150))
            self.last_day['bluetooth_communications'] = day

        else:
            self.contacts['n_bluetooth_contacts']['total'][human1.age, human2.age] += 1
            self.contacts['n_bluetooth_contacts']['total'][human2.age, human1.age] += 1
            self.contacts['n_bluetooth_contacts']['n_people'][(human1.age, human2.age)].add(human1.name)
            self.contacts['n_bluetooth_contacts']['n_people'][(human2.age, human1.age)].add(human2.name)

    def track_encounter_distance(self, type, packing_term, encounter_term, social_distancing_term, distance, location=None):
        if location:
            str = '{}\t{}\t{}\t{}\t{}\t{}'.format(type, location, packing_term, encounter_term, social_distancing_term, distance)
        else:
            str = 'B\t{}\t{}\t{}\t{}'.format(packing_term, encounter_term, social_distancing_term, distance)
        self.encounter_distances.append(str)

    def track_encounter_events(self, human1, human2, location, distance, duration):
        """
        Counts encounters that qualify to be in contact_condition.
        Keeps average of daily

        Args:
            human1 (Human): One of the `Human` involved in encounter
            human2 (Human): One of the `Human` involved in encounter
            location (Location): Location at which encounter took place
            distance (float): Distance at which encounter took place (cm)
            duration (float): time duration for which the encounter took place (minutes)
        """
        for i, (l,u) in enumerate(self.age_bins):
            if l <= human1.age <= u:
                bin1 = (i,(l,u))
            if l <= human2.age <= u:
                bin2 = (i, (l,u))

        self.contacts["all_encounters"][human1.age, human2.age] += 1
        self.contacts["all_encounters"][human2.age, human1.age] += 1
        self.contacts["location_all_encounters"][location.location_type][human1.age, human2.age] += 1
        self.contacts["location_all_encounters"][location.location_type][human2.age, human1.age] += 1
        self.n_contacts += 1

        # bins of 50
        dist_bin = (
            math.floor(distance / 50)
            if distance <= self.city.conf.get("INFECTION_RADIUS")
            else math.floor(self.city.conf.get("INFECTION_RADIUS") / 50)
        )

        # bins of 15 mins
        time_bin = math.floor(duration/15) if duration <= 60 else 4

        hour = self.env.hour_of_day
        day = self.env.day_of_week
        if self.last_encounter_day != day:
            n, avg, last_day_count = self.day_encounters[self.last_encounter_day]
            self.day_encounters[self.last_encounter_day] = [n+1, (avg * n + last_day_count)/(n + 1), 0]

            # per age bin
            for bin in self.age_bins:
                n, avg, last_day_count  = self.daily_age_group_encounters[bin]
                self.daily_age_group_encounters[bin] = [n+1, (avg * n + last_day_count)/(n+1), 0]

            self.last_encounter_day = day

        if self.last_encounter_hour != hour:
            n, avg, last_hour_count = self.hour_encounters[self.last_encounter_hour]
            self.hour_encounters[self.last_encounter_hour] = [n+1, (avg * n + last_hour_count)/(n + 1), 0]
            self.last_encounter_hour = hour

        self.day_encounters[self.last_encounter_day][-1] += 1
        self.hour_encounters[self.last_encounter_hour][-1] += 1
        self.daily_age_group_encounters[bin1[1]][-1] += 1
        self.daily_age_group_encounters[bin2[1]][-1] += 1
        self.dist_encounters[dist_bin] += 1
        self.time_encounters[time_bin] += 1

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
        try:
            return total_infections / total_contacts
        except Exception:
            return 0.

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

        conf = self.city.conf
        days = conf['simulation_days']
        if since_intervention and conf['INTERVENTION_DAY'] > 0 :
            days = conf['simulation_days'] - conf['INTERVENTION_DAY']

        return all_effective_contacts / (days * self.n_people), all_healthy_effective_contacts / all_healthy_days


    def write_metrics(self, logfile=None):
        """
        Writes various metrics to logfile.
        Prints them if logfile is None.

        Args:
            logfile (str, optional): filename where these logs will be dumped
        """
        log("######## DEMOGRAPHICS #########", logfile)
        log(f"age distribution\n {self.age_distribution.describe()}", logfile)
        log(f"house age distribution\n {self.house_age.describe()}", logfile )
        log(f"house size distribution\n {self.house_size.describe()}", logfile )
        log(f"Fraction of asymptomatic {self.frac_asymptomatic}", logfile )

        log("######## COVID PROPERTIES #########", logfile)
        log(f"Avg. incubation days {self.covid_properties['incubation_days'][1]: 5.2f}", logfile)
        log(f"Avg. recovery days {self.covid_properties['recovery_days'][1]: 5.2f}", logfile)
        log(f"Avg. infectiousnes onset days {self.covid_properties['infectiousness_onset_days'][1]: 5.2f}", logfile)

        log("######## COVID SPREAD #########", logfile)
        x = 0
        if len(self.infection_monitor) > 0:
            x = 1.0*self.n_env_infection/len(self.infection_monitor)
        log(f"human-human transmissions {len(self.infection_monitor)}", logfile )
        log(f"environment-human transmissions {self.n_env_infection}", logfile )
        log(f"environmental transmission ratio {x:5.3f}", logfile )
        r0 = self.get_R0(logfile)
        log(f"Ro {r0}", logfile)
        log(f"Generation times {self.get_generation_time()} ", logfile)
        log(f"Cumulative Incidence {self.cumulative_incidence}", logfile )
        log(f"R : {self.r}", logfile)

        log("******** R0 *********", logfile)
        if self.r_0['asymptomatic']['infection_count'] > 0:
            x = 1.0 * self.r_0['asymptomatic']['infection_count']/len(self.r_0['asymptomatic']['humans'])
        else:
            x = 0.0
        log(f"Asymptomatic R0 {x}", logfile)

        if self.r_0['presymptomatic']['infection_count'] > 0:
            x = 1.0 * self.r_0['presymptomatic']['infection_count']/len(self.r_0['presymptomatic']['humans'])
        else:
            x = 0.0
        log(f"Presymptomatic R0 {x}", logfile)

        if self.r_0['symptomatic']['infection_count'] > 0 :
            x = 1.0 * self.r_0['symptomatic']['infection_count']/len(self.r_0['symptomatic']['humans'])
        else:
            x = 0.0
        log(f"Symptomatic R0 {x}", logfile )

        log("******** Transmission Ratios *********", logfile)
        total = sum(self.r_0[x]['infection_count'] for x in ['symptomatic','presymptomatic', 'asymptomatic'])
        total += self.n_env_infection

        x = self.r_0['asymptomatic']['infection_count']
        if total > 0:
            log(f"% asymptomatic transmission {100*x/total :5.2f}%", logfile)

        x = self.r_0['presymptomatic']['infection_count']
        if total > 0:
            log(f"% presymptomatic transmission {100*x/total :5.2f}%", logfile)

        x = self.r_0['symptomatic']['infection_count']
        if total > 0:
            log(f"% symptomatic transmission {100*x/total :5.2f}%", logfile)

        log("******** R0 LOCATIONS *********", logfile)
        for loc_type, v in self.r_0.items():
            if loc_type in ['asymptomatic', 'presymptomatic', 'symptomatic']:
                continue
            if v['infection_count']  > 0:
                x = 1.0 * v['infection_count']/len(v['humans'])
                log(f"{loc_type} R0 {x}", logfile)

        log("######## SYMPTOMS #########", logfile)
        self.track_symptoms(count_all=True)
        total = self.symptoms['covid']['n']
        tmp_s = {}
        for s,v in self.symptoms['covid'].items():
            if s == 'n':
                continue
            if total > 0:
                tmp_s[s] = v/total
            else:
                tmp_s[s] = 0.
        print_dict("P(symptoms = x | covid patient), where x is", tmp_s, is_sorted="desc", top_k=10, logfile=logfile)

        total = self.symptoms['all']['n']
        tmp_s = {}
        for s,v in self.symptoms['covid'].items():
            if s == 'n':
                continue
            if total > 0:
                tmp_s[s] = v/total
            else:
                tmp_s[s] = 0.
        print_dict("P(symptoms = x | human had some sickness e.g. cold, flu, allergies, covid), where x is", tmp_s, is_sorted="desc", top_k=10, logfile=logfile)

        log("######## MOBILITY #########", logfile)
        log("Day - ", logfile)
        total = sum(v[1] for v in self.day_encounters.values())
        x = ['Mon', "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun"]
        for c,day in enumerate(x):
            v = self.day_encounters[c]
            if total > 0:
                log(f"{day} #avg: {v[1]/self.n_people} %:{100*v[1]/total:5.2f} ", logfile)
        #
        # log("Hour - ", logfile)
        # total = sum(v[1] for v in self.hour_encounters.values())
        # for hour, v in self.hour_encounters.items():
        #     log(f"{hour} #avg: {v[1]} %:{100*v[1]/total:5.2f} ", logfile)
        #
        # log("Distance (cm) - ", logfile)
        # x = ['0 - 50', "50 - 100", "100 - 150", "150 - 200", ">= 200"]
        # total = sum(self.dist_encounters.values())
        # for c, dist in enumerate(x):
        #     v = self.dist_encounters[c]
        #     log(f"{dist} #avg: {v} %:{100*v/total:5.2f} ", logfile)
        #
        # log("Time (min) ", logfile)
        # x = ['0 - 15', "15 - 30", "30 - 45", "45 - 60", ">= 60"]
        # total = sum(self.time_encounters.values())
        # for c, bin in enumerate(x):
        #     v = self.time_encounters[c]
        #     log(f"{bin} #avg: {v} %:{100*v/total:5.2f} ", logfile)
        #
        log("Average Daily Contacts ", logfile)
        total = sum(x[1] for x in self.daily_age_group_encounters.values())
        for bin in self.age_bins:
            x = self.city.age_histogram[bin]
            v = self.daily_age_group_encounters[bin][1]
            if total > 0:
                log(f"{bin} #avg: {v/x} %:{100*v/total:5.2f} ", logfile)
        #
        # for until_days in [30, None]:
        #     log("******** Risk Precision/Recall *********", logfile)
        #     prec, lift, recall = self.compute_risk_precision(daily=False, until_days=until_days)
        #     top_k = [0.01, 0.03, 0.05, 0.10]
        #     type_str = ["all", "no test", "no test and symptoms"]
        #
        #     log(f"*** Precision (until days={until_days}) ***", logfile)
        #     idx = 0
        #     for k_values in zip(*prec):
        #         x,y,z= k_values
        #         log(f"Top-{100*top_k[idx]:2.2f}% all: {100*x:5.2f}% no_test:{100*y:5.2f}% no_test_and_symptoms: {100*z:5.2f}%", logfile)
        #         idx += 1
        #
        #     log(f"*** Lift (until days={until_days}) ***", logfile)
        #     idx = 0
        #     for k_values in zip(*lift):
        #         x,y,z = k_values
        #         log(f"Top-{100*top_k[idx]:2.2f}% all: {x:5.2f} no_test:{y:5.2f} no_test_and_symptoms: {z:5.2f}", logfile)
        #         idx += 1
        #
        #     log(f"*** Recall (until days={until_days}) ***", logfile)
        #     x,y,z = recall
        #     log(f"all: {100*x:5.2f}% no_test: {100*y:5.2f}% no_test_and_symptoms: {100*z:5.2f}%", logfile)
        #
        # log("*** Avg daily precision ***", logfile)
        # prec = [x[0] for x in self.risk_precision_daily]
        # lift = [x[1] for x in self.risk_precision_daily]
        # recall = [x[2] for x in self.risk_precision_daily]
        #
        # all = list(zip(*[x[0] for x in prec]))
        # no_test = list(zip(*[x[1] for x in prec]))
        # no_test_symptoms = list(zip(*[x[2] for x in prec]))
        # idx = 0
        # for k in top_k:
        #     log(f"Top-{100*top_k[idx]:2.2f}% all: {100*np.mean(all[idx]):5.2f}% no_test:{100*np.mean(no_test[idx]):5.2f}% no_test_and_symptoms: {100*np.mean(no_test_symptoms[idx]):5.2f}%", logfile)
        #     idx += 1
        #
        # log("*** Avg daily lift ***", logfile)
        # all = list(zip(*[x[0] for x in lift]))
        # no_test = list(zip(*[x[1] for x in lift]))
        # no_test_symptoms = list(zip(*[x[2] for x in lift]))
        # idx = 0
        # for k in top_k:
        #     log(f"Top-{100*top_k[idx]:2.2f}% all: {np.mean(all[idx]):5.2f} no_test:{np.mean(no_test[idx]):5.2f} no_test_and_symptoms: {np.mean(no_test_symptoms[idx]):5.2f}", logfile)
        #     idx += 1
        #
        # log("*** Avg. daily recall ***", logfile)
        # x,y,z = zip(*recall)
        # log(f"all: {100*np.mean(x):5.2f}% no_test: {100*np.mean(y):5.2f} no_test_and_symptoms: {100*np.mean(z):5.2f}", logfile)

        self.compute_test_statistics(logfile)

        log("######## Effective Contacts & % infected #########", logfile)
        p_infected = 100 * sum(self.cases_per_day) / len(self.city.humans)
        effective_contacts, healthy_effective_contacts = self.compute_effective_contacts()
        p_transmission = self.compute_probability_of_transmission()
        infectiousness_duration = self.covid_properties['recovery_days'][1] - self.covid_properties['infectiousness_onset_days'][1]
        # R0 = Susceptible population x Duration of infectiousness x p_transmission
        # https://www.youtube.com/watch?v=wZabMDS0CeA
        # valid only when small portion of population is infected
        r0 = p_transmission * effective_contacts * infectiousness_duration

        log(f"Eff. contacts: {effective_contacts:5.3f} \t Healthy Eff. Contacts {healthy_effective_contacts:5.3f} \th % infected: {p_infected: 2.3f}%", logfile)
        # log(f"Probability of transmission: {p_transmission:2.3f}", logfile)
        # log(f"Ro (valid only when small proportion of population is infected): {r0: 2.3f}", logfile)
        # log("definition of small might be blurry for a population size of less than 1000  ", logfile)
        # log(f"Serial interval: {self.get_serial_interval(): 5.3f}", logfile)

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

    def _get_metrics_data(self):
        data = dict()
        data['intervention_day'] = self.city.conf.get('INTERVENTION_DAY')
        data['intervention'] = self.city.conf.get('INTERVENTION')
        data['risk_model'] = self.city.conf.get('RISK_MODEL')
        data['expected_mobility'] = self.expected_mobility
        data['mobility'] = self.mobility
        data['n_init_infected'] = self.n_infected_init
        data['contacts'] = dict(self.contacts)
        data['cases_per_day'] = self.cases_per_day
        data['ei_per_day'] = self.ei_per_day
        data['r_0'] = self.r_0
        data['R'] = self.r
        data['n_humans'] = self.n_humans
        data['s'] = self.s_per_day
        data['e'] = self.e_per_day
        data['i'] = self.i_per_day
        data['r'] = self.r_per_day
        data['avg_infectiousness_per_day'] = self.avg_infectiousness_per_day
        data['risk_precision_global'] = self.compute_risk_precision(False)
        data['risk_precision'] = self.risk_precision_daily
        data['human_monitor'] = self.human_monitor
        data['infection_monitor'] = self.infection_monitor
        data['infector_infectee_update_messages'] = self.infector_infectee_update_messages
        data['encounter_distances'] = self.encounter_distances
        return data

    def dump_metrics(self):
        data = self._get_metrics_data()
        os.makedirs("logs3", exist_ok=True)
        with open(f"logs3/{self.filename}", 'wb') as f:
            dill.dump(data, f)

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
