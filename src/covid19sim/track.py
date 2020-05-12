
"""
[summary]
"""
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import networkx as nx
import datetime
import dill
import copy

from covid19sim.configs.config import HUMAN_DISTRIBUTION, LOCATION_DISTRIBUTION, INFECTION_RADIUS, EFFECTIVE_R_WINDOW
from covid19sim.utils import log
from covid19sim.configs.exp_config import ExpConfig


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
        self.env = env
        self.city = city
        # filename to store intermediate results; useful for bigger simulations;
        timenow = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        if ExpConfig.get('INTERVENTION_DAY') == -1:
            name = "unmitigated"
        else:
            name = ExpConfig.get('RISK_MODEL')
        self.filename = f"tracker_data_n_{len(city.humans)}_{timenow}_{name}.pkl"

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
                'n_contacts': {'avg': (0, np.zeros((150,150))), 'total': np.zeros((150,150))}

                }

        self.infection_graph = nx.DiGraph()
        self.s_per_day = [sum(h.is_susceptible for h in self.city.humans)]
        self.e_per_day = [sum(h.is_exposed for h in self.city.humans)]
        self.i_per_day = [sum(h.is_infectious for h in self.city.humans)]
        self.r_per_day = [sum(h.is_removed for h in self.city.humans)]

        # R0 and Generation times
        self.avg_infectious_duration = 0
        self.n_recovery = 0
        self.n_infectious_contacts = 0
        self.n_contacts = 0
        self.avg_generation_times = (0,0)
        self.generation_time_book = {}
        self.n_env_infection = 0
        self.recovered_stats = []
        self.covid_properties = defaultdict(lambda : [0,0])

        # cumulative incidence
        day = self.env.timestamp.strftime("%d %b")
        self.last_day = {'track_recovery':day, "track_infection":day, 'social_mixing':day}
        self.cumulative_incidence = []
        self.cases_per_day = [0]
        self.r_0 = defaultdict(lambda : {'infection_count':0, 'humans':set()})
        self.r = []

        # testing & hospitalization
        self.cases_positive_per_day = [0]
        self.hospitalization_per_day = [0]
        self.critical_per_day = [0]

        # demographics
        self.age_bins = sorted(HUMAN_DISTRIBUTION.keys(), key = lambda x:x[0])
        self.n_humans = len(self.city.humans)

        # track encounters
        self.last_encounter_day = self.env.day_of_week()
        self.last_encounter_hour = self.env.hour_of_day()
        self.day_encounters = defaultdict(lambda : [0.,0.,0.])
        self.hour_encounters = defaultdict(lambda : [0.,0.,0.])
        self.daily_age_group_encounters = defaultdict(lambda :[0.,0.,0.])

        self.dist_encounters = defaultdict(int)
        self.time_encounters = defaultdict(int)

        # symptoms
        self.symptoms = {'covid': defaultdict(int), 'all':defaultdict(int)}
        self.symptoms_set = {'covid': defaultdict(set), 'all': defaultdict(set)}

        # mobility
        self.n_outside_daily_contacts = 0
        self.transition_probability = get_nested_dict(4)
        M, G, B, O, R, EM, F = self.compute_mobility()
        self.mobility = [M]
        self.expected_mobility = [EM]
        self.summarize_population()
        self.feelings = [F]
        self.rec_feelings = []
        self.outside_daily_contacts = []

        # risk models
        self.risk_precision_daily = [self.compute_risk_precision()]
        self.recommended_levels_daily = [[G, B, O, R]]
        self.ei_per_day = []
        self.risk_values = []
        self.avg_infectiousness_per_day = []
        self.risk_attributes = []

        # monitors
        self.human_monitor = {}
        self.infection_monitor = []

        # update messages
        self.infector_infectee_update_messages = defaultdict(lambda :defaultdict(dict))

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
        time_since_start =  (self.env.timestamp - self.env.initial_timestamp).total_seconds() / 86400 # DAYS
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
            for x in self.r:
                if x >0:
                    return x
        else:
            log("not enough data points to estimate r0. Falling back to average")
            return self.get_R()

    def get_generation_time(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return self.avg_generation_times[1]

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

        # Rt
        self.r.append(self.get_R())

        # recovery stats
        self.recovered_stats.append([0,0])
        if len(self.recovered_stats) > EFFECTIVE_R_WINDOW:
            self.recovered_stats = self.recovered_stats[1:]

        # test_per_day
        self.cases_positive_per_day.append(0)
        self.hospitalization_per_day.append(0)
        self.critical_per_day.append(0)

        # mobility
        M, G, B, O, R, EM, F = self.compute_mobility()
        self.mobility.append(M)
        self.expected_mobility.append(EM)
        self.feelings.append(F)
        self.rec_feelings.extend([(h.rec_level, h.how_am_I_feeling()) for h in self.city.humans])
        self.outside_daily_contacts.append(self.n_outside_daily_contacts/len(self.city.humans))

        # risk models
        prec, lift, recall = self.compute_risk_precision(daily=True)
        self.risk_precision_daily.append((prec,lift, recall))
        self.recommended_levels_daily.append([G, B, O, R])
        self.risk_values.append([(h.risk, h.is_exposed or h.is_infectious, h.test_result, len(h.symptoms) == 0) for h in self.city.humans])
        row = []
        for h in self.city.humans:
            x = { "infection_timestamp": h.infection_timestamp,
                  "n_infectious_contacts": h.n_infectious_contacts,
                  "risk": h.risk,
                  "risk_level": h.risk_level,
                  "rec_level": h.rec_level,
                  "state": h.state.index(1),
                  "test_result": h.test_result,
                  "n_symptoms": len(h.symptoms),
                  "test_orders":copy.deepcopy(h.message_info['n_contacts_tested_positive']),
                  "symptom_orders":copy.deepcopy(h.message_info['n_contacts_symptoms']),
                 }
            row.append(x)

        self.human_monitor[self.env.timestamp.date()-datetime.timedelta(days=1)] = row

        #
        self.avg_infectiousness_per_day.append(np.mean([h.infectiousness for h in self.city.humans]))

        if len(self.city.humans) > 5000:
            self.dump_metrics()

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

    def compute_risk_precision(self, daily=True, threshold=0.5, until_days=None):
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

                top_k_prec[idx].append(pred/len(xy))
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

    def track_risk_attributes(self, humans):
        for h in humans:
            if h.is_removed:
                continue

            _tmp = {
                "risk": h.risk,
                "risk_level": h.risk_level,
                "rec_level": h.rec_level,
                "exposed": h.is_exposed,
                "infectious": h.is_infectious,
                "symptoms": len(h.symptoms),
                "test": h.test_result,
                "recovered": h.is_removed,
                "timestamp": self.env.timestamp,
                "test_recommended": h.test_recommended,
                "name":h.name
            }

            order_1_is_exposed = False
            order_1_is_infectious = False
            order_1_is_presymptomatic = False
            order_1_is_symptomatic = False
            order_1_is_tested = False
            for order_1_human in h.contact_book.book:
                order_1_is_exposed = order_1_is_exposed or order_1_human.is_exposed
                order_1_is_infectious = order_1_is_infectious or order_1_human.is_infectious
                order_1_is_presymptomatic = order_1_is_presymptomatic or (order_1_human.is_infectious and len(order_1_human.symptoms) == 0)
                order_1_is_symptomatic = order_1_is_symptomatic or (order_1_human.is_infectious and len(order_1_human.symptoms) > 0)
                order_1_is_tested = order_1_is_tested or order_1_human.test_result == "positive"

            _tmp["order_1_is_exposed"] = order_1_is_exposed
            _tmp["order_1_is_presymptomatic"] = order_1_is_presymptomatic
            _tmp["order_1_is_infectious"] = order_1_is_infectious
            _tmp["order_1_is_symptomatic"] = order_1_is_symptomatic
            _tmp["order_1_is_tested"] = order_1_is_tested

            self.risk_attributes.append(_tmp)

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

    def track_infection(self, type, from_human, to_human, location, timestamp):
        """
        [summary]

        Args:
            type ([type]): [description]
            from_human ([type]): [description]
            to_human ([type]): [description]
            location ([type]): [description]
            timestamp ([type]): [description]
        """
        for i, (l,u) in enumerate(self.age_bins):
            if from_human and l <= from_human.age < u:
                from_bin = i
            if l <= to_human.age < u:
                to_bin = i

        self.cases_per_day[-1] += 1

        if type == "human":
            self.contacts["human_infection"][from_human.age, to_human.age] += 1
            self.contacts["location_human_infection"][location.location_type][from_human.age, to_human.age] += 1

            delta = timestamp - from_human.infection_timestamp
            self.infection_graph.add_node(from_human.name, bin=from_bin, time=from_human.infection_timestamp)
            self.infection_graph.add_node(to_human.name, bin=to_bin, time=timestamp)
            self.infection_graph.add_edge(from_human.name, to_human.name,  timedelta=delta)

            self.infection_monitor.append([from_human.name, from_human.risk, from_human.risk_level, from_human.rec_level, to_human.name, to_human.risk, to_human.risk_level, to_human.rec_level, timestamp.date()])

            if from_human.symptom_start_time is not None:
                self.generation_time_book[to_human.name] = from_human.symptom_start_time

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
            self.infection_monitor.append([None, to_human.name, timestamp.date()])

    def track_update_messages(self, from_human, to_human, payload):
        if self.infection_graph.has_edge(from_human.name, to_human.name):
            reason = payload['reason']
            model = self.city.intervention.risk_model
            if  model == "transformer":
                if  reason != "risk_update":
                    return # transformer only sends risks
                x = {'method':model, 'reason':payload['reason'], 'new_risk_level':payload['new_risk_level']}
                self.infector_infectee_update_messages[from_human.name][to_human.name][self.env.timestamp] = x
            else:
                if not self.city.intervention.propagate_risk and reason == "risk_update":
                    return
                if not self.city.intervention.propagate_symptoms and reason == "symptoms":
                    return
                x = {'method':model, 'reason':payload['reason']}
                self.infector_infectee_update_messages[from_human.name][to_human.name][self.env.timestamp] = x

    def track_generation_times(self, human_name):
        """
        [summary]

        Args:
            human_name ([type]): [description]
        """
        if human_name not in self.generation_time_book:
            return

        generation_time = (self.env.timestamp - self.generation_time_book.pop(human_name)).total_seconds() / 86400 # DAYS
        n, avg_gen_time = self.avg_generation_times
        self.avg_generation_times = (n+1, 1.0*(avg_gen_time * n + generation_time)/(n+1))

    def track_tested_results(self, human, test_result, test_type):
        """
        [summary]

        Args:
            human ([type]): [description]
            test_result ([type]): [description]
            test_type ([type]): [description]
        """
        if test_result == "positive":
            self.cases_positive_per_day[-1] += 1

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

    def track_symptoms(self, human):
        """
        [summary]

        Args:
            human ([type]): [description]
        """
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

    def track_social_mixing(self, **kwargs):
        """
        [summary]
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
                # duration
                n, M = self.contacts['duration']['avg']
                where = self.contacts['duration']['n'] != 0
                m = np.divide(self.contacts['duration']['total'], self.contacts['duration']['n'], where=where)
                self.contacts['duration']['avg'] = (n+1, (n*M + m)/(n+1))

                self.contacts['duration']['total'] = np.zeros((150,150))
                self.contacts['duration']['n'] = np.zeros((150,150))

                # n_contacts
                n, M = self.contacts['n_contacts']['avg']
                m = self.contacts['n_contacts']['total']
                self.contacts['n_contacts']['avg'] = (n+1, (n*M + m)/(n+1))

                self.contacts['n_contacts']['total'] = np.zeros((150,150))

                self.n_outside_daily_contacts = 0
                self.last_day['social_mixing'] = day

            else:
                human1 = kwargs.get('human1', None)
                human2 = kwargs.get('human2', None)
                if human1 is not None and human2 is not None:
                    self.contacts['duration']['total'][human1.age, human2.age] += duration
                    self.contacts['duration']['n'][human1.age, human2.age] += 1

                    self.contacts['duration']['total'][human2.age, human1.age] += duration
                    self.contacts['duration']['n'][human2.age, human1.age] += 1

                    self.contacts['n_contacts']['total'][human1.age, human2.age] += 1
                    self.contacts['n_contacts']['total'][human2.age, human1.age] += 1
                    if human1.location != human1.household:
                        self.n_outside_daily_contacts += 1

        if location is not None:
            x = len(self.contacts['location_duration'][location.location_type])
            if bin >= x:
                self.contacts['location_duration'][location.location_type].extend([0 for _ in range(bin - x + 1)])
            self.contacts['location_duration'][location.location_type][bin] += 1

    def track_encounter_events(self, human1, human2, location, distance, duration):
        """
        [summary]

        Args:
            human1 ([type]): [description]
            human2 ([type]): [description]
            location ([type]): [description]
            distance ([type]): [description]
            duration ([type]): [description]
        """
        for i, (l,u) in enumerate(self.age_bins):
            if l <= human1.age < u:
                bin1 = (i,(l,u))
            if l <= human2.age < u:
                bin2 = (i, (l,u))

        self.contacts["all_encounters"][human1.age, human2.age] += 1
        self.contacts["all_encounters"][human2.age, human1.age] += 1
        self.contacts["location_all_encounters"][location.location_type][human1.age, human2.age] += 1
        self.contacts["location_all_encounters"][location.location_type][human2.age, human1.age] += 1
        self.n_contacts += 1

        # bins of 50
        dist_bin = math.floor(distance/50) if distance <= INFECTION_RADIUS else math.floor(INFECTION_RADIUS/50)

        # bins of 15 mins
        time_bin = math.floor(duration/15) if duration <= 60 else 4

        hour = self.env.hour_of_day()
        day = self.env.day_of_week()
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

    def write_metrics(self, logfile):
        """
        [summary]

        Args:
            logfile ([type]): [description]
        """
        log("######## DEMOGRAPHICS #########", logfile)
        log(f"age distribution\n {self.age_distribution.describe()}", logfile)
        log(f"house age distribution\n {self.house_age.describe()}", logfile )
        log(f"house size distribution\n {self.house_size.describe()}", logfile )
        log(f"Fraction of asymptomatic {self.frac_asymptomatic}", logfile )

        log("######## COVID PROPERTIES #########", logfile)
        print("Avg. incubation days", self.covid_properties['incubation_days'][1])
        print("Avg. recovery days", self.covid_properties['recovery_days'][1])
        print("Avg. infectiousnes onset days", self.covid_properties['infectiousness_onset_days'][1])

        log("######## COVID SPREAD #########", logfile)
        x = 1.0*self.n_env_infection/self.n_infectious_contacts if self.n_infectious_contacts else 0.0
        log(f"environmental transmission ratio {x}", logfile )
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
        log(f"% asymptomatic transmission {100*x/total :5.2f}%", logfile)

        x = self.r_0['presymptomatic']['infection_count']
        log(f"% presymptomatic transmission {100*x/total :5.2f}%", logfile)

        x = self.r_0['symptomatic']['infection_count']
        log(f"% symptomatic transmission {100*x/total :5.2f}%", logfile)

        log("******** R0 LOCATIONS *********", logfile)
        for loc_type, v in self.r_0.items():
            if loc_type in ['asymptomatic', 'presymptomatic', 'symptomatic']:
                continue
            if v['infection_count']  > 0:
                x = 1.0 * v['infection_count']/len(v['humans'])
                log(f"{loc_type} R0 {x}", logfile)

        # log("######## SYMPTOMS #########", logfile)
        # total = self.symptoms['covid']['n']
        # for s,v in self.symptoms['covid'].items():
        #     if s == 'n':
        #         continue
        #     log(f"{s} {100*v/total:5.2f}%")
        #
        # log("######## MOBILITY #########", logfile)
        # log("Day - ", logfile)
        # total = sum(v[1] for v in self.day_encounters.values())
        # x = ['Mon', "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun"]
        # for c,day in enumerate(x):
        #     v = self.day_encounters[c]
        #     log(f"{day} #avg: {v[1]} %:{100*v[1]/total:5.2f} ", logfile)
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
        # log("Average Daily Contacts ", logfile)
        # total = sum(x[1] for x in self.daily_age_group_encounters.values())
        # for bin in self.age_bins:
        #     v = self.daily_age_group_encounters[bin][1]
        #     log(f"{bin} #avg: {v} %:{100*v/total:5.2f} ", logfile)
        #
        for until_days in [30, None]:
            log("******** Risk Precision/Recall *********", logfile)
            prec, lift, recall = self.compute_risk_precision(daily=False, until_days=until_days)
            top_k = [0.01, 0.03, 0.05, 0.10]
            type_str = ["all", "no test", "no test and symptoms"]

            log(f"*** Precision (until days={until_days}) ***", logfile)
            idx = 0
            for k_values in zip(*prec):
                x,y,z= k_values
                log(f"Top-{100*top_k[idx]:2.2f}% all: {100*x:5.2f}% no_test:{100*y:5.2f}% no_test_and_symptoms: {100*z:5.2f}%", logfile)
                idx += 1

            log(f"*** Lift (until days={until_days}) ***", logfile)
            idx = 0
            for k_values in zip(*lift):
                x,y,z = k_values
                log(f"Top-{100*top_k[idx]:2.2f}% all: {x:5.2f} no_test:{y:5.2f} no_test_and_symptoms: {z:5.2f}", logfile)
                idx += 1

            log(f"*** Recall (until days={until_days}) ***", logfile)
            x,y,z = recall
            log(f"all: {100*x:5.2f}% no_test: {100*y:5.2f}% no_test_and_symptoms: {100*z:5.2f}%", logfile)

        log("*** Avg daily precision ***", logfile)
        prec = [x[0] for x in self.risk_precision_daily]
        lift = [x[1] for x in self.risk_precision_daily]
        recall = [x[2] for x in self.risk_precision_daily]

        all = list(zip(*[x[0] for x in prec]))
        no_test = list(zip(*[x[1] for x in prec]))
        no_test_symptoms = list(zip(*[x[2] for x in prec]))
        idx = 0
        for k in top_k:
            log(f"Top-{100*top_k[idx]:2.2f}% all: {100*np.mean(all[idx]):5.2f}% no_test:{100*np.mean(no_test[idx]):5.2f}% no_test_and_symptoms: {100*np.mean(no_test_symptoms[idx]):5.2f}%", logfile)
            idx += 1

        log("*** Avg daily lift ***", logfile)
        all = list(zip(*[x[0] for x in lift]))
        no_test = list(zip(*[x[1] for x in lift]))
        no_test_symptoms = list(zip(*[x[2] for x in lift]))
        idx = 0
        for k in top_k:
            log(f"Top-{100*top_k[idx]:2.2f}% all: {np.mean(all[idx]):5.2f} no_test:{np.mean(no_test[idx]):5.2f} no_test_and_symptoms: {np.mean(no_test_symptoms[idx]):5.2f}", logfile)
            idx += 1

        log("*** Avg. daily recall ***", logfile)
        x,y,z = zip(*recall)
        log(f"all: {100*np.mean(x):5.2f}% no_test: {100*np.mean(y):5.2f} no_test_and_symptoms: {100*np.mean(z):5.2f}", logfile)

    def plot_metrics(self, dirname):
        """
        [summary]

        Args:
            dirname ([type]): [description]
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import seaborn as sns
        import glob, os

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
        types = sorted(LOCATION_DISTRIBUTION.keys())
        ages = sorted(HUMAN_DISTRIBUTION.keys(), key = lambda x:x[0])
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

    def dump_metrics(self):
        data = dict()
        data['intervention_day'] = ExpConfig.get('INTERVENTION_DAY')
        data['intervention'] = ExpConfig.get('INTERVENTION')
        data['risk_model'] = ExpConfig.get('RISK_MODEL')

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

        with open(f"logs3/{self.filename}", 'wb') as f:
            dill.dump(data, f)
