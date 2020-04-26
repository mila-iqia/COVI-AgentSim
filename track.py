import pandas as pd
import numpy as np
import math
from collections import defaultdict
from config import HUMAN_DISTRIBUTION, LOCATION_DISTRIBUTION, INFECTION_RADIUS, INFECTION_DURATION
import networkx as nx
from utils import log

def get_nested_dict(nesting):
    if nesting == 1:
        return defaultdict(int)
    elif nesting == 2:
        return defaultdict(lambda : defaultdict(int))
    elif nesting == 3:
        return defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    elif nesting == 4:
        return defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(int))))

class Tracker(object):
    def __init__(self, env, city):
        self.env = env
        self.city = city

        # infection & contacts
        self.contacts = {
                'all':get_nested_dict(2),
                'location_all': get_nested_dict(3),
                'human_infection': get_nested_dict(2),
                'env_infection':get_nested_dict(1),
                'location_env_infection': get_nested_dict(2),
                'location_human_infection': get_nested_dict(3),
                'duration': defaultdict(lambda : defaultdict(lambda :[0,0]))
                }

        self.infection_graph = nx.DiGraph()

        # R0 and Generation times
        self.avg_infectious_duration = 0
        self.n_recovery = 0
        self.n_infectious_contacts = 0
        self.n_contacts = 0
        self.avg_generation_times = (0,0)
        self.generation_time_book = {}
        self.n_env_infection = 0
        self.recovered_stats = []

        # cumulative incidence
        day = self.env.timestamp.strftime("%d %b")
        self.last_day = {'track_recovery':day, "track_infection":day}
        self.cumulative_incidence = []
        self.n_susceptible = sum(h.is_susceptible for h in city.humans)
        self.cases_per_day = [0]
        self.r_0 = defaultdict(lambda : {'infection_count':0, 'humans':set()})
        self.r = []

        # demographics
        self.age_bins = sorted(HUMAN_DISTRIBUTION.keys(), key = lambda x:x[0])

        # track encounters
        self.last_encounter_day = self.env.day_of_week()
        self.last_encounter_hour = self.env.hour_of_day()
        self.day_encounters = defaultdict(lambda : [0.,0.,0.])
        self.hour_encounters = defaultdict(lambda : [0.,0.,0.])
        self.daily_age_group_encounters = defaultdict(lambda :[0.,0.,0.])

        self.dist_encounters = defaultdict(int)
        self.time_encounters = defaultdict(int)

        # symptoms
        self.symptoms = {'covid': defaultdict(int), 'others':defaultdict(int)}

        # mobility
        self.transition_probability = get_nested_dict(4)
        self.summarize_population()

    def summarize_population(self):
        self.n_infected_init = sum([h.is_exposed for h in self.city.humans])
        print(f"initial infection {self.n_infected_init}")

        self.age_distribution = pd.DataFrame([h.age for h in self.city.humans])
        print("age distribution\n", self.age_distribution.describe())

        self.house_age = pd.DataFrame([np.mean([h.age for h in house.residents]) for house in self.city.households])
        self.house_size = pd.DataFrame([len(house.residents) for house in self.city.households])
        print("house age distribution\n", self.house_age.describe())
        print("house size distribution\n", self.house_size.describe())

        self.frac_asymptomatic = sum(h.is_asymptomatic for h in self.city.humans)/len(self.city.humans)
        print("asymptomatic fraction", self.frac_asymptomatic)

    def get_R(self):
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
        if len(self.r) > 0:
            return self.r[0]
        else:
            log("not enough data points to estimate r0. Falling back to average")
            x = [h.n_infectious_contacts for h in self.city.humans if h.state.index(1) >= 2]
            if x:
                return np.mean(x)
            return -1

    def get_generation_time(self):
        return self.avg_generation_times[1]

    def track_infection(self, type, from_human, to_human, location, timestamp):
        for i, (l,u) in enumerate(self.age_bins):
            if from_human and l <= from_human.age < u:
                from_bin = i
            if l <= to_human.age < u:
                to_bin = i

        day = self.env.timestamp.strftime("%d %b")
        if self.last_day['track_infection'] != day:
            self.cumulative_incidence.append(self.cases_per_day[-1] / self.n_susceptible)
            self.last_day['track_infection'] = day
            self.n_susceptible = sum(h.is_susceptible for h in self.city.humans)
            self.cases_per_day.append(0)
        else:
            self.cases_per_day[-1] += 1

        if type == "human":
            self.contacts["human_infection"][from_bin][to_bin] += 1
            self.contacts["location_human_infection"][location.location_type][from_bin][to_bin] += 1

            delta = timestamp - from_human.infection_timestamp
            self.infection_graph.add_node(from_human.name, bin=from_bin, time=from_human.infection_timestamp)
            self.infection_graph.add_node(to_human.name, bin=to_bin, time=timestamp)
            self.infection_graph.add_edge(from_human.name, to_human.name,  timedelta=delta)

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

    def track_generation_times(self, human_name):
        if human_name not in self.generation_time_book:
            return

        generation_time = (self.env.timestamp - self.generation_time_book.pop(human_name)).total_seconds() / 86400 # DAYS
        n, avg_gen_time = self.avg_generation_times
        self.avg_generation_times = (n+1, 1.0*(avg_gen_time * n + generation_time)/(n+1))

    def track_tested_results(self, human, test_result, test_type):
        pass

    def track_recovery(self, n_infectious_contacts, duration):
        self.n_infectious_contacts += n_infectious_contacts
        self.avg_infectious_duration = (self.n_recovery * self.avg_infectious_duration + duration) / (self.n_recovery + 1)
        self.n_recovery += 1

        day = self.env.timestamp.day
        if self.last_day['track_recovery'] != day:
            self.last_day['track_recovery'] = day
            if len(self.recovered_stats) > 10:
                self.r.append(self.get_R())
                self.recovered_stats = self.recovered_stats[1:]
            self.recovered_stats.append([0, 0])
        else:
            n, total = self.recovered_stats[-1]
            self.recovered_stats[-1] = [n+1, total + n_infectious_contacts]

    def track_trip(self, from_location, to_location, age, hour):
        bin = None
        for i, (l,u) in enumerate(self.age_bins):
            if l <= age < u:
                bin = i

        self.transition_probability[hour][bin][from_location][to_location] += 1

    def track_initialized_covid_params(self, humans):
        days = [(h.incubation_days, h.recovery_days, h.infectiousness_onset_days) for h in humans]
        print("Avg. incubation days", np.mean([x[0] for x in days]))
        print("Avg. recovery days", np.mean([x[1] for x in days]))
        print("Avg. infectiousnes onset days", np.mean([x[2] for x in days]))

    def track_symptoms(self, symptoms, covid=True):
        # called after logging the test
        self.symptoms['covid']['n'] += 1
        for s in symptoms:
            self.symptoms['covid'][s] += 1

    def track_social_mixing(self, human1, human2, location, distance, duration):
        n, avg = self.contacts['duration'][human1.age][human2.age]
        self.contacts['duration'][human1.age][human2.age] = (n+1, (avg*n + duration)/(n+1))

        # self.contacts['location_duration'][location.location_type] binning

    def track_encounter_events(self, human1, human2, location, distance, duration):
        for i, (l,u) in enumerate(self.age_bins):
            if l <= human1.age < u:
                bin1 = (i,(l,u))
            if l <= human2.age < u:
                bin2 = (i, (l,u))

        self.contacts["all"][human1.age][human2.age] += 1
        self.contacts["location_all"][location.location_type][bin1[0]][bin2[0]] += 1
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
        log("######## DEMOGRAPHICS #########", logfile)
        log(f"age distribution\n {self.age_distribution.describe()}", logfile)
        log(f"house age distribution\n {self.house_age.describe()}", logfile )
        log(f"house size distribution\n {self.house_size.describe()}", logfile )
        log(f"Fraction of asymptomatic {self.frac_asymptomatic}", logfile )

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

        log("******** R0 LOCATIONS *********", logfile)
        for loc_type, v in self.r_0.items():
            if loc_type in ['asymptomatic', 'presymptomatic', 'symptomatic']:
                continue
            if v['infection_count']  > 0:
                x = 1.0 * v['infection_count']/len(v['humans'])
                log(f"{loc_type} R0 {x}", logfile)

        log("######## SYMPTOMS #########", logfile)
        total = self.symptoms['covid']['n']
        for s,v in self.symptoms['covid'].items():
            if s == 'n':
                continue
            log(f"{s} {100*v/total:5.2f}%")

        log("######## MOBILITY #########", logfile)
        log("Day - ", logfile)
        total = sum(v[1] for v in self.day_encounters.values())
        x = ['Mon', "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun"]
        for c,day in enumerate(x):
            v = self.day_encounters[c]
            log(f"{day} #avg: {v[1]} %:{100*v[1]/total:5.2f} ", logfile)

        log("Hour - ", logfile)
        total = sum(v[1] for v in self.hour_encounters.values())
        for hour, v in self.hour_encounters.items():
            log(f"{hour} #avg: {v[1]} %:{100*v[1]/total:5.2f} ", logfile)

        log("Distance (cm) - ", logfile)
        x = ['0 - 50', "50 - 100", "100 - 150", "150 - 200", ">= 200"]
        total = sum(self.dist_encounters.values())
        for c, dist in enumerate(x):
            v = self.dist_encounters[c]
            log(f"{dist} #avg: {v} %:{100*v/total:5.2f} ", logfile)

        log("Time (min) ", logfile)
        x = ['0 - 15', "15 - 30", "30 - 45", "45 - 60", ">= 60"]
        total = sum(self.time_encounters.values())
        for c, bin in enumerate(x):
            v = self.time_encounters[c]
            log(f"{bin} #avg: {v} %:{100*v/total:5.2f} ", logfile)

        log("Average Daily Contacts ", logfile)
        total = sum(x[1] for x in self.daily_age_group_encounters.values())
        for bin in self.age_bins:
            v = self.daily_age_group_encounters[bin][1]
            log(f"{bin} #avg: {v} %:{100*v/total:5.2f} ", logfile)

    def plot_metrics(self, dirname):
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
