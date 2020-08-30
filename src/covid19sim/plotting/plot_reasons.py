import numpy as np
import os
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def run(data, path, comparison_key):


    label2pkls = list()
    for method in data:
        for key in data[method]:
            label = f"{key}"
            pkls = [r["pkl"] for r in data[method][key].values()]
            label2pkls.append((label, method, pkls))

    for label, method, pkls in label2pkls:
        # heuristic version risk msg levels
        path = path / method
        if "heuristicv1" in str(path):
            high_risk_threshold = 12
            moderate_risk_threshold = 10
            mild_risk_threshold = 6
        elif "heuristicv3" in str(path):
            high_risk_threshold = 15
            moderate_risk_threshold = 13
            mild_risk_threshold = 10

        for pkl in pkls:
            data = pkl["risk_attributes"]

            # Compute the set of all reasons for changing rec level

            symptomatic_infected = defaultdict(list)
            symptomatic_uninfected = defaultdict(list)
            asymptomatic_infected = defaultdict(list)
            asymptomatic_uninfected = defaultdict(list)
            symptomatic_infected_msgs = defaultdict(list)
            symptomatic_uninfected_msgs = defaultdict(list)
            asymptomatic_infected_msgs = defaultdict(list)
            asymptomatic_uninfected_msgs = defaultdict(list)

            for d in tqdm.tqdm(data):
                reason = ", ".join(d['reason'])
                if reason == "":
                    reason = "no reason"
                rec_level = d['rec_level']
                symptoms = d['symptoms']
                exposed = d['exposed']
                infectious = d['infectious']
                reported_symptoms = d['symptom_names']
                prevalence = d['current_prevalence']
                clusters = d['clusters']
                # print(reported_symptoms)
                # print(prevalence)
                # if len(d['clusters']) != 0:
                #     print(clusters) # day, level, num encounters

                # symptomatic people
                if symptoms != 0 and (exposed or infectious):
                    symptomatic_infected[rec_level].append(reason)
                if symptoms != 0 and not (exposed or infectious):
                    symptomatic_uninfected[rec_level].append(reason)

                # asymptomatic people
                if symptoms == 0 and (exposed or infectious):
                    asymptomatic_infected[rec_level].append(reason)
                if symptoms == 0 and not (exposed or infectious):
                    asymptomatic_uninfected[rec_level].append(reason)

                # make new plots that show the distribution of risk messages received by asymptomatic people, split into infected/uninfected, and for varying prevalence
                high_risks = sum([encs for day, level, encs in clusters if level >= high_risk_threshold])
                med_risks = sum([encs for day, level, encs in clusters if level >= moderate_risk_threshold and level < high_risk_threshold])
                low_risks = sum([encs for day, level, encs in clusters if level >= mild_risk_threshold and level < moderate_risk_threshold])
                no_risks = sum([encs for day, level, encs in clusters if level < mild_risk_threshold])
                if symptoms != 0 and (exposed or infectious):
                    symptomatic_infected_msgs["high"].append(high_risks)
                    symptomatic_infected_msgs["moderate"].append(med_risks)
                    symptomatic_infected_msgs["mild"].append(low_risks)
                    symptomatic_infected_msgs["low"].append(no_risks)
                if symptoms != 0 and not (exposed or infectious):
                    symptomatic_uninfected_msgs["high"].append(high_risks)
                    symptomatic_uninfected_msgs["moderate"].append(med_risks)
                    symptomatic_uninfected_msgs["mild"].append(low_risks)
                    symptomatic_uninfected_msgs["low"].append(no_risks)
                if symptoms == 0 and (exposed or infectious):
                    asymptomatic_infected_msgs["high"].append(high_risks)
                    asymptomatic_infected_msgs["moderate"].append(med_risks)
                    asymptomatic_infected_msgs["mild"].append(low_risks)
                    asymptomatic_infected_msgs["low"].append(no_risks)
                if symptoms == 0 and not (exposed or infectious):
                    asymptomatic_uninfected_msgs["high"].append(high_risks)
                    asymptomatic_uninfected_msgs["moderate"].append(med_risks)
                    asymptomatic_uninfected_msgs["mild"].append(low_risks)
                    asymptomatic_uninfected_msgs["low"].append(no_risks)

            plot_risk_messages(symptomatic_infected_msgs.values(), path, title="Symptomatic Infected Risk Messages")
            plot_risk_messages(symptomatic_uninfected_msgs.values(), path, title="Symptomatic Uninfected Risk Messages")
            plot_risk_messages(asymptomatic_infected_msgs.values(), path, title="Asymptomatic Infected Risk Messages")
            plot_risk_messages(asymptomatic_uninfected_msgs.values(), path, title="Asymptomatic Uninfected Risk Messages")

            plot_reasons(symptomatic_infected, path, title="Symptomatic Infected")
            plot_reasons(symptomatic_uninfected, path, title="Symptomatic Uninfected")

            plot_reasons(asymptomatic_infected, path, title="Asymptomatic Infected")
            plot_reasons(asymptomatic_uninfected, path, title="Asymptomatic Uninfected")

def plot_risk_messages(data, path, title):
    # Plotting
    fig = plt.figure(figsize=(10,10))
    plt.tight_layout()
    ax = fig.add_subplot(111)

    ax.hist(data, 100, density=False, histtype='bar', stacked=True)

    if not os.path.isdir(path):
        os.mkdir(path)

    # Format and save the figure
    ax.set_ylabel('Num Samples')
    ax.set_title(title)
    ax.set_xlabel("Num Encounters with Risk")
    ax.legend(labels=["low risk messages", "mild risk messages", "moderate risk messages", "high risk messages"])
    fig.subplots_adjust(bottom=0.3)
    plt.savefig(f"{path}/{title}.png")

def plot_reasons(data, path, title=""):
    # Plotting
    fig = plt.figure(figsize=(10,10))
    plt.tight_layout()
    ax = fig.add_subplot(111)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    if not os.path.isdir(path):
        os.mkdir(path)

    # Extract set of reasons
    reasons = set()
    for all_reasons in data.values():
        for r in all_reasons:
            reasons.add(r)
    reasons = list(reasons)

    # plot the data
    X = np.arange(len(reasons))
    plot_data = np.zeros((5, len(reasons)))
    rec_levels = range(-1, 4)
    prev = np.zeros(len(reasons))
    for rec_level in rec_levels:
        c = Counter(data[rec_level])
        for idx, reason in enumerate(reasons):
            plot_data[rec_level, idx] = c[reason]
        ax.bar(X, list(plot_data)[rec_level], bottom=prev, color=colors[rec_level], width=0.8)
        prev += list(plot_data)[rec_level]

    # Format and save the figure
    ax.set_ylabel('Num Samples')
    ax.set_title(title)
    ax.set_xticks(X)
    ax.set_xticklabels(list(reasons))
    plt.xticks(rotation=90)
    ax.legend(labels=[str(x) for x in rec_levels])
    fig.subplots_adjust(bottom=0.3)
    plt.savefig(f"{path}/{title}.png")
