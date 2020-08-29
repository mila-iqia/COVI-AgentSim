import numpy as np
import os
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def statistics(pkl_data):

    data = pkl_data["risk_attributes"]

    # Compute the set of all reasons for changing rec level
    rec_level_reasons = defaultdict(list)
    infectious_reasons = defaultdict(list)

    symptomatic_infected = defaultdict(list)
    symptomatic_uninfected = defaultdict(list)
    asymptomatic_infected = defaultdict(list)
    asymptomatic_uninfected = defaultdict(list)
    for d in tqdm.tqdm(data):
        reason = ", ".join(d['reason'])
        if reason == "":
            reason = "no reason"
        rec_level = d['rec_level']
        symptoms = d['symptoms']
        exposed = d['exposed']
        infectious = d['infectious']

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



    plot(symptomatic_infected, title="Symptomatic Infected")
    plot(symptomatic_uninfected, title="Symptomatic Uninfected")

    plot(asymptomatic_infected, title="Asymptomatic Infected")
    plot(asymptomatic_uninfected, title="Asymptomatic Uninfected")


def plot(data, title=""):
    # Plotting
    fig = plt.figure(figsize=(10,10))
    plt.tight_layout()
    ax = fig.add_subplot(111)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

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
    plt.savefig(f"{title}.png")


def run(data, path, comparison_key):

    # Options:
    # 1. "times" is a list, indicating the times we are interested in.
    # For example, [-1, -2, -3] means we want to get the plot for Day-1, Day-2, Day-3.
    # 2. "mode" is a string, which can be either 'risk' or 'rec' if None,
    # it will be both.

    label2pkls = list()
    for method in data:
        for key in data[method]:
            label = f"{key}"
            pkls = [r["pkl"] for r in data[method][key].values()]
            label2pkls.append((label, pkls))

    results = list()
    for label, pkls in label2pkls:
        all_positive, all_negative = [], []
        for pkl in pkls:
            result = statistics(pkl)
        # results.append((all_positive, all_negative, all_positive - all_negative))
