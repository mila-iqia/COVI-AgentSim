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
    for d in tqdm.tqdm(data):
        reason = ", ".join(d['reason'])
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

        # people w rec level 3

        # people w rec level 2
    fig = plt.figure()
    ax = fig.add_subplot(111)

    reason_set = set()
    for all_reasons in  symptomatic_infected.values():
        for r in all_reasons:
            reason_set.add(r)
    # import pdb; pdb.set_trace()

    # Plotting
    X = np.arange(len(reason_set))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    reasons = np.zeros((5, len(reason_set)))
    rec_levels = range(-1, 4)
    for rec_level in rec_levels:
        c = Counter(symptomatic_infected[rec_level])
        for idx, reason in enumerate(reason_set):
            reasons[rec_level, idx] = c[reason]
        ax.bar(X + rec_level * 0.25, list(reasons)[rec_level], color=colors[rec_level], width=0.25)
    title = "Symptomatic Infected"
    ax.set_ylabel('Num Samples')
    ax.set_title(title)
    import pdb; pdb.set_trace()
    # ax.set_xticks(X, list(reason_set))
    ax.set_xticklabels(list(reason_set))
    # ax.set_yticks(np.arange(0, 81, 10))
    plt.xticks(rotation=45)
    ax.legend(labels=[str(x) for x in rec_levels])

    fig.show()
    import pdb; pdb.set_trace()
    return rec_level_reasons, infectious_reasons


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
        results.append((all_positive, all_negative, all_positive - all_negative))
