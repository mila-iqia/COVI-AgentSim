import numpy as np
import os
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def run(data, path, comparison_key):
    do_reasons = True
    do_messages = True
    do_symptoms = True
    limit = 10000000

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
        elif "heuristicv3" in str(path) or "heuristicv4" in str(path):
            high_risk_threshold = 15
            moderate_risk_threshold = 13
            mild_risk_threshold = 10

        for pkl in pkls:
            data = pkl["risk_attributes"]

            # Reasons
            symptomatic_infected = defaultdict(list)
            symptomatic_uninfected = defaultdict(list)
            asymptomatic_infected = defaultdict(list)
            asymptomatic_uninfected = defaultdict(list)

            # Messages
            symptomatic_infected_msgs = {"low": [], "mild": [], "moderate": [], "high": []}
            symptomatic_uninfected_msgs = {"low": [], "mild": [], "moderate": [], "high": []}
            asymptomatic_infected_msgs = {"low": [], "mild": [], "moderate": [], "high": []}
            asymptomatic_uninfected_msgs = {"low": [], "mild": [], "moderate": [], "high": []}

            # Symptoms
            symptomatic_infected_symptoms = defaultdict(int)
            symptomatic_uninfected_symptoms = defaultdict(int)

            for d in tqdm.tqdm(data[:limit]):
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

                if do_reasons:
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
                if do_messages:

                    high_risks = sum([encs for day, level, encs in clusters if level >= high_risk_threshold])
                    med_risks = sum([encs for day, level, encs in clusters if level >= moderate_risk_threshold and level < high_risk_threshold])
                    low_risks = sum([encs for day, level, encs in clusters if level >= mild_risk_threshold and level < moderate_risk_threshold])
                    no_risks = sum([encs for day, level, encs in clusters if level < mild_risk_threshold])
                    if symptoms != 0 and (exposed or infectious):
                        if high_risks != 0: symptomatic_infected_msgs["high"].append(high_risks)
                        if med_risks != 0: symptomatic_infected_msgs["moderate"].append(med_risks)
                        if low_risks != 0: symptomatic_infected_msgs["mild"].append(low_risks)
                        if no_risks != 0: symptomatic_infected_msgs["low"].append(no_risks)
                    if symptoms != 0 and not (exposed or infectious):
                        if high_risks != 0: symptomatic_uninfected_msgs["high"].append(high_risks)
                        if med_risks != 0:symptomatic_uninfected_msgs["moderate"].append(med_risks)
                        if low_risks != 0: symptomatic_uninfected_msgs["mild"].append(low_risks)
                        if no_risks != 0:symptomatic_uninfected_msgs["low"].append(no_risks)
                    if symptoms == 0 and (exposed or infectious):
                        if high_risks != 0: asymptomatic_infected_msgs["high"].append(high_risks)
                        if med_risks != 0:asymptomatic_infected_msgs["moderate"].append(med_risks)
                        if low_risks != 0: asymptomatic_infected_msgs["mild"].append(low_risks)
                        if no_risks != 0:asymptomatic_infected_msgs["low"].append(no_risks)
                    if symptoms == 0 and not (exposed or infectious):
                        if high_risks != 0: asymptomatic_uninfected_msgs["high"].append(high_risks)
                        if med_risks != 0: asymptomatic_uninfected_msgs["moderate"].append(med_risks)
                        if low_risks != 0: asymptomatic_uninfected_msgs["mild"].append(low_risks)
                        if no_risks != 0: asymptomatic_uninfected_msgs["low"].append(no_risks)

                if do_symptoms:
                    for symptom in reported_symptoms:

                        # symptomatic people
                        if symptoms != 0 and (exposed or infectious):
                            symptomatic_infected_symptoms[symptom] += 1
                        if symptoms != 0 and not (exposed or infectious):
                            symptomatic_uninfected_symptoms[symptom] += 1

            if do_symptoms:
                plot_symptoms(symptomatic_infected_symptoms, symptomatic_uninfected_symptoms, path, "symptoms_probs")



            if do_messages:
                datasets = {"symptomatic_infected_msgs": symptomatic_infected_msgs,
                            "symptomatic_uninfected_msgs": symptomatic_uninfected_msgs,
                            "asymptomatic_infected_msgs": asymptomatic_infected_msgs,
                            "asymptomatic_uninfected_msgs": asymptomatic_uninfected_msgs}

                for title, dataset in datasets.items():
                    counters = [Counter(list(x)) for x in dataset.values()]
                    msgs_np = np.zeros((4,  max([max(x.keys()) for x in counters])))
                    for idx1, counter in enumerate(counters):
                        for idx2, value in counter.items():
                            msgs_np[idx1, idx2-1] = value
                    datasets[title] = msgs_np
                    msgs_np_normed = msgs_np / msgs_np.sum(axis=0)
                    msgs_np_normed[np.isnan(msgs_np_normed)] = 0
                    plot_risk_messages_normed(msgs_np_normed, path, title)

                compare_risk(datasets["asymptomatic_infected_msgs"], datasets["asymptomatic_uninfected_msgs"], path)

                symptomatic_infected_msgs = np.array(list(symptomatic_infected_msgs.values()))
                symptomatic_uninfected_msgs = np.array(list(symptomatic_uninfected_msgs.values()))
                asymptomatic_infected_msgs = np.array(list(asymptomatic_infected_msgs.values()))
                asymptomatic_uninfected_msgs = np.array(list(asymptomatic_uninfected_msgs.values()))

                plot_risk_messages(symptomatic_infected_msgs, path, title="Symptomatic Infected Risk Messages Normed")
                plot_risk_messages(symptomatic_uninfected_msgs, path, title="Symptomatic Uninfected Risk Messages Normed")
                plot_risk_messages(asymptomatic_infected_msgs, path, title="Asymptomatic Infected Risk Messages Normed")
                plot_risk_messages(asymptomatic_uninfected_msgs, path, title="Asymptomatic Uninfected Risk Messages Normed")

            if do_reasons:
                plot_reasons(symptomatic_infected, path, title="Symptomatic Infected")
                plot_reasons(symptomatic_uninfected, path, title="Symptomatic Uninfected")

                plot_reasons(asymptomatic_infected, path, title="Asymptomatic Infected")
                plot_reasons(asymptomatic_uninfected, path, title="Asymptomatic Uninfected")

def plot_symptoms(symptomatic_infected_symptoms, symptomatic_uninfected_symptoms, path, title):
    all_symptoms = set(symptomatic_infected_symptoms.keys()).union(set(symptomatic_uninfected_symptoms))
    probs = {}
    abs_inf = {}
    abs_uninf = {}
    for symptom in all_symptoms:
        probs[symptom] = 100 * symptomatic_infected_symptoms[symptom] / (
                    symptomatic_infected_symptoms[symptom] + symptomatic_uninfected_symptoms[symptom])
        abs_inf[symptom] = symptomatic_infected_symptoms[symptom]
        abs_uninf[symptom] = symptomatic_uninfected_symptoms[symptom]

    # Plotting infected / total (abs)

    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout()
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(len(abs_inf)))

    ax.bar(np.arange(len(abs_inf)), list(abs_inf.values()), color="r", width=1.)
    ax.bar(np.arange(len(abs_uninf)), list(abs_uninf.values()), bottom=list(abs_inf.values()), color="b", width=1.)
    ax.legend(labels=["infected", "uninfected"])

    ax.set_xticklabels(list(abs_uninf.keys()))
    plt.xticks(rotation=90)
    ax.set_ylabel('Samples (Infected / All)')
    ax.set_title("Symptoms of Covid Infectees and Others")
    ax.set_xlabel("Symptom name")
    fig.subplots_adjust(bottom=0.3)

    plt.savefig(f"{path}/symptoms_abs.png")



    # Plotting infected / total (percentage)
    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout()
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(len(probs)))
    ax.bar(np.arange(len(probs)), list(probs.values()), color="b", width=1.)
    ax.set_xticklabels(list(probs.keys()))
    plt.xticks(rotation=90)
    ax.set_ylabel('Samples % (Infected / All)')
    ax.set_title(title)
    ax.set_xlabel("Symptom name")
    fig.subplots_adjust(bottom=0.3)

    plt.savefig(f"{path}/{title}.png")



def compare_risk(infected, uninfected, path, title="risk message comparison (infected on total)"):
    # infected / total
    proportions = np.zeros(infected.shape)
    for rec_level in range(0, 4):
        for num_encounters in range(0, infected.shape[1]):
            proportions[rec_level, num_encounters] = infected[rec_level, num_encounters] / \
                                                     (uninfected[rec_level, num_encounters] + infected[rec_level, num_encounters])
    proportions[np.isnan(proportions)] = 0

    # Plotting infected / total
    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout()
    ax = fig.add_subplot(111)
    colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
    X = np.arange(proportions.shape[0])
    prev = np.zeros(proportions.shape[1])

    for rec_level in X:
        ax.bar(np.arange(proportions.shape[1]), list(proportions)[rec_level], bottom=prev, color=colors[rec_level], width=0.8)
        prev += list(proportions)[rec_level]

    ax.set_ylabel('Proportion of Samples')
    ax.set_title(title)
    ax.set_xlabel("Num Encounters")
    ax.legend(labels=["low risk messages", "mild risk messages", "moderate risk messages", "high risk messages"])
    fig.subplots_adjust(bottom=0.3)

    plt.savefig(f"{path}/{title}.png")


def plot_risk_messages_normed(data, path, title):

    if not os.path.isdir(path):
        os.mkdir(path)

    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout()
    ax = fig.add_subplot(111)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    X = np.arange(data.shape[0])
    prev = np.zeros(data.shape[1])
    for rec_level in X:
        ax.bar(np.arange(data.shape[1]), list(data)[rec_level], bottom=prev, color=colors[rec_level], width=0.8)
        prev += list(data)[rec_level]
    ax.set_ylabel('Proportion of Samples')
    ax.set_title(title)
    ax.set_xlabel("Num Encounters")
    ax.legend(labels=["low risk messages", "mild risk messages", "moderate risk messages", "high risk messages"])
    fig.subplots_adjust(bottom=0.3)

    plt.savefig(f"{path}/{title}.png")


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
