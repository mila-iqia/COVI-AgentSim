import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt


def statistics(pkl_data, times, mode):

    data = pkl_data["human_monitor"]
    data = sorted(data.items(), key=lambda x: x[0])
    start_date = data[0][0]
    data = [((k - start_date).days, v) for k, v in data]

    human2symptom_day = dict()
    human_day2infectious = dict()
    human_day2susceptible = dict()
    human_day2risk_level = dict()
    human_day2rec_level = dict()

    if mode == "risk":
        positive = np.zeros((len(times), 16))
        negative = np.zeros((len(times), 16))
    if mode == "rec":
        positive = np.zeros((len(times), 4))
        negative = np.zeros((len(times), 4))

    for day, items in data:
        for item in items:
            human = item["name"]

            if item["n_reported_symptoms"] != 0:
                if human not in human2symptom_day:
                    human2symptom_day[human] = day
                else:
                    human2symptom_day[human] = min(human2symptom_day[human], day)

            if item["state"] == 2:
                human_day2infectious[(human, day)] = True

            if item["state"] == 0:
                human_day2susceptible[(human, day)] = True

            human_day2risk_level[(human, day)] = item["risk_level"]
            human_day2rec_level[(human, day)] = item["rec_level"]

    for k, t in enumerate(times):

        for human, day in human2symptom_day.items():
            past_day = day + t
            if past_day < 0:
                continue
            if human_day2infectious.get((human, past_day), False) is True:
                if mode == "risk":
                    r = human_day2risk_level[(human, past_day)]
                if mode == "rec":
                    r = human_day2rec_level[(human, past_day)]
                if r != -1:
                    positive[k][r] += 1

        for human, day in human_day2susceptible.keys():
            if mode == "risk":
                r = human_day2risk_level[(human, day)]
            if mode == "rec":
                r = human_day2rec_level[(human, day)]
            if r != -1:
                negative[k][r] += 1

    return positive, negative


def run(data, path, comparison_key, times=[-1, -2, -3], mode=None):

    # Options:
    # 1. "times" is a list, indicating the times we are interested in.
    # For example, [-1, -2, -3] means we want to get the plot for Day-1, Day-2, Day-3.
    # 2. "mode" is a string, which can be either 'risk' or 'rec' if None,
    # it will be both.
    if mode in {"rec", "risk"}:
        modes = [mode]
    elif mode is None:
        modes = ["rec", "risk"]
    else:
        raise ValueError("Unknown mode {}".format(mode))

    for mode in modes:
        print("Preparing data for times {} and mode {}...".format(times, mode))
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
                positive, negative = statistics(pkl, times, mode)
                positive = np.expand_dims(positive, 0)
                negative = np.expand_dims(negative, 0)
                all_positive.append(positive)
                all_negative.append(negative)
            all_positive = np.concatenate(all_positive, 0).sum(0)
            all_negative = np.concatenate(all_negative, 0).sum(0)
            all_positive = all_positive / np.expand_dims(all_positive.sum(1), 1)
            all_negative = all_negative / np.expand_dims(all_negative.sum(1), 1)
            results.append((all_positive, all_negative, all_positive - all_negative))

        for k, t in enumerate(times):
            i = 0
            j = 0

            for method in data:
                j = j + i if i == 0 else j + i + 1

                for i, adoption in enumerate(data[method].keys()):
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 10))
                    ax.set_title(f"{method} Day{t} (Adoption: {float(label) * 100}%)", fontsize=30)

                    label = label2pkls[i + j][0]
                    result = results[i + j]

                    # change the font size
                    font = {"family": "DejaVu Sans", "size": 18}
                    matplotlib.rc("font", **font)
                    plt.rcParams["axes.labelsize"] = 18

                    if mode == "risk":
                        length = 16
                        xlabel = "Risk Level"
                        xticklabels = [
                            0,
                            "",
                            "",
                            "",
                            4,
                            "",
                            "",
                            "",
                            8,
                            "",
                            "",
                            "",
                            12,
                            "",
                            "",
                            "",
                        ]
                    if mode == "rec":
                        length = 4
                        xlabel = "Rec Level"
                        xticklabels = [0, 1, 2, 3]

                    ax.bar(
                        [_ * 7 + 2 for _ in range(length)],
                        result[0][k],
                        color="darkorange",
                        label=f"Presymptomatic Rec Levels",
                    )
                    ax.bar(
                        [_ * 7 + 3 for _ in range(length)],
                        result[1][k],
                        color="darkslateblue",
                        label=f"Susceptible Rec Levels",
                    )
                    ax.bar(
                        [_ * 7 + 4 for _ in range(length)],
                        result[2][k],
                        color="orangered",
                        label=f"Delta Rec Levels",
                    )
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("% Population", size="medium")
                    ax.set_xticks([_ * 7 + 3 for _ in range(length)])
                    ax.set_xticklabels(xticklabels)
                    ax.set_ylim(-1, 1)
                    ax.set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    ax.set_yticklabels(["-100", "-80", "-60", "-40", "-20", "0", "20", "40", "60", "80", "100"])
                    ax.plot([0, length * 7 - 1], [0, 0], color="b", linewidth=1.0)
                    ax.legend()

                    plt.subplots_adjust(
                        wspace=0.5,
                        hspace=0.3,  # left=0.05, bottom=0.05, right=0.99, top=0.95,
                    )
                    dir_path = (
                        path
                        / "presymptomatic"
                        / f"statistics_day{t}"
                        / f"adoption_{label}"
                    )
                    fig_path = dir_path / f"{method}_{mode}.png"
                    print(f"Saving Figure {str(fig_path)}...", end="", flush=True)
                    os.makedirs(dir_path, exist_ok=True)
                    plt.savefig(fig_path)
                    plt.close("all")
                    plt.clf()
                    print("Done.")
