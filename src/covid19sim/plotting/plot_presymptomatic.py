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
                print(label)
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
            print(f"{method} {label}: {all_positive}")
            all_negative = all_negative / np.expand_dims(all_negative.sum(1), 1)
            results.append((all_positive, all_negative, all_positive - all_negative))

        for k, t in enumerate(times):
            i = 0
            j = 0

            for method in data:
                fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7.5, 20))
                fig.suptitle(method, fontsize=30)
                j = j + i if i == 0 else j + i + 1

                for i, adoption in enumerate(data[method].keys()):
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

                    axs[0].bar(
                        list(range(length)),
                        result[0][k],
                        color="darkorange",
                        label="Day{}".format(t),
                    )
                    axs[0].set_title(
                        f"Presymptomatic Rec Levels (Adoption: {float(label)*100}%)",
                        y=1.06,
                    )
                    axs[0].set_xlabel(xlabel)
                    axs[0].set_ylabel("% Population", size="medium")
                    axs[0].set_xticks(list(range(length)))
                    axs[0].set_xticklabels(xticklabels)
                    axs[0].set_ylim(0, 1)
                    axs[0].set_yticks(list(range(0, 5)))
                    axs[0].set_yticklabels(["0", "20", "40", "60", "80", "100"])
                    axs[0].plot([0, length - 1], [0, 0], color="b", linewidth=1.0)

                    axs[1].bar(
                        list(range(length)),
                        result[1][k],
                        color="darkorange",
                        label="Day{}".format(t),
                    )
                    axs[1].set_title(
                        f"Susceptible Rec Levels (Adoption Rate: {float(label)*100}%)",
                        y=1.06,
                    )
                    axs[1].set_xlabel(xlabel)
                    axs[1].set_ylabel("% Population", size="medium")
                    axs[1].set_xticks(list(range(length)))
                    axs[1].set_xticklabels(xticklabels)
                    axs[1].set_ylim(0, 1)
                    axs[1].set_yticks(list(range(0, 5)))
                    axs[1].set_yticklabels(["0", "20", "40", "60", "80", "100"])
                    axs[1].plot([0, length - 1], [0, 0], color="b", linewidth=1.0)

                    axs[2].bar(
                        list(range(length)),
                        result[2][k],
                        color="darkorange",
                        label="Day{}".format(t),
                    )
                    axs[2].set_title(
                        f"Delta Rec Levels (Adoption Rate: {float(label)*100}%)", y=1.06
                    )
                    axs[2].set_xlabel(xlabel)
                    axs[2].set_ylabel("% Population", size="medium")
                    axs[2].set_xticks(list(range(length)))
                    axs[2].set_xticklabels(xticklabels)
                    axs[2].set_ylim(-1, 1)
                    # axs[0].set_yticks(list(range(0, 5)))
                    # axs[0].set_yticklabels(["0", "20", "40", "60", "80", "100"])
                    axs[2].plot([0, length - 1], [0, 0], color="b", linewidth=1.0)

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
                    print("Done.")
