import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from covid19sim.plotting.utils import (
    get_all,
    get_metrics,
)

def run(data, path, comparison_key):
    """
    data is a dictionnary that maps methods (bdt1, bdt1_norm, transformer etc.)
    to another dictionnary which has keys the values of the comparing key and
    values a dictionnary with the run's simulation configuration and pkl path

    e.g.
    comparison_key=APP_UPTAKE
    data:
      bdt1:
        -1:
          pkl: loaded_tracker.pkl
          conf: configuration_dict
        0.8415:
          pkl: loaded_tracker.pkl
          conf: configuration_dict
      bdt2:
        -1:
          pkl: loaded_tracker.pkl
          conf: configuration_dict
        0.8415:
          pkl: loaded_tracker.pkl
          conf: configuration_dict


    Args:
        data (dict): the data as method -> comparing value -> conf, pkl
        comparison_key (str): the key used to compare runs, like APP_UPTAKE
    """
    print("Preparing data...")
    pkls = []
    labels = []
    pkls_norm = []
    labels_norm = []

    for method in data:
        for key in data[method]:
            if "_norm" in method:
                pkls_norm.append([r["pkl"] for r in data[method][key].values()])
                labels_norm.append(f"{method}_{key}")
            else:
                pkls.append([r["pkl"] for r in data[method][key].values()])
                labels.append(f"{method}_{key}")

    for idx, label in enumerate(labels):
        print(f"{len(pkls[idx])} seeds for {label}")

    rows = get_all(pkl_types=pkls, labels=labels, normalized=False)
    lrows = set([r[0] for r in rows])
    labels = [label for label in labels if label in lrows]

    rows_norm = get_all(pkl_types=pkls_norm, labels=labels_norm, normalized=True)
    lrows_norm = set([r[0] for r in rows_norm])
    labels_norm = [label for label in labels_norm if label in lrows_norm]

    rows = rows + rows_norm

    n_seeds = None
    for mv in data.values():
        for cv in mv.values():
            n_seeds = len(cv)
            break
        break
    assert n_seeds is not None, "Could not find the number of seeds"
    df = pd.DataFrame(rows, columns=["type", "metric"] + list(np.arange(n_seeds) + 1))
    df["mean"] = df[list(np.arange(n_seeds) + 1)].mean(axis=1)
    df["stderr"] = df[list(np.arange(n_seeds) + 1)].sem(axis=1)

    ############
    ### /!\ Ordering should be consistent everywhere. i.e. _70, _60, _40
    ############

    xmetrics = [
        "f3",
        "f2",
        "f1",
        "f2_up",
        "effective_contacts",
        "outside_daily_contacts",
    ]
    ymetric = "percent_infected"

    colormap = [
        "mediumvioletred",
        "darkorange",
        "green",
        "red",
        "blue",
        "brown",
        "cyan",
        "gray",
        "olive",
        "orange",
        "pink",
        "purple",
        "royalblue",
    ]

    fig, axs = plt.subplots(
        nrows=math.ceil(len(xmetrics) / 2),
        ncols=2,
        figsize=(20, 20),
        dpi=100,
        sharey=True,
    )

    base_methods = sorted(data.keys())

    results = []
    print("| Method | % Rec Level 3 | % Rec Level 2 | % Rec Level 1 | % Infected |")
    print("|---|---|---|---|---|")
    for idx, method in enumerate(sorted(base_methods)):
        current_labels = sorted([lab for lab in labels if lab.startswith(method)])
        for i, lab in enumerate(current_labels):
            result = f"{lab} | "

            for axis_idx, xmetric in enumerate(xmetrics):
                if xmetric in ["f1", "f2", "f3"]:
                    x, xe = get_metrics(df, lab, xmetric)
                    y, ye = get_metrics(df, lab, ymetric)
                    result += f" {round(float(x)*100, 1)}% +/- {round(float(xe*100), 1)}% | "
            infected = f" {round(float(y * 100), 1)}% +/- {round(float(ye * 100), 1)}% |"
            result += infected
            print(result)
            results.append(result)

    print("\n\n\n")
    print("| Method | Restriction | % Infected | infection over baseline | restriction over baseline | Efficiency |")
    print("|---|---|---|---|---|---|")
    baseline_restriction = 1.6
    baseline_infection = 88.0
    for idx, method in enumerate(sorted(base_methods)):
        current_labels = sorted([lab for lab in labels if lab.startswith(method)])
        for i, lab in enumerate(current_labels):
            result = f"{lab} | "
            method_restriction = 0

            for axis_idx, xmetric in enumerate(xmetrics):
                x, xe = get_metrics(df, lab, xmetric)
                y, ye = get_metrics(df, lab, ymetric)
                if xmetric == "f1":
                    method_restriction += 0.25 * round(float(x * 100), 1)

                elif xmetric == "f2":
                    method_restriction += 0.5 * round(float(x * 100), 1)

                elif xmetric == "f3":
                    method_restriction += 1 * round(float(x * 100), 1)

            result += f"{method_restriction}% | "
            infected = f" {round(float(y * 100), 1)}% +/- {round(float(ye * 100), 1)}% |"
            result += infected

            infection_over_baseline = f" {float(((y * 100) - baseline_infection))} |"
            result += infection_over_baseline

            restriction_over_basleine = f" {float((method_restriction - baseline_restriction))} |"
            result += restriction_over_basleine

            efficiency = f" {float((y - baseline_infection) / (method_restriction - baseline_restriction) )}"
            result += efficiency
            print(result)
            results.append(result)
