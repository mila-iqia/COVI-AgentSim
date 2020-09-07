import math
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from covid19sim.plotting.utils import (
    get_title,
    get_all,
    get_metrics,
)

ms = 6 * 2 * 1.5
capsize = 4
markers = ["P", "s", "X", "d", ".", "h", "^", "*", "v", "p", "x", "1"]
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

metric_name_map = {
    "fq": "False Quarantine",
    "f3": "False level-3",
    "f2": "False level-2",
    "f1": "False level-1",
    "f2_up": "False level >= 2",
    "effective_contacts": "Effective Contacts",
    "outside_daily_contacts": "Outside Daily Contacts",
    "f0": "False Susceptible or Recovered"
}


def plot_all_metrics(
    axs,
    data,
    label,
    color,
    marker,
    xmetrics,
    ymetric,
    normalized=False,
    capsize=4,
    ms=6 * 2 * 1.5,
):
    alpha = 1.0
    if normalized:
        alpha = 0.5
    for axis_idx, xmetric in enumerate(xmetrics):
        x, xe = get_metrics(data, label, xmetric)
        y, ye = get_metrics(data, label, ymetric)
        axs[axis_idx].errorbar(
            x=x,
            y=y,
            xerr=xe,
            yerr=ye,
            linestyle="None",
            capsize=capsize,
            c=color,
            marker=marker,
            ms=ms,
            label=label,
            alpha=alpha,
        )
    return axs


def get_line2D(value, color, marker, is_method=True):

    if is_method:
        return Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markeredgecolor="k",
            markerfacecolor=color,
            markersize=15,
            label=get_title(value),
        )

    return Line2D(
        [0],
        [0],
        color="none",
        lw=2,
        marker=marker,
        markerfacecolor="black",
        markersize=15,
        label=get_title(value),
    )

def plot_and_save_ymetric(data, ymetric, xmetrics, base_methods, labels, labels_norm, path):
    """
    Plots `ymetric` for all `xmetrics` in pareto.

    Args:
        data (pd.DataFrame): Dataframe with rows containing type, metric and values for each seed
        ymetric (str): metric that should be in `metric` column of `data`
        xmetrics (list): list of strings where each element will be plotted on the x-axis
        base_methods (list): list of strings where each element is an intervention name (without any key for comparison)
        labels (list): each element is a string representing the `method_key`
        labels_norm (list): each element is a string representing the `method_key` for normalized runs
        path (str): path where to save the figure
    """
    fig, axs = plt.subplots(
        nrows=math.ceil(len(xmetrics) / 2),
        ncols=2,
        figsize=(20, 20),
        dpi=100,
        sharey=True,
    )
    axs = [i for j in axs for i in j]

    method_legend = []
    compare_legend = []
    for idx, method in enumerate(sorted(base_methods)):
        print("Plotting", method, "...")
        current_labels = sorted([lab for lab in labels if lab.startswith(method)])
        current_labels_norm = sorted(
            [lab for lab in labels_norm if lab.startswith(method)]
        )
        current_color = colormap[idx] if method != "unmitigated" else "#34495e"
        method_legend.append(
            get_line2D(method, current_color, None, True)
        )
        for i, lab in enumerate(current_labels):

            current_marker = markers[i]
            if idx == 0:
                compare_legend.append(
                    get_line2D(
                        lab.split("_")[-1],
                        current_color,
                        current_marker,
                        False,
                    )
                )
            plot_all_metrics(
                axs,
                data,
                lab,
                current_color,
                current_marker,
                xmetrics,
                ymetric,
                capsize=capsize,
                ms=ms,
                normalized=False,
            )
        for i, lab in enumerate(current_labels_norm):
            current_marker = markers[i]
            plot_all_metrics(
                axs,
                data,
                lab,
                current_color,
                current_marker,
                xmetrics,
                ymetric,
                capsize=capsize,
                ms=ms,
                normalized=True,
            )

    # grids
    for axis_id, ax in enumerate(axs):
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(True, axis="y", alpha=0.3)

        ax.set_xlabel(metric_name_map[xmetrics[axis_id]], size=40)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)

    # ylabel
    if ymetric == "percent_infected":
        ylabel = "Fraction infected"
    elif ymetric == "proxy_r":
        ylabel = "Proxy $\hat{R_t}$"
    elif ymetric == "r":
        ylabel = "$R_t$"
    elif ymetric == "f0":
        ylabel = "False Susceptible or Recoverd"
    ylab = fig.text(-0.05, 0.5, ylabel, va="center", rotation="vertical", size=50)

    if ymetric in ["proxy_r", "r"]:
        for ax in axs:
            ax.plot([0, ax.get_xlim()[1]], [1.0, 1.0], "-.", c="gray", alpha=0.3, label="Rt = 1.0")

    spttl = fig.suptitle(
        "Comparison of tracing methods across different adoption rates",
        fontsize=50,
        y=1.15,
    )
    if len(method_legend) % 2 != 0:
        legend = (
            method_legend
            + [
                Line2D(
                    [0],
                    [0],
                    color="none",
                    marker="",
                    markeredgecolor="k",
                    markerfacecolor=None,
                    markersize=1,
                    label=" ",
                )
            ]
            + compare_legend
        )
    else:
        legend = method_legend + compare_legend
    lgd = fig.legend(
        handles=legend,
        loc="upper center",
        ncol= 3,# idx, # + 1,
        fontsize=25,
        bbox_to_anchor=(0.5, 1.1),
    )
    plt.tight_layout()
    save_path = Path(path) / f"pareto_adoption/pareto_adoption_all_metrics_vs_{ymetric}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    print("Saving Figure {}...".format(save_path.name), end="", flush=True)
    fig.savefig(
        str(save_path),
        dpi=100,
        bbox_extra_artists=(lgd, ylab, spttl),
        bbox_inches="tight",
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
        "fq",
        "f3",
        "f2",
        # "f1",
        "f2_up",
        "effective_contacts",
        "outside_daily_contacts",
    ]
    base_methods = sorted(data.keys())
    plot_and_save_ymetric(df, "proxy_r", xmetrics, base_methods, labels, labels_norm, path)
    plot_and_save_ymetric(df, "percent_infected", xmetrics, base_methods, labels, labels_norm, path)
    plot_and_save_ymetric(df, "f0", xmetrics, base_methods, labels, labels_norm, path)


    print("\n\nLine Plot")
    fig, axs = plt.subplots(figsize=(20, 20), dpi=100, sharey=True)
    save_path = Path(path) / "pareto_adoption/pareto_front.png"
    method_legend = []
    # Make a lineplot version of the plot
    for idx, method in enumerate(sorted(base_methods)):
        print("Plotting", method, "...")
        xs = []
        ys = []
        yerrs = []
        current_labels = sorted([lab for lab in labels if lab.startswith(method)])
        current_color = colormap[idx] if method != "unmitigated" else "#34495e"
        method_legend.append(
            get_line2D(method, current_color, None, True)
        )
        for i, lab in enumerate(current_labels):
            x, xe = get_metrics(df, lab, "healthy_effective_contacts")
            y, ye = get_metrics(df, lab, "proxy_r")
            xs.append(x.item())
            ys.append(y.item())
            yerrs.append(ye.item())

        plt.errorbar(xs, ys, yerr=yerrs, label=lab)
        axs.fill_between(xs, ys - np.array(yerrs)/2, ys + np.array(yerrs)/2, alpha=0.2)

    lgd = axs.legend(
        loc="upper center",
        ncol= 3,
        fontsize=25,
        bbox_to_anchor=(0.5, 1.1),
    )
    spttl = plt.title(
        "Comparison of tracing methods across different GLOBAL_EFFECTIVE_MOBILITY_SCALES",
        fontsize=50,
        y=1.15,
    )
    axs.set_xlabel("Healthy Effective Contacts", size=40)
    axs.set_ylabel("R_t (infectees / recovered infectors)", size=40)
    for tick in axs.xaxis.get_major_ticks():
        tick.label.set_fontsize(30)

    for tick in axs.yaxis.get_major_ticks():
        tick.label.set_fontsize(30)

    plt.savefig(str(save_path))

    print("Done.")
