import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats as sps

from covid19sim.plotting.utils.extract_data import (
    get_data,
    get_human_rec_levels,
    get_human_states,
)


def get_all_false(filename=None, data=None, normalized=False):
    data = get_data(filename, data)
    intervention_day = data["intervention_day"]
    if intervention_day < 0:
        intervention_day = 0
    states = get_human_states(data=data)
    states = states[:, intervention_day:]
    rec_levels = get_human_rec_levels(data=data, normalized=normalized)

    false_level3 = np.sum(((states == 0) | (states == 3)) & (rec_levels == 3), axis=0)
    false_level2 = np.sum(((states == 0) | (states == 3)) & (rec_levels == 2), axis=0)
    false_level1 = np.sum(((states == 0) | (states == 3)) & (rec_levels == 1), axis=0)
    false_level1_above = np.sum(
        ((states == 0) | (states == 3))
        & ((rec_levels == 1) | (rec_levels == 2) | (rec_levels == 3)),
        axis=0,
    )
    false_level2_above = np.sum(
        ((states == 0) | (states == 3)) & ((rec_levels == 2) | (rec_levels == 3)),
        axis=0,
    )
    return (
        false_level3 / states.shape[0],
        false_level2 / states.shape[0],
        false_level1 / states.shape[0],
        false_level1_above / states.shape[0],
        false_level2_above / states.shape[0],
    )


def get_proxy_r(data):
    total_infected = 0
    for k in data["humans_state"].keys():
        total_infected += any(z == "I" for z in data["humans_state"][k][5:])
    return sum(data["cases_per_day"][5:]) / total_infected


def get_fq_r(filename=None, data=None, normalized=False):
    assert filename is not None or data is not None
    if data is None:
        data = pickle.load(open(filename, "rb"))

    f3, f2, f1, f1_up, f2_up = get_all_false(data=data, normalized=normalized)
    x = [i[-5:].mean() for i in [f3, f2, f1, f1_up, f2_up]]

    intervention_day = data["intervention_day"]
    od = np.mean(data["outside_daily_contacts"][intervention_day:])
    ec = data["effective_contacts_since_intervention"]

    # percent_infected
    y = sum(data["cases_per_day"]) / data["n_humans"]

    # R
    z = get_effective_R(data)

    # proxy_r
    a = get_proxy_r(data)

    return x, y, z, a, od, ec


def get_mean_fq_r(filenames=None, pkls=None, normalized=False):
    assert filenames is not None or pkls is not None
    if pkls is not None:
        _tmp = [(None, {"pkl": pkl}) for pkl in pkls]
    elif filenames is not None:
        _tmp = [(filename, None) for filename in filenames]
    else:
        raise ValueError("filenames and pkls are None")

    metrics = {
        "f3": [],
        "f2": [],
        "f1": [],
        "f1_up": [],
        "f2_up": [],
        "percent_infected": [],
        "r": [],
        "proxy_r": [],
        "outside_daily_contacts": [],
        "effective_contacts": [],
    }
    for filename, pkl in _tmp:
        x, y, z, a, od, ec = get_fq_r(
            filename=filename, data=pkl["pkl"], normalized=normalized
        )
        metrics["f3"].append(x[0])
        metrics["f2"].append(x[1])
        metrics["f1"].append(x[2])
        metrics["f1_up"].append(x[3])
        metrics["f2_up"].append(x[4])
        metrics["percent_infected"].append(y)
        metrics["r"].append(z)
        metrics["proxy_r"].append(a)
        metrics["outside_daily_contacts"].append(od)
        metrics["effective_contacts"].append(ec)

    return metrics


def get_effective_R(data):
    GT = data["generation_times"]
    a = 4
    b = 0.5
    window_size = 5
    ws = [sps.gamma.pdf(x, a=GT, loc=0, scale=0.9) for x in range(window_size)]
    last_ws = ws[::-1]
    cases_per_day = data["cases_per_day"]

    lambda_s = []
    rt = []
    for i in range(len(cases_per_day)):
        if i < window_size:
            last_Is = cases_per_day[:i]
        else:
            last_Is = cases_per_day[(i - window_size) : i]

        lambda_s.append(sum(x * y for x, y in zip(last_Is, last_ws)))
        last_lambda_s = sum(lambda_s[-window_size:])
        rt.append((a + sum(last_Is)) / (1 / b + last_lambda_s))
    return np.mean(rt[-5:])


def get_all(filename_types=None, pkl_types=None, labels=[], normalized=False):
    if pkl_types is not None:
        tmp = [(None, pkls) for pkls in pkl_types]
    elif filename_types is not None:
        tmp = [(filenames, None) for filenames in filename_types]
    else:
        raise ValueError("filename_types and pkl_types are None")

    _rows = []

    for i, (filenames, pkls) in enumerate(tmp):
        metrics = get_mean_fq_r(filenames=filenames, pkls=pkls, normalized=normalized)
        for key, val in metrics.items():
            _rows.append([labels[i], key] + val)
    return _rows


def get_metrics(data, label, metric):
    tmp = data[(data["type"] == label) & (data["metric"] == metric)]
    return tmp["mean"], tmp["stderr"]


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


def get_line2D(value, color, marker, is_method=True, compare="APP_UPTAKE"):
    legends = {
        "APP_UPTAKE": lambda x: f"{float(x) * 100 * 0.71203:.0f}% Adoption Rate",
        "method": {
            "bdt1": "1st Order Binary Tracing",
            "bdt2": "2nd Order Binary Tracing",
            "heuristicv1": "Heuristic (v1)",
            "heuristicv2": "Heuristic (v2)",
            "transformer": "Transformer",
            "transformer-[0, 0, 0]": "Transformer-[0, 0, 0]",
            "transformer-[0, 1, 2]": "Transformer-[0, 1, 2]",
            "transformer-[1, 3, 5]": "Transformer-[1, 3, 5]",
            "transformer>twilight-[0, 1, 2]": "Transformer-twilight-[0, 1, 2]",
            "transformer>rosy-[0, 1, 2]": "Transformer-rosy-[0, 1, 2]",
            "transformer>vague-[0, 1, 2]": "Transformer-vague-[0, 1, 2]",
            "linreg": "Linear Regression",
            "mlp": "MLP",
            "unmitigated": "Unmitigated",
            "oracle": "Oracle",
        },
        "REC_LEVEL_THRESHOLDS": lambda x: f"{''.join(x)} Thresholds",
    }

    if is_method:
        return Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markeredgecolor="k",
            markerfacecolor=color,
            markersize=15,
            label=legends["method"][value],
        )

    return Line2D(
        [0],
        [0],
        color="none",
        lw=2,
        marker=marker,
        markerfacecolor="black",
        markersize=15,
        label=legends[compare](value),
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

    ms = 6 * 2 * 1.5
    capsize = 4
    markers = ["P", "s", "X", "d", ".", "h", "^", "*", "v"]
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
    axs = [i for j in axs for i in j]

    base_methods = sorted(set([lab.split("_")[0] for lab in labels + labels_norm]))

    method_legend = []
    compare_legend = []
    for idx, method in enumerate(sorted(base_methods)):
        print("Plotting", method, "...")
        # import pdb; pdb.set_trace()
        current_labels = sorted([lab for lab in labels if lab.startswith(method)])
        current_labels_norm = sorted(
            [lab for lab in labels_norm if lab.startswith(method)]
        )
        current_color = colormap[idx] if method != "unmitigated" else "#34495e"
        method_legend.append(
            get_line2D(method, current_color, None, True, comparison_key)
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
                        comparison_key,
                    )
                )
            plot_all_metrics(
                axs,
                df,
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
                df,
                lab,
                current_color,
                current_marker,
                xmetrics,
                ymetric,
                capsize=capsize,
                ms=ms,
                normalized=True,
            )

    metric_name_map = {
        "f3": "False level-3",
        "f2": "False level-2",
        "f1": "False level-1",
        "f2_up": "False level >= 2",
        "effective_contacts": "Effective Contacts",
        "outside_daily_contacts": "Outside Daily Contacts",
    }

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
    ylab = fig.text(-0.05, 0.5, ylabel, va="center", rotation="vertical", size=50)

    if ymetric in ["proxy_r", "r"]:
        for ax in axs:
            ax.plot([0, 1.0], [1.0, 1.0], "-.", c="gray", alpha=0.3, label="Rt = 1.0")

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
    save_path = Path(path) / "pareto_adoption_all_metrics.png"
    print("Saving Figure {}...".format(save_path.name), end="", flush=True)
    fig.savefig(
        str(save_path),
        dpi=100,
        bbox_extra_artists=(lgd, ylab, spttl),
        bbox_inches="tight",
    )
    print("Done.")
