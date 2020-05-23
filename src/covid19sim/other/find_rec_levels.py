"""
Runnable script to find the best thresholds to assign
recommendation levels from risks or risk_levels.

Instrunctions below in __main__ section.
Further details in docs/find_rec-levels.md
"""
import numpy as np
import dill
from collections import defaultdict
import datetime
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pdb
import matplotlib
import matplotlib.pyplot as plt


def proba_to_risk(probas, mapping):
    """
    Given risk probabilities and a reference risk mapping, give the risk_level

    Args:
        probas (list | np.array): the risks to convert to risk_level
        mapping (list | np.array): the reference bins

    Returns:
        list | np.array: risk_levels
    """
    return np.maximum(np.searchsorted(mapping, probas, side="left") - 1, 0)


def custom_f1_weighted(y_true, y_pred, costs):
    """
    Function to make f1_score support a dummy argument (costs)
    so that f1 computations and error computations have the same interface
    """
    return f1_score(y_true, y_pred, average="weighted")


def custom_f1_micro(y_true, y_pred, costs):
    """
    Function to make f1_score support a dummy argument (costs)
    so that f1 computations and error computations have the same interface
    """
    return f1_score(y_true, y_pred, average="micro")


def custom_f1_macro(y_true, y_pred, costs):
    """
    Function to make f1_score support a dummy argument (costs)
    so that f1 computations and error computations have the same interface
    """
    return f1_score(y_true, y_pred, average="macro")


def report(y_true, y_pred, score):
    """
    Create a report containing:
        * confusion matrix
        * 3 f1 scores
        * sklearn's classification report with per-class precision, recall, f1
        * current best score (either f1-based or error-based)

    Args:
        y_true (list): true color indices
        y_pred (list): threshold-based predictions of color indices
        score (float): current best score

    Returns:
        str: the report to print or write
    """
    cm = confusion_matrix(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_mic = f1_score(y_true, y_pred, average="micro")
    target_names = sorted(color_to_id.keys(), key=lambda x: color_to_id[x])
    cr = classification_report(y_true, y_pred, target_names=target_names)

    s = "\n>>> ----------------------------------------------------- <<<\n"
    s += "\n{}\n{}\n".format(cm, cr)
    s += "{:.3f} f1_weighted".format(f1_weighted) + "\n"
    s += "{:.3f} f1_mic".format(f1_mic) + "\n"
    s += "{:.3f} f1_mac".format(f1_mac) + "\n"
    s += "=> NEW best score, for threshold {}: {}\n".format(model, score)
    s += ">>> ----------------------------------------------------- <<<\n\n"
    return s


def get_hist(y_best, risk_attributes, risk_levels=False):
    """
    Compute count-based histogram for predicted labels

    Args:
        y_best (list): predicted color indices
        risk_attributes (list): list of simulation's risk_attributes
        risk_levels (bool, optional): whether to use risk or risk_level. Defaults to False.

    Returns:
        dict: for each color, a defaultdict(int) = {risk | risk_level : count_for_risk}
    """
    histograms = {c: defaultdict(int) for c in id_to_color}
    for i, r in enumerate(risk_attributes):
        yb = y_best[i]
        histograms[yb][r["risk_level" if risk_levels else "risk"]] += 1
    return histograms


def data_str(v):
    """
    Get a string representation of a data value:
      v itself if not list or dict or tuple
      len(v) otherwise

    Args:
        v (Any): value to print

    Returns:
        str: string for v
    """
    return v if not isinstance(v, (list, dict, tuple)) else "{} items".format(len(v))


def get_category(risk_attr):
    """
    Assigns a category to a given risk attribute,
    that is to a human at a given hour of a given day

    Args:
        risk_attr (dict): dictionnary representing a human's risk at a given time

    Returns:
        str: category for this person
    """

    if risk_attr["test"]:
        return "D"

    if risk_attr["infectious"] and risk_attr["symptoms"] == 0:
        return "B"

    if risk_attr["infectious"] and risk_attr["symptoms"] > 0:
        return "C"

    if risk_attr["exposed"]:
        return "A"

    if risk_attr["order_1_is_tested"]:
        return "J"

    if risk_attr["order_1_is_symptomatic"]:
        return "I"

    if risk_attr["order_1_is_presymptomatic"]:
        return "H"

    if risk_attr["order_1_is_exposed"]:
        return "E"

    return "K"


def sample_thresholds(n=1):
    """
    Samples uniformly a set of thresholds:
    r1 = uniform(0, 1)
    r2 = uniform(r1, 1)
    r3 = uniform(r2, 1)

    test:
    for r1, r2, r3 in sample_thresholds(1000):
        assert 0 < r1 < r2 < r3 < 1

    Args:
        n (int, optional): Number of thresholds triplets to sample. Defaults to 1.

    Returns:
        np.array: nx3 array of sampled thresholds
    """
    r1 = np.random.uniform(0, 0.4, size=n)
    r2 = np.random.uniform(r1, 1, size=n)
    r3 = np.random.uniform(r2, 1, size=n)
    return np.array([r1, r2, r3]).T


def predict_color_id(risks, thresholds):
    """
    Get the color_id prediction for the risks according to thresholds:

        risk < thresholds[0] => 0
        risk >= thresholds[0] => 1
        risk >= thresholds[1] => 2
        risk >= thresholds[2] => 3

    Args:
        risks (list): float, risks
        thresholds (list): 3 values describing the risk-thresholds for colors

    Returns:
        np.array: array as long as risks where they have been categorized as ints
    """
    predictions = np.zeros_like(risks)
    for i, r in enumerate(thresholds):
        predictions[risks > r] = i + 1
    return predictions.astype(int)


def risk_level_thresholds(n_levels=16):
    """
    Compute all the possible risk-level thresholds.
    Values range in [0:n_levels[:

    thresholds = [2, 5, 9]:
             risk < 2  => green
        2 <= risk < 5  => yellow
        5 <= risk < 9  => orange
        9 <= risk      => red

    Args:
        n_levels (int, optional): Number of risk-levels. Defaults to 16.

    Returns:
        np.array: nx3 array of all thresholds
    """
    models = []
    for i in range(n_levels):
        for j in range(i + 1, n_levels):
            for k in range(j + 1, n_levels):
                models.append([i, j, k])
    return np.array(models)


def error_rate(y_true, y_pred, costs):
    """
    Compute the weighted error-rate depending on the categories' weights

    Args:
        y_true (list): true color indices
        y_pred (list): predicted color indices
        costs (list): cost-per prediction error, created from the cost of error
            per category

    Returns:
        float: total weighted error
    """
    error = 0
    error = (y_pred != y_true).astype(int) * costs
    error = error.sum()  # / costs.sum()
    return error


if __name__ == "__main__":
    """
                                      HOW TO USE

    Data:
        This script expects pickled data that was output by ~ `python run.py ... tune=True`,
        meaning the city.tracker's data parsed by covid19sim.utils.extract_tracker_data

    Usage:

        $ python find_rec_levels.py --data path/to/data.pkl --risk_levels
        or
        $ python find_rec_levels.py --data path/to/data.pkl --samples 10000
        or
        $ python find_rec_levels.py --data path/to/data.pkl --risk_levels --plot_model "[1, 4, 8]"
        or
        $ python find_rec_levels.py --help

    File org:
        1. Constants used in exectution
        2. Available command-line arguments
        3. Parse & prepare data
        4. Sample thresholds (risks) or create list of all possible ones (risk_levels)
        5. Evaluate each sample
        6. Plot best model performance with 4 lines:
            a. For each ground-truth color (get_category & category_to_color)
               plot the risk/risk_level distribution
            b. Same, log-scale
            c. For each threshold color, plot the risk distribution
            d. Same, log-scale
    """

    # -----------------------
    # -----  Constants  -----
    # -----------------------
    ptrace = pdb.set_trace  # debug only
    filename = "/Users/victor/Downloads/"
    filename += "tracker_data_n_300_seed_0_20200511-202514_transformer_no_modif.pkl"

    category_to_color = {
        "A": "RED",
        "B": "RED",
        "C": "RED",
        "D": "RED",
        "E": "YELLOW",
        "H": "YELLOW",
        "I": "ORANGE",
        "J": "ORANGE",
        "K": "GREEN",
    }
    color_to_id = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
    id_to_color = {v: k for k, v in color_to_id.items()}
    risk_level_mapping = [
        0.0,
        0.0001,
        0.02790058831805188,
        0.056507214366051786,
        0.09964329702354373,
        0.12181110126553577,
        0.18458727162787497,
        0.28600519445013506,
        0.3628629063698628,
        0.44121387005841506,
        0.4954098649943365,
        0.5819137774823899,
        0.6232547997952577,
        0.7266351205574721,
        0.8874807435526698,
        1.0,
        1.065126768029105,
    ]

    # How much we care about errors made by the predictor (thresholds) on these
    # categories. If all set to 1 then the error is just the sum of all
    # misclassifications ~= 1 - accuracy
    costs_for_category = {
        "A": 1.5,
        "B": 2.5,
        "C": 2,
        "D": 2,
        "E": 1,
        "H": 1,
        "I": 1,
        "J": 1.5,
        "K": 1,
    }

    # -----------------------------
    # -----  Parse Arguments  -----
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=filename,
        help="Path to covid19sim.run's tune pickled data",
    )
    parser.add_argument(
        "--score",
        type=str,
        default="error",
        help="How to measure success? f1_micro, f1_macro, f1_weighted "
        + "or custom error (costs_for_category) ?",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="If randomly searching for risk thresholds, "
        + "how many random thresholds to samples",
    )
    parser.add_argument(
        "--day", type=int, default=0, help="Find rec-level for a specific day"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="When plotting histograms, number of bins to use",
    )
    parser.add_argument(
        "--risk_levels",
        default=False,
        action="store_true",
        help="If true, find rec_levels from risk_levels, otherwise it's from risks",
    )
    parser.add_argument(
        "--bar",
        default=False,
        action="store_true",
        help="Default behavior is to plot hists but you may want to use bar plots",
    )
    parser.add_argument(
        "--plot_model",
        default="",
        type=str,
        help="Bypass the search and get the plots for a specific model which should"
        + " be a string representing a python list like '[1, 2, 3]' ",
    )
    parser.add_argument(
        "--ignore_categories",
        default="AE",
        type=str,
        help="Categories to ignore in the data (such samples are filtered out)",
    )
    opts = parser.parse_args()

    day = opts.day

    assert Path(opts.data).exists()
    assert opts.samples > 0
    if opts.plot_model:
        opts.samples = 1

    if opts.score == "f1_micro":
        score_function = custom_f1_micro
        compare = "max"
        best_score = 0
    elif opts.score == "f1_macro":
        score_function = custom_f1_macro
        compare = "max"
        best_score = 0
    elif opts.score == "f1_weighted":
        score_function = custom_f1_weighted
        compare = "max"
        best_score = 0
    elif opts.score == "error":
        score_function = error_rate
        compare = "min"
        best_score = 1e12
    else:
        raise ValueError("unknown score function: {}".format(opts.score))

    # ---------------------------------------
    # -----  Prepare Ground Truth Data  -----
    # ---------------------------------------
    data = dill.load(open(opts.data, "rb"))
    print("Loaded", opts.data)
    print(
        "\n".join("{:35}: {}".format(k, data_str(v)) for k, v in sorted(data.items()))
    )

    start_stamp = data["risk_attributes"][0]["timestamp"]
    risk_attributes = [
        r
        for r in data["risk_attributes"]
        if (r["timestamp"] - start_stamp).days >= day
        and (r["timestamp"] - start_stamp).days < (day + 1 if day else 1e12)
        and get_category(r) not in set(opts.ignore_categories)
    ]

    categories = [get_category(r) for r in risk_attributes]
    colors = [category_to_color[c] for c in categories]
    y_true = np.array([color_to_id[c] for c in colors])
    costs = np.array([costs_for_category[c] for c in categories])

    # -------------------------------
    # -----  Sample Thresholds  -----
    # -------------------------------
    if opts.risk_levels:
        models = risk_level_thresholds()
        X = np.array([r["risk_level"] for r in risk_attributes])
    else:
        models = sample_thresholds(opts.samples)
        X = np.array([r["risk"] for r in risk_attributes])

    best_model = None

    # ---------------------------------
    # -----  Evaluate Thresholds  -----
    # ---------------------------------
    if not opts.plot_model:
        for i, model in enumerate(models):
            y_pred = predict_color_id(X, model)
            # https://datascience.stackexchange.com/questions/15989/micro-
            # average-vs-macro-average-performance-in-a-multiclass-classification-settin
            score = score_function(y_true, y_pred, costs)
            if i % 25 == 0:
                print("Test model {} / {}".format(i, len(models - 1)), end="\r")

            if (compare == "min" and score < best_score) or (
                compare == "max" and score > best_score
            ):
                f1_weighted = f1_score(y_true, y_pred, average="weighted")
                # By definition a confusion matrix C is such that C_ij is equal to
                # the number of observations known to be in group i and predicted
                # to be in group j.
                best_score = score
                best_model = model
                print(report(y_true, y_pred, score))

        print("\nEnd of search, plotting")
    # ---------------------------
    # -----  END OF SEARCH  -----
    # ---------------------------

    # -----------------------------
    # -----  PLOT BEST MODEL  -----
    # -----------------------------

    # matplotlib.use("MacOSX")

    if opts.plot_model:
        best_model = (
            opts.plot_model.replace("[", "")
            .replace("]", "")
            .replace(" ", "")
            .split(",")
        )
        best_model = [float(m) for m in best_model]

    y_best = predict_color_id(X, best_model)
    error = score_function(y_true, y_best, costs)
    if opts.bar:
        histograms = {c: defaultdict(int) for c in id_to_color}
        true_histograms = {c: defaultdict(int) for c in id_to_color}
        for i, r in enumerate(risk_attributes):
            yb = y_best[i]
            yt = y_true[i]
            histograms[yb][r["risk_level" if opts.risk_levels else "risk"]] += 1
            true_histograms[yt][r["risk_level" if opts.risk_levels else "risk"]] += 1
    else:
        histograms = {c: [] for c in id_to_color}
        true_histograms = {c: [] for c in id_to_color}
        for i, r in enumerate(risk_attributes):
            yb = y_best[i]
            yt = y_true[i]
            histograms[yb].append(r["risk_level" if opts.risk_levels else "risk"])
            true_histograms[yt].append(r["risk_level" if opts.risk_levels else "risk"])

    fig, axes = plt.subplots(4)
    ax1, ax2, ax3, ax4 = axes
    for hist in [3, 0, 2, 1]:
        for i, ax in enumerate(axes):
            if i > 1:
                print("plot histograms", hist, id_to_color[hist].lower())
                if opts.bar:
                    ax.bar(
                        histograms[hist].keys(),
                        histograms[hist].values() or [0],
                        width=0.1,
                        color=id_to_color[hist].lower(),
                    )
                else:
                    ax.hist(
                        histograms[hist] or [0],
                        bins=opts.bins if not opts.risk_levels else 16,
                        color=id_to_color[hist].lower(),
                        alpha=0.5,
                    )
            else:
                print("plot true_histograms", hist, id_to_color[hist].lower())
                if opts.bar:
                    ax.bar(
                        true_histograms[hist].keys(),
                        true_histograms[hist].values() or [0],
                        width=0.1 * (4 - hist),
                        color=id_to_color[hist].lower(),
                    )
                else:
                    ax.hist(
                        true_histograms[hist] or [0],
                        bins=opts.bins if not opts.risk_levels else 16,
                        color=id_to_color[hist].lower(),
                        alpha=0.5,
                    )
    ax2.set_yscale("log")
    ax4.set_yscale("log")

    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_title(
                "{} Reports (1 / human / hour)  |  Thresholds: {:.3f} {:.3f} {:.3f} | {} {:.3f}".format(
                    len(risk_attributes), *best_model, opts.score, error
                )
            )
        ax.set_xticks(
            np.arange(0, 1.1, 0.1) if not opts.risk_levels else np.arange(0, 16)
        )
        if i == 3:
            ax.set_xlabel("risk" if not opts.risk_levels else "risk_level")
        log = "Log-" if i in {1, 3} else ""
        true = "TRUE " if i < 2 else ""
        ax.set_ylabel(true + log + "Reports")
    if opts.score == "error":
        fig.text(
            0.5,
            0.02,
            "Error Costs     "
            + "  ".join("{}:{}".format(k, v) for k, v in costs_for_category.items()),
            ha="center",
            fontsize=9,
        )

    plt.show(block=False)
