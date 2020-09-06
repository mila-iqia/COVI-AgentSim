"""
Basic plotting functions.
"""
import os
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

COLORS = ["#34495e",  "mediumvioletred", "orangered", "royalblue", "darkorange", "green", "red"]


def get_color(idx):
    """
    Returns color at idx position in `COLORS`.

    Args:
        idx (int): index in the list

    Returns:
        (str): color that is recognized by matplotlib
    """
    return COLORS[idx]

def _plot_mean_with_stderr_bands_of_series(ax, series, label, color, **kwargs):
    """
    Plots mean of `series` and a band of standard error around the mean.

    Data:
        ax (matplotlib.axes.Axes): Axes on which to plot the series
        series (list): each element is an np.array series
        label (str): label for the series to appear in legend
        color (str): color to plot this series with
        **kwargs (key=value): see below for keyword arguments used

    Returns:
        ax (matplotlib.axes.Axes): Axes with the plot of series
    """
    # params
    linestyle = kwargs.get("linestyle", "-")
    alpha = kwargs.get("alpha", 0.8)
    marker = kwargs.get("marker", None)
    markersize = kwargs.get("markersize", 1)
    linewidth = kwargs.get("linewidth", 1)

    # plot only up until minimum length
    min_len = min(len(x) for x in series)
    out = [x[:min_len] for x in series]
    df = np.array(out)

    index = np.array(list(range(df.shape[1])))
    #
    mean = df.mean(axis=0)
    #
    stderr = df.std(axis=0)/np.sqrt(df.shape[0])
    lows = mean - stderr
    highs = mean + stderr
    lowfn = interp1d(index, lows, bounds_error=False, fill_value='extrapolate')
    highfn = interp1d(index, highs, bounds_error=False, fill_value='extrapolate')
    #
    ax.plot(mean, color=color, alpha=alpha, linestyle=linestyle,
                linewidth=linewidth, label=label, marker=marker, ms=markersize)
    ax.fill_between(index, lowfn(index), highfn(index), color=color, alpha=.3, lw=0, zorder=3)

    return ax

def add_bells_and_whistles(ax, y_title=None, x_title=None, **kwargs):
    """
    Adds / enhances elements of axes for better readability of plots

    Args:
        ax (matplotlib.axes.Axes): Axes that is to be enhanced
        y_title (str): y title for the axes
        x_title (str): x title for the axes
        **kwargs (key=value): see below for keyword arguments used

    Returns:
        ax (matplotlib.axes.Axes): Axes with enhanced features
    """
    TICKSIZE = kwargs.get("TICKSIZE", 20)
    TITLEPAD = kwargs.get("TITLEPAD", 1.0)
    TITLESIZE = kwargs.get("TITLESIZE", 25)
    legend_loc = kwargs.get('legend_loc', None)

    x_tick_gap = kwargs.get("x_tick_gap", 5)
    n_ticks = np.arange(0, ax.get_xlim()[1], x_tick_gap)

    if x_title is not None:
        ax.set_xlabel(x_title, fontsize=TITLESIZE)

    if y_title is not None:
        ax.set_ylabel(y_title, labelpad=TITLEPAD, fontsize=TITLESIZE)

    # grids
    ax.grid(True, axis='x', alpha=0.3)
    ax.grid(True, axis='y', alpha=0.3)

    # xticks
    ax.set_xticks(n_ticks)

    # tick size
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(TICKSIZE)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(TICKSIZE)

    # legend
    if legend_loc is not None:
        ax.legend(prop={"size":20}, loc=legend_loc)

    return ax

def save_figure(figure, basedir, folder, filename):
    """
    Saves figure at `basedir/folder/filename`. Creates `folder` if it doesn't exist.

    Args:
        figure (matplotlib.pyplot.Figure): figure to save
        basedir (str): existing directory
        folder (str): folder name. If None, save it in basedir.
        filename (str): filename of figure

    Returns:
        filepath (str): A full path where figure is saved.
    """
    assert basedir is not None, "No base directory specified"
    basedir = Path(basedir)
    assert basedir.resolve().is_dir, f"{basedir} is not a directory"

    figure.tight_layout()
    #
    if folder is None:
        filepath = str(basedir / filename)
        figure.savefig(filepath)
        return filepath
    #
    folder = basedir / folder
    os.makedirs(str(folder), exist_ok=True)
    #
    filepath = str(folder / filename)
    figure.savefig(filepath)
    return filepath

def get_adoption_rate_label_from_app_uptake(uptake):
    """
    Maps uptake to app adoption string for folder names

    Args:
        uptake (str or int): uptake for which adoption rate is to be determined.

    Returns:
        (str): A string of rounded integer representing adoption rate for the population
    """
    assert type(uptake) in [str, int], f"{uptake} is of type {type(uptake)} not str or int"
    uptake = eval(uptake)
    if uptake == -1:
     return ""
    if uptake == 0.9831:
     return "70"
    if uptake == 0.8415:
     return "60"
    if uptake == 0.5618:
     return "40"
    if uptake == 0.4215:
     return "30"

def get_intervention_label(intervention):
    """
    Maps raw intervention name to a readable name.

    Args:
        intervention (str): raw intervention name

    Returns:
        (str): a readable name for the intervention
    """
    assert type(intervention) == str, f"improper intervention type: {type(intervention)}"

    without_hhld = "wo_hhld" in intervention
    without_hhld_string = "" if without_hhld else ""
    base_method = intervention.replace("_wo_hhld", "")
    if base_method == "bdt1":
        return "Binary Digital Tracing" + without_hhld_string

    if base_method == "bdt2":
        return "Binary Digital Tracing (recursive)" + without_hhld_string

    if base_method == "post-lockdown-no-tracing":
        return "Post Lockdown No Tracing"

    if base_method == "oracle":
        return "Oracle" + without_hhld_string

    if "heuristic" in base_method:
        version = intervention.replace("heuristic", "")
        return f"Heuristic{version}" + without_hhld_string

    if base_method == "transformer":
        return "Transformer" + without_hhld_string

    raise ValueError(f"Unknown raw intervention name: {intervention}")
