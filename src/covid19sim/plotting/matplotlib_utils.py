"""
Basic plotting functions.
"""
import os
import yaml
import math
import colorsys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from scipy.interpolate import interp1d
from matplotlib.colors import TwoSlopeNorm, is_color_like, colorConverter, to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable

from covid19sim.plotting.curve_fitting import bootstrap_series, ewma

# base colors for each method
COLOR_MAP = {
    "bdt1": "mediumvioletred",
    "post-lockdown-no-tracing": "red",
    "bdt2": "darkorange",
    "heuristicv1": "royalblue",
    "transformer": "green",
    "oracle": "green",
    "heuristicv4": "pink"
}
COLORS = ["#34495e",  "mediumvioletred", "orangered", "royalblue", "darkorange", "green", "red"]

def scale_lightness(rgb, scale_l):
    """
    Scales lightness of color `rgb`
    https://stackoverflow.com/a/60562502/3413239
    """
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def make_color_transparent(color, bg_rgb=[1,1,1], alpha=0.5):
    """
    https://stackoverflow.com/a/33375738/3413239
    """
    rgb = colorConverter.to_rgb(color)
    return to_rgba([alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)])

def get_color(idx=None, method=None):
    """
    Returns color at idx position in `COLORS`.

    Args:
        idx (int):
        method (str): `intervention` method

    Returns:
        (str): color that is recognized by matplotlib
    """
    assert idx is not None or method is not None, "either of idx or method are required"
    if idx is not None:
        return COLORS[idx]

    return COLOR_MAP[method]

def get_colormap(methods_and_base_confs, path):
    """
    Generates consistent colors for very method in folder_names.

    Args:
        methods_and_base_confs (list): list of tuples where each element in a tuple is as follows -
            folder_name (str): method name as found in the experimental directory
            base_intervention_name (str): intervention configuration filename (without .yaml) in `configs/simulation/intervention/` folder.
        path (str): path of the folder where results will be saved. It looks for LABELMAP.yaml and uses label explicitly if present.

    Returns:
        (dict): a color (value) for each method (key)
    """
    colormap_file = Path(path).resolve() / "COLORMAP.yaml"
    explicit_colormap = {}
    if colormap_file.exists():
        with open(str(colormap_file), "rb") as f:
            explicit_colormap = yaml.safe_load(f)

    # unique counts for each base conf
    counts = {}
    for method, base_conf in methods_and_base_confs:
        counts[base_conf] = counts.get(base_conf, 0) + 1

    # generate range of scales for each base conf
    colors = {}
    for base_conf, count in counts.items():
        scale_range = np.linspace(1, 2, count)
        base_color = matplotlib.colors.ColorConverter.to_rgb(COLOR_MAP[base_conf])
        colors[base_conf] = iter([scale_lightness(base_color, scale) for scale in scale_range])

    colormap = {}
    for method, base_conf in methods_and_base_confs:
        color = explicit_colormap.get(method, "")
        if (
            method not in explicit_colormap
            or not is_color_like(color)
        ):
            color = next(colors[base_conf])

        colormap[method] = color

    return colormap

def get_labelmap(methods_and_base_confs, path):
    """
    Generates consistent labels for every method in folder_names.

    Args:
        methods_and_base_confs (list): list of tuples where each element in a tuple is as follows -
            folder_name (str): method name as found in the experimental directory
            base_intervention_name (str): intervention configuration filename (without .yaml) in `configs/simulation/intervention/` folder.
        path (str): path of the folder where results will be saved. It looks for LABELMAP.yaml and uses label explicitly if present.

    Returns:
        (dict): a label (value) for each folder_name (key)
    """
    labelmap_file = Path(path).resolve() / "LABELMAP.yaml"
    explicit_labelmap = {}
    if labelmap_file.exists():
        with open(str(labelmap_file), "rb") as f:
            explicit_labelmap = yaml.safe_load(f)

    labelmap = {}
    for method, base_conf in methods_and_base_confs:
        label = explicit_labelmap.get(method, "")
        if method not in explicit_labelmap:
            label = get_intervention_label(method, base_conf)

        labelmap[method] = label

    return labelmap

def _plot_mean_with_stderr_bands_of_series(ax, series, label, color, bootstrap=False, plot_quantiles=False, window=None, **kwargs):
    """
    Plots mean of `series` and a band of standard error around the mean.

    Args:
        ax (matplotlib.axes.Axes): Axes on which to plot the series
        series (list): each element is an np.array series
        label (str): label for the series to appear in legend
        color (str): color to plot this series with
        bootstrap (bool): If True, uses bootstrapped sampled to estimate stderrors and mean.
        plot_quantiles (bool): If True, plot quantiles. quantiles can be provided in the keyword arg - `quantiles`
        window (int): If not None, it takes `window` step moving average of the series.
        **kwargs (key=value): see `_plot_mean_and_stderr_bands` for keyword arguments used

    Returns:
        ax (matplotlib.axes.Axes): Axes with the plot of series
    """
    # plot only up until minimum length
    min_len = min(len(x) for x in series)
    out = [x[:min_len] for x in series]

    # moving window
    if window is not None:
        out = [ewma(cc, window) for cc in out]

    df = np.array(out)
    index = np.array(list(range(df.shape[1])))
    if bootstrap:
        num_bootstrap_samples = kwargs.get('num_bootstrap_samples', 1000)
        mode = kwargs.get('mode', 'mean')
        df =  bootstrap_series(df, num_bootstrap_samples=num_bootstrap_samples, mode=mode)

    if plot_quantiles:
        QS = kwargs.get("quantiles", [5, 50, 95])
        quantiles = np.array([np.percentile(df, q, axis=0) for q in QS])
        half = len(quantiles) // 2
        ax.plot(index, quantiles[half][:], color=color, label=label)
        for i in range(half):
            ax.fill_between(index, quantiles[i, :], quantiles[-(i + 1), :], color=color, alpha=(0.35 * (i+1)/half))

        return ax

    #
    mean = df.mean(axis=0)
    #
    if bootstrap:
        stderr = df.std(axis=0)
    else:
        stderr = df.std(axis=0)/np.sqrt(df.shape[0])

    ax = plot_mean_and_stderr_bands(ax, index, mean, stderr, label, color, **kwargs)
    return ax

def plot_mean_and_stderr_bands(ax, index, mean, stderr, label, color, fill=True, **kwargs):
    """
    Plots `mean` and `stderr` bounds on `ax`

    Args:
        ax (matplotlib.axes.Axes): Axes on which to plot the series
        index (pd.Series or no.array): 1D. Values on x axis for which mean and stderr is to be plotted
        mean (pd.Series): mean along x that is to be plotted
        stderr (pd.Series): error bounds around mean. Pass `confidence_level` to decide the bounds.
        label (str): label for the series to appear in legend
        color (str): color of line and the filling of polygon
        fill (bool): If True, fills the polygon between std error bands.
        **kwargs (key=value): see below for keyword arguments used

    Returns:
        ax (matplotlib.axes.Axes): Axes with a filled polygon corresponding to mean and stderr bounds
    """
    # params
    linestyle = kwargs.get("linestyle", "-")
    mean_alpha = kwargs.get("mean_alpha", 1.0)
    stderr_alpha = kwargs.get("stderr_alpha", 0.3)
    marker = kwargs.get("marker", None)
    markersize = kwargs.get("markersize", 1)
    linewidth = kwargs.get("linewidth", 1)
    confidence_level = kwargs.get('confidence_level', 1.0) # z-value corresponding to a significance level
    capsize = kwargs.get('capsize', 5.0)

    #
    ax.plot(index, mean, color=color, alpha=mean_alpha, linestyle=linestyle,
                linewidth=linewidth, label=label, marker=marker, ms=markersize)
    if fill:
        lows = mean - confidence_level * stderr
        highs = mean + confidence_level * stderr
        lowfn = interp1d(index, lows, bounds_error=False, fill_value='extrapolate')
        highfn = interp1d(index, highs, bounds_error=False, fill_value='extrapolate')
        ax.fill_between(index, lowfn(index), highfn(index), color=color, alpha=stderr_alpha, lw=0, zorder=3)
    else:
        ax.errorbar(index, mean, yerr=confidence_level * stderr, color=color, alpha=stderr_alpha, capsize=capsize)

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
    XY_TITLEPAD = kwargs.get("XY_TITLEPAD", 1.0)
    XY_TITLESIZE = kwargs.get("XY_TITLESIZE", 25)
    LEGENDSIZE = kwargs.get("LEGENDSIZE", 20)
    legend_loc = kwargs.get('legend_loc', None)
    percent_fmt_on_x = kwargs.get('percent_fmt_on_x', None)

    # xticks
    x_ticks = kwargs.get('x_ticks', None)
    if x_ticks is not None:
        ax.set_xticks(np.array(x_ticks))
    else:
        lower_lim = kwargs.get("x_lower_lim", math.floor(ax.get_xlim()[0] / 2.) * 2.0)
        upper_lim = kwargs.get("x_upper_lim", math.ceil(ax.get_xlim()[1] / 2.) * 2.0)
        x_tick_gap = kwargs.get("x_tick_gap", (upper_lim - lower_lim)/5.0)
        n_ticks = np.arange(lower_lim, upper_lim, x_tick_gap)
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_xticks(n_ticks)

    if percent_fmt_on_x is not None:
        xticks = mtick.StrMethodFormatter(percent_fmt_on_x)
        ax.xaxis.set_major_formatter(xticks)

    # yticks
    lower_lim = kwargs.get("y_lower_lim", None)
    upper_lim = kwargs.get("y_upper_lim", None)
    if lower_lim or upper_lim:
        ax.set_ylim(lower_lim, upper_lim)

    if x_title is not None:
        ax.set_xlabel(x_title, fontsize=XY_TITLESIZE)

    if y_title is not None:
        ax.set_ylabel(y_title, labelpad=XY_TITLEPAD, fontsize=XY_TITLESIZE)

    # grids
    ax.grid(True, axis='x', alpha=0.3)
    ax.grid(True, axis='y', alpha=0.3)

    # tick size
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(TICKSIZE)
        tick.set_pad(8.)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(TICKSIZE)
        tick.set_pad(8.)

    # legend
    if legend_loc is not None:
        ax.legend(prop={"size":LEGENDSIZE}, loc=legend_loc)

    return ax

def plot_heatmap_of_advantages(data, labelmap, USE_MATH_NOTATION=False):
    """
    Plots heatmap of values in matrix with a diverging color scheme i.e. only positive values are colored.

    Args:
        data (pd.DataFrame): A dataframe with following columns - 'method1', 'method2', 'advantage', 'stddev', 'P(advantage > 0)'
        labelmap (dict): a label (value) for each folder_name (key)
        USE_MATH_NOTATION (bool): if True, uses math notations for labels

    Returns:
        (matplotlib.figure.Figure): a canvas with heatmap on it
    """
    XY_TITLESIZE = 40
    XY_LABELSIZE = 30
    ANNOTATION_SIZE=28
    TICKSIZE = 20

    ordered_ref_methods = data['method1'].value_counts().index.tolist()
    ordered_comp_methods = data['method2'].value_counts().index.tolist()

    assert len(ordered_ref_methods) == len(ordered_comp_methods), "# ref methods:{len(ordered_ref_methods)}, and #comp methods:{len(ordered_comp_methods)}. Expected same!"
    N = len(ordered_comp_methods)
    if N == 0:
        return plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=200)[0] # empty figure

    CELL_MULTIPLIER = 4 if N > 3 else 8

    advs = -np.ones((N, N))
    stds = -np.ones((N, N))
    pvs = -np.ones((N, N))
    annotation = np.ndarray((N, N), dtype="object")

    # each column is a reference method i.e. we find the value of comparator at x for which reference R = 1
    for col, method1 in enumerate(ordered_ref_methods):
        for row, method2 in enumerate(ordered_comp_methods):
            selector = data['method1'] == method1
            selector &= data['method2'] == method2
            if data[selector].shape[0] > 0:
                advs[row, col] = data[selector]['advantage'].tolist()[0]
                stds[row, col] = data[selector]['stddev'].tolist()[0]
                pvs[row, col] = data[selector]['P(advantage > 0)'].tolist()[0]
                significance_95_str = "$^*$" if pvs[row, col] > 0.95 else ""
                annotation[row, col] = f"{advs[row, col]: 0.3f}\n($\pm${1.96 * stds[row, col]: 0.3f}){significance_95_str}"

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(N*CELL_MULTIPLIER, N*CELL_MULTIPLIER), dpi=200)
    im = ax.imshow(advs, cmap='summer_r', norm=TwoSlopeNorm(advs[advs > 0].mean(), vmin=0, vmax=None), interpolation='none', alpha=0.5)

    # set positioning
    ax.set(adjustable='box', aspect=1)

    # xticks and labels
    ax.set_yticks(np.arange(N))
    ax.set_xticks(np.arange(N))
    ax.set_yticklabels([labelmap[x] for x in ordered_comp_methods], fontdict=dict(fontsize=XY_LABELSIZE))
    ax.set_xticklabels([labelmap[x] for x in ordered_ref_methods], rotation=45, fontdict=dict(fontsize=XY_LABELSIZE))

    # labels
    ax.set_xlabel("Reference Method", fontsize=XY_TITLESIZE)
    ax.set_ylabel("Comparator Method", fontsize=XY_TITLESIZE, labelpad=0.5)

   # colorbar for each row
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.50)
    cbar = fig.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=ANNOTATION_SIZE)

    # annotation
    for col in range(N):
        for row in range(N):
            text = ax.text(col, row, annotation[row, col], ha="center", va="center", color="black", size=ANNOTATION_SIZE, fontweight="bold")

    if USE_MATH_NOTATION:
        ax.set_title("$\Delta \hat{R}$ ($\pm$ 2$\sigma$)", fontsize=XY_TITLESIZE, y=1.03)
    else:
        ax.set_title("Advantages ($\pm$ 2$\sigma$) of Tracing Methods (* 95% Significance level)", fontsize=XY_TITLESIZE, y=1.03)

    return fig

def save_figure(figure, basedir, folder, filename, bbox_extra_artists=None, bbox_inches='tight', pad_inches=None):
    """
    Saves figure at `basedir/folder/filename`. Creates `folder` if it doesn't exist.

    Args:
        figure (matplotlib.pyplot.Figure): figure to save
        basedir (str): existing directory
        folder (str): folder name. If None, save it in basedir.
        filename (str): filename of figure
        bbox_extra_artists (tuple): a tuple of out of canvas objects that need to be saved together

    Returns:
        filepath (str): A full path where figure is saved.
    """
    assert basedir is not None, "No base directory specified"
    basedir = Path(basedir)
    assert basedir.resolve().is_dir, f"{basedir} is not a directory"

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

    figure.savefig(filepath, bbox_inches=bbox_inches, bbox_extra_artists=bbox_extra_artists, pad_inches=pad_inches)
    return filepath

def get_adoption_rate_label_from_app_uptake(uptake):
    """
    Maps uptake to app adoption string for folder names

    Args:
        uptake (str or int): uptake for which adoption rate is to be determined.

    Returns:
        (str): A string of rounded integer representing adoption rate for the population
    """
    assert type(uptake) in [str, float, int], f"{uptake} is of type {type(uptake)} not str or int"
    uptake = eval(uptake) if type(uptake) == str else uptake
    if uptake == -1:
        return ""
    if uptake == 0.9831:
        return "70"
    if uptake == 0.8415:
        return "60"
    if uptake == 0.7170:
        return "50"
    if uptake == 0.6415:
        return "45"
    if uptake == 0.6425:
        return "45"
    if uptake == 0.5618:
        return "40"
    if uptake == 0.4215:
        return "30"
    if uptake == 0.3580:
        return "25"
    if uptake == 0.2850:
        return "20"
    if uptake == 0.2140:
        return "15"
    return uptake

def get_intervention_label(method_name, base_intervention_name):
    """
    Maps raw method name to a readable name.

    Args:
        method_name (str): name of the folder in the experiments.
        base_intervention_name (str): filename in `configs/simulation/intervention/` folder.

    Returns:
        (str): a readable name for the intervention
    """
    assert type(method_name) == str, f"improper intervention type: {type(method_name)}"

    # when experimental runs are named something else other than the intervention config filename
    if base_intervention_name != method_name:
        return method_name.upper()

    without_hhld_string = " wo hhld" if "wo_hhld" in method_name else ""
    base_method = method_name.replace("_wo_hhld", "")
    if base_method == "bdt1":
        return "Binary Digital Tracing" + without_hhld_string

    if base_method == "bdt2":
        return "Binary Digital Tracing (recursive)" + without_hhld_string

    if base_method == "post-lockdown-no-tracing":
        return "Post Lockdown No Tracing"

    if base_method == "oracle":
        return "Oracle" + without_hhld_string

    if "heuristic" in base_method:
        version = method_name.replace("heuristic", "")
        return f"Heuristic{version}" + without_hhld_string

    if "transformer" in base_method:
        return "Transformer" + without_hhld_string

    raise ValueError(f"Unknown raw intervention name: {method_name}")

def get_base_intervention(intervention_conf):
    """
    Maps `conf` to the configuration filename in `configs/simulation/intervention/` folder.

    Args:
        intervention_conf (dict): an experimental configuration.

    Returns:
        (str): filename in `configs/simulation/intervention/` folder.
    """

    # this key is added later
    if "INTERVENTION_NAME" in intervention_conf:
        return intervention_conf['INTERVENTION_NAME']

    # for old runs, base_intervention needs to be inferred from the conf parameters.
    if intervention_conf['RISK_MODEL'] == "":
        if intervention_conf['N_BEHAVIOR_LEVELS'] > 2:
            return "post-lockdown-no-tracing"

        if intervention_conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']:
            return "lockdown"

        return "no_intervention"

    risk_model = intervention_conf['RISK_MODEL']
    hhld_behavior = intervention_conf['MAKE_HOUSEHOLD_BEHAVE_SAME_AS_MAX_RISK_RESIDENT']

    if risk_model == "digital":
        order = intervention_conf['TRACING_ORDER']
        x = f"bdt{order}"
    else:
        x = f"{risk_model}"

    if hhld_behavior:
        return f"{x}"
    return f"{x}_wo_hhld"

def get_sensitivity_label(name):
    """
    Return the label for sensitivty parameter for the plot

    Args:
        name (str): name of config parameter

    Returns:
        (str): name of the x-label
    """
    if name == "ALL_LEVELS_DROPOUT":
        return "Daily Behavior Adherence\n(% of app users)"

    if name == "PROPORTION_LAB_TEST_PER_DAY":
        return "Daily Testing Capacity\n(% of population)"

    if name == "P_DROPOUT_SYMPTOM":
        return "Quality of Self-diagnosis\n(% of daily symptoms)"

    if name == "BASELINE_P_ASYMPTOMATIC": #????
        return "Asymptomaticity\n(% of population)"

    raise ValueError(f"Invalid name: {name}")
