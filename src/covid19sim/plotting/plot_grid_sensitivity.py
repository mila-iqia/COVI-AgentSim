"""
Plots a scatter plot showing trade-off between metrics of different simulations across varying mobility.
"""
import os
import yaml
import operator
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
from copy import deepcopy
from pathlib import Path

from covid19sim.plotting.utils import load_plot_these_methods_config
from covid19sim.plotting.matplotlib_utils import add_bells_and_whistles, save_figure, get_color, get_adoption_rate_label_from_app_uptake, get_intervention_label, \
                                plot_mean_and_stderr_bands, get_base_intervention, get_labelmap, get_colormap, plot_heatmap_of_advantages, get_sensitivity_label, make_color_transparent
from covid19sim.plotting.curve_fitting import GPRFit, bootstrap
from covid19sim.plotting.plot_normalized_mobility_scatter import get_metric_label

DPI=300
TITLESIZE = 40
TICKSIZE = 25
LEGENDSIZE = 30
TITLEPAD = 25
LABELSIZE = 30
SCENARIO_LABELSIZE=25
SCENARIO_LABELPAD=85
LABELPAD = 30
LINESTYLES = ["-", "--", ":"]

INTERPOLATION_FN = GPRFit

NUM_BOOTSTRAP_SAMPLES=1000
SUBSET_SIZE=100

CONTACT_RANGE =[0, 10]
# CONTACT_RANGE=None # [x1, x2] if provided GPRFit is not used i.e `TARGET_R_FOR_NO_TRACING` and `MARGIN` are not used
TARGET_R_FOR_NO_TRACING = 1.2 # find the performance of simulations around (defined by MARGIN) the number of contacts where NO_TRACING has R of 1.2
MARGIN = 0.75

# (optimistic, mdoerate, pessimistic)
# default str_formatter = lambda x: f"{100 * x: 2.0f}"
# NOTE: Following is taken from job_scripts/sensitivity_launch_mpic.py
SENSITIVITY_PARAMETER_RANGE ={
    "ASYMPTOMATIC_RATIO": {
        "values": [0.20, 0.30, 0.40], # 0.20 0.30 0.40
        "no-effect":[]
    },
    "BASELINE_P_ASYMPTOMATIC": {
        # "values": [0.20, 0.30, 0.40], # 0.20 0.30 0.40
        "values": [0.5, 0.75, 1.0], # asymptomatic-ratio =  0.20 0.30 0.40
        "no-effect":[]
    },
    "ALL_LEVELS_DROPOUT": {
        "values": [0.02, 0.08, 0.16, 0.32, 0.64], # 0.02 0.08 0.16
        # "values": [0.02, 0.05, 0.10],
        "no-effect":["post-lockdown-no-tracing"]
    },
    "P_DROPOUT_SYMPTOM": {
        "values": [0.20, 0.40, 0.60, 0.80, 1.0], # 0.20 0.40 0.60
        "no-effect":["post-lockdown-no-tracing", "bdt1"]
    },
    "PROPORTION_LAB_TEST_PER_DAY": {
        # "values": [0.004, 0.003, 0.0025, 0.002, 0.001],
        "values": [0.005, 0.003, 0.0015, 0.001, 0.0005],
        "no-effect":[]
    },
    "adoption_rate": {
        "values": [60, 50, 40, 30, 20],
        "no-effect": ['post-lockdown-no-tracing']
    },
    "ASYMPTOMATIC_INFECTION_RATIO": {
        "values": [0.29, 0.65, 1.0, 1.35],
        "no-effect": []
    }
}

MAIN_SENSITIVITY_PARAMETERS = ["ALL_LEVELS_DROPOUT", "P_DROPOUT_SYMPTOM", "PROPORTION_LAB_TEST_PER_DAY", "ASYMPTOMATIC_INFECTION_RATIO"]

DEFAULT_PARAMETER_VALUES = {
    "PROPORTION_LAB_TEST_PER_DAY": 0.001,
    "ALL_LEVELS_DROPOUT": 0.02,
    "P_DROPOUT_SYMPTOM": 0.20,
    "adoption_rate": 60,
    "ASYMPTOMATIC_INFECTION_RATIO": 0.29
}

NO_TRACING_METHOD = "post-lockdown-no-tracing"
REFERENCE_METHOD = "bdt1"
OTHER_METHODS = ["heuristicv4"]
HUE_ORDER = [NO_TRACING_METHOD, REFERENCE_METHOD] + OTHER_METHODS

REFERENCE_R=1.2
USE_MATH_NOTATION=False

# fix the seed
np.random.seed(123)

class StringFormatter(object):
    def __init__(self, fn):
        self.fn = fn

    def format(self, x, pos):
        return self.fn(x)

def get_y_label(y_metric, y_metric_denom):
    """
    Returns a label for y-axis

    Args:
        y_metric (str): main metric
        y_metric_denom (str): a metric label for denominator

    Returns:
        (str) : label for y-axis
    """
    if y_metric == "r":
        y_label = "$R$"
    elif y_metric == "percentage_infected":
        y_label = "% infected"
    else:
        y_label = y_metric

    if y_metric_denom == "effective_contacts":
        y_label += " / contacts (per day per human)"

    return y_label


def get_bootstrapped_values(df, y_metric, y_metric_denom=None):
    """
    Returns a list of bootstrapped means from `df`

    Args:
        y_metric (str): main metric to boostrap.
        y_metric_denom (str): if specified devides the bootstrapped values with the bootstrapped values for this metric

    Returns:
        (list): bootstrapped means
    """
    out = bootstrap(df[y_metric], num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES)
    if y_metric_denom is not None:
        denom = bootstrap(df[y_metric_denom], num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES)
        out = [i/j for i,j in zip(out, denom)]

    return out


def plot_stable_frames_line(ax, df, y_metric, sensitivity_parameter, colormap, y_metric_denom=None):
    """
    Plots means (and error bars) of `y_metric` obtained from boostrapping

    Args:
        ax (matploltib.ax.Axes): axes on which distribution will be plotted
        df (pd.DataFrame): dataframe which has `y_metric` as its column
        y_metric (str): metric for which distribution of means need to be computed
        sensitivity_parameter (str): parameter to decide the position on x_axis
        colormap (dict): mapping from method to color

    Returns:
        ax (matploltib.ax.Axes): axes with distribution plotted on it

    """
    np.random.seed(1234)
    rng = np.random.RandomState(1)

    no_tracing_plotted=False
    outer_loop = [None] if sensitivity_parameter == "adoption_rate" else sorted(df['adoption_rate'].unique(), key=lambda x:-x)

    for i, adoption_rate in enumerate(outer_loop):
        for method in df['method'].unique():
            # plot No Tracing only once
            if method == NO_TRACING_METHOD and no_tracing_plotted:
                continue
            no_tracing_plotted |= (method == NO_TRACING_METHOD) and (adoption_rate == -1)

            xs = sorted(df[sensitivity_parameter].unique())
            ys, yerrs = [], []
            for x_val in xs:
                if sensitivity_parameter == "adoption_rate":
                    selector = (df[['method', sensitivity_parameter]] == [method, x_val]).all(1)
                else:
                    selector = (df[['method', 'adoption_rate', sensitivity_parameter]] == [method, adoption_rate,  x_val]).all(1)
                y_vals = get_bootstrapped_values(df[selector], y_metric, y_metric_denom)
                ys.append(np.mean(y_vals))
                yerrs.append(np.std(y_vals))
                print(f"{method} @ {adoption_rate} {sensitivity_parameter} = {x_val} has {sum(selector)} samples")

            ax.errorbar(x=xs, y=ys, yerr=yerrs, color=colormap[method], fmt='-o', linestyle=LINESTYLES[i], linewidth=3)

    ax.legend().remove()
    ax.set(xlabel=None)
    ax.grid(True, which="minor", axis='x')
    ax.set(ylabel=None)
    ax.xaxis.set_major_locator(ticker.FixedLocator(xs))
    return ax


def plot_stable_frames(ax, df, y_metric, sensitivity_parameter, colormap, y_metric_denom=None):
    """
    Plots distribution of means of `y_metric` obtained from boostrapping

    Args:
        ax (matploltib.ax.Axes): axes on which distribution will be plotted
        df (pd.DataFrame): dataframe which has `y_metric` as its column
        y_metric (str): metric for which distribution of means need to be computed
        sensitivity_parameter (str): parameter to decide the position on x_axis
        colormap (dict): mapping from method to color
        y_metric_denom (str): metric for ratio

    Returns:
        ax (matploltib.ax.Axes): axes with distribution plotted on it

    """
    np.random.seed(1234)
    rng = np.random.RandomState(1)

    color_scale = 1.0

    no_tracing_plotted=False
    outer_loop = [None] if sensitivity_parameter == "adoption_rate" else sorted(df['adoption_rate'].unique(), key=lambda x:-x)
    for adoption_rate in outer_loop:
        bootstrapped_df = pd.DataFrame(columns=['method', 'y', sensitivity_parameter])
        _colormap = copy.deepcopy(colormap)
        _colormap = {m: make_color_transparent(color, alpha=color_scale) for m, color in colormap.items()}
        color_scale *= 0.5
        for method in df['method'].unique():
            # plot No Tracing only once
            if method == NO_TRACING_METHOD and no_tracing_plotted:
                continue
            no_tracing_plotted |= (method == NO_TRACING_METHOD) and (adoption_rate == -1)

            for x_val in df[sensitivity_parameter].unique():
                tmp_df = pd.DataFrame()
                if sensitivity_parameter == "adoption_rate":
                    selector = (df[['method', sensitivity_parameter]] == [method, x_val]).all(1)
                else:
                    selector = (df[['method', 'adoption_rate', sensitivity_parameter]] == [method, adoption_rate,  x_val]).all(1)
                tmp_df['y'] = get_bootstrapped_values(df[selector], y_metric, y_metric_denom)
                tmp_df['method'] = method
                tmp_df[sensitivity_parameter] = x_val
                bootstrapped_df = pd.concat([bootstrapped_df, tmp_df], axis=0, ignore_index=True)

                print(f"{method} @ {adoption_rate} {sensitivity_parameter} = {x_val} has {sum(selector)} samples")

        hue_order = [x for x in HUE_ORDER if x in bootstrapped_df['method'].unique()]
        ax = sns.violinplot(x=sensitivity_parameter, y='y', hue='method', palette=_colormap,
                        data=bootstrapped_df, inner="quartile", cut=2, ax=ax, width=0.8,
                        hue_order=hue_order, order=sorted(df[sensitivity_parameter].unique()))

    ax.legend().remove()
    ax.set(xlabel=None)
    ax.grid(True, which="minor", axis='x')
    ax.set(ylabel=None)
    return ax


def find_stable_frames(df, contact_range=None):
    """
    Finds `effective_contacts` where `r` for `NO_TRACING_METHOD` is `TARGET_R_FOR_NO_TRACING`

    Args:
        df (pd.DataFrame):

    Returns:
        stable_frames (pd.DataFrame): filtered rows from df that have `effective_contacts` in the range of `stable_point` +- `MARGIN`
        stable_point (float): number of `effective_contacts` where `NO_TRACING_METHOD` has `r` of 1.2
    """
    if contact_range is not None:
        stable_frames = df[df["effective_contacts"].between(*contact_range)]
        return stable_frames, np.mean(contact_range)

    assert NO_TRACING_METHOD in df['method'].unique(), f"{NO_TRACING_METHOD} not found in methods "
    assert df.groupby(SENSITIVITY_PARAMETERS).size().reset_index().shape[0] == 1, "same sensitivity parameters expected"

    selector = df['method'] == NO_TRACING_METHOD
    x = df[selector]['effective_contacts'].to_numpy()
    y = df[selector]['r'].to_numpy()
    fitted_fn = INTERPOLATION_FN().fit(x, y)
    stable_point = fitted_fn.find_x_for_y(TARGET_R_FOR_NO_TRACING).item()
    stable_frames = df[df["effective_contacts"].between(stable_point - MARGIN, stable_point + MARGIN)]
    return stable_frames, stable_point


def plot_and_save_grid_sensitivity_analysis(results, path, y_metric, SENSITIVITY_PARAMETERS, violin_plot=True, contact_range=None, y_metric_denom=None):
    """
    Plots and saves grid sensitivity for various SCENARIOS.

    Args:
        results (pd.DataFrame): Dataframe with rows extracted from plotting scripts of normalized_mobility i.e. stable_frames.csv (check plot_normalized_mobility_scatter.py)
        path (str): path of the folder where results will be saved
        y_metric (str): metric that needs to be plotted on y-axis
        SENSITIVITY_PARAMETERS (list): list of parameters
        violin_plot (bool): True if violin plots need to be plotted.
    """
    TICKGAP=2
    ANNOTATION_FONTSIZE=15

    assert y_metric in results.columns, f"{y_metric} not found in columns :{results.columns}"
    assert y_metric_denom is None or y_metric_denom in results.columns, f"{y_metric_denom} not found in columns: {results.columns}"

    plot_fn = plot_stable_frames if violin_plot else plot_stable_frames_line

    methods = results['method'].unique()
    methods_and_base_confs = results.groupby(['method', 'intervention_conf_name']).size().index
    labelmap = get_labelmap(methods_and_base_confs, path)
    colormap = get_colormap(methods_and_base_confs, path)

    # find if only specific folders (methods) need to be plotted
    plot_these_methods = load_plot_these_methods_config(path)
    nrows = 1
    ncols = len(SENSITIVITY_PARAMETERS)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(27, 18), sharex='col', sharey=True, dpi=DPI, constrained_layout=True, squeeze=False)

    #
    SCENARIO_PARAMETERS = [DEFAULT_PARAMETER_VALUES[param] for param in  SENSITIVITY_PARAMETERS]
    default_scenario_df = results[(results[SENSITIVITY_PARAMETERS] == SCENARIO_PARAMETERS).all(1)]

    stable_frames_scenario_df, stable_point_scenario = find_stable_frames(default_scenario_df, contact_range=contact_range)

    for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
        values = SENSITIVITY_PARAMETER_RANGE[parameter]['values']
        no_effect_on_methods = SENSITIVITY_PARAMETER_RANGE[parameter]['no-effect']

        ax_df = pd.DataFrame()
        ax = axs[0, i]
        for param_index, value in enumerate(values):
            tmp_params = copy.deepcopy(SCENARIO_PARAMETERS)
            tmp_params[i] = value
            df = results[(results[SENSITIVITY_PARAMETERS] == tmp_params).all(1)]

            if df.shape[0] == 0:
                continue

            cell_methods = df['method'].unique()
            if parameter != "P_DROPOUT_SYMPTOM" or True:
                for method in no_effect_on_methods:
                    if method not in cell_methods:
                        tmp_df = stable_frames_scenario_df[stable_frames_scenario_df['method'] == method]
                        tmp_df[parameter] = value
                        df = pd.concat([df, tmp_df], axis=0)

            if NO_TRACING_METHOD not in cell_methods:
                if contact_range is None:
                    stable_frames_df = df[df["effective_contacts"].between(stable_point_scenario - MARGIN, stable_point_scenario + MARGIN)]
                else:
                    stable_frames_df = df[df["effective_contacts"].between(*contact_range)]
            else:
                stable_frames_df, stable_point = find_stable_frames(df, contact_range=contact_range)

            ax_df = pd.concat([ax_df, stable_frames_df], ignore_index=True, axis=0)
        plot_fn(ax, ax_df, y_metric, parameter, colormap, y_metric_denom)

        ar_legend = []
        if parameter != "adoption_rate":
            adoption_rates = sorted(results['adoption_rate'].unique(), key = lambda x:-x)
            ar_legend.append(Line2D([0, 1], [0, 0], color="black", linestyle=LINESTYLES[0], label=f"{adoption_rates[0]} %", linewidth=5))
            ar_legend.append(Line2D([0, 1], [0, 0], color="black", linestyle=LINESTYLES[1], label=f"{adoption_rates[1]} %", linewidth=5))
            ar_legend.append(Line2D([0, 1], [0, 0], color="black", linestyle=LINESTYLES[2], label=f"N/A", linewidth=5))
            ar_lgd = ax.legend(handles=ar_legend, ncol=len(ar_legend), loc="upper right", fontsize=30, fancybox=True, title="Adoption Rate")
            ar_lgd.get_title().set_fontsize(30)

    # set row and column headers
    for col, name in enumerate(SENSITIVITY_PARAMETERS):
        tmp_ax = axs[0, col].twiny()
        tmp_ax.set_xticks([])
        tmp_ax.set_xlabel(get_sensitivity_label(name), labelpad=LABELPAD, fontsize=TITLESIZE)

    #
    y_label = get_y_label(y_metric, y_metric_denom)
    axs[0, 0].set_ylabel(y_label, labelpad=LABELPAD, fontsize=LABELSIZE, rotation=90)

    # legends
    legends = []
    for method in results['method'].unique():
        method_label = labelmap[method]
        color = colormap[method]
        legends.append(Line2D([0, 1], [0, 0], color=color, linestyle="-", label=method_label, linewidth=5))

    lgd = fig.legend(handles=legends, ncol=len(legends), fontsize=30, loc="lower center", fancybox=True, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)

    # ylim
    y_min = np.min([np.min(ax.get_ylim()) for ax in axs.flatten()])
    y_max = np.max([np.max(ax.get_ylim()) for ax in axs.flatten()])
    _ = [ax.set_ylim(y_min, y_max) for ax in axs.flatten()]

    ref = 1.0 if y_metric == "r" else None
    for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
        ax = axs[0, i]
        ax.grid(True, axis='x', alpha=0.3)
        ax.grid(True, axis='y', alpha=0.3)

        # tick size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(TICKSIZE)
            tick.set_pad(8.)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(TICKSIZE)
            tick.set_pad(8.)

        ax.plot(ax.get_xlim(),  [ref, ref],  linestyle=":", color="gray", linewidth=2)

    # save
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    filename = f"{y_metric}"
    filename += f"_by_{y_metric_denom}" if y_metric_denom is not None else ""
    filename += "_violin" if violin_plot else "_line"
    filename += f"_C_{contact_range[0]}-{contact_range[1]}" if contact_range is not None else ""
    filename = filename.replace(".", "_") # to allow decimals in contact range string
    filepath = save_figure(fig, basedir=path, folder="grid_sensitivity", filename=f'{filename}', bbox_extra_artists=(lgd,), bbox_inches='tight')
    print(f"Sensitivity analysis saved at {filepath}")

def run(data, plot_path, compare=None, **kwargs):
    """
    Plots and saves grid form of sensitivity various configurations across different methods.

    It requires subfolders to be experiments from normalized_mobility.yaml with different combinations of `SENSITIVITY_PARAMETERS`

    Args:
        data (NoneType):
        plot_path (str): path where to save plots
    """
    use_extracted_data = kwargs.get('use_extracted_data', False)
    sensitivity_parameter = kwargs.get('sensitivity_parameter', None)
    MAIN_FOLDER = plot_path.parent.name
    if (
        sensitivity_parameter == "user-behavior"
        or "sensitivity_LxSx" in MAIN_FOLDER
    ):
        SENSITIVITY_PARAMETERS = ['ALL_LEVELS_DROPOUT', 'P_DROPOUT_SYMPTOM']
    elif (
        sensitivity_parameter == "user-behavior-Lx"
        or "sensitivity_Lx" in MAIN_FOLDER
    ):
        SENSITIVITY_PARAMETERS = ['ALL_LEVELS_DROPOUT']
    elif (
        sensitivity_parameter == "user-behavior-Sx"
        or "sensitivity_Sx" in MAIN_FOLDER
    ):
        SENSITIVITY_PARAMETERS = ['P_DROPOUT_SYMPTOM']
    elif (
        sensitivity_parameter == "test-quantity"
        or "sensitivity_Tx" in MAIN_FOLDER
    ):
        SENSITIVITY_PARAMETERS = ['PROPORTION_LAB_TEST_PER_DAY']
    elif (
        sensitivity_parameter == "adoption-rate"
        or "sensitivity_ARx" in MAIN_FOLDER
    ):
        SENSITIVITY_PARAMETERS = ['adoption_rate']
    elif (
        sensitivity_parameter == "ASYMPTOMATIC_INFECTION_RATIO"
        or "sensitivity_AIRx" in MAIN_FOLDER
    ):
        SENSITIVITY_PARAMETERS = ['ASYMPTOMATIC_INFECTION_RATIO']
    else:
        raise ValueError("Sensitivity parameter not specified..")

    print(f"sensitivity parameters : {SENSITIVITY_PARAMETERS}")

    folder_name = Path(plot_path).resolve() / "grid_sensitivity"
    os.makedirs(str(folder_name), exist_ok=True)

    # merge all csvs
    filename = folder_name / "all_extracted_data.csv"
    if use_extracted_data:
        results = pd.read_csv(str(filename))
    else:
        print(f"Plot path: {str(plot_path)}")
        remaining_sensitivity_parameters = [x for x in MAIN_SENSITIVITY_PARAMETERS if x not in SENSITIVITY_PARAMETERS]
        default_parameter_values = [DEFAULT_PARAMETER_VALUES[x] for x in remaining_sensitivity_parameters]
        results = pd.DataFrame()
        for sensitivity_dir in plot_path.parent.parent.iterdir():
            if (
                sensitivity_dir.name == "main_scenario"
                or not sensitivity_dir.is_dir()
            ):
                continue # uses different mobility factors for methods
            print(sensitivity_dir.name)
            for scenario_folder in sensitivity_dir.iterdir():
                if not scenario_folder.is_dir():
                    continue

                for subfolder in scenario_folder.iterdir():
                    if "scatter" not in subfolder.name:
                        continue
                    print(f"Currently at: {str(subfolder)}.")
                    all_runs = subfolder / "normalized_mobility/plots/normalized_mobility/"
                    all_runs = list(all_runs.glob("full_extracted_data_AR_*.csv"))
                    for _run in all_runs:
                        df = pd.read_csv(str(_run))
                        if "ASYMPTOMATIC_INFECTION_RATIO" not in df:
                            df['ASYMPTOMATIC_INFECTION_RATIO'] = DEFAULT_PARAMETER_VALUES["ASYMPTOMATIC_INFECTION_RATIO"]
                        selector = (df[remaining_sensitivity_parameters] == default_parameter_values).all(1)
                        if selector.shape[0] > 0:
                            results = pd.concat([results, df[selector]], axis=0, ignore_index=True)

        results.loc[results['app_based'] == False, 'adoption_rate'] = -1
        results.to_csv(str(filename))

    print("Unique adoption rates: ", results['adoption_rate'].unique())
    # plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='r', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=True, contact_range=CONTACT_RANGE)
    # plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='percentage_infected', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=True, contact_range=CONTACT_RANGE)
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='r', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=False, contact_range=CONTACT_RANGE)
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='percentage_infected', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=False, contact_range=CONTACT_RANGE)

    # ratios
    # plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='r', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=True, contact_range=[0, 10], y_metric_denom="effective_contacts")
    # plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='percentage_infected', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=True, contact_range=[0, 10], y_metric_denom="effective_contacts")
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='r', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=False, contact_range=[0, 10], y_metric_denom="effective_contacts")
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='percentage_infected', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=False, contact_range=[0, 10], y_metric_denom="effective_contacts")
