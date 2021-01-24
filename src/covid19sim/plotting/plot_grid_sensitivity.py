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

DPI=100
TITLESIZE = 35
TICKSIZE = 22
LEGENDSIZE = 25
TITLEPAD = 25
LABELSIZE = 24
SCENARIO_LABELSIZE=25
SCENARIO_LABELPAD=85
LABELPAD = 30
LINESTYLES = ["-", "--"]

INTERPOLATION_FN = GPRFit

NUM_BOOTSTRAP_SAMPLES=1000
SUBSET_SIZE=100

CONTACT_RANGE =[4, 7]
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
        "values": [0.02, 0.08, 0.16, 0.32], # 0.02 0.08 0.16
        # "values": [0.02, 0.05, 0.10],
        "no-effect":["post-lockdown-no-tracing"]
    },
    "P_DROPOUT_SYMPTOM": {
        "values": [0.20, 0.40, 0.60], # 0.20 0.40 0.60
        "no-effect":["post-lockdown-no-tracing", "bdt1"]
    },
    "PROPORTION_LAB_TEST_PER_DAY": {
        "values": [0.004, 0.0025, 0.001],
        "no-effect":[]
    }
}

_SCENARIOS_NAME = ["Optimistic", "Moderate", "Pessimistic"]
SCENARIO_PARAMETERS_IDX={
    "Optimistic" : 0,
    "Moderate": 1,
    "Pessimistic": 2
}

DEFAULT_PARAMETER_VALUES = {
    "PROPORTION_LAB_TEST_PER_DAY": 0.001,
    "ALL_LEVELS_DROPOUT": 0.02,
    "P_DROPOUT_SYMPTOM": 0.20,
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

def plot_stable_frames_line(ax, df, y_metric, sensitivity_parameter, colormap):
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
    for i, adoption_rate in enumerate(sorted(df['adoption_rate'].unique(), key=lambda x:-x)):
        for method in df['method'].unique():
            # plot No Tracing only once
            if method == NO_TRACING_METHOD and no_tracing_plotted:
                continue
            no_tracing_plotted |= method == NO_TRACING_METHOD

            xs = sorted(df[sensitivity_parameter].unique())
            ys, yerrs = [], []
            for x_val in xs:
                # tmp_df = pd.DataFrame()
                selector = (df[['method', 'adoption_rate', sensitivity_parameter]] == [method, adoption_rate,  x_val]).all(1)
                y_vals = bootstrap(df[selector][y_metric], num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES)
                ys.append(np.mean(y_vals))
                yerrs.append(np.std(y_vals))
                print(f"{method} @ {adoption_rate} {sensitivity_parameter} = {x_val} has {sum(selector)} samples")

            ax.errorbar(x=xs, y=ys, yerr=yerrs, color=colormap[method], fmt='-o', linestyle=LINESTYLES[i])

    ax.legend().remove()
    ax.set(xlabel=None)
    ax.grid(True, which="minor", axis='x')
    ax.set(ylabel=None)
    ax.xaxis.set_major_locator(ticker.FixedLocator(xs))
    return ax


def plot_stable_frames(ax, df, y_metric, sensitivity_parameter, colormap):
    """
    Plots distribution of means of `y_metric` obtained from boostrapping

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

    color_scale = 1.0

    no_tracing_plotted=False
    for adoption_rate in sorted(df['adoption_rate'].unique(), key=lambda x:-x):
        bootstrapped_df = pd.DataFrame(columns=['method', y_metric, sensitivity_parameter])
        _colormap = copy.deepcopy(colormap)
        _colormap = {m: make_color_transparent(color, alpha=color_scale) for m, color in colormap.items()}
        color_scale *= 0.5
        for method in df['method'].unique():
            # plot No Tracing only once
            if method == NO_TRACING_METHOD and no_tracing_plotted:
                continue
            no_tracing_plotted |= method == NO_TRACING_METHOD

            for x_val in df[sensitivity_parameter].unique():
                tmp_df = pd.DataFrame()
                selector = (df[['method', 'adoption_rate', sensitivity_parameter]] == [method, adoption_rate,  x_val]).all(1)
                tmp_df[y_metric] = bootstrap(df[selector][y_metric], num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES)
                tmp_df['method'] = method
                tmp_df[sensitivity_parameter] = x_val
                bootstrapped_df = pd.concat([bootstrapped_df, tmp_df], axis=0, ignore_index=True)

                print(f"{method} @ {adoption_rate} {sensitivity_parameter} = {x_val} has {sum(selector)} samples {selector.shape[0]}")

        hue_order = [x for x in HUE_ORDER if x in bootstrapped_df['method'].unique()]
        ax = sns.violinplot(x=sensitivity_parameter, y=y_metric, hue='method', palette=_colormap,
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


def plot_and_save_grid_sensitivity_analysis(results, path, y_metric, SENSITIVITY_PARAMETERS, violin_plot=True):
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

    # plot all
    # scenario specific results
    # idx = SCENARIO_PARAMETERS_IDX['Moderate']
    # SCENARIO_PARAMETERS = [SENSITIVITY_PARAMETER_RANGE[param]['values'][idx] for param in SENSITIVITY_PARAMETERS]
    SCENARIO_PARAMETERS = [DEFAULT_PARAMETER_VALUES[param] for param in  SENSITIVITY_PARAMETERS]
    scenario_df = results[(results[SENSITIVITY_PARAMETERS] == SCENARIO_PARAMETERS).all(1)]

    stable_frames_scenario_df, stable_point_scenario = find_stable_frames(scenario_df, contact_range=CONTACT_RANGE)

    for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
        values = SENSITIVITY_PARAMETER_RANGE[parameter]['values']
        no_effect_on_methods = SENSITIVITY_PARAMETER_RANGE[parameter]['no-effect']
        str_formatter = SENSITIVITY_PARAMETER_RANGE[parameter].get("str_formatter", lambda x: f"{100 * x: 2.0f}")

        ax_df = pd.DataFrame()
        ax = axs[0, i]
        for param_index, value in enumerate(values):
            tmp_params = copy.deepcopy(SCENARIO_PARAMETERS)
            tmp_params[i] = value
            df = results[(results[SENSITIVITY_PARAMETERS] == tmp_params).all(1)]

            if df.shape[0] == 0:
                continue

            cell_methods = df['method'].unique()
            for method in no_effect_on_methods:
                if method not in cell_methods:
                    tmp_df = stable_frames_scenario_df[stable_frames_scenario_df['method'] == method]
                    tmp_df[parameter] = value
                    df = pd.concat([df, tmp_df], axis=0)

            if NO_TRACING_METHOD not in cell_methods:
                if CONTACT_RANGE is None:
                    stable_frames_df = df[df["effective_contacts"].between(stable_point_scenario - MARGIN, stable_point_scenario + MARGIN)]
                else:
                    stable_frames_df = df[df["effective_contacts"].between(*CONTACT_RANGE)]
            else:
                stable_frames_df, stable_point = find_stable_frames(df, contact_range=CONTACT_RANGE)

            ax_df = pd.concat([ax_df, stable_frames_df], ignore_index=True, axis=0)
        plot_fn(ax, ax_df, y_metric, parameter, colormap)

    # set row and column headers
    for col, name in enumerate(SENSITIVITY_PARAMETERS):
        tmp_ax = axs[0, col].twiny()
        tmp_ax.set_xticks([])
        tmp_ax.set_xlabel(get_sensitivity_label(name), labelpad=LABELPAD, fontsize=LABELSIZE)

    if y_metric == "r":
        y_label = "$R$"
    elif y_metric == "percentage_infected":
        y_label = "% infected"
    else:
        y_label = y_metric

    # for row, name in enumerate(SCENARIOS_NAME):
    axs[0, 0].set_ylabel(y_label, labelpad=LABELPAD, fontsize=LABELSIZE, rotation=90)
    # tmp_ax = axs[row, -1].twinx()
    # tmp_ax.set_ylabel(name+"\nScenario", fontsize=SCENARIO_LABELSIZE, fontweight="bold", rotation=0, labelpad=SCENARIO_LABELPAD )
    # tmp_ax.set_yticks([])

    # legends
    legends = []
    for method in results['method'].unique():
        method_label = labelmap[method]
        color = colormap[method]
        legends.append(Line2D([0, 1], [0, 0], color=color, linestyle="-", label=method_label, linewidth=3))

    lgd = fig.legend(handles=legends, ncol=len(legends), fontsize=30, loc="lower center", fancybox=True, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)

    # ylim
    y_min = np.min([np.min(ax.get_ylim()) for ax in axs.flatten()])
    y_max = np.max([np.max(ax.get_ylim()) for ax in axs.flatten()])
    _ = [ax.set_ylim(y_min, y_max) for ax in axs.flatten()]

    ref = 1.0 if y_metric == "r" else None
    # for j in range(len(SCENARIOS_NAME)):
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
    filename += "_violin" if violin_plot else "_line"
    filename += f"_C_{CONTACT_RANGE[0]}-{CONTACT_RANGE[1]}" if CONTACT_RANGE is not None else ""
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
        sensitivity_parameter == "test-quantity"
        or "sensitivity_Tx" in MAIN_FOLDER
    ):
        SENSITIVITY_PARAMETERS = ['PROPORTION_LAB_TEST_PER_DAY']
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
        results = pd.DataFrame()
        for scenario_folder in plot_path.parent.iterdir():
            for subfolder in scenario_folder.iterdir():
                if "scatter" not in subfolder.name:
                    continue
                print(f"Currently at: {str(subfolder)}.")
                all_runs = subfolder / "normalized_mobility/plots/normalized_mobility/"
                all_runs = list(all_runs.glob("full_extracted_data_AR_*.csv"))
                for _run in all_runs:
                    ar = str(_run).split(".csv")[0].split("_")[-1]
                    y = pd.read_csv(str(_run))
                    y['adoption_rate'] = float(ar) if ar else 100
                    results = pd.concat([results, y], axis=0, ignore_index=True)
        results.to_csv(str(filename))

    print("Unique adoption rates: ", results['adoption_rate'].unique())
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='r', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=True)
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='percentage_infected', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=True)
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='r', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=False)
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='percentage_infected', SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, violin_plot=False)
