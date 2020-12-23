"""
Plots a scatter plot showing trade-off between metrics of different simulations across varying mobility.
"""
import os
import yaml
import operator
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
from copy import deepcopy
from pathlib import Path

from covid19sim.plotting.utils import load_plot_these_methods_config
from covid19sim.plotting.matplotlib_utils import add_bells_and_whistles, save_figure, get_color, get_adoption_rate_label_from_app_uptake, get_intervention_label, \
                                plot_mean_and_stderr_bands, get_base_intervention, get_labelmap, get_colormap, plot_heatmap_of_advantages
from covid19sim.plotting.curve_fitting import GPRFit
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
TARGET_R_FOR_NO_TRACING = 1.2 # find the performance of simulations around (defined by MARGIN) the number of contacts where NO_TRACING has R of 1.2
MARGIN = 0.5
NUM_BOOTSTRAP_SAMPLES=1000
SUBSET_SIZE=100

METRICS = ['r', 'effective_contacts', 'healthy_contacts']
# SENSITIVITY_PARAMETERS = ['ASYMPTOMATIC_RATIO', 'ALL_LEVELS_DROPOUT', 'P_DROPOUT_SYMPTOM',  'PROPORTION_LAB_TEST_PER_DAY'] #????
SENSITIVITY_PARAMETERS = ['BASELINE_P_ASYMPTOMATIC', 'ALL_LEVELS_DROPOUT', 'P_DROPOUT_SYMPTOM',  'PROPORTION_LAB_TEST_PER_DAY']
XMETRICS = ['effective_contacts'] + SENSITIVITY_PARAMETERS

# (optimistic, mdoerate, pessimistic)
# default str_formatter = lambda x: f"{100 * x: 2.0f}"
# NOTE: Following is taken from job_scripts/sensitivity_launch_mpic.py
SENSITIVITY_PARAMETER_RANGE ={
    "ASYMPTOMATIC_RATIO": {
        "values": [0.20, 0.30, 0.40], # 0.20 0.30 0.40
        "no-effect":[]
    },
    "BASELINE_P_ASYMPTOMATIC": {
        "values": [0.20, 0.30, 0.40], # 0.20 0.30 0.40
        # "values": [0.1475, 0.2525, 0.3575], # asymptomatic-ratio =  0.20 0.30 0.40
        "no-effect":[]
    },
    "ALL_LEVELS_DROPOUT": {
        "values": [0.02, 0.08, 0.16], # 0.02 0.08 0.16
        "no-effect":["post-lockdown-no-tracing"]
    },
    "P_DROPOUT_SYMPTOM": {
        "values": [0.20, 0.40, 0.60], # 0.20 0.40 0.60
        "no-effect":["post-lockdown-no-tracing", "bdt1"]
    },
    "PROPORTION_LAB_TEST_PER_DAY": {
        "values": [0.004, 0.002, 0.001], # 0.004 0.002 0.001
        "no-effect":[]
    }
}

SCENARIOS_NAME = ["Optimistic", "Moderate", "Pessimistic"]
SCENARIO_PARAMETERS_IDX={
    "Optimistic" : 0,
    "Moderate": 1,
    "Pessimistic": 2
}

NO_TRACING_METHOD = "post-lockdown-no-tracing"
REFERENCE_METHOD = "bdt1"
OTHER_METHODS = ["heuristicv4"]

# our scenario - 0.02, 0.001, 0.20, 0.23

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


REFERENCE_R=1.2
USE_MATH_NOTATION=False

# fix the seed
np.random.seed(123)

class StringFormatter(object):
    def __init__(self, fn):
        self.fn = fn

    def format(self, x, pos):
        return self.fn(x)

def plot_stable_frames(ax, df, sensitivity_parameter, colormap, mode='mean'):
    """
    """
    x_val = df[sensitivity_parameter].unique().item()

    # bootstrap
    rng = np.random.RandomState(1)
    bootstrapped_df = {
        'method': [],
        'r': []
    }

    for method in df['method'].unique():
        r = []
        tmp_df = df[df['method'] == method]
        for bootstrap_sample in range(NUM_BOOTSTRAP_SAMPLES):
            estimand = tmp_df.sample(SUBSET_SIZE, replace=True, random_state=rng)['r'].to_numpy()
            if mode == "median":
                estimate = np.median(estimand)
            elif mode == "mean":
                estimate = np.mean(estimand)
            else:
                raise NotImplementedError
            bootstrapped_df['method'].append(method)
            bootstrapped_df['r'].append(estimate)
            r.append(estimate)

        # ax.errorbar(x=x_val, y=np.mean(r), yerr=np.std(r), color=colormap[method], fmt='-o')
        scale = int(f"{x_val:.2E}".split("E")[-1][1:])
        parts = ax.violinplot(r, positions=[x_val], showmeans=False, widths=0.5 * (10 ** -scale))
        for pc in parts['bodies']:
            pc.set_facecolor(colormap[method])
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        # ax.boxplot(x=r, positions=[x_val], showfliers=False, widths=0.1)
    return ax


def find_stable_frames(df):
    """
    Finds `effective_contacts` where `r` for `NO_TRACING_METHOD` is `TARGET_R_FOR_NO_TRACING`

    Args:
        df (pd.DataFrame):

    Returns:
        stable_frames (pd.DataFrame): filtered rows from df that have `effective_contacts` in the range of `stable_point` +- `MARGIN`
        stable_point (float): number of `effective_contacts` where `NO_TRACING_METHOD` has `r` of 1.2
    """
    assert NO_TRACING_METHOD in df['method'].unique(), f"{NO_TRACING_METHOD} not found in methods "
    assert df.groupby(SENSITIVITY_PARAMETERS).size().reset_index().shape[0] == 1, "same sensitivity parameters expected"

    selector = df['method'] == NO_TRACING_METHOD
    x = df[selector]['effective_contacts'].to_numpy()
    y = df[selector]['r'].to_numpy()
    fitted_fn = INTERPOLATION_FN().fit(x, y)
    stable_point = fitted_fn.find_x_for_y(TARGET_R_FOR_NO_TRACING).item()
    stable_frames = df[df["effective_contacts"].between(stable_point - MARGIN, stable_point + MARGIN)]
    return stable_frames, stable_point


def plot_and_save_grid_sensitivity_analysis(results, path):
    """
    Plots and saves grid sensitivity for various SCENARIOS.

    Args:
        results (pd.DataFrame): Dataframe with rows extracted from plotting scripts of normalized_mobility i.e. stable_frames.csv (check plot_normalized_mobility_scatter.py)
        path (str): path of the folder where results will be saved
    """
    TICKGAP=2
    ANNOTATION_FONTSIZE=15

    methods = results['method'].unique()
    methods_and_base_confs = results.groupby(['method', 'intervention_conf_name']).size().index
    labelmap = get_labelmap(methods_and_base_confs, path)
    colormap = get_colormap(methods_and_base_confs, path)
    ALL_METHODS = OTHER_METHODS
    ALL_METHODS += [REFERENCE_METHOD, NO_TRACING_METHOD]

    # find if only specific folders (methods) need to be plotted
    plot_these_methods = load_plot_these_methods_config(path)
    fig, axs = plt.subplots(nrows=len(SCENARIOS_NAME), ncols=len(SENSITIVITY_PARAMETERS), figsize=(27, 18), sharex='col', sharey=True, dpi=DPI, constrained_layout=True)

    # set row and column headers
    for col, name in enumerate(SENSITIVITY_PARAMETERS):
        tmp_ax = axs[0, col].twiny()
        tmp_ax.set_xticks([])
        tmp_ax.set_xlabel(get_sensitivity_label(name), labelpad=LABELPAD, fontsize=LABELSIZE)

    y_label = "$R$"
    for row, name in enumerate(SCENARIOS_NAME):
        axs[row, 0].set_ylabel(y_label, labelpad=LABELPAD, fontsize=LABELSIZE, rotation=0)
        tmp_ax = axs[row, -1].twinx()
        tmp_ax.set_ylabel(name+"\nScenario", fontsize=SCENARIO_LABELSIZE, fontweight="bold", rotation=0, labelpad=SCENARIO_LABELPAD )
        tmp_ax.set_yticks([])

    # plot all
    for j, scenario in enumerate(SCENARIOS_NAME[:1]):
        # scenario specific results
        idx = SCENARIO_PARAMETERS_IDX[scenario]
        SCENARIO_PARAMETERS = [SENSITIVITY_PARAMETER_RANGE[param]['values'][idx] for param in SENSITIVITY_PARAMETERS]
        scenario_df = results[(results[SENSITIVITY_PARAMETERS] == SCENARIO_PARAMETERS).all(1)]

        stable_frames_scenario_df, stable_point_scenario = find_stable_frames(scenario_df)

        for i, ax in enumerate(axs[j, :]):
            plot_stable_frames(ax, stable_frames_scenario_df, SENSITIVITY_PARAMETERS[i], colormap)

        for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
            values = SENSITIVITY_PARAMETER_RANGE[parameter]['values']
            no_effect_on_methods = SENSITIVITY_PARAMETER_RANGE[parameter]['no-effect']
            str_formatter = SENSITIVITY_PARAMETER_RANGE[parameter].get("str_formatter", lambda x: f"{100 * x: 2.0f}")

            ax = axs[j, i]
            for param_index, value in enumerate(values):
                tmp_params = copy.deepcopy(SCENARIO_PARAMETERS)
                tmp_params[i] = value
                df = results[(results[SENSITIVITY_PARAMETERS] == tmp_params).all(1)]

                cell_methods = df['method'].unique()
                for method in no_effect_on_methods:
                    if method not in cell_methods:
                        tmp_df = stable_frames_scenario_df[stable_frames_scenario_df['method'] == method]
                        tmp_df[parameter] = value
                        df = pd.concat([df, tmp_df], axis=0)

                if NO_TRACING_METHOD not in cell_methods:
                    stable_frames_df = df[df["effective_contacts"].between(stable_point_scenario - MARGIN, stable_point_scenario + MARGIN)]
                else:
                    stable_frames_df, stable_point = find_stable_frames(df)

                plot_stable_frames(ax, stable_frames_df, parameter, colormap)

    # legends
    legends = []
    for method in ALL_METHODS:
        method_label = labelmap[method]
        color = colormap[method]
        legends.append(Line2D([0, 1], [0, 0], color=color, linestyle="-", label=method_label, linewidth=3))

    lgd = fig.legend(handles=legends, ncol=len(legends), fontsize=30, loc="lower center", fancybox=True, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)

    # ylim
    y_min = np.min([np.min(ax.get_ylim()) for ax in axs.flatten()])
    y_max = np.max([np.max(ax.get_ylim()) for ax in axs.flatten()])
    _ = [ax.set_ylim(y_min, y_max) for ax in axs.flatten()]

    ref = 1.0
    for j in range(len(SCENARIOS_NAME)):
        for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
            ax = axs[j, i]
            ax = add_bells_and_whistles(ax, x_ticks=SENSITIVITY_PARAMETER_RANGE[parameter]['values'])
            ax.plot(ax.get_xlim(),  [ref, ref],  linestyle=":", color="gray", linewidth=2)

    # save
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    filename = "grid_sensitivity_R"
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

    folder_name = Path(plot_path).resolve() / "grid_sensitivity"
    os.makedirs(str(folder_name), exist_ok=True)

    # merge all csvs
    filename = folder_name / "all_extracted_data.csv"
    if use_extracted_data:
        results = pd.read_csv(str(filename))
    else:
        results = pd.DataFrame()
        for scenario_folder in plot_path.iterdir():
            for subfolder in scenario_folder.iterdir():
                if "scatter" not in subfolder.name:
                    continue

                all_runs = subfolder / "normalized_mobility/plots/normalized_mobility/full_extracted_data_AR_60.csv"
                assert all_runs.exist(), f"{subfolder.name} hasn't been plotted yet"
                results = pd.concat([results, pd.read_csv(str(all_runs))], axis=0, ignore_index=True)
        results.to_csv(str(filename))

    plot_and_save_grid_sensitivity_analysis(results, path=plot_path)
