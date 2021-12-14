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
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
from copy import deepcopy
from pathlib import Path

from covid19sim.plotting.utils import load_plot_these_methods_config
from covid19sim.plotting.matplotlib_utils import add_bells_and_whistles, save_figure, get_color, get_adoption_rate_label_from_app_uptake, get_intervention_label, \
                                plot_mean_and_stderr_bands, get_base_intervention, get_labelmap, get_colormap, plot_heatmap_of_advantages, get_sensitivity_label
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

# CONTACT_RANGE = [6 - 0.5, 6 + 0.5]
CONTACT_RANGE=None # [x1, x2] if provided GPRFit is not used i.e `TARGET_R_FOR_NO_TRACING` and `MARGIN` are not used
TARGET_R_FOR_NO_TRACING = 1.2 # find the performance of simulations around (defined by MARGIN) the number of contacts where NO_TRACING has R of 1.2
MARGIN = 0.5

METRICS = ['r', 'effective_contacts', 'healthy_contacts']
SENSITIVITY_PARAMETERS = ['ALL_LEVELS_DROPOUT', 'P_DROPOUT_SYMPTOM',  'PROPORTION_LAB_TEST_PER_DAY']
# SENSITIVITY_PARAMETERS = ['ALL_LEVELS_DROPOUT', 'P_DROPOUT_SYMPTOM']
XMETRICS = ['effective_contacts'] + SENSITIVITY_PARAMETERS

# (optimistic, mdoerate, pessimistic)
# default str_formatter = lambda x: f"{100 * x: 2.0f}"
# NOTE: Following is taken from job_scripts/sensitivity_launch_mpic.py
SENSITIVITY_PARAMETER_RANGE ={
    "ASYMPTOMATIC_RATIO": {
        "values": [0.29, 0.35, 0.40, 0.45], # 0.20 0.30 0.40
        "no-effect":[]
    },
    "BASELINE_P_ASYMPTOMATIC": {
        # "values": [0.20, 0.30, 0.40], # 0.20 0.30 0.40
        "values": [0.1475, 0.2525, 0.3575], # asymptomatic-ratio =  0.20 0.30 0.40
        "no-effect":[]
    },
    "ALL_LEVELS_DROPOUT": {
        # "values": [0.02, 0.08, 0.16], # 0.02 0.08 0.16
        "values": [0.10, 0.30, 0.50],
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
HUE_ORDER = [NO_TRACING_METHOD, REFERENCE_METHOD] + OTHER_METHODS

# our scenario - 0.02, 0.001, 0.20, 0.23

REFERENCE_R=1.2
USE_MATH_NOTATION=False

# fix the seed
np.random.seed(123)

class StringFormatter(object):
    def __init__(self, fn):
        self.fn = fn

    def format(self, x, pos):
        return self.fn(x)

def plot_fitted_fns_at_stable_point(ax, fitted_fns, x_value, target_r, colormap):
    """
    """
    assert NO_TRACING_METHOD in fitted_fns, f"{NO_TRACING_METHOD} not found in {fitted_fns.keys()}"

    no_tracing_fn = fitted_fns[NO_TRACING_METHOD]
    stable_point = no_tracing_fn.find_x_for_y(target_r)

    for method, fn in fitted_fns.items():
        y = fn.evaluate_y_for_x(stable_point)
        yerr = fn.stderr_for_x(stable_point, return_var=False)
        ax.errorbar(x=x_value, y=y, yerr=yerr, color=colormap[method], fmt='-o')

    return ax


def find_fitted_fns(df):
    """
    Fits GP regression (`r` vs `effective_contacts`) to each function in `df`

    Args:
        df (pd.DataFrame): these columns are required - `method`, `effective_contacts`, `r`

    Returns:
        (dict): where each key is a method and value is a `curve_fitting.GPRFit` with optimized parameters
    """
    fitted_fns = {}
    for method in df['method'].unique():
        selector = df['method'] == method
        x = df[selector]['effective_contacts'].to_numpy()
        y = df[selector]['r'].to_numpy()
        fitted_fns[method] = GPRFit().fit(x, y)

    return fitted_fns

def plot_and_save_grid_sensitivity_analysis(results, path, y_metric):
    """
    Plots and saves grid sensitivity for various SCENARIOS.

    Args:
        results (pd.DataFrame): Dataframe with rows extracted from plotting scripts of normalized_mobility i.e. stable_frames.csv (check plot_normalized_mobility_scatter.py)
        path (str): path of the folder where results will be saved
        y_metric (str): metric that needs to be plotted on y-axis
    """
    TICKGAP=2
    ANNOTATION_FONTSIZE=15

    assert y_metric in results.columns, f"{y_metric} not found in columns :{results.columns}"

    methods = results['method'].unique()
    methods_and_base_confs = results.groupby(['method', 'intervention_conf_name']).size().index
    labelmap = get_labelmap(methods_and_base_confs, path)
    colormap = get_colormap(methods_and_base_confs, path)

    # find if only specific folders (methods) need to be plotted
    plot_these_methods = load_plot_these_methods_config(path)
    fig, axs = plt.subplots(nrows=len(SCENARIOS_NAME), ncols=len(SENSITIVITY_PARAMETERS), figsize=(27, 18), sharex='col', sharey=True, dpi=DPI, constrained_layout=True)

    # when none of the sensitivity parameter affects no tracing method. pick any..
    no_tracing_fitted_fn = find_fitted_fns(results[results['method'] == NO_TRACING_METHOD])

    # plot all
    for j, scenario in enumerate(SCENARIOS_NAME):
        # scenario specific results
        idx = SCENARIO_PARAMETERS_IDX[scenario]
        SCENARIO_PARAMETERS = [SENSITIVITY_PARAMETER_RANGE[param]['values'][idx] for param in SENSITIVITY_PARAMETERS]
        scenario_df = results[(results[SENSITIVITY_PARAMETERS] == SCENARIO_PARAMETERS).all(1)]

        if scenario_df.shape[0] == 0:
            continue

        scenario_fitted_fns = find_fitted_fns(scenario_df)
        if NO_TRACING_METHOD not in scenario_fitted_fns:
            scenario_fitted_fns.update(no_tracing_fitted_fn)

        for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
            values = SENSITIVITY_PARAMETER_RANGE[parameter]['values']
            no_effect_on_methods = SENSITIVITY_PARAMETER_RANGE[parameter]['no-effect']
            str_formatter = SENSITIVITY_PARAMETER_RANGE[parameter].get("str_formatter", lambda x: f"{100 * x: 2.0f}")

            # ax_df = pd.DataFrame()
            fitted_fns = {}
            # ax_df = pd.concat([ax_df, scenario_df], ignore_index=True, axis=0)
            ax = axs[j, i]
            for param_index, value in enumerate(values):
                tmp_params = copy.deepcopy(SCENARIO_PARAMETERS)
                tmp_params[i] = value
                df = results[(results[SENSITIVITY_PARAMETERS] == tmp_params).all(1)]

                if df.shape[0] == 0:
                    continue

                fitted_fns = find_fitted_fns(df)
                cell_methods = df['method'].unique()
                for method in no_effect_on_methods:
                    if method not in cell_methods:
                        fitted_fns[method] = scenario_fitted_fns[method]

                plot_fitted_fns_at_stable_point(ax, fitted_fns, value, TARGET_R_FOR_NO_TRACING, colormap)

    # set row and column headers
    for col, name in enumerate(SENSITIVITY_PARAMETERS):
        tmp_ax = axs[0, col].twiny()
        tmp_ax.set_xticks([])
        tmp_ax.set_xlabel(get_sensitivity_label(name), labelpad=LABELPAD, fontsize=LABELSIZE)

    if y_metric == "r":
        y_label = "$R$"
    elif y_metric == "percentage_infected":
        y_label = "% infected"

    for row, name in enumerate(SCENARIOS_NAME):
        axs[row, 0].set_ylabel(y_label, labelpad=LABELPAD, fontsize=LABELSIZE, rotation=90)
        tmp_ax = axs[row, -1].twinx()
        tmp_ax.set_ylabel(name+"\nScenario", fontsize=SCENARIO_LABELSIZE, fontweight="bold", rotation=0, labelpad=SCENARIO_LABELPAD )
        tmp_ax.set_yticks([])

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
    for j in range(len(SCENARIOS_NAME)):
        for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
            ax = axs[j, i]
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
    filename = f"grid_sensitivity_{y_metric}"
    filename += f"_C_{CONTACT_RANGE[0]}-{CONTACT_RANGE[1]}" if CONTACT_RANGE is not None else ""
    filename = filename.replace(".", "_") # to allow decimals in contact range string
    filepath = save_figure(fig, basedir=path, folder="grid_sensitivity_absolute", filename=f'{filename}', bbox_extra_artists=(lgd,), bbox_inches='tight')
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

    folder_name = Path(plot_path).resolve() / "grid_sensitivity_absolute"
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
                all_runs = subfolder / "normalized_mobility/plots/normalized_mobility/full_extracted_data_AR_60.csv"
                # assert all_runs.exists(), f"{subfolder.name} hasn't been plotted yet"
                if all_runs.exists():
                    results = pd.concat([results, pd.read_csv(str(all_runs))], axis=0, ignore_index=True)
        results.to_csv(str(filename))

    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='r')
    plot_and_save_grid_sensitivity_analysis(results, path=plot_path, y_metric='percentage_infected')
