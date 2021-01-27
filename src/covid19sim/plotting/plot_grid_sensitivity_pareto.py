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
from collections import defaultdict

from covid19sim.plotting.utils import load_plot_these_methods_config
from covid19sim.plotting.matplotlib_utils import add_bells_and_whistles, save_figure, get_color, get_adoption_rate_label_from_app_uptake, get_intervention_label, \
                                plot_mean_and_stderr_bands, get_base_intervention, get_labelmap, get_colormap, plot_heatmap_of_advantages, get_sensitivity_label, make_color_transparent
from covid19sim.plotting.curve_fitting import GPRFit, bootstrap
from covid19sim.plotting.plot_normalized_mobility_scatter import get_metric_label, get_polyfit_str

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
SHAPES = ["^", "s", "P", "D"]

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
    },
    "adoption_rate": {
        "values": [60, 50, 40, 30, 20],
        "no-effect": ['post-lockdown-no-tracing']
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

def plot_and_save_grid_sensitivity_paretos(results, path, SENSITIVITY_PARAMETERS, trend_fit):
    """
    Plots and saves "\Delta R (method and NO_TRACING_METHOD)"" vs "Contacts at which method controlled the outbreak" paretos for various SCENARIOS.

    Args:
        results (pd.DataFrame): Dataframe with rows extracted from plotting scripts of normalized_mobility i.e. stable_frames.csv (check plot_normalized_mobility_scatter.py)
        path (str): path of the folder where results will be saved
        SENSITIVITY_PARAMETERS (list): list of parameters
        trend_fit (str): type of fit that models have used
    """
    TICKGAP=2
    ANNOTATION_FONTSIZE=15

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
    SCENARIO_PARAMETERS = [DEFAULT_PARAMETER_VALUES[param] for param in  SENSITIVITY_PARAMETERS]
    scenario_df = results[(results[SENSITIVITY_PARAMETERS] == SCENARIO_PARAMETERS).all(1)]

    for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
        values = SENSITIVITY_PARAMETER_RANGE[parameter]['values']
        no_effect_on_methods = SENSITIVITY_PARAMETER_RANGE[parameter]['no-effect']

        ax_df = pd.DataFrame()
        ax = axs[0, i]
        method_ar_pareto = defaultdict(list)
        for param_index, value in enumerate(values):
            tmp_params = copy.deepcopy(SCENARIO_PARAMETERS)
            tmp_params[i] = value
            df = results[(results[SENSITIVITY_PARAMETERS] == tmp_params).all(1)]

            if df.shape[0] == 0:
                continue

            #
            cell_methods = df['method'].unique()
            for method in no_effect_on_methods:
                if method not in cell_methods:
                    tmp_df = scenario_df[scenario_df['method'] == method]
                    tmp_df[parameter] = value
                    df = pd.concat([df, tmp_df], axis=0)

            color_scale = 2.0
            value_shape = SHAPES[param_index] if parameter != "adoption_rate" else "o"
            xs, ys, yerrs, cs, sigs = [], [], [], [], []
            for ar in sorted(df['adoption_rate'].unique(), key=lambda x:-x):
                color_scale *= 0.5
                for method in df['method'].unique():
                    selector = (df[['adoption_rate', 'method']] == [ar, method]).all(1)
                    xs.append(df[selector]['threshold_contacts'].item())
                    ys.append(df[selector]['advantage'].item())
                    yerrs.append(df[selector]['advantage_stderr'].item())
                    sigs.append(df[selector]['p-value'].item())
                    cs.append(make_color_transparent(colormap[method], alpha=color_scale))
                    method_ar_pareto[(method, ar, cs[-1])].append([xs[-1], ys[-1]])
                    ax.errorbar(x=xs[-1], y=ys[-1], yerr=yerrs[-1], color=cs[-1], marker=value_shape, ms=50)

        # line connecting method-ar points across different sensitivity values
        for (method, ar, color), series in method_ar_pareto.items():
            xs, ys = list(zip(*series))
            ax.plot(xs, ys, linestyle=":", color=color)

        # legend for shapes - sensitivity values
        if parameter != "adoption_rate":
            legend0 = []
            for param_index, value in enumerate(values):
                legend0.append(Line2D([], [], color='black', marker=SHAPES[param_index], linestyle="None", markersize=10, label=value))

        ax.legend(handles=legend0, ncol=1, fontsize=30, loc="top right", fancybox=True)

    # set plot header
    for col, name in enumerate(SENSITIVITY_PARAMETERS):
        tmp_ax = axs[0, col].twiny()
        tmp_ax.set_xticks([])
        tmp_ax.set_xlabel(get_sensitivity_label(name), labelpad=LABELPAD, fontsize=LABELSIZE)

    # ylabels
    axs[0, 0].set_ylabel("\delta R", labelpad=LABELPAD, fontsize=LABELSIZE, rotation=90)

    # legends for color - method
    legend1 = []
    for method in results['method'].unique():
        method_label = labelmap[method]
        color = colormap[method]
        legend1.append(Line2D([0, 1], [0, 0], color=color, linestyle="-", label=method_label, linewidth=3))

    # legend for transparency - adoption rate
    legend2 = []
    ar_color = [(x[1], x[2]) for x in method_ar_pareto.keys() if x[0] == method]
    for ar, color in ar_color:
        legend2.append(Line2D([0,1], [0,0], color=color, linestyle="-", label=ar, linewidth=6))

    lgd1 = fig.legend(handles=legend1, ncol=len(legend1), fontsize=30, loc="lower center", fancybox=True, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)
    lgd2 = fig.legend(handles=legend1, ncol=len(legend2), fontsize=30, loc="lower center", fancybox=True, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)

    # ylim
    y_min = np.min([np.min(ax.get_ylim()) for ax in axs.flatten()])
    y_max = np.max([np.max(ax.get_ylim()) for ax in axs.flatten()])
    _ = [ax.set_ylim(y_min, y_max) for ax in axs.flatten()]

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

    # save
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    filename = "paretos_r_C" + f"_{trend_fit}_fit"
    filepath = save_figure(fig, basedir=path, folder="grid_sensitivity", filename=f'{filename}', bbox_extra_artists=(lgd1, lgd2,), bbox_inches='tight')
    print(f"Sensitivity analysis  (Paretos) saved at {filepath}")

def load_models(models_folder, trend_fit=""):
    """
    Loads all GP models fit using `trend_fit` args in `models_folder`

    Args:
        models_folder (pathlib.Path): path where models are saved
        trend_fit (str): fit for the models

    Returns:
        (dict): keys are method name, values are loaded models
    """
    POLYFIT_STR = get_polyfit_str(trend_fit)
    BASESTR = f"GP{POLYFIT_STR}_model_"
    folders = models_folder.glob(f"{BASESTR}*") # GP_PolyFit_model_*, GP_Linear_model_* or GP_model_*
    loaded_models = {}
    for path in folders:
        method = path.name.replace(BASESTR, "")
        model = INTERPOLATION_FN(fit=trend_fit).load(path)
        loaded_models[method] = model

    return loaded_models

def get_offsets_and_contacts(loaded_models, sensitivity_parameters):
    """
    Computes advantages of methods over no tracing.

    Args:
        loaded_models (dict): keys are tuple of adoption_rate, sensitivity_parameter values, method_conf_name, method name
        sensitivity_parameters (list): list of string denotign the name of parameters

    Returns:
        (pd.DataFrame): resulting values of advantages, their stderrs, and p-values at the threshold point of contacts.
    """
    def get_best_match(no_tracing_models, ar, sp_values, sensitivity_parameters):
        if len(no_tracing_models) == 1:
            return list(no_tracing_models.values())[0]

        match = {key: value for key, value in no_tracing_models.items() if key[1] == sp_values}
        if len(match) == 1:
            return list(match.values())[0]
        return [value for key, value in match.items() if key[0] == ar][0]

    no_tracing_models = {key: value for key, value in loaded_models.items() if key[-1] == NO_TRACING_METHOD}

    obs = []
    for key, model in loaded_models.items():
        if key[3] == NO_TRACING_METHOD:
            continue

        ar = key[0]
        sp_values = key[1]
        conf_name = key[2]
        try: # ??
            no_tracing_model = get_best_match(no_tracing_models, ar, sp_values, sensitivity_parameters)
        except:
            breakpoint()
        c0 = model.find_x_for_y(1.0).item() # contacts at which disease is under control
        delta_r, delta_r_stderr, cdf = model.find_offset_and_stderr_at_x(c0, no_tracing_model, analytical=True)
        obs.append([ar, *sp_values, conf_name, key[3], c0, delta_r, delta_r_stderr, cdf])

    return pd.DataFrame(obs, columns=['adoption_rate', *sensitivity_parameters, "intervention_conf_name", "method", "threshold_contacts", "advantage", "advantage_stderr", "p-value"])

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
    TREND_FIT = "linear" # linear, ""
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
    elif (
        sensitivity_parameter == "adoption-rate"
        or "sensitivity_ARx" in MAIN_FOLDER
    ):
        SENSITIVITY_PARAMETERS = ['adoption_rate']

    else:
        raise ValueError("Sensitivity parameter not specified..")

    print(f"sensitivity parameters : {SENSITIVITY_PARAMETERS}")

    folder_name = Path(plot_path).resolve() / "grid_sensitivity"
    os.makedirs(str(folder_name), exist_ok=True)

    # merge all csvs
    filename = folder_name / "all_extracted_offsets.csv"
    if use_extracted_data:
        results = pd.read_csv(str(filename))
    else:
        print(f"Plot path: {str(plot_path)}")
        results = {}
        for scenario_folder in plot_path.parent.iterdir():
            for subfolder in scenario_folder.iterdir():
                if "scatter" not in subfolder.name:
                    continue
                print(f"Currently at: {str(subfolder)}.")
                results_folder = subfolder / "normalized_mobility/plots/normalized_mobility/"
                models_folder = results_folder / "models" #"models_r_vs_effective_contacts"
                models = load_models(models_folder, trend_fit=TREND_FIT)
                full_data = pd.read_csv(list(results_folder.glob("full_extracted_data_AR_*.csv"))[-1])

                try: # ??
                    ar = full_data[full_data['method'] != NO_TRACING_METHOD]['adoption_rate'].unique()
                except:
                    continue
                assert len(ar) <= 1, f"Expected atmost one adoption rate per folder. Got {len(ar)} adoption rates:{ar}"
                sp = []
                for x in SENSITIVITY_PARAMETERS:
                    _x = full_data[x].unique()
                    assert len(_x) == 1, f"Expected one unique value for sensitivity parameter. Got {len(_x)} values: {_xq}"
                    sp.append(_x.item())

                for method, loaded_model in models.items():
                    conf_name = full_data[full_data['method'] == method]['intervention_conf_name'].unique()
                    if method == NO_TRACING_METHOD: # ??
                        conf_name = np.array([NO_TRACING_METHOD])
                    assert len(conf_name) == 1, f"Expected one intervention_conf file for {method}. Got {len(conf_name)} files : {conf_name}"
                    results[(ar.item(), tuple(sp), conf_name.item(), method)] = loaded_model

        # extract points for pareto
        results  = get_offsets_and_contacts(results, SENSITIVITY_PARAMETERS)
        results.to_csv(str(filename))

    print("Unique adoption rates: ", results['adoption_rate'].unique())
    plot_and_save_grid_sensitivity_paretos(results, path=plot_path, SENSITIVITY_PARAMETERS=SENSITIVITY_PARAMETERS, trend_fit=TREND_FIT)
