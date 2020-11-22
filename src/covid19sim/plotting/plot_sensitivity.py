"""
Plots a scatter plot showing trade-off between metrics of different simulations across varying mobility.
"""
import os
import yaml
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
from copy import deepcopy
from pathlib import Path

from covid19sim.utils.utils import is_app_based_tracing_intervention
from covid19sim.plotting.utils import get_proxy_r, split_methods_and_check_validity, load_plot_these_methods_config
from covid19sim.plotting.extract_tracker_metrics import _daily_false_quarantine, _daily_false_susceptible_recovered, _daily_fraction_risky_classified_as_non_risky, \
                                _daily_fraction_non_risky_classified_as_risky, _daily_fraction_quarantine
from covid19sim.plotting.extract_tracker_metrics import _mean_effective_contacts, _mean_healthy_effective_contacts, _percentage_total_infected, _positivity_rate
from covid19sim.plotting.matplotlib_utils import add_bells_and_whistles, save_figure, get_color, get_adoption_rate_label_from_app_uptake, get_intervention_label, \
                                plot_mean_and_stderr_bands, get_base_intervention, get_labelmap, get_colormap, plot_heatmap_of_advantages
from covid19sim.plotting.curve_fitting import LinearFit, GPRFit
from covid19sim.plotting.plot_normalized_mobility_scatter import get_metric_label

DPI=300
TITLESIZE = 35
TICKSIZE = 22
LEGENDSIZE = 25
TITLEPAD = 25
LABELSIZE = 24
SCENARIO_LABELSIZE=25
SCENARIO_LABELPAD=85
LABELPAD = 30
LINESTYLES = ["-", "--"]

METRICS = ['r', 'effective_contacts', 'healthy_contacts']
SENSITIVITY_PARAMETERS = ['ASYMPTOMATIC_RATIO', 'ALL_LEVELS_DROPOUT', 'P_DROPOUT_SYMPTOM',  'PROPORTION_LAB_TEST_PER_DAY']
XMETRICS = ['effective_contacts'] + SENSITIVITY_PARAMETERS
SCENARIOS = [
    [0.02, 0.004, 0.20, 0.15], # optimistic scenario (** 5% or 95% of the bounds in log domain)
    [0.06, 0.0025, 0.40, 0.25], # intermediate scenaraio (arithmetic average in log domain)
    [0.18, 0.0015, 0.60, 0.40], # worse scenario
]

# (optimistic, pessimistic)
# default str_formatter = lambda x: f"{100 * x: 2.0f}"
SENSITIVITY_PARAMETER_RANGE = {
    "ASYMPTOMATIC_RATIO" : {
        "range": [0.15, 0.45],
        "x_tick_gap": 0.05
    },
    "ALL_LEVELS_DROPOUT": {
        "range": [0.01, 0.20],
        "x_tick_gap": 0.03
    },
    "P_DROPOUT_SYMPTOM": {
        "range": [0.1, 0.60],
        "x_tick_gap": 0.10
    },
    "PROPORTION_LAB_TEST_PER_DAY": {
        "range": [0.0005, 0.005],
        "x_tick_gap": 0.001,
        "str_formatter" : lambda x: f"{100 * x: 0.2f}"
    }
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

    if name == "ASYMPTOMATIC_RATIO":
        return "Asymptomaticity\n(% of population)"

    raise ValueError(f"Invalid name: {name}")


def get_scenarios(parameter_ranges):
    """
    Constructs scenarios by selecting parameter values from parameter ranges

    Args:
        parameter_ranges (list): each element is a 2-tuple with corresponding pessimistic and optimistic (in terms of disease control) values of the parameter

    Returns:
        (list): each element is a n-tuple with values for parameters as per the parameter range
    """
    n_parameters = len(parameter_ranges)
    scenarios = [[None]*n_parameters for i in range(3)]
    scenarios_name = ["Optimistic", "Moderate", "Pessimistic"]
    idx = 0
    for parameter in SENSITIVITY_PARAMETERS:
        o, p = parameter_ranges[parameter]['range']
        D = np.abs(np.log(p) - np.log(o))
        optim_op = operator.add if o < p else operator.sub
        scenarios[0][idx] = np.exp(optim_op(np.log(o), D * 0.05)) # optimistic
        scenarios[2][idx] = np.exp(optim_op(np.log(o), D * 0.95)) # pessimistic
        scenarios[1][idx] = np.exp(optim_op(np.log(o), D * 0.50)) # intermediate
        idx += 1

    return scenarios, scenarios_name

SCENARIOS, SCENARIOS_NAME = get_scenarios(SENSITIVITY_PARAMETER_RANGE)
str_to_print = " "*10 + " ".join(SENSITIVITY_PARAMETERS) + "\n"
str_to_print += "Pessimistic: " + " ".join(map(lambda x: f"{x:0.3f}",SCENARIOS[2])) + "\n"
str_to_print += "Intermediate: " + " ".join(map(lambda x: f"{x:0.3f}",SCENARIOS[1])) + "\n"
str_to_print += "Optimistic: " + " ".join(map(lambda x: f"{x:0.3f}",SCENARIOS[0])) + "\n"
print(str_to_print)
REFERENCE_R=1.2
USE_MATH_NOTATION=False


# fix the seed
np.random.seed(123)

class StringFormatter(object):
    def __init__(self, fn):
        self.fn = fn

    def format(self, x, pos):
        return self.fn(x)

def estimate_y_and_std(x_input, fitted_fns, method, plot_advantage):
    """
    Estimates y-value and corresponding stderr to be plotted at x_input.

    Args:
        x_input (np.array):
        fitted_fns (dict): method --> GPRfit
        method (str): method for which it needs to be calculated
        plot_advantage (bool): True if the difference in R needs to be estimated. False if R is to be returned

    Returns:
        y (float): y-value
        y_std (float): std error of estimation
    """
    if plot_advantage:
        delta_r = fitted_fns[REFERENCE_METHOD].evaluate_y_for_x(x_input) - fitted_fns[method].evaluate_y_for_x(x_input)
        stderr_delta_r = np.sqrt(fitted_fns[REFERENCE_METHOD].stderr_for_x(x_input, return_var=True) + fitted_fns[method].stderr_for_x(x_input, return_var=True))
        return delta_r, stderr_delta_r

    r = fitted_fns[method].evaluate_y_for_x(x_input)
    r_std = fitted_fns[method].stderr_for_x(x_input, return_var=False)
    return r, r_std


def find_c(fitted_fn, partial_x, target_r=1.2):
    """
    Finds `effective_contacts` for which fitted_fn gives R=1.2 at other values of input given by `partial_x`.

    Args:
        fitted_fn (curve_fitting.GPRFit): GP function
        partial_x (np.array): all values of input to `fitted_fn` except the first index  (check `XMETRICS`)
        target_r (float): desired value of `fitted_fn`

    Returns:
        (float): value of `effective_contacts` at which `fitted_fn` is `target_r`
    """
    def func(x, partial_x):
        x = np.array([[x] + partial_x])
        return np.abs(fitted_fn.evaluate_y_for_x(x) - target_r).item()

    res = optimize.minimize_scalar(fun=func, args=(partial_x), bounds=(2,10), tol=1e-4)
    return res.x, res.fun

def plot_and_save_sensitivity_analysis(results, uptake_rates, path, plot_advantage=True):
    """
    Plots and saves sensitivity for SCENARIOS for data obtained from `configs/experiment/sensitivity.yaml`.

    Args:
        results (pd.DataFrame): Dataframe with rows as methods and corresponding simulation metrics.
        uptake_rate (list): APP_UPTAKE for all the methods. Assumed to be same for all app-based methods.
        path (str): path of the folder where results will be saved
    """
    TICKGAP=2
    ANNOTATION_FONTSIZE=15

    methods = results['method'].unique()
    methods_and_base_confs = results.groupby(['method', 'intervention_conf_name']).size().index
    labelmap = get_labelmap(methods_and_base_confs, path)
    colormap = get_colormap(methods_and_base_confs, path)
    INTERPOLATION_FN = GPRFit
    ALL_METHODS = OTHER_METHODS
    ALL_METHODS += [] if plot_advantage else [REFERENCE_METHOD]

    # find if only specific folders (methods) need to be plotted
    plot_these_methods = load_plot_these_methods_config(path)

    fitted_fns = {uptake_rate: {} for uptake_rate in uptake_rates}
    for i, method in enumerate(methods):
        if (
            len(plot_these_methods) > 0
            and method not in plot_these_methods
        ):
            continue

        for uptake_rate in uptake_rates:
            # function fitting
            selector = (results['method'] == method) * (results['uptake_rate'] == uptake_rate)
            x = results[selector][XMETRICS].to_numpy()
            y = results[selector]['r'].to_numpy()
            fitted_fns[uptake_rate][method] = INTERPOLATION_FN().fit(x, y)
            print(f"R-squared for {method}: {fitted_fns[method].r_squared:3.3f}")

    fig, axs = plt.subplots(nrows=len(SCENARIOS), ncols=len(SENSITIVITY_PARAMETERS), figsize=(27, 18), sharex='col', sharey=True, dpi=DPI)

    # set row and column headers
    for col, name in enumerate(SENSITIVITY_PARAMETERS):
        tmp_ax = axs[0, col].twiny()
        tmp_ax.set_xticks([])
        tmp_ax.set_xlabel(get_sensitivity_label(name), labelpad=LABELPAD, fontsize=LABELSIZE)

    y_label = "$\Delta R$" if plot_advantage else "$R$"
    for row, name in enumerate(SCENARIOS_NAME):
        axs[row, 0].set_ylabel(y_label, labelpad=LABELPAD, fontsize=LABELSIZE, rotation=0)
        tmp_ax = axs[row, -1].twinx()
        tmp_ax.set_ylabel(name+"\nScenario", fontsize=SCENARIO_LABELSIZE, fontweight="bold", rotation=0, labelpad=SCENARIO_LABELPAD )
        tmp_ax.set_yticks([])

    # plot all
    for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
        o, p = SENSITIVITY_PARAMETER_RANGE[parameter]['range']
        str_formatter = SENSITIVITY_PARAMETER_RANGE[parameter].get("str_formatter", lambda x: f"{100 * x: 2.0f}")
        (l, u) = (o, p) if o < p else (p, o)
        xs = np.linspace(l, u, 5)

        for j, scenario in enumerate(SCENARIOS):
            ax = axs[j, i]
            ax = add_bells_and_whistles(ax, y_title=None, x_title=None, TICKSIZE=TICKSIZE, x_tick_gap=SENSITIVITY_PARAMETER_RANGE[parameter]['x_tick_gap'],
                                             x_lower_lim=l, x_upper_lim=u, percent_fmt_on_x=StringFormatter(str_formatter))

            for k, uptake_rate in enumerate(uptake_rates):
                for method in ALL_METHODS:
                    y, y_std = [], []
                    for value in xs:
                        # find c0 such that R_nt = 1.2 under this scenario
                        partial_x = deepcopy(scenario)
                        partial_x[i] = value
                        c0, error = find_c(fitted_fns[uptake_rate][NO_TRACING_METHOD], partial_x, target_r=REFERENCE_R)
                        assert error < 1e-4, f"Error in minimization: {error}"

                        # find delta r
                        x_input = np.array([[c0] + partial_x])
                        val, std = estimate_y_and_std(x_input, fitted_fns[uptake_rate], method, plot_advantage=plot_advantage)
                        y.append(val)
                        y_std.append(std)

                    method_label = labelmap[method]
                    color = colormap[method]
                    ax = plot_mean_and_stderr_bands(ax, xs, np.array(y).reshape(-1), np.array(y_std).reshape(-1), \
                                        label=method_label, color=color, confidence_level=1, stderr_alpha=0.2. \
                                        linestyle=LINESTYLES[k])

    # spacing between plots
    plt.subplots_adjust(left=0.125, wspace=0.2, hspace=0.2, bottom=0.15)

    # legends
    legends = []
    for method in ALL_METHODS:
        method_label = labelmap[method]
        color = colormap[method]
        legends.append(Line2D([0, 1], [0, 0], color=color, linestyle="-", label=method_label))

    adoption_rates = [get_adoption_rate_label_from_app_uptake(uptake_rate) for uptake_rate in uptake_rates]
    if len(uptake_rates) > 1:
        for uptake_rate in uptake_rates:
            legends.append(Line2D([0, 1], [0, 0], color="black", linestyle=LINESTYLES[k], label=adoption_rates[k]))

    lgd = fig.legend(handles=legends, ncol=len(legends), fontsize=30, loc="lower center", fancybox=True, bbox_to_anchor=(0.5, -0.03, 0, 0.5))

    # ylim
    y_min = np.min([np.min(ax.get_ylim()) for ax in axs.flatten()])
    y_max = np.max([np.max(ax.get_ylim()) for ax in axs.flatten()])
    _ = [ax.set_ylim(y_min, y_max) for ax in axs.flatten()]

    ref = 0.0 if plot_advantage else 1.0
    for ax in axs.flatten():
        ax.plot(ax.get_xlim(),  [ref, ref],  linestyle=":", color="gray", linewidth=2)

    # save
    fig.tight_layout()
    filename = f"sensitivity_deltaR" if plot_advantage else "sensitivity_R"
    AR_str = "all" if len(adoption_rates) > 1 else adoption_rates[0]
    filepath = save_figure(fig, basedir=path, folder="sensitivity", filename=f'{filename}_AR_{AR_str}', bbox_extra_artists=(lgd,), bbox_inches=None)
    print(f"Sensitivity analysis saved at {filepath}")

def _extract_metrics(data, conf):
    """
    Extracts `METRICS` from data corresponding to a single simulation run.

    Args:
        data (dict): tracker files for the simulation
        conf (dict): an experimental configuration.

    Returns:
        (list): a list of scalars representing metrics in `METRICS` for the simulations
    """
    out = []
    out.append(get_proxy_r(data, verbose=False))
    out.append(_mean_effective_contacts(data))
    out.append(_mean_healthy_effective_contacts(data))

    out.append(conf['ALL_LEVELS_DROPOUT'])
    out.append(conf['PROPORTION_LAB_TEST_PER_DAY'])
    out.append(conf['P_DROPOUT_SYMPTOM'])
    out.append(1.0 * sum(h['asymptomatic'] for h in data['humans_demographics']) / len(data['humans_demographics']))

    return out

def _extract_data(simulation_runs, method):
    """
    Extracts all metrics from simulation runs.

    Args:
        simulation_runs (dict): folder_name --> {'conf': yaml file, 'pkl': tracker file}
        method (str): name of the method for which this extraction is being done.

    Returns:
        (pd.DataFrame): Each row is method specific information and extracted scalar metrics.
    """
    all_data = []
    for simname, sim in simulation_runs.items():
        data = sim['pkl']
        intervention_name = get_base_intervention(sim['conf'])
        mobility_factor = sim['conf']['GLOBAL_MOBILITY_SCALING_FACTOR']
        row =  [method, simname, mobility_factor, intervention_name, is_app_based_tracing_intervention(intervention_conf=sim['conf'])] + _extract_metrics(data, sim['conf'])
        all_data.append(row)

    columns = ['method', 'dir', 'mobility_factor', 'intervention_conf_name','app_based'] + METRICS + SENSITIVITY_PARAMETERS
    return pd.DataFrame(all_data, columns=columns)

def run(data, plot_path, compare=None, **kwargs):
    """
    Plots and saves mobility scatter with various configurations across different methods.
    Outputs are -
        1. CSV files of extracted metrics
        2. A txt file for each adoption rate showing mobility factors for reasonable R
        3. several plots for mobility and other metrics. Check the code down below.
        4. all_advantages.csv containing pairwise advantages of methods derived from curve fit.

    Args:
        data (dict): intervention_name --> APP_UPTAKE --> folder_name --> {'conf': yaml file, 'pkl': tracker file}
        plot_path (str): path where to save plots
    """
    use_extracted_data = kwargs.get('use_extracted_data', False)

    folder_name = Path(plot_path).resolve() / "sensitivity"
    os.makedirs(str(folder_name), exist_ok=True)

    uptake_key_filepath = folder_name / "uptake_keys.csv"
    if use_extracted_data:
        uptake_keys = pd.read_csv(str(uptake_key_filepath))['uptake'].tolist()
    else:
        app_based_methods, other_methods, uptake_keys = split_methods_and_check_validity(data)
        pd.DataFrame(uptake_keys, columns=['uptake']).to_csv(str(uptake_key_filepath))

    all_uptake_data = pd.DataFrame()
    ## data preparation
    for uptake in uptake_keys:
        adoption_rate = get_adoption_rate_label_from_app_uptake(uptake)
        extracted_data_filepath = folder_name / f"full_extracted_data_AR_{adoption_rate}.csv"
        if not use_extracted_data:
            no_app_df = pd.DataFrame([])
            for method in other_methods:
                key = list(data[method].keys())[0]
                no_app_df = pd.concat([no_app_df, _extract_data(data[method][key], method)], axis='index', ignore_index=True)

            all_data = deepcopy(no_app_df)
            for method in app_based_methods:
                all_data = pd.concat([all_data, _extract_data(data[method][uptake], method)], axis='index', ignore_index=True)

            all_data['uptake_rate'] = uptake
            all_uptake_data = pd.concat([all_uptake_data, all_data], axis=0)
            all_data.to_csv(str(extracted_data_filepath))
        else:
            assert extracted_data_filepath.exists(), f"{extracted_data_filepath} do not exist"
            all_data = pd.read_csv(str(extracted_data_filepath))
            all_uptake_data = pd.concat([all_uptake_data, all_data], axis=0)

        # plot
        plot_and_save_sensitivity_analysis(all_data, [uptake], path=plot_path, plot_advantage=True)
        plot_and_save_sensitivity_analysis(all_data, [uptake], path=plot_path, plot_advantage=False)

    # plot all
    plot_and_save_sensitivity_analysis(all_uptake_data, uptake_keys, path=plot_path, plot_advantage=True)
    plot_and_save_sensitivity_analysis(all_uptake_data, uptake_keys, path=plot_path, plot_advantage=False)
