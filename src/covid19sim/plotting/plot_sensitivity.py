"""
Plots a scatter plot showing trade-off between metrics of different simulations across varying mobility.
"""
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
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
TITLESIZE = 25
LABELPAD = 0.50
LABELSIZE = 25
TICKSIZE = 20
LEGENDSIZE = 25
ANNOTATION_FONTSIZE=15

METRICS = ['r', 'effective_contacts', 'healthy_contacts']
SENSITIVITY_PARAMETERS = ['ALL_LEVELS_DROPOUT', 'PROPORTION_LAB_TEST_PER_DAY', 'P_DROPOUT_SYMPTOM', 'BASELINE_P_ASYMPTOMATIC']
SCENARIOS = [
    [0.10, 0.004, 0.20, 0.15], # optimistic scenario
    [0.30, 0.025, 0.40, 0.30], # intermediate scenaraio
    [0.50, 0.001, 0.60, 0.50], # worse scenario
]
SENSITIVITY_PARAMETER_RANGE = [
    [0.05, 0.50],
    [0.0015, 0.005],
    [0.20, 0.65],
    [0.09, 0.36]
]
USE_MATH_NOTATION=False

# fix the seed
np.random.seed(123)

def plot_and_save_sensitivity_analysis(results, uptake_rate, path):
    """
    Plots and saves sensitivity for SCENARIOS for data obtained from `configs/experiment/sensitivity.yaml`.

    Args:
        results (pd.DataFrame): Dataframe with rows as methods and corresponding simulation metrics.
        uptake_rate (str): APP_UPTAKE for all the methods. Assumed to be same for all app-based methods.
        path (str): path of the folder where results will be saved
    """
    assert xmetric in METRICS and ymetric in METRICS, f"Unknown metrics: {xmetric} or {ymetric}. Expected one of {METRICS}."
    TICKGAP=2
    ANNOTATION_FONTSIZE=15

    adoption_rate = get_adoption_rate_label_from_app_uptake(uptake_rate)
    methods = results['method'].unique()
    methods_and_base_confs = results.groupby(['method', 'intervention_conf_name']).size().index
    labelmap = get_labelmap(methods_and_base_confs, path)
    colormap = get_colormap(methods_and_base_confs, path)
    INTERPOLATION_FN = _get_interpolation_kind(xmetric, ymetric, use_gp=USE_GP)

    # find if only specific folders (methods) need to be plotted
    plot_these_methods = load_plot_these_methods_config(path)

    # fit functions
    fitted_fns = {}
    for i, method in enumerate(methods):
        # to include only certain methods to avoid busy plots
        # but still involve pairwise comparison of individual methods
        if (
            len(plot_these_methods) > 0
            and method not in plot_these_methods
        ):
            continue

        # function fitting
        selector = results['method'] == method
        x = results[selector][xmetric].to_numpy()
        y = results[selector][ymetric].to_numpy()
        fitted_fns[method] = INTERPOLATION_FN().fit(x, y)

    # set up subplot grid
    fig = plt.figure(num=1, figsize=(15,10), dpi=DPI)
    gridspec.GridSpec(len(SCENARIOS), len(SENSITIVITY_PARAMETERS))


    for k, scenario in enumerate(SCENARIOS):
        for i, parameter in enumerate(SENSITIVITY_PARAMETERS):
            x = np.linspace(*SENSITIVITY_PARAMETER_RANGE[i], 10)
            y, y_std = [], []
            for value in x:
                # find c0 such that R_nt = 1.2 under this scenario
                c0 = fitted_fns['post-lockdown-no-tracing']
                # find delta r
                delta_r, delta_r_std = fitted_fns['bdt1'].find_y_for_x(c0) - fitted_fns['heuristic'].find_y_for_x(c0)
                y.append(delta_r)
                y_std.append(delta_r_std)

            method_label = labelmap[method]
            color = colormap[method]
            ax = plt.subplot2grid(shape=(1,1), loc=(k,i), rowspan=1, colspan=1, fig=fig)
            ax = plot_mean_and_stderr_bands(ax, x, y, y_std, label=label, color=colormap[method], confidence_level=1.96, stderr_alpha=0.3)

    # plot fitted fns
    x = np.arange(results[xmetric].min(), results[xmetric].max(), 0.05)
    for method, fn in fitted_fns.items():
        y = fn.evaluate_y_for_x(x)
        label = labelmap[method] if not plot_scatter else None
        stderr = fn.stderr_for_x(x, analytical=True)
        ax = plot_mean_and_stderr_bands(ax, x, y, stderr, label=label, color=colormap[method], confidence_level=1.96, stderr_alpha=0.3)

    xlabel = get_metric_label(xmetric)
    ylabel = get_metric_label(ymetric)
    ax = add_bells_and_whistles(ax, y_title=ylabel, x_title=None if plot_residuals else xlabel, XY_TITLEPAD=LABELPAD, \
                    XY_TITLESIZE=LABELSIZE, TICKSIZE=TICKSIZE, legend_loc='upper left', \
                    LEGENDSIZE=LEGENDSIZE, x_tick_gap=TICKGAP, x_lower_lim=2, x_upper_lim=10.5, y_lower_lim=0.25)

    # figure title
    fig.suptitle(f"Sensitivity Analysis", fontsize=TITLESIZE, y=1.05)

    # save
    fig.tight_layout()
    filename = f"{USE_GP_STR}{ymetric}_{xmetric}_mobility_scatter"
    filepath = save_figure(fig, basedir=path, folder="sensitivity", filename=f'{filename}_AR_{adoption_rate}')
    print(f"Sensitivity analysis saved at {filepath}")

def _extract_metrics(data):
    """
    Extracts `METRICS` from data corresponding to a single simulation run.

    Args:
        data (dict): tracker files for the simulation

    Returns:
        (list): a list of scalars representing metrics in `METRICS` for the simulations
    """
    out = []
    out.append(get_proxy_r(data, verbose=False))
    out.append(_mean_effective_contacts(data))
    out.append(_mean_healthy_effective_contacts(data))
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
        row =  [method, simname, mobility_factor, intervention_name, is_app_based_tracing_intervention(intervention_conf=sim['conf'])] + _extract_metrics(data)
        row += [sim['conf'][key] for key in SENSITIVITY_PARAMETERS]
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

            all_data.to_csv(str(extracted_data_filepath))
        else:
            assert extracted_data_filepath.exists(), f"{extracted_data_filepath} do not exist"
            all_data = pd.read_csv(str(extracted_data_filepath))

        # plot
        plot_and_save_sensitivity_analysis(all_data, uptake, path=plot_path)
