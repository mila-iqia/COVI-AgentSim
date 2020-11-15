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
USE_MATH_NOTATION=False

# fix the seed
np.random.seed(123)

def plot_and_save_mobility_scatter(results, uptake_rate, xmetric, ymetric, path, USE_GP=False, plot_residuals=False, display_r_squared=False, annotate_advantages=True, plot_scatter=True, plot_heatmap=True):
    """
    Plots and saves scatter plot for data obtained from `configs/experiment/normalized_mobility.yaml` showing a trade off between health and mobility.

    Args:
        results (pd.DataFrame): Dataframe with rows as methods and corresponding simulation metrics.
        uptake_rate (str): APP_UPTAKE for all the methods. Assumed to be same for all app-based methods.
        xmetric (str): metric on the x-axis
        ymetric (str): metrix on the y-axis
        path (str): path of the folder where results will be saved
        USE_GP (bool): if True, uses GP regression to fit x and y.
        plot_residuals (bool): If True, plot a scatter plot of residuals at the bottom.
        display_r_squared (bool): If True, show R-squared value in the legend.
        annotate_advantages (bool): if True, annotates the plot with advantages
        plot_scatter (bool): if True, plots scatter points corresponding to each experiment.
        plot_heatmap (bool): if True, plots heatmap of pairwise advantages.
    """
    assert xmetric in METRICS and ymetric in METRICS, f"Unknown metrics: {xmetric} or {ymetric}. Expected one of {METRICS}."
    TICKGAP=2
    ANNOTATION_FONTSIZE=15
    USE_GP_STR = "GP_" if USE_GP else ""

    adoption_rate = get_adoption_rate_label_from_app_uptake(uptake_rate)
    methods = results['method'].unique()
    methods_and_base_confs = results.groupby(['method', 'intervention_conf_name']).size().index
    labelmap = get_labelmap(methods_and_base_confs, path)
    colormap = get_colormap(methods_and_base_confs, path)
    INTERPOLATION_FN = _get_interpolation_kind(xmetric, ymetric, use_gp=USE_GP)

    # find if only specific folders (methods) need to be plotted
    plot_these_methods = load_plot_these_methods_config(path)

    # set up subplot grid
    fig = plt.figure(num=1, figsize=(15,10), dpi=DPI)
    gridspec.GridSpec(3,1)
    if plot_residuals:
        ax = plt.subplot2grid(shape=(3,1), loc=(0,0), rowspan=2, colspan=1, fig=fig)
        res_ax = plt.subplot2grid(shape=(3,1), loc=(2,0), rowspan=1, colspan=1, fig=fig)
    else:
        ax = plt.subplot2grid(shape=(3,1), loc=(0,0), rowspan=3, colspan=1, fig=fig)

    fitted_fns = {}
    for i, method in enumerate(methods):
        # to include only certain methods to avoid busy plots
        # but still involve pairwise comparison of individual methods
        if (
            len(plot_these_methods) > 0
            and method not in plot_these_methods
        ):
            continue

        #
        if _filter_out_irrelevant_method(xmetric, ymetric, method, results):
            continue

        # function fitting
        selector = results['method'] == method
        x = results[selector][xmetric].to_numpy()
        y = results[selector][ymetric].to_numpy()
        fitted_fns[method] = INTERPOLATION_FN().fit(x, y)

        size = results[selector]['mobility_factor'].to_numpy()
        method_label = labelmap[method]
        if display_r_squared:
            method_label = f"{method_label} ($r^2 = {fitted_fns[method].r_squared: 0.3f}$)"
        color = colormap[method]

        #
        if plot_scatter:
            ax.scatter(x, y, s=size*75, color=color, label=method_label, alpha=0.5)

        # residuals
        if (
            plot_residuals
            and plot_scatter
        ):
            res = fitted_fns[method].res
            res_ax.scatter(x, res, s=size*75, color=color)
            for _x, _res in zip(x, res):
                res_ax.plot([_x, _x], [0, _res], color=color, linestyle=":")

    # plot fitted fns
    x = np.arange(results[xmetric].min(), results[xmetric].max(), 0.05)
    for method, fn in fitted_fns.items():
        y = fn.evaluate_y_for_x(x)
        label = labelmap[method] if not plot_scatter else None
        stderr = fn.stderr_for_x(x, analytical=True)
        ax = plot_mean_and_stderr_bands(ax, x, y, stderr, label=label, color=colormap[method], confidence_level=1.96, stderr_alpha=0.3)

    # compute and plot offset and its confidence bounds
    if ymetric == "r":
        points = find_all_pairs_offsets_and_stddev(fitted_fns)
        table_to_save = []
        x_offset = 0.0
        for p1, p2, res1, m1, m2, res2, plot in points:
            table_to_save.append([m1, m2, labelmap[m1], labelmap[m2], *res1, *res2])
            if (
                not annotate_advantages
                or not plot
                or len(methods) == 3 # if there are 3 methods, annotation will not be cluttered
            ):
                continue

            x_noise = np.random.normal(0.01, 0.01)
            y_noise = 0.2
            p3 = [p1[0] + x_noise + 0.01, (p1[1] + p2[1])/2.0]
            # arrow
            ax.annotate(s='', xy=(p1[0] + x_noise, p1[1]), xytext=(p2[0] + x_noise, p2[1]), arrowprops=dict(arrowstyle='<->', linestyle="-", linewidth=1, zorder=1000, mutation_scale=20))
            text=f"{res1[0]:0.2f} $\pm$ {res1[1]: 0.2f}\n{1-res1[2]:0.1e}"
            ax.annotate(s=text, xy=p3, xytext=(p3[0] + x_offset, p3[1]-y_noise), fontsize=ANNOTATION_FONTSIZE, \
                        fontweight='black', bbox=dict(facecolor='none', edgecolor='black'), zorder=1000, verticalalignment="center", \
                        arrowprops=dict(arrowstyle="->"))
            x_offset += 1.0

        # save the table
        table_to_save = pd.DataFrame(table_to_save, columns=['method1', 'method2', 'label1', 'label2', 'advantage', 'stddev', 'P(advantage > 0)', 'rnd_advantage', 'rnd_stderr', 'P(rnd_advantage > 0)'])
        table_to_save.to_csv(str(Path(path).resolve() / f"normalized_mobility/{USE_GP_STR}R_all_advantages_{xmetric}{USE_GP_STR}.csv"))

        # make a heatmap
        if plot_heatmap:
            heatmap = plot_heatmap_of_advantages(table_to_save, labelmap, USE_MATH_NOTATION)
            filepath = save_figure(heatmap, basedir=path, folder="normalized_mobility", filename=f'{USE_GP_STR}Heatmap_{xmetric}_advantages_AR_{adoption_rate}')
            print(f"Heatmap of advantages @ {adoption_rate}% Adoption saved at {filepath}")

        # reference lines
        ax.plot(ax.get_xlim(), [1.0, 1.0], '-.', c="gray", alpha=0.5)
        ax.set_ylim(0, 2)
        ax.set_xlim(left=results[xmetric].min(), right=results[xmetric].max())

        # add legend for the text box
        if USE_MATH_NOTATION:
            text = "$\Delta \hat{R} \pm \sigma$\np-value"
        else:
            text = "advantage $\pm$ stderr\np-value"
        ax.annotate(s=text, xy=(ax.get_xlim()[1]-2, 0.5), fontsize=ANNOTATION_FONTSIZE, fontweight='normal', bbox=dict(facecolor='none', edgecolor='black'), zorder=10)

    xlabel = get_metric_label(xmetric)
    ylabel = get_metric_label(ymetric)
    ax = add_bells_and_whistles(ax, y_title=ylabel, x_title=None if plot_residuals else xlabel, XY_TITLEPAD=LABELPAD, \
                    XY_TITLESIZE=LABELSIZE, TICKSIZE=TICKSIZE, legend_loc='upper left', \
                    LEGENDSIZE=LEGENDSIZE, x_tick_gap=TICKGAP, x_lower_lim=2, x_upper_lim=10.5, y_lower_lim=0.25)

    if (
        plot_residuals
        and plot_scatter
    ):
        res_ax.plot(res_ax.get_xlim(), [0.0, 0.0], '-.', c="gray", alpha=0.5)
        res_ax = add_bells_and_whistles(res_ax, y_title="Residuals", x_title=xlabel, XY_TITLEPAD=LABELPAD, \
                        XY_TITLESIZE=LABELSIZE, TICKSIZE=TICKSIZE, x_tick_gap=TICKGAP)

    # figure title
    fig.suptitle(f"Tracing Operating Characteristics @ {adoption_rate}% Adoption Rate", fontsize=TITLESIZE, y=1.05)

    # save
    fig.tight_layout()
    filename = f"{USE_GP_STR}{ymetric}_{xmetric}_mobility_scatter"
    filename += "_w_r_squared" if display_r_squared else ""
    filename += "_w_residuals" if plot_residuals else ""
    filename += "_w_annotations" if annotate_advantages else ""
    filename += "_w_scatter" if plot_scatter else ""
    filepath = save_figure(fig, basedir=path, folder="normalized_mobility", filename=f'{filename}_AR_{adoption_rate}')
    print(f"Scatter plot of mobility and R @ {adoption_rate}% Adoption saved at {filepath}")

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

def save_relevant_csv_files(results, adoption_rate, extract_path, good_factors_path):
    """
    Saves csv files for the entire result to be viewed later.

    Args:
        results (pd.DataFrame): Dataframe with rows as methods and corresponding simulation metrics.
        adoption_rate (str): Adoption rate. Assumed to be same for all app-based methods.
        extract_path (pathlib.Path): path of the file where extracted data will be saved
        good_factors_path (pathlib.Path): path of the file where good mobility factors (as per R) will be saved
    """
    R_LOWER = 0.5
    R_UPPER = 1.5

    # full data
    filename = str(extract_path)
    results.to_csv(filename )
    print(f"All extracted metrics for adoption rate: {adoption_rate}% saved at {filename}")

    # relevant mobility factors
    methods = results['method'].unique()
    filename = str(good_factors_path)
    with open(filename, "w") as f:
        for method in methods:
            f.write(method)
            x = _get_mobility_factor_counts_for_reasonable_r(results, method, lower_R=R_LOWER, upper_R=R_UPPER)
            f.write(str(x))
    print(f"All relevant mobility factors for {R_LOWER} <= R <= {R_UPPER} at adoption rate: {adoption_rate}% saved at {filename}")

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

    folder_name = Path(plot_path).resolve() / "normalized_mobility"
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
        good_mobility_factor_filepath = folder_name / f"mobility_factor_vs_R_AR_{adoption_rate}.txt"
        if not use_extracted_data:
            no_app_df = pd.DataFrame([])
            for method in other_methods:
                key = list(data[method].keys())[0]
                no_app_df = pd.concat([no_app_df, _extract_data(data[method][key], method)], axis='index', ignore_index=True)

            all_data = deepcopy(no_app_df)
            for method in app_based_methods:
                all_data = pd.concat([all_data, _extract_data(data[method][uptake], method)], axis='index', ignore_index=True)
            save_relevant_csv_files(all_data, adoption_rate, extract_path=extracted_data_filepath, good_factors_path=good_mobility_factor_filepath)
        else:
            assert extracted_data_filepath.exists(), f"{extracted_data_filepath} do not exist"
            all_data = pd.read_csv(str(extracted_data_filepath))

        for USE_GP in [True]:
            for ymetric in ['r']:
                for xmetric in ['effective_contacts', 'healthy_contacts']:
                    plot_heatmap = True
                    for annotate_advantages in [True]:
                        for plot_scatter in [False, True]:
                            plot_and_save_mobility_scatter(all_data, uptake, xmetric=xmetric, path=plot_path, \
                                ymetric=ymetric, plot_residuals=False, display_r_squared=False, \
                                annotate_advantages=annotate_advantages, plot_scatter=plot_scatter, USE_GP=USE_GP, plot_heatmap=plot_heatmap)
                            plot_heatmap = False # dont' plotheatmap again
