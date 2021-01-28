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
from covid19sim.plotting.utils import get_proxy_r, split_methods_and_check_validity, load_plot_these_methods_config, get_simulation_parameter
from covid19sim.plotting.extract_tracker_metrics import _daily_false_quarantine, _daily_false_susceptible_recovered, _daily_fraction_risky_classified_as_non_risky, \
                                _daily_fraction_non_risky_classified_as_risky, _daily_fraction_quarantine
from covid19sim.plotting.extract_tracker_metrics import _mean_effective_contacts, _mean_healthy_effective_contacts, _percentage_total_infected, _positivity_rate
from covid19sim.plotting.matplotlib_utils import add_bells_and_whistles, save_figure, get_color, get_adoption_rate_label_from_app_uptake, get_intervention_label, \
                                plot_mean_and_stderr_bands, get_base_intervention, get_labelmap, get_colormap, plot_heatmap_of_advantages
from covid19sim.plotting.curve_fitting import LinearFit, GPRFit

DPI=300
TITLESIZE = 25
LABELPAD = 0.50
LABELSIZE = 25
TICKSIZE = 20
LEGENDSIZE = 25
ANNOTATION_FONTSIZE=15

METRICS = ['r', 'false_quarantine', 'false_sr', 'effective_contacts', 'healthy_contacts', 'percentage_infected', \
            'fraction_false_non_risky', 'fraction_false_risky', 'positivity_rate', 'fraction_quarantine']

SENSITIVITY_PARAMETERS = ['ASYMPTOMATIC_RATIO', 'ALL_LEVELS_DROPOUT', 'P_DROPOUT_SYMPTOM',  'PROPORTION_LAB_TEST_PER_DAY', 'BASELINE_P_ASYMPTOMATIC', 'GLOBAL_MOBILITY_SCALING_FACTOR']# used for sensitivity plots

USE_MATH_NOTATION=False

# fix the seed
np.random.seed(123)


def get_metric_label(label):
    """
    Maps label to a readable title

    Args:
        label (str): label for which a readable title is to be returned

    Returns:
        (str): a readable title for label
    """
    assert label in METRICS, f"unknown label: {label}"

    if label == "r":
        if USE_MATH_NOTATION:
            return "$\hat{R}$"
        return "$R$"

    if label == "effective_contacts":
        return "# Contacts per day per human ($C$)"

    if label == "false_quarantine":
        return "False Quarantine"

    if label == "false_sr":
        return "False S/R"

    if label == "healthy_contacts":
        return "Healthy Contacts"

    if label == "fraction_false_risky":
        # return "Risk to Economy ($\frac{False Quarantine}{Total Non-Risky}$)"
        return "Risk to Economy \n (False Quarantine / Total Non-Risky)"

    if label == "fraction_false_non_risky":
        # return "Risk to Healthcare ($\frac{False Non-Risky}{Total Infected}$)"
        return "Risk to Healthcare \n (False Non-Risky / Total Infected)"

    if label == "positivity_rate":
        return "Positivity Rate"

    if label == "percentage_infected":
        return "Fraction Infected"

    if label == "fraction_quarantine":
        return "Fraction Quarantine"

    if label == "mobility_factor":
        return "Global Mobility Factor"

    raise ValueError(f"Unknown label:{label}")

def get_polyfit_str(trend_fit):
    """
    Returns a string to add to saved files to distnguish between kernels used for underlying models

    Args:
        trend_fit (str): Type of fit.
    """
    if trend_fit == "polynomial":
        return "_PolyFit_"
    elif trend_fit == "linear":
        return "_Linear_"
    else:
        return ""

def _get_mobility_factor_counts_for_reasonable_r(results, method, lower_R=0.5, upper_R=1.5):
    """
    Returns the count of mobility factors for which method resulted in an R value between `lower_R` and `upper_R`.

    Args:
        results (pd.DataFrame): Dataframe with rows as methods and corresponding simulation metrics.
        method (str): name of the intervention method for which range is to be found
        lower_R (float): a valid lower value of R
        upper_R (float): a valid upper value of R

    Returns:
        (pd.Series): A series where each row is a mobility factor and corresponding number of simulations that resulted in valid R.
    """
    selector = results['method'] == method
    correct_R = (lower_R <= results[selector]['r']) & (results[selector]['r'] <= upper_R)
    return results[selector][correct_R][['mobility_factor']].value_counts()

def _get_interpolation_kind(xmetric, ymetric, use_gp=False):
    """
    Returns a valid interpolation function between xmetric and ymetric.

    Args:
        xmetric (str): one of `METRICS`
        ymetric (str): one of `METRICS`
        use_gp (str): return GP regression fit if True.

    Returns:
        (function): a function that accepts an np.array. Examples - `_linear`
    """
    assert xmetric != ymetric, "x and y can't be same"
    assert xmetric in METRICS and ymetric in METRICS, f"unknown metrics - xmetric: {xmetric} or ymetric:{ymetric}. Expected one of {METRICS}"

    if use_gp:
        return GPRFit

    metrics = [xmetric, ymetric]
    if sum(metric in ["effective_contacts", "r"] for metric in metrics) == 2:
        return LinearFit
    return LinearFit

def _filter_out_irrelevant_method(xmetric, ymetric, method, results):
    """
    Checks whether `xmetric` and `ymetric` are suitable to be plotted for `method`.
    if any metric is specific to app-based methods, filter out non-app based method

    Args:
        xmetric (str): one of `METRICS`
        ymetric (str): one of `METRICS`
        method (str): method for which validity is to be checked
        results (pd.DataFrame): Dataframe with rows as methods and corresponding simulation metrics. Only `app_based` column is used.

    Returns:
        (bool): True if `method` is not suitable to be plotted for `xmetric` and `ymetric` comparison
    """
    assert xmetric != ymetric, "x and y can't be same"
    assert xmetric in METRICS and ymetric in METRICS, f"unknown metrics - xmetric: {xmetric} or ymetric:{ymetric}. Expected one of {METRICS}"

    is_app_based = results[results['method'] == method]['app_based'].unique()
    assert len(is_app_based) == 1, "Same method is expected to be app based and non app based. This can't happen!"
    is_app_based = is_app_based.item()

    metrics = [xmetric, ymetric]
    # if any metric is specific to app-based methods, filter out non-app based method
    if (
        sum(metric in ["false_quarantine", "false_sr", "fraction_false_non_risky","fraction_false_risky" ] for metric in metrics) > 0
        and not is_app_based
    ):
        return True

    return False

def find_all_pairs_offsets_and_stddev(fitted_fns):
    """
    Computes offset estimates and their stddev for all pairs of methods.
    NOTE: Only applicable when ymetric is "r".

    Args:
        fitted_fns (dict): method --> FittedFn

    Returns:
        (list): list of lists where each list corresponds to pairwise comparison of methods has following elements -
            1. (x1, y1): lower point to indicate the start of offset
            2. (x1, y2): upper point to indicate the end of offset. Note: it's a vertical line.
            3. (offset, stddev, cdf): Computed analytically. Offset value: y2 - y1, stderr of this offset, P(offset > 0)
            4. method1: name of the reference method
            5. method2: name of the method that reference method is being compared to.
            6. (offset, stddev, cdf): Computed from random draws.
    """
    def mss(res):
        return np.mean(res ** 2)

    # for R = 1, find the value on x-axis.
    method_x = []
    for method, fn in fitted_fns.items():
        x = fn.find_x_for_y(1.0)
        method_x.append((x, method))

    # for the x of reference method, how much is the offset from other methods.
    all_pairs = []
    method_x = sorted(method_x, key=lambda x:-x[0]) # larger x is the reference
    for idx, (x1, method1) in enumerate(method_x):
        plot = True
        reference_fn = fitted_fns[method1]
        y1 = reference_fn.evaluate_y_for_x(x1)
        assert abs(y1 - 1.0) < 1e-2, f"encountered incorrect y cordinate. Expected 1.0. Got {y1}"
        for _, method2 in method_x[idx+1:]:
            comparator_fn = fitted_fns[method2]
            y2 = comparator_fn.evaluate_y_for_x(x1)
            offset_rnd, stddev_rnd, cdf_rnd = reference_fn.find_offset_and_stderr_at_x(x1, other_fn=comparator_fn, analytical=False)
            offset, stddev, cdf = reference_fn.find_offset_and_stderr_at_x(x1, other_fn=comparator_fn, analytical=True)

            #
            all_pairs.append([(x1, y1), (x1, y2), (offset, stddev, cdf), method1, method2, (offset_rnd, stddev_rnd, cdf_rnd), plot])
            plot = False

    return sorted(all_pairs, key=lambda x: (x[0][0], x[2][0]))

def plot_and_save_mobility_scatter(results, uptake_rate, xmetric, ymetric, path, USE_GP=False, plot_residuals=False, display_r_squared=False, annotate_advantages=True, plot_scatter=True, plot_heatmap=True, trend_fit=""):
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
        trend_fit (str): Type of trend to fit.
    """
    assert xmetric in METRICS and ymetric in METRICS, f"Unknown metrics: {xmetric} or {ymetric}. Expected one of {METRICS}."
    TICKGAP=2
    ANNOTATION_FONTSIZE=15
    USE_GP_STR = "GP_" if USE_GP else ""
    POLYFIT_STR = get_polyfit_str(trend_fit)

    # save models
    model_dir = Path(path).resolve() / f"normalized_mobility/models_{ymetric}_vs_{xmetric}"
    os.makedirs(str(model_dir), exist_ok=True)

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
        fitted_fns[method] = INTERPOLATION_FN(fit=trend_fit).fit(x, y)

        # save for sensitivity analysis (if not already saved)
        model_path = model_dir / f"GP{POLYFIT_STR}_model_{method}"
        fitted_fns[method].save(path=str(model_path))

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
            table_to_save.append([m1, m2, p1[0], p1[1], p2[1], labelmap[m1], labelmap[m2], *res1, *res2])
            if (
                not annotate_advantages
                or not plot
                or len(methods) > 3 # if there are 3 methods, annotation will not be cluttered
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
        table_to_save = pd.DataFrame(table_to_save, columns=['method1', 'method2', 'contacts', 'm1_R', 'm2_R', 'label1', 'label2', 'advantage', 'stddev', 'P(advantage > 0)', 'rnd_advantage', 'rnd_stderr', 'P(rnd_advantage > 0)'])
        table_to_save.to_csv(str(Path(path).resolve() / f"normalized_mobility/{USE_GP_STR}{POLYFIT_STR}R_all_advantages_{xmetric}{USE_GP_STR}.csv"))

        # make a heatmap
        if plot_heatmap:
            heatmap = plot_heatmap_of_advantages(table_to_save, labelmap, USE_MATH_NOTATION)
            filepath = save_figure(heatmap, basedir=path, folder="normalized_mobility", filename=f'{USE_GP_STR}{POLYFIT_STR}Heatmap_{xmetric}_advantages_AR_{adoption_rate}')
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
    filename = f"{USE_GP_STR}{POLYFIT_STR}{ymetric}_{xmetric}_mobility_scatter"
    filename += "_w_r_squared" if display_r_squared else ""
    filename += "_w_residuals" if plot_residuals else ""
    filename += "_w_annotations" if annotate_advantages else ""
    filename += "_w_scatter" if plot_scatter else ""
    filepath = save_figure(fig, basedir=path, folder=f"normalized_mobility/{ymetric}_vs_{xmetric}{POLYFIT_STR}", filename=f'{filename}_AR_{adoption_rate}')
    print(f"Scatter plot of mobility and R @ {adoption_rate}% Adoption saved at {filepath}")

def _extract_metrics(data, conf):
    """
    Extracts `METRICS` and `SENSITIVITY_PARAMETERS` from data corresponding to a single simulation run.

    Args:
        data (dict): tracker files for the simulation
        conf (dict): an experimental configuration.

    Returns:
        (list): a list of scalars representing metrics in `METRICS` for the simulations
    """
    out = []
    out.append(get_proxy_r(data, verbose=False))
    out.append(_daily_false_quarantine(data).mean())
    out.append(_daily_false_susceptible_recovered(data).mean())
    out.append(_mean_effective_contacts(data))
    out.append(_mean_healthy_effective_contacts(data))
    out.append(_percentage_total_infected(data))
    out.append(_daily_fraction_risky_classified_as_non_risky(data).mean())
    out.append(_daily_fraction_non_risky_classified_as_risky(data).mean())
    out.append(_positivity_rate(data))
    out.append(_daily_fraction_quarantine(data).mean())

    for x in SENSITIVITY_PARAMETERS:
        out.append(get_simulation_parameter(x, data, conf))

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
        conf = sim['conf']
        intervention_name = get_base_intervention(sim['conf'])
        mobility_factor = sim['conf']['GLOBAL_MOBILITY_SCALING_FACTOR']
        row =  [method, simname, mobility_factor, intervention_name, is_app_based_tracing_intervention(intervention_conf=sim['conf'])] + _extract_metrics(data, conf)
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
        if len(uptake_keys) == 0:
            uptake_keys = [-1] # when only no-app interventions are present
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
            all_data['adoption_rate'] = adoption_rate
            save_relevant_csv_files(all_data, adoption_rate, extract_path=extracted_data_filepath, good_factors_path=good_mobility_factor_filepath)
        else:
            assert extracted_data_filepath.exists(), f"{extracted_data_filepath} do not exist"
            all_data = pd.read_csv(str(extracted_data_filepath))

        # filter those simulations that had full outbreak (>=50% infected)
        all_data_truncated = all_data[all_data['percentage_infected'] < 30]
        plot_path_truncated = str(Path(plot_path).resolve() / "normalized_mobility_truncated")

        for results, path in [(all_data, plot_path), (all_data_truncated, plot_path_truncated)]:
            for ymetric in ['r', 'percentage_infected']:
                for xmetric in ['effective_contacts', 'healthy_contacts', 'GLOBAL_MOBILITY_SCALING_FACTOR']:
                    plot_heatmap = True
                    for annotate_advantages in [True]:
                        for plot_scatter in [False, True]:
                            # with non linear fit
                            plot_and_save_mobility_scatter(all_data, uptake, xmetric=xmetric, path=path, \
                                ymetric=ymetric, plot_residuals=False, display_r_squared=False, \
                                annotate_advantages=annotate_advantages, plot_scatter=plot_scatter, USE_GP=USE_GP, plot_heatmap=plot_heatmap)

                            # with plynomial fit
                            plot_and_save_mobility_scatter(all_data, uptake, xmetric=xmetric, path=path, \
                                ymetric=ymetric, plot_residuals=False, display_r_squared=False, \
                                annotate_advantages=annotate_advantages, plot_scatter=plot_scatter, USE_GP=USE_GP, plot_heatmap=plot_heatmap, trend_fit="polynomial")

                            # with linear fit
                            plot_and_save_mobility_scatter(all_data, uptake, xmetric=xmetric, path=path, \
                                ymetric=ymetric, plot_residuals=False, display_r_squared=False, \
                                annotate_advantages=annotate_advantages, plot_scatter=plot_scatter, USE_GP=USE_GP, plot_heatmap=plot_heatmap, trend_fit="linear")

                            plot_heatmap = False # dont' plot heatmap again
