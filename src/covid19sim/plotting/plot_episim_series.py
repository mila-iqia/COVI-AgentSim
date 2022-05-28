"""
Plots various time series obtained from simulations for a fixed app adoption (if app is required)
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from covid19sim.plotting.utils import split_methods_and_check_validity, load_plot_these_methods_config
from covid19sim.plotting.matplotlib_utils import _plot_mean_with_stderr_bands_of_series, add_bells_and_whistles, save_figure, \
            get_color, get_adoption_rate_label_from_app_uptake, get_intervention_label, get_base_intervention, get_labelmap, get_colormap
from covid19sim.plotting.extract_tracker_metrics import _daily_fraction_cumulative_cases, _daily_incidence, _daily_prevalence, \
                            _daily_false_quarantine, _daily_fraction_of_population_infected, _daily_fraction_quarantine
from covid19sim.plotting.extract_tracker_metrics import _cumulative_infected_by_recovered_people, _proxy_R_estimated_by_recovered_people

DPI = 300
TICKGAP = 5
TITLESIZE = 25

def plot_all(ax, list_of_all_data, list_of_all_methods, key, labelmap, colormap, plot_percentages=False):
    """
    Plots various series (mean and stderr) corresponding to different methods in `list_of_all_methods`

    Args:
        ax (matplotlib.axes.Axes):Axes on which to plot the series
        list_of_all_data (list): a list of dicts where each dict contains a series specified by `key`
        list_of_all_methods (list): a list of tuples of strings where each tuple is a raw method name (folder name) and intervention config name
        key (str): name of attribute extracted in `_extract_data`
        labelmap (dict): method --> label
        colormap (dict): method --> color
        plot_percentages (bool): plots percentages if True. else keeps it as is.

    Returns:
        ax (matplotlib.axes.Axes): Axes with the series plotted on it
    """
    assert len(list_of_all_data) == len(list_of_all_methods), f"Number of series {len(list_of_all_data)} is not equal to number of methods {len(list_of_all_methods)}"
    percent = 100 if plot_percentages else 1

    list_of_all_series = [[y * percent for y in x[key]] for x in list_of_all_data]
    for idx, all_series in enumerate(list_of_all_series):
        method = list_of_all_methods[idx][0]
        label = labelmap[method]
        color = colormap[method]
        ax = _plot_mean_with_stderr_bands_of_series(ax, all_series, label, color, plot_quantiles=False, bootstrap=True, window=4, confidence_level=1)
    return ax

def plot_and_save_single_metric(ax, list_of_all_data, list_of_all_methods, key, labelmap, colormap, y_title, path, adoption_rate, plot_percentages=False, save_tiff=False):
    """
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10), dpi=DPI)

    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, plot_percentages=plot_percentages)
    ax = add_bells_and_whistles(ax, y_title=y_title, x_tick_gap=TICKGAP, x_lower_lim=0, y_lower_lim=0, legend_loc="upper left", x_title="Days since outbreak")
    filepath = save_figure(fig, basedir=path, folder='episim_series', filename=f'{key}_AR_{adoption_rate}', save_tiff=save_tiff)


def plot_and_save_r_characteristics(list_of_all_data, list_of_all_methods, uptake_rate, path):
    """
    Plot and save the figure of variation in R and infected numbers as measured by the number of people recovered.

    Args:
        list_of_all_data (list): a list of dicts where each dict contains a series specified by `key`
        list_of_all_methods (list): a list of tuples of strings where each tuple is a raw method name (folder name) and intervention config name
        uptake_rate (str): APP_UPTAKE value
        path (str): path where to save the figure
    """
    TICKGAP = 25
    TITLESIZE = 25
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15,15), sharex=True, dpi=100)

    labelmap = get_labelmap(list_of_all_methods, path)
    colormap = get_colormap(list_of_all_methods, path)

    # recovered cumulative
    ax = axs[0]
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key='cumulative_infected_by_recovered_people', labelmap=labelmap, colormap=colormap)
    ax = add_bells_and_whistles(ax, y_title="% Cumulative infected \nby recovered", x_tick_gap=TICKGAP, legend_loc="upper left")

    for R in [1.0, 2.5, 3.5]:
        benchmark_x = [x*R  for x in np.arange(0, ax.get_xlim()[1], 1)]
        ax.plot(benchmark_x, alpha=0.3, linestyle="--", color="grey")

    # running R
    ax = axs[1]
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key='proxy_R_estimated_by_recovered_people', labelmap=labelmap, colormap=colormap)
    ax = add_bells_and_whistles(ax, y_title="Running 'mean infected'\n by recovered", x_title="Number Recovered", x_tick_gap=TICKGAP)

    for R in [1.0, 2.5, 3.5]:
        benchmark_x = [R for x in np.arange(0, ax.get_xlim()[1], 1)]
        ax.plot(benchmark_x, alpha=0.3, linestyle="--", color="grey")

    # add title to the figure
    adoption_rate = get_adoption_rate_label_from_app_uptake(uptake_rate)
    fig.suptitle(f"Infection statistics as per recovered infectors @ {adoption_rate}% Adoption", fontsize=TITLESIZE, y=1.05)

    # save
    fig.tight_layout()
    filepath = save_figure(fig, basedir=path, folder='episim_series', filename=f'estimated_R_by_recovered_population_AR_{adoption_rate}')
    print(f"Plots on R and Recovered stats @ {adoption_rate}% Adoption saved at {filepath}")

def plot_and_save_epi_characteristics(list_of_all_data, list_of_all_methods, uptake_rate, path, save_each_axes_as_figure=True):
    """
    Plot and save various time series characteristics of the simulation runs across different methods.

    Args:
        list_of_all_data (list): a list of dicts where each dict contains a series specified by `key`
        list_of_all_methods (list): a list of tuples of strings where each tuple is a raw method name (folder name) and intervention config name
        uptake_rate (str): APP_UPTAKE value
        path (str): path where to save the figure
    """

    adoption_rate = get_adoption_rate_label_from_app_uptake(uptake_rate)
    save_axes = lambda ax: ax
    if save_each_axes_as_figure:
        def save_axes(ax, fig, key):
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            _ = save_figure(fig, basedir=path, folder='episim_series', filename=f'{key}_AR_{adoption_rate}', bbox_inches=extent.expanded(1.1, 1.2))

    labelmap = get_labelmap(list_of_all_methods, path)
    colormap = get_colormap(list_of_all_methods, path)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,15), sharex=True, dpi=DPI)

    ax, key = axs[0, 0], 'daily_fraction_of_population_infected'
    y_title = "Daily cases (%)"
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, plot_percentages=True)
    ax = add_bells_and_whistles(ax, y_title=y_title, x_tick_gap=TICKGAP, x_lower_lim=0, y_lower_lim=0, legend_loc="upper left")
    plot_and_save_single_metric(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, y_title=y_title, path=path, adoption_rate=adoption_rate, plot_percentages=True)

    ax, key = axs[0, 1], 'daily_fraction_cumulative_cases'
    y_title = "Cumulative cases (%)"
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, plot_percentages=True)
    ax = add_bells_and_whistles(ax, y_title=y_title, x_tick_gap=TICKGAP, x_lower_lim=0, y_lower_lim=0)
    plot_and_save_single_metric(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, y_title=y_title, path=path, adoption_rate=adoption_rate, plot_percentages=True, save_tiff=True)

    ax, key = axs[1, 0], 'daily_prevalence'
    y_title = "Prevalence (%)"
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, plot_percentages=True)
    ax = add_bells_and_whistles(ax, y_title=y_title, x_tick_gap=TICKGAP, x_title="Days since outbreak", x_lower_lim=0, y_lower_lim=0)
    ax.set_ylim(0, None)
    plot_and_save_single_metric(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, y_title=y_title, path=path, adoption_rate=adoption_rate, plot_percentages=True)

    ax, key = axs[1, 1], 'daily_incidence'
    y_title = "Incidence (per 1000 people)"
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap)
    ax = add_bells_and_whistles(ax, y_title=y_title, x_tick_gap=TICKGAP,  x_title="Days since outbreak", x_lower_lim=0, y_lower_lim=0)
    plot_and_save_single_metric(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, y_title=y_title, path=path, adoption_rate=adoption_rate, plot_percentages=False)

    # add title to the figure
    fig.suptitle(f"Simulated dynamics of DCT methods @ {adoption_rate}% adoption rate", fontsize=TITLESIZE, y=1.05)

    # save
    fig.tight_layout(pad=3.0)
    filepath = save_figure(fig, basedir=path, folder='episim_series', filename=f'case_curves_AR_{adoption_rate}')
    print(f"Case Curves @ {adoption_rate}% Adoption saved at {filepath}")

    # other plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10), dpi=DPI)
    key = 'daily_fraction_quarantine'
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key=key, labelmap=labelmap, colormap=colormap, plot_percentages=True)
    ax = add_bells_and_whistles(ax, y_title="Quarantined agents (%)", x_tick_gap=TICKGAP, x_title="Days since outbreak",  legend_loc="upper left", y_lower_lim=0, x_lower_lim=0)
    # fig.suptitle(f"Quarantined agents in simulations of DCT methods @ {adoption_rate}% adoption rate", fontsize=TITLESIZE, y=1.05)
    filepath = save_figure(fig, basedir=path, folder='episim_series', filename=f'quarantine_AR_{adoption_rate}', save_tiff=True)

def _extract_data(simulation_runs):
    """
    Extracts relevant data from a list of trackers to be used for plotting.

    Args:
        trackers (dict): keys are `str` representing the path of the tracker files, values are `dict` holding raw tracker attributes.
                        (all these trackers are assumed to belong to the same category to be plotted together i.e. mean and stderr).

    Returns:
        (dict): keys are name of an attribute, values are a list of extracted values for each tracker.
    """
    series = {
        "daily_fraction_cumulative_cases": [],
        "daily_incidence": [],
        "daily_prevalence": [],
        "daily_false_quarantine": [],
        "cumulative_infected_by_recovered_people": [],
        "proxy_R_estimated_by_recovered_people": [],
        "APP_UPTAKE": [],
        "daily_fraction_of_population_infected": [],
        "daily_fraction_quarantine": []
    }
    #
    for data in simulation_runs.values():
        tracker = data['pkl']
        series['daily_fraction_of_population_infected'].append(_daily_fraction_of_population_infected(tracker))
        series["daily_fraction_cumulative_cases"].append(_daily_fraction_cumulative_cases(tracker))
        series["daily_incidence"].append(_daily_incidence(tracker))
        series['daily_prevalence'].append(_daily_prevalence(tracker))
        series['daily_false_quarantine'].append(_daily_false_quarantine(tracker))
        series['daily_fraction_quarantine'].append(_daily_fraction_quarantine(tracker))
        series['cumulative_infected_by_recovered_people'].append(_cumulative_infected_by_recovered_people(tracker))
        series['proxy_R_estimated_by_recovered_people'].append(_proxy_R_estimated_by_recovered_people(tracker))
        series['APP_UPTAKE'].append(data['conf']['APP_UPTAKE'])
    return series

def run(data, plot_path, compare=None, **kwargs):
    """
    Plots and saves comparison of various simulation characteristics across same adoption rate (if applicable) and no-tracing.
    Following are included -
        1. (time series) Cumulative Cases
        2. (time series) True Prevalence
        3. (time series) True Incidence
        4. (time series) False Quarantine
        5. (series) Compuation of R vs number of recovered agents
        6. (series) Cumulative infected vs number of recovered agents

    Args:
        data (dict): intervention_name --> APP_UPTAKE --> folder_name --> {'conf': yaml file, 'pkl': tracker file}
        plot_path (str): path where to save plots
    """
    app_based_methods, other_methods, uptake_keys = split_methods_and_check_validity(data)
    plot_these_methods = load_plot_these_methods_config(plot_path)

    ## data preparation
    list_of_no_app_data = []
    other_base_intervention_conf = []
    for method in other_methods:
        if (
            len(plot_these_methods) > 0
            and method not in plot_these_methods
        ):
            continue

        key = list(data[method].keys())[0]
        list_of_no_app_data.append(_extract_data(data[method][key]))
        intervention_conf = next(iter(data[method][key].values()))['conf']
        other_base_intervention_conf.append(get_base_intervention(intervention_conf))

    for uptake_rate in uptake_keys:
        extracted_data = {}
        list_of_all_data = deepcopy(list_of_no_app_data)
        list_of_all_methods = deepcopy(other_methods)
        list_of_all_base_confs = deepcopy(other_base_intervention_conf)
        for method in app_based_methods:
            if (
                len(plot_these_methods) > 0
                and method not in plot_these_methods
            ):
                continue

            list_of_all_data.append(_extract_data(data[method][uptake_rate]))
            list_of_all_methods.append(method)
            intervention_conf = next(iter(data[method][uptake_rate].values()))['conf']
            list_of_all_base_confs.append(get_base_intervention(intervention_conf))

        list_of_all_methods = list(zip(list_of_all_methods, list_of_all_base_confs))
        plot_and_save_r_characteristics(list_of_all_data, list_of_all_methods, uptake_rate=uptake_rate, path=plot_path)
        plot_and_save_epi_characteristics(list_of_all_data, list_of_all_methods, uptake_rate=uptake_rate, path=plot_path)
