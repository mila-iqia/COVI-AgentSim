"""
Plots various time series obtained from simulations for a fixed app adoption (if app is required)
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from covid19sim.utils.utils import is_app_based_tracing_intervention
from covid19sim.plotting.matplotlib_utils import _plot_mean_with_stderr_bands_of_series, add_bells_and_whistles, save_figure, get_color, get_adoption_rate_label_from_app_uptake, get_intervention_label
from covid19sim.plotting.extract_tracker_metrics import _daily_fraction_cumulative_cases, _daily_incidence, _daily_prevalence, _daily_false_quarantine
from covid19sim.plotting.extract_tracker_metrics import _cumulative_infected_by_recovered_people, _proxy_R_estimated_by_recovered_people

def plot_all(ax, list_of_all_data, list_of_all_methods, key):
    """
    Plots various series (mean and stderr) corresponding to different methods in `list_of_all_methods`

    Args:
        ax (matplotlib.axes.Axes):Axes on which to plot the series
        list_of_all_data (list): a list of dicts where each dict contains a series specified by `key`
        list_of_all_methods (list): a list of strings where each element is a raw method name
        key (str): name of attribute extracted in `_extract_data`

    Returns:
        ax (matplotlib.axes.Axes): Axes with the series plotted on it
    """
    assert len(list_of_all_data) == len(list_of_all_methods), f"Number of series {len(list_of_all_data)} is not equal to number of methods {len(list_of_all_methods)}"

    list_of_all_series = [x[key] for x in list_of_all_data]
    for idx, all_series in enumerate(list_of_all_series):
        label = get_intervention_label(list_of_all_methods[idx])
        color = get_color(idx)
        ax = _plot_mean_with_stderr_bands_of_series(ax, all_series, label, color)
    return ax

def plot_and_save_r_characteristics(list_of_all_data, list_of_all_methods, uptake_rate, path):
    """
    Plot and save the figure of variation in R and infected numbers as measured by the number of people recovered.

    Args:
        list_of_all_data (list): a list of dicts where each dict contains a series specified by `key`
        list_of_all_methods (list): a list of strings where each element is a raw method name
        uptake_rate (str): APP_UPTAKE value
        path (str): path where to save the figure
    """
    TICKGAP = 25
    TITLESIZE = 25
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15,15), sharex=True, dpi=100)

    # recovered cumulative
    ax = axs[0]
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key='cumulative_infected_by_recovered_people')
    ax = add_bells_and_whistles(ax, y_title="% Cumulative infected \nby recovered", x_tick_gap=TICKGAP, legend_loc="upper left")

    for R in [1.0, 2.5, 3.5]:
        benchmark_x = [x*R  for x in np.arange(0, ax.get_xlim()[1], 1)]
        ax.plot(benchmark_x, alpha=0.3, linestyle="--", color="grey")

    # running R
    ax = axs[1]
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key='proxy_R_estimated_by_recovered_people')
    ax = add_bells_and_whistles(ax, y_title="Running 'mean infected'\n by recovered", x_title="Number Recovered", x_tick_gap=TICKGAP)

    for R in [1.0, 2.5, 3.5]:
        benchmark_x = [R for x in np.arange(0, ax.get_xlim()[1], 1)]
        ax.plot(benchmark_x, alpha=0.3, linestyle="--", color="grey")

    # add title to the figure
    adoption_rate = get_adoption_rate_label_from_app_uptake(uptake_rate)
    fig.suptitle(f"Infection statistics as per recovered infectors @ {adoption_rate}% Adoption", fontsize=TITLESIZE, y=1.0)

    # save
    fig.tight_layout()
    filepath = save_figure(fig, basedir=path, folder='episim_series', filename=f'estimated_R_by_recovered_population_AR_{adoption_rate}')
    print(f"Plots on R and Recovered stats @ {adoption_rate}% Adoption saved at {filepath}")

def plot_and_save_epi_characteristics(list_of_all_data, list_of_all_methods, uptake_rate, path):
    """
    Plot and save various time series characteristics of the simulation runs across different methods.

    Args:
        list_of_all_data (list): a list of dicts where each dict contains a series specified by `key`
        list_of_all_methods (list): a list of strings where each element is a raw method name
        uptake_rate (str): APP_UPTAKE value
        path (str): path where to save the figure
    """
    TICKGAP = 5
    TITLESIZE = 25

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,15), sharex=True, dpi=100)

    ax = axs[0, 0]
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key='daily_fraction_cumulative_cases')
    ax = add_bells_and_whistles(ax, y_title="% Cumulative Cases", x_tick_gap=TICKGAP)

    ax = axs[0, 1]
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key='daily_incidence')
    ax = add_bells_and_whistles(ax, y_title="incidence", x_tick_gap=TICKGAP, legend_loc="upper right")

    ax = axs[1, 0]
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key='daily_prevalence')
    ax = add_bells_and_whistles(ax, y_title="prevalence", x_tick_gap=TICKGAP, x_title="Days since outbreak")

    ax = axs[1, 1]
    ax = plot_all(ax, list_of_all_data, list_of_all_methods, key='daily_false_quarantine')
    ax = add_bells_and_whistles(ax, y_title="False Quarantine", x_tick_gap=TICKGAP, x_title="Days since outbreak")

    # add title to the figure
    adoption_rate = get_adoption_rate_label_from_app_uptake(uptake_rate)
    fig.suptitle(f"Case curves @ {adoption_rate}% Adoption", fontsize=TITLESIZE, y=1.0)

    # save
    fig.tight_layout()
    filepath = save_figure(fig, basedir=path, folder='episim_series', filename=f'case_curves_AR_{adoption_rate}')
    print(f"Case Curves @ {adoption_rate}% Adoption saved at {filepath}")

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
        "APP_UPTAKE": []
    }
    #
    for data in simulation_runs.values():
        tracker = data['pkl']
        series["daily_fraction_cumulative_cases"].append(_daily_fraction_cumulative_cases(tracker))
        series["daily_incidence"].append(_daily_incidence(tracker))
        series['daily_prevalence'].append(_daily_prevalence(tracker))
        series['daily_false_quarantine'].append(_daily_false_quarantine(tracker))
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
        data (dict):
        plot_path (str):
        compare (str):
    """
    # prepare series broken down by adoption rates
    methods = list(data.keys())
    app_based_methods = [x for x in methods if is_app_based_tracing_intervention(x)]
    other_methods = list(set(methods) - set(app_based_methods))

    uptake_keys = [list(data[x].keys()) for x in app_based_methods]

    ## experiment correctness checks
    assert len(set(frozenset(x) for x in uptake_keys)) == 1, "found different adoption rates across tracing based methods"
    uptake_keys = list(list(set([frozenset(x) for x in uptake_keys]))[0])
    for uptake_rate in uptake_keys:
        assert len(set([len(data[method][uptake_rate]) for method in app_based_methods])) == 1, f"Found different number of seeds across {adoption_rate}. Methods: {methods}"

    ## data preparation
    list_of_no_app_data = []
    for method in other_methods:
        key = list(data[method].keys())[0]
        list_of_no_app_data.append(_extract_data(data[method][key]))

    for uptake_rate in uptake_keys:
        extracted_data = {}
        list_of_all_data = deepcopy(list_of_no_app_data)
        list_of_all_methods = deepcopy(other_methods)
        for method in app_based_methods:
            list_of_all_data.append(_extract_data(data[method][uptake_rate]))
            list_of_all_methods.append(method)

        plot_and_save_r_characteristics(list_of_all_data, list_of_all_methods, uptake_rate=uptake_rate, path=plot_path)
        plot_and_save_epi_characteristics(list_of_all_data, list_of_all_methods, uptake_rate=uptake_rate, path=plot_path)
