import argparse
import datetime
import pickle
import os
from typing import Dict, List, Tuple
import zipfile

from matplotlib import pyplot as plt

from src.covid19sim.simulator import Human


def plot_locations(axe,
                   locations: List[int],
                   locations_names: List[str],
                   timestamps: List[datetime.datetime]):
    for location, timestamp in zip(locations, timestamps):
        axe.broken_barh([(timestamp, 1)], (location, 1))
    axe.set_title("Location")
    axe.set_ylim((0, len(locations_names)))
    axe.set_yticklabels(locations_names)


def plot_recommendation_levels(axe,
                               recommendation_levels: List[int],
                               timestamps: List[datetime.datetime]):
    axe.step(timestamps, recommendation_levels)
    axe.set_title("Recommendation Level")
    axe.set_ylim((0, 3))
    axe.set_yticks([tick for tick in range(4)])


def plot_risks(axe,
               risks: List[int],
               timestamps: List[datetime.datetime]):
    axe.step(timestamps, risks)
    axe.set_title("Risk")
    axe.set_ylim((0, 15))
    axe.set_yticks([tick * 4 for tick in range(4+1)])


def plot_viral_loads(axe,
                     viral_loads: List[int],
                     timestamps: List[datetime.datetime]):
    axe.plot(timestamps, viral_loads)
    axe.set_title("Viral Load Curve")
    axe.set_ylim((0, 1))
    axe.set_yticks([tick / 4 for tick in range(4+1)])


def get_location_history(human_snapshots: List[Human],
                         timestamps: List[datetime.datetime],
                         all_locations: Dict[Tuple[int, int], Tuple[str, int]],
                         time_begin: datetime.datetime,
                         time_end: datetime.datetime) -> \
        Tuple[List[int], List[datetime.datetime]]:
    locations = []
    l_timestamps = []

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        locations.append(all_locations[(human.location.lon, human.location.lat)][1])
        l_timestamps.append(timestamp)

    return locations, l_timestamps


def get_recommendation_level_history(human_snapshots: List[Human],
                                     timestamps: List[datetime.datetime],
                                     time_begin: datetime.datetime,
                                     time_end: datetime.datetime) -> \
        Tuple[List[int], List[datetime.datetime]]:
    recommendation_levels = []
    rl_timestamps = []

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        recommendation_levels.append(human.rec_level)
        rl_timestamps.append(timestamp)

    return recommendation_levels, rl_timestamps


def get_risk_history(human_snapshots: List[Human],
                     timestamps: List[datetime.datetime],
                     time_begin: datetime.datetime,
                     time_end: datetime.datetime) -> \
        Tuple[List[int], List[datetime.datetime]]:
    risks = []
    r_timestamps = []

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        risks.append(human.risk)
        r_timestamps.append(timestamp)

    return risks, r_timestamps


def get_viral_load_history(human_snapshots: List[Human],
                           timestamps: List[datetime.datetime],
                           time_begin: datetime.datetime,
                           time_end: datetime.datetime) -> \
        Tuple[List[int], List[datetime.datetime]]:
    viral_loads = []
    vl_timestamps = []

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        viral_loads.append(human.viral_load_for_day(timestamp))
        vl_timestamps.append(timestamp)

    return viral_loads, vl_timestamps


def generate_human_centric_plots(debug_data, output_folder):
    human_backups = debug_data['human_backups']
    timestamps = list(human_backups.keys())
    nb_humans = len(human_backups[timestamps[0]].keys())

    begin = timestamps[0]
    end = timestamps[-1]

    # Treat each human individually
    for idx_human in range(1, nb_humans + 1):

        # Get all the backups of this human for all the timestamps
        h_key = "human:%i" % idx_human
        h_backup = [human_backups[t][h_key] for t in timestamps]

        fig, axes = plt.subplots(3, 1, sharex="col")
        plt.xlabel("Time")
        plt.gcf().autofmt_xdate()

        risks, r_timestamps = get_risk_history(h_backup, timestamps, begin, end)
        viral_loads, vl_timestamps = get_viral_load_history(h_backup, timestamps, begin, end)
        recommendation_levels, rl_timestamps = get_viral_load_history(h_backup, timestamps, begin, end)

        plot_risks(axes[0], risks, timestamps)
        plot_viral_loads(axes[1], viral_loads, timestamps)
        plot_recommendation_levels(axes[2], recommendation_levels, timestamps)

        plt.savefig(os.path.join(output_folder, f"{str(begin)}-{str(end)}_{h_key}.png"))


def generate_location_centric_plots(debug_data, output_folder):
    pass


def generate_debug_plots(debug_data, output_folder):
    generate_human_centric_plots(debug_data, output_folder)
    generate_location_centric_plots(debug_data, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_data")
    parser.add_argument("--output_folder")
    args = parser.parse_args()

    # Load the debug data
    debug_data = None
    with zipfile.ZipFile(args.debug_data) as zf:
        for filename in zf.namelist():
            day_debug_data = pickle.load(zf.open(filename))
            if debug_data is None:
                debug_data = day_debug_data
            else:
                for k in debug_data:
                    debug_data[k].extend(day_debug_data[k])

    # Ensure that the output folder does exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    generate_debug_plots(debug_data, args.output_folder)
