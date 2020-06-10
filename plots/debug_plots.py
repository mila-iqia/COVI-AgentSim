import argparse
import datetime
import pickle
import os
from typing import Any, Dict, List, Tuple

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
        Tuple[List[int], List[datetime.datetime], Any]:
    locations = []
    l_timestamps = []
    min_max = None

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        locations.append(all_locations[(human.location.lon, human.location.lat)][1])
        l_timestamps.append(timestamp)

    return locations, l_timestamps, min_max


def get_recommendation_level_history(human_snapshots: List[Human],
                                     timestamps: List[datetime.datetime],
                                     time_begin: datetime.datetime,
                                     time_end: datetime.datetime) -> \
        Tuple[List[int], List[datetime.datetime], Any]:
    recommendation_levels = []
    r_timestamps = []
    min_max = (0, 3)

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        recommendation_levels.append(human.rec_level)
        r_timestamps.append(timestamp)

    return recommendation_levels, r_timestamps, min_max


def get_risk_history(human_snapshots: List[Human],
                     timestamps: List[datetime.datetime],
                     time_begin: datetime.datetime,
                     time_end: datetime.datetime) -> \
        Tuple[List[int], List[datetime.datetime], Any]:
    risks = []
    r_timestamps = []
    min_max = (0.0, 1.0)

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        risks.append(human.risk)
        r_timestamps.append(timestamp)

    return risks, r_timestamps, min_max


def get_viral_load_history(human_snapshots: List[Human],
                           timestamps: List[datetime.datetime],
                           time_begin: datetime.datetime,
                           time_end: datetime.datetime) -> \
        Tuple[List[int], List[datetime.datetime], Any]:
    viral_loads = []
    vl_timestamps = []
    min_max = (0.0, 1.0)

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        viral_loads.append(human.viral_load_for_day(timestamp))
        vl_timestamps.append(timestamp)

    return viral_loads, vl_timestamps, min_max


def generate_human_centric_plots(debug_data, output_folder):
    pass


def generate_location_centric_plots(debug_data, output_folder):
    import pdb; pdb.set_trace()


def generate_debug_plots(debug_data, output_folder):
    generate_human_centric_plots(debug_data, output_folder)
    generate_location_centric_plots(debug_data, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_data")
    parser.add_argument("--output_folder")
    args = parser.parse_args()

    # Load the debug data
    with open(args.debug_data, "rb") as f:
        debug_data = pickle.load(f)

    # Ensure that the output folder does exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    generate_debug_plots(debug_data, args.output_folder)
