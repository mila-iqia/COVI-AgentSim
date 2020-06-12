import argparse
import datetime
import pickle
import os
from typing import Dict, List, Tuple
import zipfile

from matplotlib import pyplot as plt

from src.covid19sim.simulator import Human
from src.covid19sim.base import Event


PLOT_EVENTS_LABEL = ["Encounters", "Contaminations", "Tests"]


def plot_events(events: Dict[str, List[int]],
                timestamps: List[datetime.datetime]):
    for event_label in PLOT_EVENTS_LABEL:
        plt.bar(timestamps, events[event_label], label=event_label)
    plt.title("Events")
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_locations(locations: List[int],
                   locations_names: List[str],
                   timestamps: List[datetime.datetime]):

    location_to_color = {"household": "tab:red",
                         "park": "tab:green",
                         "hospital": "tab:blue",
                         "store": "tab:purple",
                         "school": "tab:olive",
                         "workplace": "tab:gray",
                         "senior_residency": "tab:brown",
                         "misc": "tab:pink"}

    """
    for l_idx, l_name in enumerate(locations_names):
        l_timestamps = [l for (l,t) in zip(locations, timestamps) if l == l_idx]
        l_plot_xs = [(t, 1./24) for t in l_timestamps]
        l_plot_ys = [(l_idx, 1)] * len(l_timestamps)
        plt.broken_barh(l_plot_xs, l_plot_ys)
    """

    for location, timestamp in zip(locations, timestamps):
        l_name = locations_names[location]
        color = location_to_color[l_name.split(":")[0]]
        plt.broken_barh([(timestamp, 1/24)], (location, 1), facecolor=color, label=l_name)
    plt.title("Location")
    plt.ylim((0, len(locations_names)))
    # plt.set_yticklabels(locations_names)
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_metadata(human):
    table_data = [
        ["name:", human.name],
        ["age:", human.age],
        ["carefulness:", human.carefulness],
        ["has_app:", human.has_app],
        ["has_allergies:", human.has_allergies],
        ["household:", human.household],
        ["workplace:", human.workplace],
        ["timeslots:", str(human.time_slots)],
    ]
    table = plt.table(cellText=table_data, loc='center')
    table.set_fontsize(14)
    table.scale(1, 3)
    plt.axis('off')


def plot_recommendation_levels(recommendation_levels: List[int],
                               timestamps: List[datetime.datetime]):
    plt.step(timestamps, recommendation_levels)
    plt.title("Recommendation Level")
    plt.ylim((0, 3))
    plt.yticks([tick for tick in range(4)])
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_risks(risks: List[int],
               timestamps: List[datetime.datetime]):
    plt.step(timestamps, risks)
    plt.title("Risk Level")
    plt.ylim((0, 15))
    plt.yticks([tick * 4 for tick in range(4+1)])
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_symptoms(true_symptoms: List[int],
                  obs_symptoms: List[int],
                  timestamps: List[datetime.datetime]):
    plt.step(timestamps, true_symptoms, color="tab:red")
    plt.step(timestamps, obs_symptoms, color="tab:blue")
    plt.title("Nb symptoms (red=true, blue=obs)")
    plt.ylim((0, 15))
    plt.yticks([tick * 4 for tick in range(4+1)])
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_viral_loads(viral_loads: List[int],
                     timestamps: List[datetime.datetime]):
    plt.plot(timestamps, viral_loads)
    plt.title("Viral Load Curve")
    plt.ylim((0, 1))
    plt.yticks([tick / 4 for tick in range(4+1)])
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def get_events_history(human_snapshots: List[Human],
                       timestamps: List[datetime.datetime],
                       time_begin: datetime.datetime,
                       time_end: datetime.datetime) -> \
        Tuple[Dict[str, List[int]], List[datetime.datetime]]:
    events = {event_label: [] for event_label in PLOT_EVENTS_LABEL}
    e_timestamps = []

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        for timestamp_events in events.values():
            timestamp_events.append(0)

        for event in human.events:
            if event["event_type"] == Event.encounter:
                events["Encounters"][-1] += 1
            elif event["event_type"] == Event.contamination:
                events["Contaminations"][-1] += 1
            elif event["event_type"] == Event.test:
                events["Tests"][-1] += 1

        e_timestamps.append(timestamp)

    return events, e_timestamps 


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
        locations.append(all_locations.index(human.location))
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


def get_symptom_history(human_snapshots: List[Human],
                        timestamps: List[datetime.datetime],
                        time_begin: datetime.datetime,
                        time_end: datetime.datetime) -> \
        Tuple[List[int], List[int], List[datetime.datetime]]:

    true_symptoms = []
    obs_symptoms = []
    rl_timestamps = []

    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]

        if timestamp > time_end:
            break

        true_symptoms.append(len(human.rolling_all_symptoms[-1]))
        obs_symptoms.append(len(human.rolling_all_reported_symptoms[-1]))
        rl_timestamps.append(timestamp)

    return true_symptoms, obs_symptoms, rl_timestamps


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

        risks.append(human.risk_level)
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

    # Get list of all locations
    all_locations = set()
    for t in timestamps:
        for h in human_backups[t].values():
            all_locations.add(h.location)
    all_locations = list(all_locations)

    # Sort locations by type and then by index
    sorted_locations_indices = sorted([(l.split(':')[0], int(l.split(':')[1])) for l in all_locations])
    sorted_all_locations = ["%s:%i" % (l[0], l[1]) for l in sorted_locations_indices]

    # Treat each human individually
    for idx_human in range(1, nb_humans + 1):

        # Get all the backups of this human for all the timestamps
        h_key = "human:%i" % idx_human
        h_backup = [human_backups[t][h_key] for t in timestamps]

        risks, r_timestamps = get_risk_history(h_backup, timestamps, begin, end)
        viral_loads, vl_timestamps = get_viral_load_history(h_backup, timestamps, begin, end)
        true_symptoms, obs_symptoms, s_timestamps = get_symptom_history(h_backup, timestamps, begin, end)
        recommendation_levels, rl_timestamps = get_recommendation_level_history(h_backup, timestamps, begin, end)
        locations, l_timestamps = get_location_history(h_backup, timestamps, sorted_all_locations, begin, end)
        events, e_timestamps = get_events_history(h_backup, timestamps, begin, end)

        fig = plt.figure()

        fig.add_subplot(6, 2, 2)
        plot_risks(risks, timestamps)

        fig.add_subplot(6, 2, 4)
        plot_viral_loads(viral_loads, timestamps)

        fig.add_subplot(6, 2, 6)
        plot_symptoms(true_symptoms, obs_symptoms, timestamps)

        fig.add_subplot(6, 2, 8)
        plot_recommendation_levels(recommendation_levels, timestamps)

        fig.add_subplot(6, 2, 10)
        plot_locations(locations, sorted_all_locations, timestamps)

        fig.add_subplot(6, 2, 12)
        plot_events(events, timestamps)

        fig.add_subplot(1, 2, 1)
        human = h_backup[0]
        table_data = [
            ["name:", human.name],
            ["age:", human.age],
            ["carefulness:", human.carefulness],
            ["has_app:", human.has_app],
            ["has_allergies:", human.has_allergies],
            ["household:", human.household],
            ["workplace:", human.workplace],
            ["timeslots:", str(human.time_slots)],
        ]
        table = plt.table(cellText=table_data, loc='center')
        table.set_fontsize(14)
        table.scale(1, 3)
        plt.axis('off')

        plot_path = os.path.join(output_folder, f"{str(begin)}-{str(end)}_{h_key}.png")
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()


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
        for fileinfo in zf.infolist():
            if fileinfo.is_dir():
                continue

            day_debug_data = pickle.loads(zf.read(fileinfo))
            timestamp = day_debug_data["human_backups"]["human:1"].env.timestamp
            human_backups = day_debug_data["human_backups"]
            if debug_data is None:
                debug_data = day_debug_data
                debug_data["human_backups"] = {}
            debug_data["human_backups"][timestamp] = human_backups

    # Ensure that the output folder does exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    generate_debug_plots(debug_data, args.output_folder)
