import argparse
import datetime
import pickle
import os
from typing import Dict, List, Tuple
import zipfile

from matplotlib import pyplot as plt

from covid19sim.simulator import Human
from covid19sim.base import Event


PLOT_EVENTS_LABEL = ["Encounters", "Contaminations", "Tests", "Positive Tests", "Negative Tests"]


LOCATION_TO_COLOR = {"household": "tab:red",
                     "park": "tab:green",
                     "hospital": "tab:blue",
                     "store": "tab:purple",
                     "school": "tab:olive",
                     "workplace": "tab:gray",
                     "senior_residency": "tab:brown",
                     "misc": "tab:pink"}


ENCOUNTER_TO_COLOR = {"non-infectuous": "tab:blue",
                      "infectuous": "tab:orange",
                      "contamination": "tab:red"}


EVENT_TO_COLOR = {"Contaminations": "tab:red",
                  "Tests": "tab:green",
                  "Encounters": "tab:blue",
                  "Positive Tests": "tab:purple",
                  "Negative Tests": "tab:olive"}


STATE_TO_COLOR = {"has_flu": "tab:red",
                  "has_cold": "tab:green",
                  "has_allergy_symptoms": "tab:blue",
                  "is_infected": "tab:purple"}


class DebugDataLoader():

    def __init__(self, debug_data_path: str):
        self.path = debug_data_path

        with zipfile.ZipFile(self.path) as zf:
            zip_files = [f for f in zf.infolist() if not f.is_dir()]

            # Nb of timestamps = nb of files (1 file per timestamp)
            self.nb_timestamps = len(zip_files)

            # Read any file to get humans names
            day_debug_data = pickle.loads(zf.read(zip_files[0]))
            human_names = list(day_debug_data["human_backups"].keys())
            self.human_names = sorted(human_names, key=lambda x: int(x.split(":")[1]))

    def get_nb_humans(self):
        return len(self.human_names)

    def get_humans_names(self):
        return self.human_names

    def get_nb_timestamps(self):
        return self.nb_timestamps

    def load_human_data(self, start_idx=None, end_idx=None):
        """
        Load human backups and event data for the specified humans.

        Ex : Calling with start_idx=1 and end_idx=4 will load data for
        the second, third and fourth humans.

        Args:
            start_idx (int, optional): Index (starting at 0) of the first human to load.
                If unspecified, loading will start at first human.
            end_idx (int, optional): Index (starting at 0) of the last human to load plus one.
                If unspecified, humans up until the last one will be loaded.

        Returns:
            [type]: [description]
        """

        if start_idx is not None and end_idx is not None:
            assert start_idx < end_idx

        # Load the human data
        human_backups = {}
        humans_events = {}
        with zipfile.ZipFile(self.path) as zf:
            fileinfos = zf.infolist()
            fileinfos.sort(key=lambda fi: fi.filename)
            for fileinfo in fileinfos:
                if fileinfo.is_dir():
                    continue

                # Read file
                day_debug_data = pickle.loads(zf.read(fileinfo))
                timestamp = day_debug_data["human_backups"]["human:1"].env.timestamp
                human_backups[timestamp] = {}

                # Extract backups for specified humans
                for human_name in self.get_humans_names()[start_idx:end_idx]:
                    human_backups[timestamp][human_name] = day_debug_data["human_backups"][human_name]

                # Extract human event data
                for human in human_backups[timestamp].values():
                    humans_events.setdefault(human.name, dict())
                    for event in human._events:
                        humans_events[human.name][(event["time"], event["event_type"])] = event
                    human._events = []

        # Ensure events are sorted by timestamp for each human
        for human_id, human_events in humans_events.items():
            events = list(human_events.values())
            events.sort(key=lambda e: e["time"])
            humans_events[human_id] = events

        return human_backups, humans_events


def set_pad_for_table(table, pad=0.1):
    for cell in table._cells.values():
        cell.PAD = pad


def plot_mapping(title: str,
                 mapping: Dict[str, str]):
    plt.title(title)
    table_data = [[k + " :", v.replace("tab:","")] for k,v in mapping.items()]
    table = plt.table(cellText=table_data, loc='center')
    table.set_fontsize(14)
    table.scale(1, 0.5)
    plt.axis('off')
    plt.gcf().autofmt_xdate()


def plot_events(events: Dict[str, List[int]],
                timestamps: List[datetime.datetime]):
    event_to_width = {"Encounters": 0.05,
                      "Contaminations": 0.05,
                      "Tests": 0.05,
                      "Positive Tests": 0.05,
                      "Negative Tests": 0.05}
    events_sum = [0] * len(timestamps)
    for event_label in PLOT_EVENTS_LABEL:
        if event_label == "Encounters":
            plt.step(timestamps, events[event_label], where='mid')
        else:
            plt.bar(timestamps, events[event_label],
                    width=0.1,
                    color=EVENT_TO_COLOR[event_label],
                    bottom=events_sum,
                    label=event_label)
            for i in range(len(events[event_label])):
                events_sum[i] += events[event_label][i]
    plt.title("Events")
    plt.ylim(0, 10)
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_encounters(encounters: Dict[str, List[int]],
                    risky_encounters: Dict[str, List[int]],
                    contamination_encounters: Dict[str, List[int]],
                    timestamps: List[datetime.datetime]):
    risky_encounters_human_ids = set()
    contamination_encounters_human_ids = set()
    for other_human in encounters:
        other_human_id = int(other_human[6:])
        for encounter, timestamp in zip(encounters[other_human], timestamps):
            if not encounter:
                continue
            safe_color = ENCOUNTER_TO_COLOR["non-infectuous"]
            plt.broken_barh([(timestamp, 1/24)], (other_human_id - 1, 1), color=safe_color, label=other_human)
        for encounter, timestamp in zip(risky_encounters[other_human], timestamps):
            if not encounter:
                continue
            risky_encounters_human_ids.add(other_human_id)
            risky_color = ENCOUNTER_TO_COLOR["infectuous"]
            plt.broken_barh([(timestamp, 1 / 21)], (other_human_id - 1, 1), color=risky_color, label=other_human)
        for encounter, timestamp in zip(contamination_encounters[other_human], timestamps):
            if not encounter:
                continue
            contamination_encounters_human_ids.add(other_human_id)
            contamination_color = ENCOUNTER_TO_COLOR["contamination"]
            plt.broken_barh([(timestamp, 1 / 12)], (other_human_id - 1, 1), color=contamination_color, label=other_human)

    plt.title("Encounters")
    plt.ylim((0, len(encounters)))
    plt.yticks([int(tick / 4 * len(encounters)) for tick in range(4+1)] +
               list(risky_encounters_human_ids) +
               list(contamination_encounters_human_ids))
    # plt.set_yticklabels(locations_names)
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_locations(locations: List[int],
                   locations_names: List[str],
                   timestamps: List[datetime.datetime]):

    for location, timestamp in zip(locations, timestamps):
        l_name = locations_names[location]
        color = LOCATION_TO_COLOR[l_name.split(":")[0]]
        plt.broken_barh([(timestamp, 1/24)], (location, 1), facecolor=color, label=l_name)
    plt.title("Location")
    plt.ylim((0, len(locations_names)))
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_recommendation_levels(recommendation_levels: List[int],
                               timestamps: List[datetime.datetime]):
    plt.step(timestamps, recommendation_levels)
    plt.title("Recommendation Level")
    plt.ylim((-1, 3))
    plt.yticks([tick for tick in range(4)])
    plt.xlabel("Time")
    plt.gcf().autofmt_xdate()


def plot_states(states: Dict[str, List[bool]],
                timestamps: List[datetime.datetime]):
    for state_name, state_vals in states.items():
        plt.step(timestamps, [int(s) for s in state_vals], color=STATE_TO_COLOR[state_name])
    plt.title("Binary States")
    plt.ylim((-0.1, 1.1))
    plt.yticks([0, 1])
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


def get_events_history(all_human_events: List[Dict],
                       humans_cnt: int,
                       timestamps: List[datetime.datetime],
                       time_begin: datetime.datetime,
                       time_end: datetime.datetime) -> \
        Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]], List[datetime.datetime]]:
    events = {event_label: [] for event_label in PLOT_EVENTS_LABEL}
    encounters = {f"human:{i+1}": [] for i in range(humans_cnt)}
    risky_encounters = {f"human:{i+1}": [] for i in range(humans_cnt)}
    contamination_encounters = {f"human:{i+1}": [] for i in range(humans_cnt)}
    e_timestamps = []

    # There is one human snapshot per time slot per day
    events_i = 0
    # Find human's first event
    while events_i < len(all_human_events) and all_human_events[events_i]["time"] < time_begin:
        events_i += 1
    for timestamp in timestamps:
        if timestamp < time_begin:
            continue
        if timestamp > time_end:
            break

        for timestamp_events in events.values():
            timestamp_events.append(0)
        for timestamp_encounters, timestamp_risky_encounters, \
            timestamp_encounters_contamination_encounters in \
                zip(encounters.values(), risky_encounters.values(),
                    contamination_encounters.values()):
            timestamp_encounters.append(0)
            timestamp_risky_encounters.append(0)
            timestamp_encounters_contamination_encounters.append(0)

        while events_i < len(all_human_events) and \
                all_human_events[events_i]["time"] <= timestamp:
            event = all_human_events[events_i]
            if event["event_type"] == Event.encounter:
                events["Encounters"][-1] += 1
                other_human = event["payload"]["unobserved"]["human2"]
                if contamination_encounters[other_human["human_id"]][-1] == 0:
                    if other_human["is_infectious"]:
                        risky_encounters[other_human["human_id"]][-1] += 1
                    else:
                        encounters[other_human["human_id"]][-1] += 1
            elif event["event_type"] == Event.contamination:
                events["Contaminations"][-1] += 1
                other_human_id = event["payload"]["unobserved"]["source"]
                if other_human_id.startswith("human"):
                    encounters[other_human_id][-1] = 0
                    risky_encounters[other_human_id][-1] = 0
                    contamination_encounters[other_human_id][-1] += 1
            elif event["event_type"] == Event.test:
                test_result = event["payload"]["unobserved"]["result"]
                if test_result == "positive":
                    events["Positive Tests"][-1] += 1
                elif test_result == "negative":
                    events["Negative Tests"][-1] += 1
                else:
                    events["Tests"][-1] += 1
            events_i += 1

        e_timestamps.append(timestamp)

    return events, encounters, risky_encounters, contamination_encounters, e_timestamps


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

        true_symptoms.append(len(human.rolling_all_symptoms[0]))
        obs_symptoms.append(len(human.rolling_all_reported_symptoms[0]))
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


def get_states_history(human_snapshots: List[Human],
                       timestamps: List[datetime.datetime],
                       time_begin: datetime.datetime,
                       time_end: datetime.datetime) -> \
        Tuple[Dict[str, List[bool]], List[datetime.datetime]]:
    states = {
        "has_flu": [],
        "has_cold": [],
        "has_allergy_symptoms": [],
        "is_infected": [],
    }
    s_timestamps = []
    # There is one human snapshot per time slot per day
    start_index = (time_begin - timestamps[0]).days * len(human_snapshots[0].time_slots)
    for i in range(max(0, start_index), len(human_snapshots)):
        human = human_snapshots[i]
        timestamp = timestamps[i]
        if timestamp > time_end:
            break
        states["has_flu"].append(human.has_flu)
        states["has_cold"].append(human.has_cold)
        states["has_allergy_symptoms"].append(human.has_allergy_symptoms)
        states["is_infected"].append(not human.is_susceptible)
        s_timestamps.append(timestamp)
    return states, s_timestamps


def get_infection_history(humans_events: Dict[str, List], human_key: str):

    # Get the list of events in which the specified human was infected by someone else
    # and list of events in which the specified human infected someone else
    infectee_events = []
    infector_events = []
    for e in humans_events[human_key]:
        if e["event_type"] == Event.contamination:
            infectee_events.append([e["payload"]["unobserved"]["source"], e["time"]])
        elif e["event_type"] == Event.encounter and \
                e['payload']['unobserved']['human1']['exposed_other']:
            infector_events.append([e["human_id"], e["time"]])

    # Sort infector events by the date of the infection (swap columns, sort, swap columns)
    infector_events = [[j[1], j[0]] for j in sorted([[i[1], i[0]] for i in infector_events])]

    return infectee_events, infector_events


def generate_human_centric_plots(human_backups, humans_events, nb_humans_in_sim, output_folder):
    timestamps = sorted(list(human_backups.keys()))
    human_names = list(human_backups[timestamps[0]].keys())
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
    for h_key in human_names:

        # Get all the backups of this human for all the timestamps
        h_backup = [human_backups[t][h_key] for t in timestamps]

        # Extract data for each plot
        risks, r_timestamps = get_risk_history(h_backup, timestamps, begin, end)
        viral_loads, vl_timestamps = get_viral_load_history(h_backup, timestamps, begin, end)
        true_symptoms, obs_symptoms, s_timestamps = get_symptom_history(h_backup, timestamps, begin, end)
        recommendation_levels, rl_timestamps = get_recommendation_level_history(h_backup, timestamps, begin, end)
        locations, l_timestamps = get_location_history(h_backup, timestamps, sorted_all_locations, begin, end)
        events, \
            encounters, \
            risky_encounters, \
            contamination_encounters, \
            e_timestamps = get_events_history(humans_events[h_key], nb_humans_in_sim, timestamps, begin, end)
        states, s_timestamps = get_states_history(h_backup, timestamps, begin, end)
        infectee_events, infector_events = get_infection_history(humans_events, h_key)

        fig = plt.figure(constrained_layout=True)

        # First row (columns 2+)
        fig.add_subplot(4, 4, 2)
        plot_risks(risks, timestamps)

        fig.add_subplot(4, 4, 3)
        plot_locations(locations, sorted_all_locations, timestamps)

        fig.add_subplot(4, 4, 4)
        plot_mapping("Location legend", LOCATION_TO_COLOR)

        # Second row (columns 2+)
        fig.add_subplot(4, 4, 6)
        plot_viral_loads(viral_loads, timestamps)

        fig.add_subplot(4, 4, 7)
        plot_events(events, timestamps)

        fig.add_subplot(4, 4, 8)
        plot_mapping("Event legend", EVENT_TO_COLOR)

        # Third row (columns 2+)
        fig.add_subplot(4, 4, 10)
        plot_symptoms(true_symptoms, obs_symptoms, timestamps)

        fig.add_subplot(4, 4, 11)
        plot_encounters(encounters, risky_encounters, contamination_encounters, timestamps)

        fig.add_subplot(4, 4, 12)
        plot_mapping("Encounter legend", ENCOUNTER_TO_COLOR)

        # Fourth row (columns 2+)
        fig.add_subplot(4, 4, 14)
        plot_recommendation_levels(recommendation_levels, timestamps)

        fig.add_subplot(4, 4, 15)
        plot_states(states, timestamps)

        fig.add_subplot(4, 4, 16)
        plot_mapping("States legend", STATE_TO_COLOR)

        # First column
        fig.add_subplot(1, 4, 1)
        human = h_backup[-1]
        table_data = [
            ["name:", human.name],
            ["age:", human.age],
            ["carefulness:", human.carefulness],
            ["has_app:", human.has_app],
            ["has_allergies:", human.has_allergies],
            ["nb preconditions", len(human.preexisting_conditions)],
            ["household:", human.household],
            ["workplace:", human.workplace],
            ["timeslots:", str(human.time_slots)],
            ["asymptomatic", human.is_asymptomatic],
            ["gets +/++ sick",
             str([human.can_get_really_sick, human.can_get_extremely_sick])],
            ["E/M/S/W mins",
             str([human.avg_exercise_time, human.avg_misc_time,
                  human.avg_shopping_time, human.avg_working_minutes])],
            ["infected by:", "\n".join(", ".join([str(i) for i in e]) for e in infectee_events)],
        ]

        if len(infector_events) == 1:
            table_data.append(["has infected:", ""])
        else:
            for idx, e in enumerate(infector_events):
                if idx == 0:
                    table_data.append(["has infected:", ", ".join([str(i) for i in e])])
                else:
                    table_data.append(["", ", ".join([str(i) for i in e])])

        table = plt.table(cellText=table_data, loc='center', colWidths=[0.33, 0.67])
        #table.auto_set_font_size(False)
        #table.set_fontsize(8)
        table.scale(1, 1.2)
        set_pad_for_table(table, pad=0.05)
        plt.axis('off')

        fig.set_size_inches(15, 10)

        plot_path = os.path.join(output_folder, f"{str(begin)}-{str(end)}_{h_key}.png")
        fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        fig.clf()
        plt.close(fig)


def generate_location_centric_plots(debug_data, output_folder):
    pass


def generate_debug_plots(data_loader, output_folder, batch_size=10):

    # Generate human-centric plots (break it down in batches to reduce mem usage)
    nb_humans_in_sim = data_loader.get_nb_humans()
    for i in range(0, nb_humans_in_sim, batch_size):
        human_backups, human_events = data_loader.load_human_data(start_idx=i, end_idx=i+batch_size)
        generate_human_centric_plots(human_backups, human_events, nb_humans_in_sim, output_folder)

    # Generate location-centric plots
    generate_location_centric_plots(data_loader, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_data")
    parser.add_argument("--output_folder")
    args = parser.parse_args()

    # Load the debug data
    data_loader = DebugDataLoader(args.debug_data)

    # Ensure that the output folder does exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    generate_debug_plots(data_loader, args.output_folder)
