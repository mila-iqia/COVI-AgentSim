# ***********************
# This file is a duplicate of data.py with improved functions.
# Manually copied in a rush.
# Some functions are also added.
# **********************

import numpy as np
import os
import pickle
from covid19sim.plotting.plot_rt import PlotRt
from pathlib import Path
import yaml
import concurrent.futures


def get_data(filename=None, data=None):
    if data:
        return data
    elif filename:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        raise ValueError("Please provide either filename, or data")


def get_states(filename=None, data=None):

    states = get_human_states(filename, data)

    def bincount(x):
        return np.bincount(x, minlength=4)

    return np.apply_along_axis(bincount, axis=0, arr=states) / states.shape[0]


def get_human_states(filename=None, data=None):

    data = get_data(filename, data)

    mapping = {"S": 0, "E": 1, "I": 2, "R": 3, "N/A": -1}

    humans_state = data["humans_state"]
    names = list(humans_state.keys())
    num_days = len(humans_state[names[0]])
    states = np.zeros((len(names), num_days), dtype=np.int_)

    for i, name in enumerate(names):
        states[i] = np.asarray(
            [mapping[state] for state in humans_state[name]], dtype=np.int_
        )
    assert np.all(states >= 0)

    return states


def get_false_quarantine(filename=None, data=None):
    data = get_data(filename, data)
    intervention_day = data["intervention_day"]
    if intervention_day < 0:
        intervention_day = 0
    states = get_human_states(data=data)
    states = states[:, intervention_day:]
    rec_levels = get_human_rec_levels(data=data)
    false_quarantine = np.sum(
        ((states == 0) | (states == 3)) & (rec_levels == 3), axis=0
    )
    return false_quarantine / states.shape[0]


def get_all_false_quarantine(filenames):
    fq = get_false_quarantine(filenames[0])
    all_fq = np.zeros((len(filenames),) + fq.shape, dtype=fq.dtype)
    all_fq[0] = fq

    for i, filename in enumerate(filenames[1:]):
        all_fq[i + 1] = get_false_quarantine(filename)

    return all_fq


def get_all_false_quarantine_mp(filenames, cpu_count=None):
    if cpu_count is None:
        cpu_count = mp.cpu_count()
    pool = mp.Pool(cpu_count)
    fqs = pool.map(get_false_quarantine, filenames)

    return fqs


def get_all_states(filenames):
    states = get_states(filenames[0])
    all_states = np.zeros((len(filenames),) + states.shape, dtype=states.dtype)
    all_states[0] = states

    for i, filename in enumerate(filenames[1:]):
        all_states[i + 1] = get_states(filename)

    return all_states


def get_rec_levels(filename=None, data=None):
    if data is None:
        if filename is not None:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError("filename and data arguments are None")

    rec_levels = get_human_rec_levels(filename, data) + 1  # account for rec_levels `-1`

    def bincount(x):
        return np.bincount(x, minlength=5)

    return np.apply_along_axis(bincount, axis=0, arr=rec_levels) / rec_levels.shape[0]


def get_human_rec_levels(filename=None, data=None, normalized=False):
    data = get_data(filename, data)

    key = "humans_rec_level"
    if normalized:
        key = "humans_intervention_level"

    humans_rec_level = data[key]
    intervention_day = data["intervention_day"]
    names = list(humans_rec_level.keys())
    num_days = len(humans_rec_level[names[0]])
    rec_levels = np.zeros((len(names), num_days), dtype=np.int_)

    for i, name in enumerate(names):
        rec_levels[i] = np.asarray(humans_rec_level[name], dtype=np.int_)
    rec_levels = rec_levels[:, intervention_day:]

    return rec_levels


def get_all_rec_levels(filenames=None, data=None):
    if data is not None:
        data = [d["pkl"] for d in data.values()]
        filenames = [None] * len(data)
    elif filenames is not None:
        data = [None] * len(filenames)
    else:
        raise ValueError("filenames and data arguments are None")

    rec_levels = get_rec_levels(filenames[0], data[0])
    all_rec_levels = np.zeros(
        (len(filenames),) + rec_levels.shape, dtype=rec_levels.dtype
    )
    all_rec_levels[0] = rec_levels

    for i, filename in enumerate(filenames[1:]):
        all_rec_levels[i + 1] = get_rec_levels(filename, data[i + 1])

    return all_rec_levels


def get_intervention_levels(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    humans_rec_level = data["humans_intervention_level"]
    intervention_day = data["intervention_day"]
    names = list(humans_rec_level.keys())
    num_days = len(humans_rec_level[names[0]])
    rec_levels = np.zeros((len(names), num_days), dtype=np.int_)

    for i, name in enumerate(names):
        rec_levels[i] = np.asarray(humans_rec_level[name], dtype=np.int_)
    rec_levels = rec_levels[:, intervention_day:]

    def bincount(x):
        return np.bincount(x, minlength=4)

    return np.apply_along_axis(bincount, axis=0, arr=rec_levels) / len(names)


def get_all_intervention_levels(filenames):
    intervention_levels = get_intervention_levels(filenames[0])
    all_intervention_levels = np.zeros(
        (len(filenames),) + intervention_levels.shape, dtype=intervention_levels.dtype
    )
    all_intervention_levels[0] = intervention_levels

    for i, filename in enumerate(filenames[1:]):
        all_intervention_levels[i + 1] = get_intervention_levels(filename)

    return all_intervention_levels


def get_Rt(filename=None, data=None):
    data = get_data(filename, data)

    # days = list(data["human_monitor"].keys())
    cases_per_day = data["cases_per_day"]

    si = np.array(data["all_serial_intervals"])

    plotrt = PlotRt(R_T_MAX=4, sigma=0.25, GAMMA=1.0 / si.mean())
    most_likely, _ = plotrt.compute(cases_per_day, r0_estimate=2.5)

    return most_likely


def absolute_file_paths(directory):
    to_return = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if ".pkl" in f:
                to_return.append(os.path.abspath(os.path.join(dirpath, f)))
    return to_return


def get_all_data(base_path, keep_pkl_keys, multi_thread=False):
    base_path = Path(base_path).resolve()
    assert base_path.exists()
    methods = [
        m
        for m in base_path.iterdir()
        if m.is_dir()
        and not m.name.startswith(".")
        and len(list(m.glob("**/tracker*.pkl"))) > 0
    ]
    assert methods, (
        "Could not find any methods. Make sure the folder structure is correct"
        + " (expecting <base_path_you_provided>/<method>/<run>/tracker*.pkl)"
    )

    all_data = {str(m): {} for m in methods}
    for m in methods:
        sm = str(m)
        runs = [
            r
            for r in m.iterdir()
            if r.is_dir()
            and not r.name.startswith(".")
            and len(list(r.glob("tracker*.pkl"))) == 1
        ]
        print(" " * 100, end="\r")
        print("Loading runs in", m.name, "...", end="\r")
        try:
            if multi_thread:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(thread_read_run, (r, keep_pkl_keys))
                        for r in runs
                    ]
                    runs_data = [f.result() for f in futures]
                for r, conf, pkl in runs_data:
                    sr = str(r)
                    all_data[sm][sr] = {}
                    all_data[sm][sr]["conf"] = conf
                    all_data[sm][sr]["pkl"] = pkl

            else:
                for r in runs:
                    r, conf, pkl = thread_read_run((r, keep_pkl_keys))
                    sr = str(r)
                    all_data[sm][sr] = {}
                    all_data[sm][sr]["conf"] = conf
                    all_data[sm][sr]["pkl"] = pkl
                # print("Loading {}/{}".format(m.name, r.name), end="\r", flush=True)

        except TypeError as e:
            print(
                f"\nCould not load pkl in {m.name}/{r.name}"
                + "\nRemember Python 3.7 cannot read 3.8 pickles and vice versa:\n"
                + f"Error: {str(e)}"
            )
            print(">>> Skipping method **{}**\n".format(m.name))
            del all_data[sm]
    return all_data


def thread_read_run(args):
    r, keep_pkl_keys = args
    with (r / "full_configuration.yaml").open("r") as f:
        conf = yaml.safe_load(f)
    with open(str(list(r.glob("tracker*.pkl"))[0]), "rb") as f:
        pkl_data = pickle.load(f)
        pkl = {k: v for k, v in pkl_data.items() if k in keep_pkl_keys}

    return (r, conf, pkl)
