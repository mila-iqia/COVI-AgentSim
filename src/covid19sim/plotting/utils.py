import numpy as np
import os
import multiprocessing as mp
import pickle
from covid19sim.plotting.plot_rt import PlotRt
from pathlib import Path
import yaml
import concurrent.futures
from scipy import stats as sps
from collections import defaultdict


def env_to_path(path):
    """Transorms an environment variable mention in a json
    into its actual value. E.g. $HOME/clouds -> /home/vsch/clouds

    Args:
        path (str): path potentially containing the env variable

    """
    path_elements = path.split("/")
    new_path = []
    for el in path_elements:
        if "$" in el:
            new_path.append(os.environ[el.replace("$", "")])
        else:
            new_path.append(el)
    return "/".join(new_path)


def get_title(method):
    method_title = {
        "bdt1": "1st Order Binary Tracing",
        "bdt2": "2nd Order Binary Tracing",
        "heuristicv1": "Heuristic (v1)",
        "heuristicv2": "Heuristic (v2)",
        "transformer": "Transformer",
        "transformer-[1, 3, 5]": "Transformer-[1, 3, 5]",
        "transformer-[0, 1, 2]": "Transformer-[0, 1, 2]",
        "transformer-[0, 0, 0]": "Transformer-[0, 0, 0]",
        "linreg": "Linear Regression",
        "mlp": "MLP",
        "unmitigated": "Unmitigated",
        "oracle": "Oracle",
    }
    if "_norm" in method:
        if method.replace("_norm", "") in method_title:
            return method_title[method.replace("_norm", "")] + " (Norm.)"
    if method in method_title:
        return method_title[method]
    if "transformer_012_" in method:
        method = method.replace("transformer_012_", "")
        if "_normed" in method:
            method = "-".join(method.split("-")[:-1])
            method += " (Norm.)"
        else:
            method = "-".join(method.split("-")[:-1])
    return method.replace("_", " ").capitalize()


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


def get_rec_levels(filename=None, data=None, normalized=False):
    if data is None:
        if filename is not None:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError("filename and data arguments are None")

    rec_levels = (
        get_human_rec_levels(filename, data, normalized=normalized) + 1
    )  # account for rec_levels `-1`

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


def get_all_rec_levels(filenames=None, data=None, normalized=False):
    if data is not None:
        data = [d["pkl"] for d in data.values()]
        filenames = [None] * len(data)
    elif filenames is not None:
        data = [None] * len(filenames)
    else:
        raise ValueError("filenames and data arguments are None")

    rec_levels = get_rec_levels(filenames[0], data[0], normalized=normalized)
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

        try:
            if multi_thread:
                print("Loading runs in", m.name, "...", end="\r")
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
                for i, r in enumerate(runs):
                    print(" " * 120, end="\r")
                    print(
                        "Loading runs in",
                        m.name,
                        "... ({}/{})".format(i + 1, len(runs)),
                        end="\r",
                    )
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


def default_to_regular_dict(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular_dict(v) for k, v in d.items()}
    return d


def thread_read_run(args):
    r, keep_pkl_keys = args
    with (r / "full_configuration.yaml").open("r") as f:
        conf = yaml.safe_load(f)
    with open(str(list(r.glob("tracker*.pkl"))[0]), "rb") as f:
        pkl_data = pickle.load(f)
        pkl = {
            k: default_to_regular_dict(v)
            for k, v in pkl_data.items()
            if k in keep_pkl_keys
        }

    return (r, conf, pkl)


def truncate_seeds(data):
    """
    If not all runs have the same seeds (typically, a run is missing some seeds),
    remove data  which is not in the intersection of all seeds.

    Args:
        data (dict): The data to prune
    """
    seeds = set()
    first = True
    for mk, mv in data.items():
        for ck, cv in mv.items():
            comparison_seeds = set()
            for rv in cv.values():
                rseed = rv["conf"]["seed"]
                if not first and rseed not in seeds:
                    print(" !!  Warning not all runs have the same number of seeds")
                    print(" !!  Pruning seeds to the interesection of all runs.")
                comparison_seeds.add(rseed)
            if first:
                seeds = comparison_seeds
                first = False
            else:
                seeds = seeds.intersection(comparison_seeds)

    for mk, mv in data.items():
        for ck, cv in mv.items():
            for rk, rv in cv.items():
                if rv["conf"]["seed"] not in seeds:
                    del data[mk][ck][rk]
    return data

def get_all(filename_types=None, pkl_types=None, labels=[], normalized=False):
    if pkl_types is not None:
        tmp = [(None, pkls) for pkls in pkl_types]
    elif filename_types is not None:
        tmp = [(filenames, None) for filenames in filename_types]
    else:
        raise ValueError("filename_types and pkl_types are None")

    _rows = []

    for i, (filenames, pkls) in enumerate(tmp):
        metrics = get_mean_fq_r(filenames=filenames, pkls=pkls, normalized=normalized)
        for key, val in metrics.items():
            _rows.append([labels[i], key] + val)
    return _rows

def get_fq_r(filename=None, data=None, normalized=False):
    assert filename is not None or data is not None
    if data is None:
        data = pickle.load(open(filename, "rb"))

    f3, f2, f1, f1_up, f2_up = get_all_false(data=data, normalized=normalized)
    x = [i[-5:].mean() for i in [f3, f2, f1, f1_up, f2_up]]

    intervention_day = data["intervention_day"]
    od = np.mean(data["outside_daily_contacts"][intervention_day:])
    ec = data["effective_contacts_since_intervention"]

    # percent_infected
    y = sum(data["cases_per_day"]) / data["n_humans"]

    # R
    z = get_effective_R(data)

    # proxy_r
    a = get_proxy_r(data)

    return x, y, z, a, od, ec


def get_mean_fq_r(filenames=None, pkls=None, normalized=False):
    assert filenames is not None or pkls is not None
    if pkls is not None:
        _tmp = [(None, {"pkl": pkl}) for pkl in pkls]
    elif filenames is not None:
        _tmp = [(filename, None) for filename in filenames]
    else:
        raise ValueError("filenames and pkls are None")

    metrics = {
        "f3": [],
        "f2": [],
        "f1": [],
        "f1_up": [],
        "f2_up": [],
        "percent_infected": [],
        "r": [],
        "proxy_r": [],
        "outside_daily_contacts": [],
        "effective_contacts": [],
    }
    for filename, pkl in _tmp:
        x, y, z, a, od, ec = get_fq_r(
            filename=filename, data=pkl["pkl"], normalized=normalized
        )
        metrics["f3"].append(x[0])
        metrics["f2"].append(x[1])
        metrics["f1"].append(x[2])
        metrics["f1_up"].append(x[3])
        metrics["f2_up"].append(x[4])
        metrics["percent_infected"].append(y)
        metrics["r"].append(z)
        metrics["proxy_r"].append(a)
        metrics["outside_daily_contacts"].append(od)
        metrics["effective_contacts"].append(ec)

    return metrics



def get_all_false(filename=None, data=None, normalized=False):
    data = get_data(filename, data)
    intervention_day = data["intervention_day"]
    if intervention_day < 0:
        intervention_day = 0
    states = get_human_states(data=data)
    states = states[:, intervention_day:]
    rec_levels = get_human_rec_levels(data=data, normalized=normalized)

    false_level3 = np.sum(((states == 0) | (states == 3)) & (rec_levels == 3), axis=0)
    false_level2 = np.sum(((states == 0) | (states == 3)) & (rec_levels == 2), axis=0)
    false_level1 = np.sum(((states == 0) | (states == 3)) & (rec_levels == 1), axis=0)
    false_level1_above = np.sum(
        ((states == 0) | (states == 3))
        & ((rec_levels == 1) | (rec_levels == 2) | (rec_levels == 3)),
        axis=0,
    )
    false_level2_above = np.sum(
        ((states == 0) | (states == 3)) & ((rec_levels == 2) | (rec_levels == 3)),
        axis=0,
    )
    return (
        false_level3 / states.shape[0],
        false_level2 / states.shape[0],
        false_level1 / states.shape[0],
        false_level1_above / states.shape[0],
        false_level2_above / states.shape[0],
    )


def get_proxy_r(data):
    total_infected = 0
    for k in data["humans_state"].keys():
        total_infected += any(z == "I" for z in data["humans_state"][k][5:])
    if total_infected > 0:
        return sum(data["cases_per_day"][5:]) / total_infected
    return 0.


def get_effective_R(data):
    GT = data["generation_times"]
    a = 4
    b = 0.5
    window_size = 5
    ws = [sps.gamma.pdf(x, a=GT, loc=0, scale=0.9) for x in range(window_size)]
    last_ws = ws[::-1]
    cases_per_day = data["cases_per_day"]

    lambda_s = []
    rt = []
    for i in range(len(cases_per_day)):
        if i < window_size:
            last_Is = cases_per_day[:i]
        else:
            last_Is = cases_per_day[(i - window_size) : i]

        lambda_s.append(sum(x * y for x, y in zip(last_Is, last_ws)))
        last_lambda_s = sum(lambda_s[-window_size:])
        rt.append((a + sum(last_Is)) / (1 / b + last_lambda_s))
    return np.mean(rt[-5:])

def get_metrics(data, label, metric):
    tmp = data[(data["type"] == label) & (data["metric"] == metric)]
    return tmp["mean"], tmp["stderr"]
