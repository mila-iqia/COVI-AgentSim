"""
Utility functions perform generic operations.
"""
import copy
import dataclasses
import datetime
import functools
import gc
import math
import os
import pathlib
import subprocess
import sys
import textwrap
import types
import typing
import zipfile
from orderedset import OrderedSet
from pathlib import Path
import time
import dill
import numpy as np
import requests
import yaml
import json
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm
from covid19sim.utils.constants import SECONDS_PER_HOUR, SECONDS_PER_MINUTE, AGE_BIN_WIDTH_5, AGE_BIN_WIDTH_10

from covid19sim.epidemiology.symptoms import STR_TO_SYMPTOMS
from covid19sim.utils.constants import AGE_BIN_WIDTH_5, AGE_BIN_WIDTH_10
if typing.TYPE_CHECKING:
    from covid19sim.human import Human


def log(str, logfile=None, timestamp=False):
    """
    [summary]

    Args:
        str ([type]): [description]
        logfile ([type], optional): [description]. Defaults to None.
        timestamp (bool, optional): [description]. Defaults to False.
    """
    if timestamp:
        str = f"[{datetime.datetime.now()}] {str}"

    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def _normalize_scores(scores):
    """
    [summary]

    Args:
        scores ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.array(scores)/np.sum(scores)


def _get_random_area(num, total_area, rng):
    """
    Using Dirichlet distribution since it generates a "distribution of probabilities"
    which will ensure that the total area allotted to a location type remains conserved
    while also maintaining a uniform distribution

    Args:
        num ([type]): [description]
        total_area ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Keeping max at area/2 to ensure no location is allocated more than half of the total area allocated to its location type
    area = np.array([total_area/num for _ in range(num)])
    return area

def draw_random_discrete_gaussian(avg, scale, rng):
    """
    [summary]

    Args:
        avg ([type]): [description]
        scale ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    # https://stackoverflow.com/a/37411711/3413239
    irange, normal_pdf = _get_integer_pdf(avg, scale, 2)
    return int(rng.choice(irange, size=1, p=normal_pdf))

def _json_serialize(o):
    """
    [summary]

    Args:
        o ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(o, datetime.datetime):
        return o.__str__()

def compute_distance(loc1, loc2):
    """
    [summary]

    Args:
        loc1 ([type]): [description]
        loc2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.sqrt((loc1.lat - loc2.lat) ** 2 + (loc1.lon - loc2.lon) ** 2)


@functools.lru_cache(500)
def _get_integer_pdf(avg, scale, num_sigmas=2):
    """
    [summary]

    Args:
        avg ([type]): [description]
        scale ([type]): [description]
        num_sigmas (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    irange = np.arange(avg - num_sigmas * scale, avg + num_sigmas * scale + 1)
    normal_pdf = norm.pdf(irange - avg)
    normal_pdf /= normal_pdf.sum()
    return irange, normal_pdf


def probas_to_risk_mapping(probas,
                           num_bins,
                           lower_cutoff=None,
                           upper_cutoff=None):
    """
    Create a mapping from probabilities returned by the model to discrete
    risk levels, with a number of predictions in each bins being approximately
    equivalent.

    Args:
        probas (np.ndarray): The array of probabilities returned by the model.
        num_bins (int): The number of bins. For example, `num_bins=16` for risk
            messages on 4 bits.
        lower_cutoff (float, optional): Ignore values smaller than `lower_cutoff`
            in the creation of the bins. This avoids any bias towards values which
            are too close to 0. If `None`, then do not cut off the small probabilities.
            Defaults to None.
        upper_cutoff (float, optional): Ignore values larger than `upper_cutoff` in the
            creation of the bins. This avoids any bias towards values which are too
            close to 1. If `None`, then do not cut off the large probabilities.
            Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: The mapping from probabilities to discrete risk levels. This mapping has
        size `num_bins + 1`, with the first values always being 0, and the last
        always being 1.
    """
    if (lower_cutoff is not None) and (upper_cutoff is not None):
        if lower_cutoff >= upper_cutoff:
            raise ValueError('The lower cutoff must have a value which is '
                             'smaller than the upper cutoff, got `lower_cutoff='
                             '{0}` and `upper_cutoff={1}`.'.format(
                             lower_cutoff, upper_cutoff))
    mask = np.ones_like(probas, dtype=np.bool_)
    num_percentiles = num_bins + 1
    # First value is always 0, last value is always 1
    cutoffs = np.zeros((num_bins + 1,), dtype=probas.dtype)
    cutoffs[-1] = 1.

    # Remove probabilities close to 0
    lower_idx = 1 if (lower_cutoff is None) else None
    if lower_cutoff is not None:
        mask = np.logical_and(mask, probas > lower_cutoff)
        num_percentiles -= 1

    # Remove probabilities close to 1
    upper_idx = -1 if (upper_cutoff is None) else None
    if upper_cutoff is not None:
        mask = np.logical_and(mask, probas <= upper_cutoff)
        num_percentiles -= 1

    percentiles = np.linspace(0, 100, num_percentiles)
    cutoffs[1:-1] = np.percentile(probas[mask],
                                  q=percentiles[lower_idx:upper_idx])

    return cutoffs


def _proba_to_risk(probas, mapping):
    """Probability to risk mapping operation. Non-lambda version, because why use a lambda?"""
    return np.maximum(np.searchsorted(mapping, probas, side='left') - 1, 0)


def proba_to_risk_fn(mapping):
    """
    Create a callable, based on a mapping, that takes probabilities (in
    [0, 1]) and returns a discrete risk level (in [0, num_bins - 1]).

    Args:
        mapping (np.ndarray): The mapping from probabilities to discrete risk levels.
        See `probas_to_risk_mapping`.

    Returns:
        callable: Function taking probabilities and returning discrete risk levels.
    """
    return functools.partial(_proba_to_risk, mapping=mapping)


def calculate_average_infectiousness(human):
    """ This is only used for the infectiousness value for a human that is written out for the ML predictor.
    We write tomorrows infectiousness (and predict tomorrows infectiousness) so that our predictor is conservative. """
    # cur_infectiousness = human.get_infectiousness_for_day(human.env.timestamp, human.is_infectious)
    is_infectious_tomorrow = True if human.infection_timestamp and human.env.timestamp - human.infection_timestamp + datetime.timedelta(days=1) >= datetime.timedelta(days=human.infectiousness_onset_days) else False
    tomorrows_infectiousness = human.get_infectiousness_for_day(human.env.timestamp + datetime.timedelta(days=1),
                                                                is_infectious_tomorrow)
    return tomorrows_infectiousness #(cur_infectiousness + tomorrows_infectiousness) / 2


def filter_open(locations):
    """Given an iterable of locations, returns a list of those that are open for business.

    Args:
        locations (iterable): a list of objects inheriting from the covid19sim.base.Location class

    Returns:
        list
    """
    return [loc for loc in locations if loc.is_open_for_business]


def filter_queue_max(locations, max_len):
    """Given an iterable of locations, will return a list of those
    with queues that are not too long.

    Args:
        locations (iterable): a list of objects inheriting from the covid19sim.base.Location class

    Returns:
        list
    """
    return [loc for loc in locations if len(loc.queue) <= max_len]


def download_file_from_google_drive(
        gdrive_file_id: typing.AnyStr,
        destination: typing.AnyStr,
        chunk_size: int = 32768
) -> typing.AnyStr:
    """
    Downloads a file from google drive, bypassing the confirmation prompt.

    Args:
        gdrive_file_id: ID string of the file to download from google drive.
        destination: where to save the file.
        chunk_size: chunk size for gradual downloads.

    Returns:
        The path to the downloaded file.
    """
    # taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': gdrive_file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': gdrive_file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return destination


def download_exp_data_if_not_exist(
        exp_data_url: typing.AnyStr,
        exp_data_destination: typing.AnyStr,
) -> typing.AnyStr:
    """
    Downloads & extract config/weights for a model if the provided destination does not exist.

    The provided URL will be assumed to be a Google Drive download URL. The download will be
    skipped entirely if the destination folder already exists. This function will return the
    path to the existing folder, or to the newly created folder.

    Args:
        exp_data_url: the zip URL (under the `https://drive.google.com/file/d/ID` format).
        exp_data_destination: where to extract the model data.

    Returns:
        The path to the model data.
    """
    assert exp_data_url.startswith("https://drive.google.com/file/d/")
    gdrive_file_id = exp_data_url.split("/")[-1]
    output_data_path = os.path.join(exp_data_destination, gdrive_file_id)
    downloaded_zip_path = os.path.join(exp_data_destination, f"{gdrive_file_id}.zip")
    if os.path.isfile(downloaded_zip_path) and os.path.isdir(output_data_path):
        return output_data_path
    os.makedirs(output_data_path, exist_ok=True)
    zip_path = download_file_from_google_drive(gdrive_file_id, downloaded_zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    return output_data_path


def extract_tracker_data(tracker, conf):
    """
    Get a dictionnary collecting interesting fields of the tracker and experimental settings

    Args:
        tracker (covid19sim.track.Tracker): Tracker toring simulation data
        conf (dict): Experimental Configuration

    returns:
        dict: the extracted data
    """
    timenow = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    data = dict()

    # scalars
    data['intervention_day'] = conf.get('INTERVENTION_DAY')
    data['intervention'] = conf.get('INTERVENTION')
    data['risk_model'] = conf.get('RISK_MODEL')
    data['COVID_SPREAD_START_TIME'] = conf['COVID_SPREAD_START_TIME']
    data['INTERVENTION_START_TIME'] = conf['INTERVENTION_START_TIME']
    data['SIMULATION_START_TIME'] = conf['SIMULATION_START_TIME']
    data['simulation_days'] = conf['simulation_days']
    data['n_humans'] = tracker.n_humans
    data['n_init_infected'] = tracker.n_infected_init
    data['adoption_rate'] = getattr(tracker, 'adoption_rate', 1.0)
    data['generation_times'] = tracker.compute_generation_time()
    data['p_transmission'] = tracker.compute_probability_of_transmission()

    # demographics
    data['age_histogram'] = tracker.city.age_histogram
    data['human_has_app'] = tracker.human_has_app
    data['known_connections'] = {
        human.name: set(h.name for h in human.known_connections)
        for human in tracker.city.humans
    } # 3MB

    # contacts
    data['contact_patterns'] = tracker.get_contact_data() # 0.7MB
    data['infectious_contact_patterns'] = tracker.get_infectious_contact_data() # 2MB
    data['expected_mobility'] = tracker.expected_mobility
    data['mobility'] = tracker.mobility
    data['infection_monitor'] = tracker.infection_monitor # 0.8MB
    data['outside_daily_contacts'] = tracker.outside_daily_contacts

    x, y, _ = tracker.compute_effective_contacts(since_intervention=True)
    data['effective_contacts_since_intervention'] = x
    data['healthy_effective_contacts_since_intervention'] = y

    x, y, _ = tracker.compute_effective_contacts(since_intervention=False)
    data['effective_contacts_all_days'] = x
    data['healthy_effective_contacts_all_days'] = y

    # spread related
    data['serial_interval'] = tracker.compute_serial_interval()
    data['all_serial_intervals'] = tracker.serial_intervals #0.1MB
    data['cases_per_day'] = tracker.cases_per_day
    data['ei_per_day'] = tracker.ei_per_day
    data['s'] = tracker.s_per_day
    data['e'] = tracker.e_per_day
    data['i'] = tracker.i_per_day
    data['r'] = tracker.r_per_day

    # epi related
    data['avg_infectiousness_per_day'] = tracker.avg_infectiousness_per_day
    data['covid_properties'] = tracker.covid_properties
    data['recovered_stats'] = tracker.recovery_stats
    data['symptoms'] = tracker.compute_symptom_prevalence()

    # testing related
    data['test_monitor'] = tracker.test_monitor #0.14MB

    # tracing related
    data['risk_precision_global'] = tracker.compute_risk_precision(False)
    data['risk_precision'] = tracker.risk_precision_daily
    data['human_monitor'] = tracker.human_monitor # 20MB
    data['infector_infectee_update_messages'] = tracker.infector_infectee_update_messages
    data['risk_attributes'] = tracker.risk_attributes # 524MB
    data['humans_state'] = tracker.humans_state #0.4MB
    data['humans_rec_level'] = tracker.humans_rec_level
    data['humans_intervention_level'] = tracker.humans_intervention_level
    data['to_human_max_msg_per_day'] = tracker.to_human_max_msg_per_day

    # behavior related
    data['daily_quarantine'] = tracker.daily_quarantine
    data['quarantine_monitor'] = tracker.quarantine_monitor
    data['humans_quarantined_state'] = tracker.humans_quarantined_state

    # economics
    data['work_hours'] = tracker.daily_work_hours_by_age_group

    return data


def dump_tracker_data(data, outdir, name):
    """
    Writes the tracker's extracted data to outdir/name using dill.

    /!\ there are know incompatibility issues between python 3.7 and 3.8 regarding the dump/loading of data with dill/pickle

    Creates the outputdir if need be, including potential missing parents.

    Args:
        data (dict): tracker's extracted data
        outdir (str): directory where to dump the file
        name (str): the dump file's name
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / name, 'wb') as f:
        dill.dump(data, f)

def parse_search_configuration(conf):
    """
    Parses the OmegaConf to native types

    Args:
        conf (OmegaConf): Hydra configuration

    Returns:
        dict: parsed conf
    """
    return OmegaConf.to_container(conf, resolve=True)


def parse_configuration(conf):
    """
    Transforms an Omegaconf object to native python dict, parsing specific fields like:
    "1-15" age bin in YAML file becomes (1, 15) tuple, and datetime is parsed from string.

    ANY key-specific parsing should have its inverse in covid19sim.utils.dump_conf()

    Args:
        conf (omegaconf.OmegaConf): Hydra-loaded configuration

    Returns:
        dict: parsed configuration to use in experiment
    """
    if isinstance(conf, (OmegaConf, DictConfig)):
        conf = OmegaConf.to_container(conf, resolve=True)
    elif not isinstance(conf, dict):
        raise ValueError("Unknown configuration type {}".format(type(conf)))

    if "GET_TESTED_SYMPTOMS_CHECKED_IN_HOSPITAL" in conf:
        conf["GET_TESTED_SYMPTOMS_CHECKED_IN_HOSPITAL"] = \
            [STR_TO_SYMPTOMS[symptom] for symptom in conf["GET_TESTED_SYMPTOMS_CHECKED_IN_HOSPITAL"]
             if symptom]

    if "GET_TESTED_SYMPTOMS_CHECKED_BY_SELF" in conf:
        conf["GET_TESTED_SYMPTOMS_CHECKED_BY_SELF"] = \
            [STR_TO_SYMPTOMS[symptom] for symptom in conf["GET_TESTED_SYMPTOMS_CHECKED_BY_SELF"]
             if symptom]

    if "start_time" in conf:
        conf["start_time"] = datetime.datetime.strptime(
            conf["start_time"], "%Y-%m-%d %H:%M:%S"
        )

    assert "RISK_MODEL" in conf and conf["RISK_MODEL"] is not None

    try:
        conf["GIT_COMMIT_HASH"] = get_git_revision_hash()
    except subprocess.CalledProcessError as e:
        print(">> Contained git error:")
        print(e)
        print(">> Ignoring git hash")
        conf["GIT_COMMIT_HASH"] = "NO_GIT"
    return conf


def dumps_conf(
        conf: dict,
):
    """
    Perform a deep copy of the configuration dictionary, preprocess the elements into strings
    to reverse the preprocessing performed by `parse_configuration`, returning the resulting dict.

    Args:
        conf (dict): configuration dictionary to be written in a file
    """

    copy_conf = copy.deepcopy(conf)

    if "GET_TESTED_SYMPTOMS_CHECKED_IN_HOSPITAL" in copy_conf:
        copy_conf["GET_TESTED_SYMPTOMS_CHECKED_IN_HOSPITAL"] = \
            [str(symptom) for symptom in copy_conf["GET_TESTED_SYMPTOMS_CHECKED_IN_HOSPITAL"]]

    if "GET_TESTED_SYMPTOMS_CHECKED_BY_SELF" in copy_conf:
        copy_conf["GET_TESTED_SYMPTOMS_CHECKED_BY_SELF"] = \
            [str(symptom) for symptom in copy_conf["GET_TESTED_SYMPTOMS_CHECKED_BY_SELF"]]

    if "start_time" in copy_conf:
        copy_conf["start_time"] = copy_conf["start_time"].strftime("%Y-%m-%d %H:%M:%S")
    if "COVID_SPREAD_START_TIME" in copy_conf:
        copy_conf["COVID_SPREAD_START_TIME"] = str(copy_conf.get("COVID_SPREAD_START_TIME", datetime.datetime.min))#.strftime("%Y-%m-%d %H:%M:%S")
    if "SIMULATION_START_TIME" in copy_conf:
        copy_conf["SIMULATION_START_TIME"] = str(copy_conf.get("SIMULATION_START_TIME", datetime.datetime.min))#.strftime("%Y-%m-%d %H:%M:%S")
    if "INTERVENTION_START_TIME" in copy_conf:
        copy_conf["INTERVENTION_START_TIME"] = str(copy_conf.get("INTERVENTION_START_TIME", datetime.datetime.min))#.strftime("%Y-%m-%d %H:%M:%S")

    return copy_conf


def dump_conf(
        conf: dict,
        path: typing.Union[str, Path],
):
    """
    Perform a deep copy of the configuration dictionary, preprocess the elements into strings
    to reverse the preprocessing performed by `parse_configuration` and then, dumps the content into a `.yaml` file.

    Args:
        conf (dict): configuration dictionary to be written in a file
        path (str | Path): `.yaml` file where the configuration is written
    """
    stringified_conf = dumps_conf(conf)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print("WARNING configuration already exists in {}. Overwriting.".format(
            str(path.parent)
        ))
    with path.open("w") as f:
        yaml.safe_dump(stringified_conf, f)


def relativefreq2absolutefreq(
        bins_fractions: dict,
        n_elements: int,
        rng,
) -> dict:
    """
    Convert relative frequencies to absolute frequencies such that the number of elements sum to n_entity.
    First, we assign `math.floor(fraction*n_entity)` to each bin and then, we assign the remaining elements randomly
    until we have `n_entity`.
    Args:
        bins_fractions (dict): each key is the bin description and each value is the relative frequency.
        n_elements (int): the total number of elements to assign.
        rng: a random generator for randomly assigning the remaining elements
    Returns:
        histogram (dict): each key is the bin description and each value is the absolute frequency.
    """
    histogram = {}
    for my_bin, fraction in bins_fractions.items():
        histogram[my_bin] = math.floor(fraction * n_elements)
    while np.sum(list(histogram.values())) < n_elements:
        bins = list(histogram.keys())
        random_bin = rng.choice(len(bins))
        histogram[bins[random_bin]] += 1

    assert np.sum(list(histogram.values())) == n_elements

    return histogram


def get_git_revision_hash():
    """Get current git hash the code is run from

    Returns:
        str: git hash
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

def get_test_false_negative_rate(test_type, days_since_exposure, conf, interpolate="step"):
    rates = conf['TEST_TYPES'][test_type]['P_FALSE_NEGATIVE']['rate']
    days = conf['TEST_TYPES'][test_type]['P_FALSE_NEGATIVE']['days_since_exposure']
    if interpolate == "step":
        for x, y in zip(days, rates):
            if days_since_exposure <= x:
                return y
        return y
    else:
        raise

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()

def zip_outdir(outdir):
    path = Path(outdir).resolve()
    assert path.exists()
    print(f"Zipping {outdir}...")
    start_time = time.time()
    command = "cd {}; zip -r -0 {}.zip {}".format(
        str(path.parent), path.name, path.name
    )
    subprocess_cmd(command)

def _random_choice(a_list, rng, size, P=None, replace=False, catch_size_error=False):
    """
    samples `size` random elements from `tuples` with probability `P`.
    NOTE: This function work arounds the internal conversion of the elements
            in `tuples` to np.ndarray.
    Args:
        tuples (list): a list of tuples
        rng (np.random.RandomState): Random number generator
        size (int): number of elements to sample from `tuples`
        P (list): probability with which to sample. Defaults to None.
        replace (bool): True if sampling is to be done with replace.
        catch_size_error (bool): uses permutation trick to prevent errors when length of a_list < size.

    Returns:
        list: sampled elements from tuples
    """
    total = len(a_list)
    if catch_size_error:
        idxs = rng.permutation(np.arange(total))[:size] # returns empty list if size > total
    else:
        idxs = rng.choice(range(total), size=size, p=P, replace=replace)

    return [a_list[x] for x in idxs]

def _get_seconds_since_midnight(time):
    """
    Returns:
        (float): number of seconds since midnight
    """
    return time.hour *  SECONDS_PER_HOUR + time.minute * SECONDS_PER_MINUTE + time.second

def normal_pdf(x, mean, std):
    proba = np.exp(-(((x - mean) ** 2) / (2 * std ** 2))) / (std * (2 * np.pi) ** 0.5)
    return proba


def copy_obj_array_except_env(array):
    """Copies a Human/City/Location object array, calling the child function below for each object."""
    if isinstance(array, dict):
        return {k: copy_obj_except_env(v) for k, v in array.items()}
    elif isinstance(array, list):
        return [copy_obj_except_env(v) for v in array]
    else:
        raise NotImplementedError


@dataclasses.dataclass(init=False)
class DummyEnv:
    """Dummy picklable constant version of the `Env` class."""
    initial_timestamp: datetime.datetime
    timestamp: datetime.datetime
    ts_initial: int
    now: int

    def __init__(self, env):
        self.initial_timestamp = env.initial_timestamp
        self.timestamp = env.timestamp
        self.ts_initial = env.ts_initial
        self.now = env.now

    def minutes(self):
        return self.timestamp.minute

    def hour_of_day(self):
        return self.timestamp.hour

    def day_of_week(self):
        return self.timestamp.weekday()

    def is_weekend(self):
        return self.day_of_week() >= 5

    def time_of_day(self):
        return self.timestamp.isoformat()


class DummyHuman:
    """Dummy picklable constant version of the `Human` class."""
    # note: since we're talking about a metric s*-ton of attributes, most will be dynamically added

    def __init__(self, human: "Human"):
        import pickle
        # "dummy" attributes replace the original attribute by a less-complex one
        self.dummy_attribs = [
            "env", "location", "household", "workplace", "last_date",
            "recommendations_to_follow", "recovered_timestamp", "intervened_behavior"# the old states contained in behaviors might break serialization
        ]
        self.env = DummyEnv(human.env)
        self.location = human.location.name if human.location else ""
        self.household = human.household.name if human.household else ""
        self.workplace = human.workplace.name if human.workplace else ""
        self.last_date = dict(human.last_date)
        self.recommendations_to_follow = [str(rec) for rec in human.recommendations_to_follow]
        # "blacklisted" attributes are overriden with `None`, no matter their original value
        self.blacklisted_attribs = [
            "conf", "city", "known_connections", "my_history", "visits", "proba_to_risk_level_map",  "mobility_planner"
        ]
        for attr_name in self.blacklisted_attribs:
            setattr(self, attr_name, None)
        # all other attributes will be copied as-is
        for attr_name in human.__dict__.keys():
            if attr_name not in self.dummy_attribs and \
                    attr_name not in self.blacklisted_attribs and \
                    not attr_name.startswith("__"):
                setattr(self, attr_name, getattr(human, attr_name))
        from covid19sim.native._native import BaseHuman
        for attr_name in BaseHuman.__dict__.keys():
            if attr_name not in self.dummy_attribs and \
                    attr_name not in self.blacklisted_attribs and \
                    not attr_name.startswith("__"):
                setattr(self, attr_name, getattr(human, attr_name))


def copy_obj_except_env(obj):
    """Copies a Human/City/Location object without its env part (which fails due to the generator)."""
    from covid19sim.human import Human
    from covid19sim.locations.location import Location
    from covid19sim.locations.city import City
    assert isinstance(obj, (Human, Location, City))
    if isinstance(obj, Human):
        return DummyHuman(obj)
    else:
        raise NotImplementedError


def get_approx_object_size_tree(obj, tree=None, seen_obj_ids=None):
    """Returns a tree structure that contains the approximate size of an object and its children.

    This function is recursive and will avoid loops, but it might still be very slow for large objects.

    TODO: make faster/more accurate with `gc.get_referents`?
    """
    if tree is None:
        tree = {None: 0}
    else:
        assert None in tree  # root cumulative size key, should always exist
    if seen_obj_ids is None:
        seen_obj_ids = set()
    obj_id = id(obj)
    if obj_id in seen_obj_ids:
        return tree
    seen_obj_ids.add(obj_id)
    obj_size = sys.getsizeof(obj)
    obj_name = textwrap.shorten(str(type(obj)) + ": " + str(obj), width=100) + " |> " + str(obj_id)
    assert obj_name not in tree
    tree[obj_name] = {None: obj_size}
    if isinstance(obj, dict):
        for attr_key, attr_val in obj.items():
            tree[obj_name][attr_key] = {None: 0}
            tree[obj_name][attr_key] = \
                get_approx_object_size_tree(attr_key, tree[obj_name][attr_key], seen_obj_ids)
            tree[obj_name][attr_key] = \
                get_approx_object_size_tree(attr_val, tree[obj_name][attr_key], seen_obj_ids)
            tree[obj_name][None] += tree[obj_name][attr_key][None]
    elif hasattr(obj, "__dict__"):
        for attr_key, attr_val in obj.__dict__.items():
            tree[obj_name][attr_key] = {None: 0}
            tree[obj_name][attr_key] = \
                get_approx_object_size_tree(attr_val, tree[obj_name][attr_key], seen_obj_ids)
            tree[obj_name][None] += tree[obj_name][attr_key][None]
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        for idx, attr_val in enumerate(obj):
            tree[obj_name][f"#{idx}"] = {None: 0}
            tree[obj_name][f"#{idx}"] = \
                get_approx_object_size_tree(attr_val, tree[obj_name][f"#{idx}"], seen_obj_ids)
            tree[obj_name][None] += tree[obj_name][f"#{idx}"][None]
    tree[None] += tree[obj_name][None]
    return tree


def get_approx_object_size(obj):
    """Returns the approximate size of an object and its children."""
    blacklisted_types = (type, types.ModuleType, types.FunctionType)
    if isinstance(obj, blacklisted_types):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, blacklisted_types) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = gc.get_referents(*need_referents)
    return size

def _convert_bin_5s_to_bin_10s(histogram_bin_5s):
    """
    Maps the count as per age ranges of 5 to age range of 10.

    Args:
        histogram_bin_5s (dict): keys are a tuple of (lower limit, upper limit) according to AGE_BIN_WIDTH_5 both inclusive and values are counts

    Returns:
        (dict): keys are a tuple of (lower limit, upper limit) according to AGE_BIN_WIDTH_10 both inclusive and values are counts
    """
    histogram_bin_10s = {}
    # combine two consecutive bins until the last one
    for i in range(0, len(histogram_bin_5s), 2):
        bin5_1 = AGE_BIN_WIDTH_5[i]
        bin5_2 = AGE_BIN_WIDTH_5[i+1]

        # AGE_BIN_WIDTH_5 have groupings of 5 years of age until the age of 74 and the last being 75 - 110
        # AGE_BIN_WIDTH_10 have groupings of 10 years of age all through i.e. 70-80
        # (assumption) break 75-110 to 75-79 and the rest assuming uniform distribution
        # 75-79 = 5/35th of 75-110
        c1 = histogram_bin_5s[bin5_1]
        if bin5_2[1] < 70:
            c2 = histogram_bin_5s[bin5_2]

            bin10 = (bin5_1[0], bin5_2[1])
            assert bin10 == AGE_BIN_WIDTH_10[i//2], f"not the right bin 10 {bin10}"
            histogram_bin_10s[bin10] = c1 + c2
        else:
            assert bin5_1[0] == 70, f"Not the right penultimate last bin {bin5_1}"

            bin10 = (bin5_1[0], 79)
            assert bin10 == AGE_BIN_WIDTH_10[i//2], f"not the right bin 10 {bin10}"
            c2 = histogram_bin_5s[bin5_2] * (5/35)
            histogram_bin_10s[bin10] = c1 + c2

            bin10 = (80, 110)
            assert bin10 == AGE_BIN_WIDTH_10[i//2+1], f"not the right bin 10 {bin10}"
            histogram_bin_10s[bin10] = c2 * 30 / 35
            break

    return histogram_bin_10s

def _sample_positive_normal(mean, sigma, rng, upper_limit=None):
    """
    Samples a positive number from gaussian distributed as (mean, sigma) by throwing away negative samples.

    Args:
        mean (float): mean of gaussian
        sigma (float): stdandard deviation of gaussian
        upper_limit (float): upper limit above which x will be rejected. None if there is no upper limit.

    Returns:
        (float): sample
    """
    _filter = lambda x: 0 <= x
    if upper_limit > 0:
        _filter = lambda x: 0 <= x <= upper_limit

    x = rng.normal(mean, sigma)
    return x if _filter(x) else _sample_positive_normal(mean, sigma, rng, upper_limit)

def is_app_based_tracing_intervention(intervention=None, intervention_conf=None):
    """
    Determines if the intervention requires an app.

    Args:
        intervention (str): name of the intervention that matches a configuration file in `configs/simulation/intervention`. Default is None.
        intervention_conf (dict): an experimental configuration. Default is None.

    Returns:
        (bool): True if an app is required.
    """
    assert intervention is not None or intervention_conf is not None, "Expects non-None intervention_conf when internvention is None. "
    if intervention_conf is not None:
        return intervention_conf['RISK_MODEL'] != ""

    intervention_yaml_file = Path(__file__).resolve().parent.parent / "configs/simulation/intervention" / f"{intervention}.yaml"
    with open(intervention_yaml_file, "r") as f:
        conf = yaml.safe_load(f)
        app_required = conf['RISK_MODEL'] != ""

    return app_required

class NpEncoder(json.JSONEncoder):
    """
    Class to convert `obj` into json encodable objects.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
