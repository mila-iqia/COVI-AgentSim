"""
Handles querying the inference server with serialized humans and their messages.
"""

import collections
import dataclasses
import datetime
import os
import functools
import typing
from joblib import Parallel, delayed

import numpy as np

from covid19sim.server_utils import InferenceClient, InferenceEngineWrapper, proc_human_batch
from covid19sim.frozen.helper import conditions_to_np, encode_age, encode_sex, \
    encode_test_result, symptoms_to_np
from covid19sim.frozen.message_utils import UpdateMessage
from covid19sim.frozen.clustering.base import ClusterManagerBase

if typing.TYPE_CHECKING:
    from covid19sim.simulator import Human
    from covid19sim.base import SimulatorMailboxType, PersonalMailboxType


def make_human_as_message(
        human: "Human",
        personal_mailbox: "PersonalMailboxType",
        conf: typing.Dict,
):
    """
    Creates a human dataclass from a super-ultra-way-too-heavy human object.

    The returned dataclass can more properly be serialized and sent to a remote server for
    clustering/inference. All update messages aimed at the given human found in the global
    mailbox will be popped and added to the dataclass.

    Args:
        human: the human object to convert.
        personal_mailbox: the personal mailbox dictionary to fetch update messages from.
        conf: YAML configuration dictionary with all relevant settings for the simulation.
    Returns:
        The nice-and-slim human dataclass object.
    """
    preexisting_conditions = conditions_to_np(human.preexisting_conditions)
    obs_preexisting_conditions = conditions_to_np(human.obs_preexisting_conditions)
    rolling_all_symptoms = symptoms_to_np(human.rolling_all_symptoms, conf)
    rolling_all_reported_symptoms = symptoms_to_np(human.rolling_all_reported_symptoms, conf)

    # TODO: we could index the global mailbox by day, it might be faster that way
    target_mailbox_keys = [key for keys in human.contact_book.mailbox_keys_by_day.values() for key in keys]
    update_messages = []
    for key in target_mailbox_keys:
        if key in personal_mailbox:
            assert isinstance(personal_mailbox[key], list)
            update_messages.extend(personal_mailbox.pop(key))

    return HumanAsMessage(
        name=human.name,
        age=encode_age(human.age),
        sex=encode_sex(human.sex),
        obs_age=encode_age(human.obs_age),
        obs_sex=encode_sex(human.obs_sex),
        preexisting_conditions=preexisting_conditions,
        obs_preexisting_conditions=obs_preexisting_conditions,

        infectiousnesses=human.infectiousnesses,
        infection_timestamp=human.infection_timestamp,
        recovered_timestamp=human.recovered_timestamp,
        test_results=human.get_test_results_array(human.env.timestamp),
        rolling_all_symptoms=rolling_all_symptoms,
        rolling_all_reported_symptoms=rolling_all_reported_symptoms,
        incubation_days=human.incubation_days,
        recovery_days=human.recovery_days,

        update_messages=update_messages,
        carefulness=human.carefulness,
        has_app=human.has_app
    )


@dataclasses.dataclass
class HumanAsMessage:
    # Static fields
    name: str
    age: int
    sex: str
    obs_age: int
    obs_sex: str
    preexisting_conditions: np.array
    obs_preexisting_conditions: np.array

    # Medical fields
    infectiousnesses: typing.Iterable
    # TODO: Should be reformatted to int timestamp
    infection_timestamp: datetime.datetime
    # TODO: Should be reformatted to int timestamp
    recovered_timestamp: datetime.datetime
    # TODO: Should be reformatted to deque of (int, int timestamp)
    test_results: collections.deque
    rolling_all_symptoms: np.array
    rolling_all_reported_symptoms: np.array
    incubation_days: int  # NOTE: FOR NOW, USED FOR TESTING/DEBUGGING ONLY
    recovery_days: int  # NOTE: FOR NOW, USED FOR TESTING/DEBUGGING ONLY

    # Risk-level-related fields
    update_messages: typing.List[UpdateMessage]
    carefulness: float
    has_app: bool


class DummyMemManager:
    """Dummy memory manager used when running in a single process."""

    global_cluster_map: typing.Dict[str, ClusterManagerBase] = {}
    global_inference_engine: InferenceEngineWrapper = None

    @classmethod
    def get_cluster_mgr_map(cls) -> typing.Dict[str, ClusterManagerBase]:
        return cls.global_cluster_map

    @classmethod
    def get_engine(cls, conf) -> InferenceEngineWrapper:
        if cls.global_inference_engine is None:
            cls.global_inference_engine = InferenceEngineWrapper(conf.get('TRANSFORMER_EXP_PATH'))
        return cls.global_inference_engine


def batch_run_timeslot_heavy_jobs(
        humans: typing.Iterable["Human"],
        init_timestamp: datetime.datetime,
        current_timestamp: datetime.datetime,
        global_mailbox: "SimulatorMailboxType",
        time_slot: int,
        conf: typing.Dict,
        data_path: typing.Optional[typing.AnyStr] = None,
        city_hash: int = 0,
) -> typing.Iterable["Human"]:
    """
    Runs the 'heavy' processes that must occur for all users in parallel.

    The heavy stuff here is the clustering and risk level inference using a 3rd party model.
    These steps can be delegated to a remote server if the simulator is configured that way.

    Args:
        humans: the list of all humans in the zone.
        init_timestamp: initialization timestamp of the simulation.
        current_timestamp: the current timestamp of the simulation.
        global_mailbox: the global mailbox dictionary used to fetch already-existing updates from.
            Note that messages can be removed from this mailbox in this function, but not added, as
            that is delegated to the caller (see the return values).
        time_slot: the current timeslot of the day (i.e. an integer that corresponds to the hour).
        conf: YAML configuration dictionary with all relevant settings for the simulation.
        data_path: Root path where to save the 'daily outputs', i.e. the training data for ML models.
        city_hash: a hash used to tag this city's humans on an inference server that may be used by
            multiple cities in parallel. Bad mojo will happen if two cities have the same hash...
    Returns:
        A tuple consisting of the updated humans & of the newly generated update messages to register.
    """
    current_day_idx = (current_timestamp - init_timestamp).days
    assert current_day_idx >= 0

    hd = next(iter(humans)).city.hd
    all_params = []

    for human in humans:
        if time_slot not in human.time_slots:
            continue

        log_path = f"{os.path.dirname(data_path)}/daily_outputs/{current_day_idx}/{human.name[6:]}/" \
            if data_path else None
        all_params.append({
            "start": init_timestamp,
            "current_day": current_day_idx,
            "human": make_human_as_message(
                human=human,
                personal_mailbox=global_mailbox[human.name],
                conf=conf
            ),
            "log_path": log_path,
            "time_slot": time_slot,
            "conf": conf,
            "city_hash": city_hash,
        })

    if conf.get('USE_INFERENCE_SERVER'):
        batch_start_offset = 0
        batch_size = conf.get('INFERENCE_REQ_BATCH_SIZE', 100)
        batched_params = []
        while batch_start_offset < len(all_params):
            batch_end_offset = min(batch_start_offset + batch_size, len(all_params))
            batched_params.append(all_params[batch_start_offset:batch_end_offset])
            batch_start_offset += batch_size
        parallel_reqs = conf.get('INFERENCE_REQ_PARALLEL_JOBS', 16)
        parallel_reqs = max(min(parallel_reqs, len(batched_params)), 1)

        def query_inference_server(params, **inf_client_kwargs):
            # lambda used to create one socket per request (so we can request in parallel)
            client = InferenceClient(**inf_client_kwargs)
            return client.infer(params)

        inference_frontend_address = conf.get('INFERENCE_SERVER_ADDRESS', None)
        query_func = functools.partial(query_inference_server, server_address=inference_frontend_address)

        with Parallel(n_jobs=parallel_reqs, prefer="threads") as parallel:
            batched_results = parallel((delayed(query_func)(params) for params in batched_params))
        results = []
        for b in batched_results:
            results.extend(b)
    else:
        cluster_mgr_map = DummyMemManager.get_cluster_mgr_map()
        engine = DummyMemManager.get_engine(conf)
        results = proc_human_batch(all_params, engine, cluster_mgr_map)

    for name, risk_history in results:
        human = hd[name]
        if conf.get('RISK_MODEL') == "transformer":
            if risk_history is not None:
                human.apply_transformer_risk_updates(
                    current_day_idx=current_day_idx,
                    risk_history=risk_history,
                )
    return humans
