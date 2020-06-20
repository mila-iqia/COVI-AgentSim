"""
Handles querying the inference server with serialized humans and their messages.
"""

import datetime
import os
import functools
import typing
from joblib import Parallel, delayed

from covid19sim.distributed_inference.server_utils import InferenceClient, InferenceEngineWrapper, proc_human_batch
from covid19sim.distributed_inference.clustering.base import ClusterManagerBase
from covid19sim.distributed_inference.human_as_message import make_human_as_message
if typing.TYPE_CHECKING:
    from covid19sim.human import Human
    from covid19sim.locations.city import SimulatorMailboxType


class DummyMemManager:
    """Dummy memory manager used when running in a single process."""

    global_cluster_map: typing.Dict[str, ClusterManagerBase] = {}
    global_inference_engine: InferenceEngineWrapper = None

    @classmethod
    def get_cluster_mgr_map(cls) -> typing.Dict[str, ClusterManagerBase]:
        return cls.global_cluster_map

    @classmethod
    def get_engine(cls, conf) -> InferenceEngineWrapper:
        if cls.global_inference_engine is None and not conf.get('USE_ORACLE', False):
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
        if not human.has_app or time_slot not in human.time_slots or human.is_dead:
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
