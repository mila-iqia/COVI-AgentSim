"""
Handles querying the inference server with serialized humans and their messages.
"""

from collections import deque
import dataclasses
from datetime import datetime, timedelta
import os
import pickle
import functools
from joblib import Parallel, delayed
from typing import Deque, List
import warnings
import numpy as np

from covid19sim.frozen.clusters import Clusters
from covid19sim.frozen.helper import conditions_to_np, encode_age, encode_sex, \
    encode_test_result, symptoms_to_np
from covid19sim.frozen.utils import Message, UpdateMessage, encode_message, encode_update_message
from covid19sim.server_utils import InferenceClient, InferenceWorker
from ctt.inference.infer import InferenceEngine
from covid19sim.configs.exp_config import ExpConfig


def make_human_as_message(human):
    preexisting_conditions = conditions_to_np(human.preexisting_conditions)

    obs_preexisting_conditions = conditions_to_np(human.obs_preexisting_conditions)

    test_results = deque(((encode_test_result(result), timestamp)
                          for result, timestamp in human.test_results))

    rolling_all_symptoms = symptoms_to_np(human.rolling_all_symptoms)

    rolling_all_reported_symptoms = symptoms_to_np(human.rolling_all_reported_symptoms)

    messages = [encode_message(message) for message in human.contact_book.messages
                # match day; ugly till refactor
                if message[2] == human.contact_book.messages[-1][2]]

    update_messages = [encode_update_message(update_message)
                       for update_message in human.contact_book.update_messages
                       # match day; ugly till refactor
                       if update_message[3] == human.contact_book.update_messages[-1][3]]

    return HumanAsMessage(name=human.name,
                          age=encode_age(human.age),
                          sex=encode_sex(human.sex),
                          obs_age=encode_age(human.obs_age),
                          obs_sex=encode_sex(human.obs_sex),
                          preexisting_conditions=preexisting_conditions,
                          obs_preexisting_conditions=obs_preexisting_conditions,

                          infectiousnesses=human.infectiousnesses,
                          infection_timestamp=human.infection_timestamp,
                          recovered_timestamp=human.recovered_timestamp,
                          test_results=test_results,
                          rolling_all_symptoms=rolling_all_symptoms,
                          rolling_all_reported_symptoms=rolling_all_reported_symptoms,

                          clusters=human.clusters,
                          messages=messages,
                          exposure_message=human.exposure_message,
                          update_messages=update_messages,
                          carefulness=human.carefulness,
                          has_app=human.has_app)


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
    infectiousnesses: deque
    # TODO: Should be reformatted to int timestamp
    infection_timestamp: datetime
    # TODO: Should be reformatted to int timestamp
    recovered_timestamp: datetime
    # TODO: Should be reformatted to deque of (int, int timestamp)
    test_results: Deque[tuple]
    rolling_all_symptoms: np.array
    rolling_all_reported_symptoms: np.array

    # Risk fields
    clusters: Clusters
    messages: List[Message]
    exposure_message: Message
    update_messages: List[UpdateMessage]
    carefulness: float
    has_app: bool


def query_inference_server(params, **inf_client_kwargs):
    """
    [summary]

    Args:
        params ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Make a request to the server
    client = InferenceClient(**inf_client_kwargs)
    results = client.infer(params)
    return results


def integrated_risk_pred(humans, start, current_day, time_slot, port=6688, n_jobs=1, data_path=None):
    """
    [summary]
    Setup and make the calls to the server

    Args:
        humans ([type]): [description]
        start ([type]): [description]
        current_day ([type]): [description]
        time_slot ([type]): [description]
        port (int, optional): [description]. Defaults to 6688.
        n_jobs (int, optional): [description]. Defaults to 1.
        data_path ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    hd = humans[0].city.hd
    all_params = []

    current_time = (start + timedelta(days=current_day, hours=time_slot))

    # We're going to send a request to the server for each human
    for human in humans:
        if time_slot not in human.time_slots:
            continue

        human_message = make_human_as_message(human)

        log_path = None
        if data_path:
            log_path = f'{os.path.dirname(data_path)}/daily_outputs/{current_day}/{human.name[6:]}/'
        all_params.append({
            "start": start,
            "current_day": current_day,
            "COLLECT_TRAINING_DATA": ExpConfig.get('COLLECT_TRAINING_DATA'),
            "human": human_message,
            "log_path": log_path,
            "time_slot": time_slot,
            "risk_model": ExpConfig.get('RISK_MODEL'),
            "oracle": ExpConfig.get("USE_ORACLE")
        })
        human.contact_book.update_messages = []
        human.contact_book.messages = []

    if ExpConfig.get('USE_INFERENCE_SERVER'):
        batch_start_offset = 0
        batch_size = 300  # @@@@ TODO: make this a high-level configurable arg?
        batched_params = []
        while batch_start_offset < len(all_params):
            batch_end_offset = min(batch_start_offset + batch_size, len(all_params))
            batched_params.append(all_params[batch_start_offset:batch_end_offset])
            batch_start_offset += batch_size
        query_func = functools.partial(query_inference_server, target_port=port)
        with Parallel(n_jobs=n_jobs, batch_size=ExpConfig.get('MP_BATCHSIZE'), backend=ExpConfig.get('MP_BACKEND'), verbose=0, prefer="threads") as parallel:
            batched_results = parallel((delayed(query_func)(params) for params in batched_params))
        results = []
        for b in batched_results:
            results.extend(b)
    else:
        # recreating an engine every time should not be too expensive... right?
        engine = InferenceEngine(ExpConfig.get('TRANSFORMER_EXP_PATH'))
        results = InferenceWorker.process_sample(all_params, engine, ExpConfig.get('MP_BACKEND'), n_jobs)

    for result in results:
        if result is not None:
            name, risk_history, clusters = result
            if ExpConfig.get('RISK_MODEL') == "transformer":
                # TODO: Fix can be None. What should be done in this case
                if risk_history is not None:
                    # risk_history = np.clip(risk_history, 0., 1.)
                    for i in range(len(risk_history)):
                        hd[name].risk_history_map[current_day - i] = risk_history[i]
                    hd[name].update_risk_level()
                    for i in range(len(risk_history)):
                        hd[name].prev_risk_history_map[current_day - i] = risk_history[i]
                elif current_day != ExpConfig.get('INTERVENTION_DAY'):
                    warnings.warn(f"risk history is none for human:{name}", RuntimeWarning)
                hd[name].last_risk_update = current_time
            hd[name].clusters = clusters
            hd[name].last_cluster_update = current_time

    # print out the clusters
    if ExpConfig.get('DUMP_CLUSTERS'):
        os.makedirs(ExpConfig.get('DUMP_CLUSTERS'), exist_ok=True)
        curr_date_str = current_time.strftime("%Y%m%d-%H%M%S")
        curr_dump_path = os.path.join(ExpConfig.get('DUMP_CLUSTERS'), curr_date_str + ".pkl")
        to_dump = {human_id: human.clusters for human_id, human in hd.items()
                   if human.last_cluster_update == current_time}
        with open(curr_dump_path, "wb") as fd:
            pickle.dump(to_dump, fd)

    return humans
