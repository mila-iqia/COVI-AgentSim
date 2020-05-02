import os
import json
import numpy as np
import functools
from joblib import Parallel, delayed

from covid19sim.frozen.inference_client import InferenceClient
from covid19sim.frozen.utils import update_uid
from covid19sim import config

# load the risk map
# TODO: load this from config (?)
risk_map = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/../frozen/log_risk_mapping.npy")
risk_map[0] = np.log(0.01)


def query_inference_server(params, **inf_client_kwargs):
    # Make a request to the server
    client = InferenceClient(**inf_client_kwargs)
    results = client.infer(params)
    return results


def integrated_risk_pred(humans, start, current_day, all_possible_symptoms, port=6688, n_jobs=1, data_path=None):
    """ Setup and make the calls to the server"""
    hd = humans[0].city.hd
    all_params = []

    # We're going to send a request to the server for each human
    for human in humans:
        log_path = None
        if data_path:
            log_path = f'{os.path.dirname(data_path)}/daily_outputs/{current_day}/{human.name[6:]}/'

        all_params.append({
            "start": start,
            "current_day": current_day,
            "all_possible_symptoms": all_possible_symptoms,
            "human": human.__getstate__(),
            "COLLECT_TRAINING_DATA": config.COLLECT_TRAINING_DATA,
            "log_path": log_path,
            "risk_model": config.RISK_MODEL,
        })
        human.uid = update_uid(human.uid, human.rng)

    # Batch the parameters for the function calls
    batch_start_offset = 0
    batch_size = 25  # @@@@ TODO: make this a high-level configurable arg?
    batched_params = []
    while batch_start_offset < len(all_params):
        batch_end_offset = min(batch_start_offset + batch_size, len(all_params))
        batched_params.append(all_params[batch_start_offset:batch_end_offset])
        batch_start_offset += batch_size
    query_func = functools.partial(query_inference_server, target_port=port)

    # make the batched requests to the server
    with Parallel(n_jobs=n_jobs, batch_size=config.MP_BATCHSIZE, backend=config.MP_BACKEND, verbose=0, prefer="threads") as parallel:
        batched_results = parallel((delayed(query_func)(params) for params in batched_params))

    # handle the results
    results = []
    for b in batched_results:
        results.extend(b)

    for result in results:
        if result is not None:
            name, risk_history, clusters = result

            if config.RISK_MODEL == "transformer":

                hd[name].prev_risk_history = hd[name].risk_history
                hd[name].risk_history = risk_history
                hd[name].update_risk_level()

            hd[name].clusters = clusters
            hd[name].contact_book.update_messages = []

    # print out the clusters
    if config.DUMP_CLUSTERS:
        clusters = []
        for human in hd.values():
            clusters.append(dict(human.clusters.clusters))
        json.dump(clusters, open(config.CLUSTER_PATH, 'w'))
    return humans
