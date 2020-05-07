import copy
from datetime import timedelta
import os
import json
import numpy as np
import functools
from joblib import Parallel, delayed
import warnings

from covid19sim.server_utils import InferenceClient, InferenceWorker
from covid19sim import config
from ctt.inference.infer import InferenceEngine


# load the risk map
risk_map = np.array(config.RISK_MAPPING)


def query_inference_server(params, **inf_client_kwargs):
    # Make a request to the server
    client = InferenceClient(**inf_client_kwargs)
    results = client.infer(params)
    return results


def integrated_risk_pred(humans, start, current_day, time_slot, all_possible_symptoms, port=6688, n_jobs=1, data_path=None):
    """ Setup and make the calls to the server"""
    hd = humans[0].city.hd
    all_params = []

    current_date = (start + timedelta(days=current_day)).date()

    # We're going to send a request to the server for each human
    for human in humans:
        human_state = human.__getstate__()
        if human.last_date['run'] != current_date:
            infectiousnesses = copy.copy(human_state["infectiousnesses"])
            # Pad missing days
            for day in range((current_date - human.last_date['run']).days):
                infectiousnesses.appendleft(0)
            human_state["infectiousnesses"] = infectiousnesses
            warnings.warn(f"Human is outdated {human.name}. Current date {current_date}, "
                          f"last_date['run'] {human.last_date['run']}",
                          RuntimeWarning)

        if time_slot not in human.time_slots:
            continue
        log_path = None
        if data_path:
            log_path = f'{os.path.dirname(data_path)}/daily_outputs/{current_day}/{human.name[6:]}/'
        all_params.append({
            "start": start,
            "current_day": current_day,
            "all_possible_symptoms": all_possible_symptoms,
            "human": human_state,
            "COLLECT_TRAINING_DATA": config.COLLECT_TRAINING_DATA,
            "log_path": log_path,
            "risk_model": config.RISK_MODEL,
        })

    if config.USE_INFERENCE_SERVER:
        batch_start_offset = 0
        batch_size = 25  # @@@@ TODO: make this a high-level configurable arg?
        batched_params = []
        while batch_start_offset < len(all_params):
            batch_end_offset = min(batch_start_offset + batch_size, len(all_params))
            batched_params.append(all_params[batch_start_offset:batch_end_offset])
            batch_start_offset += batch_size
        query_func = functools.partial(query_inference_server, target_port=port)
        with Parallel(n_jobs=n_jobs, batch_size=config.MP_BATCHSIZE, backend=config.MP_BACKEND, verbose=0, prefer="threads") as parallel:
            batched_results = parallel((delayed(query_func)(params) for params in batched_params))
        results = []
        for b in batched_results:
            results.extend(b)
    else:
        # recreating an engine every time should not be too expensive... right?
        engine = InferenceEngine(config.TRANSFORMER_EXP_PATH)
        results = InferenceWorker.process_sample(all_params, engine, config.MP_BACKEND, n_jobs)

    if config.RISK_MODEL != "transformer":
        for result in results:
            if result is not None:
                name, risk_history, clusters = result
                hd[name].clusters = clusters
                hd[name].last_risk_update = current_day
                hd[name].contact_book.update_messages = []
                hd[name].contact_book.messages = []

        return humans

    for result in results:
        if result is not None:
            name, risk_history, clusters = result

            # TODO: Fix can be None. What should be done in this case
            if risk_history is not None:
                for i in range(config.TRACING_N_DAYS_HISTORY):
                    hd[name].risk_history_map[current_day - i] = risk_history[i]

                hd[name].update_risk_level()

                for i in range(config.TRACING_N_DAYS_HISTORY):
                    hd[name].prev_risk_history_map[current_day - i] = risk_history[i]
            else:
                warnings.warn(f"risk history is none for human:{name}", RuntimeWarning)

            hd[name].clusters = clusters
            hd[name].last_risk_update = current_day
            hd[name].contact_book.update_messages = []
            hd[name].contact_book.messages = []

    # print out the clusters
    if config.DUMP_CLUSTERS:
        clusters = []
        for human in hd.values():
            clusters.append(dict(human.clusters.clusters))
        json.dump(clusters, open(config.CLUSTER_PATH, 'w'))
    return humans
