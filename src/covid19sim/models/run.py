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

    current_time = (start + timedelta(days=current_day, hours=time_slot))
    current_date = current_time.date()

    # We're going to send a request to the server for each human
    for human in humans:
        if time_slot not in human.time_slots:
            continue

        human_state = human.get_message_dict()
        if human.last_date['run'] != current_date:
            infectiousnesses = copy.copy(human_state["infectiousnesses"])
            # Pad missing days
            # TODO: Reduce the current date by 1 hour since human with time slot at hour 0
            #  did not had the time to update. Update the human data at the same time it is
            #  sent to the inference server would properly fix this
            pad_count = ((current_time + timedelta(hours=-1)).date() - human.last_date['run']).days
            for day in range(pad_count):
                infectiousnesses.appendleft(0)
            human_state["infectiousnesses"] = infectiousnesses
            warnings.warn(f"{human.name} is outdated. Padding infectiousnesses array with {pad_count} zeros. "
                          f"Current time {current_time}, last_date['run'] {human.last_date['run']}",
                          RuntimeWarning)

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
            "time_slot": time_slot,
            "risk_model": config.RISK_MODEL,
        })
        human.contact_book.update_messages = []
        human.contact_book.messages = []

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

    for result in results:
        if result is not None:
            name, risk_history, clusters = result
            human = hd[name]
            if config.RISK_MODEL == "transformer":
                # TODO: Fix can be None. What should be done in this case
                if risk_history is not None:
                    for i in range(len(risk_history)):
                        human.risk_history_map[current_day - i] = risk_history[i]
                    human.update_risk_level()
                    for i in range(len(risk_history)):
                        human.prev_risk_history_map[current_day - i] = risk_history[i]
                else:
                    warnings.warn(f"risk_history is None for human {name}", RuntimeWarning)
                human.last_risk_update = current_day
            human.clusters = clusters

    # print out the clusters
    if config.DUMP_CLUSTERS:
        clusters = []
        for human in hd.values():
            clusters.append(dict(human.clusters.clusters))
        json.dump(clusters, open(config.CLUSTER_PATH, 'w'))
    return humans
