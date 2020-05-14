import os
import pickle

import numpy as np
import plotly.express as px

import covid19sim.config

pred_cluster_data_path = "output/clusters-500-naive"
ref_cluster_data_path = "output/clusters-500-perfect"

# parse all timeslot outputs
ref_pickle_paths = [os.path.join(ref_cluster_data_path, p)
                    for p in os.listdir(ref_cluster_data_path)]
pred_pickle_paths = [os.path.join(pred_cluster_data_path, p)
                     for p in os.listdir(pred_cluster_data_path)]
ref_pickle_paths.sort()
pred_pickle_paths.sort()

# the reference & target outputs should match in lengths + timestamps
assert len(pred_pickle_paths) == len(ref_pickle_paths)
assert all([
    os.path.basename(p1) == os.path.basename(p2)
    for p1, p2 in zip(pred_pickle_paths, ref_pickle_paths)
])

# we need to extract the data points we need for plotting
data_points = {}  # hash = alice_uid + timestamp + encounter_idx, val = dict of metrics stuff

# for each timestep (i.e. each time slot update in the simulation)
for time_step, (ref_pickle_path, pred_pickle_path) in \
        enumerate(zip(ref_pickle_paths, pred_pickle_paths)):

    # parse raw data structs (should be dicts where key = human, val = clustering manager)
    with open(ref_pickle_path, "rb") as fd:
        raw_ref_data = pickle.load(fd)
    with open(pred_pickle_path, "rb") as fd:
        raw_pred_data = pickle.load(fd)

    # for each human (alice), we will dissect its cluster manager to create data points
    for human_id, ref_cluster_mgr in raw_ref_data.items():
        ref_cluster_count = ref_cluster_mgr.get_cluster_count()
        if not ref_cluster_count:
            continue  # if the reference manager says there was no data, skip
        assert human_id in raw_pred_data  # humans in both tracks should always overlap
        pred_cluster_mgr = raw_pred_data[human_id]
        data_hash_base = str(human_id) + f"@t={time_step}"  # create a base hash for the data points
        # we will now extract all the encounters that alice had with various bobs
        ref_encounters = ref_cluster_mgr.get_encounters_cluster_mapping()
        pred_encounters = pred_cluster_mgr.get_encounters_cluster_mapping()
        already_encountered_bobs = []
        for encounter_idx, (encounter, cluster_id) in enumerate(pred_encounters):
            uid = encounter._sender_uid  # this is the real uid of the targeted bob
            # assert uid is not None  # TODO figure out why we get some messages with a missing real uid
            if uid is None or uid in already_encountered_bobs:
                continue
            # we will now create a set of all encounters for this bob (across all clusters)
            real_contacts = [e[0] for e in ref_encounters if e[0]._sender_uid == uid]
            real_contact_timestamps = set([e.encounter_time for e in real_contacts])
            # the length of the set is the first quantity we need for the metric
            # (i.e. how many recent encounters 'bob' has truly had with alice)
            real_contact_occurrences = len(real_contact_timestamps)
            # now let's find out how many times the clustering algo though we met that bob
            pred_cluster = next((c for c in pred_cluster_mgr.clusters if c.cluster_id == cluster_id), None)
            assert pred_cluster is not None
            pred_contact_occurrences = len(pred_cluster.get_timestamps())
            data_hash = data_hash_base + f"+encounter:{encounter_idx}"
            data_points[data_hash] = {
                "real_contact_occurrences": real_contact_occurrences,
                "pred_contact_occurrences": pred_contact_occurrences,
            }
            already_encountered_bobs.append(uid)

max_contact_count = max(max([
    max(p["real_contact_occurrences"], p["pred_contact_occurrences"])
    for p in data_points.values()
]), covid19sim.config.TRACING_N_DAYS_HISTORY)

heatmap_data = np.zeros((max_contact_count, max_contact_count), dtype=np.int32)
for p in data_points.values():
    heatmap_data[
        p["real_contact_occurrences"] - 1,
        p["pred_contact_occurrences"] - 1,
    ] += 1

fig = px.imshow(
    heatmap_data,
    labels=dict(
        x=f"Predicted number of days encountered over past {max_contact_count} days",
        y=f"Real number of days encountered over past {max_contact_count} days",
        color="Count"
    ),
    x=[str(n) for n in range(1, max_contact_count + 1)],
    y=[str(n) for n in range(1, max_contact_count + 1)],
)
fig.update_xaxes(side="top")
fig.show()
