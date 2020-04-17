import json
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from matplotlib import pyplot as plt
from utils import _decode_message
from collections import defaultdict, Counter
import networkx as nx
from models.helper import group_to_majority_id

np.random.seed(0)
""" Running this file will produce plots of the cluster statistics and sample graph"""


CLUSTER_ACC_PATH = "plots/cluster/hist_cluster_accuracies.png"
CLUSTER_PATH = "output/clusters.json"
CLUSTER_SIZE_PATH = "plots/cluster/cluster_size_hist.png"
CLUSTER_NUMBER_PATH = "plots/cluster/cluster_number_freq.png"
MESSAGE_NUMBER_PATH = "plots/cluster/message_number_freq.png"
INDIVIDUAL_CLUSTER_PATH = "plots/cluster/"
if not os.path.isdir( INDIVIDUAL_CLUSTER_PATH):
    os.mkdir(INDIVIDUAL_CLUSTER_PATH)

# load the cluster data
everyones_clustered_messages = json.load(open(CLUSTER_PATH, 'r'))

# gather some high level statistics about the clusters (how many groups, total and unique contacts)
all_groups = []
all_total_num_contacts = []
all_unique_people_contacted = []
for someones_clustered_messages in everyones_clustered_messages:
    groups = defaultdict(list)
    unique_people_contacted = set()
    total_num_contacts = 0
    for m_enc, assignment in someones_clustered_messages.items():
        obs_uid, obs_risk, m_sent, encounter_time, unobs_uid = _decode_message(m_enc)
        groups[assignment].append(unobs_uid)
        unique_people_contacted.add(unobs_uid)
        total_num_contacts += 1
    all_groups.append(dict(groups))
    all_unique_people_contacted.append(unique_people_contacted)
    all_total_num_contacts.append(total_num_contacts)


# count the number of people in each group
all_count_people_in_group = []
all_number_of_groups = [len(groups) for groups in all_groups]
for group in all_groups:
    count_people_in_group = []
    for g, ps in group.items():
        cnt = Counter()
        num_people_in_group = len(ps)
        for p in ps:
            cnt[p] += 1
        count_people_in_group.append(num_people_in_group)
    all_count_people_in_group.extend(count_people_in_group)

# plot the number of people in each group
plt.figure(1)
plt.hist(np.array(all_count_people_in_group).flatten(), 100, label='num_people_in_group')
plt.xlabel("Number of Messages in a Group")
plt.ylabel("Frequency Count of Messages")
plt.title("Histogram of Group Size")
plt.savefig(CLUSTER_SIZE_PATH)

# plot the number of groups per user
plt.figure(2)
plt.hist(np.array(all_number_of_groups).flatten(), 10, label='groups per user')
plt.hist(np.array([len(x) for x in all_unique_people_contacted]).flatten(), 10, label='num unique people actually contacted')
plt.legend()
plt.xlabel("number of contacts")
plt.ylabel("people with that number of contacts")
plt.title("Histogram of contacts and grouping frequency")
plt.savefig(CLUSTER_NUMBER_PATH)

# plot the total number of messages per person as a histogram
plt.figure(3)
plt.hist(np.array(all_total_num_contacts).flatten(), 10, label='messages per user')
plt.xlabel("number of messages")
plt.ylabel("people with that number of messages")
plt.title("Histogram of person <> message number frequency")
plt.savefig(MESSAGE_NUMBER_PATH)
plt.clf()

# helper function to create unique nodes for networkx
def hash_uid(group, uid, idx):
    return str(group) + "-" + str(uid) + "-" + str(idx)

# "rename" the groups. We need to figure out which group should be assigned to which true person in order to calculate an accuracy
all_groups = group_to_majority_id(all_groups)

# create and plot networkx graphs for the clusterings of individual's contact histories
all_group_accuracies = []
for group_idx, groups in enumerate(all_groups):
    G = nx.Graph()
    group_accuracies = []
    all_uids = set()
    for group, uids in groups.items():
        num_right = 0
        for idx, uid in enumerate(uids):
            G.add_node(hash_uid(group, uid, idx))
            all_uids.add(uid)
        for idx1, uid1 in enumerate(uids):
            if uid1 == group:
                num_right += 1
            for idx2, uid2 in enumerate(uids):
                G.add_edge(hash_uid(group, uid1, idx1), hash_uid(group, uid2, idx2))
        if len(uids) > 1:
            group_accuracies.append(num_right/len(uids))
    all_group_accuracies.extend(group_accuracies)

    if group_idx < 10:
        plt.figure(4+group_idx, figsize=(12,12))

        node_color = []
        for node in G.nodes():
            node_color.append(int(node.split("-")[1]))
        pos = nx.spring_layout(G, weight=.3, k=0.5, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color=node_color, cmap=plt.cm.hsv)
        plt.savefig(f"{INDIVIDUAL_CLUSTER_PATH}clusters_{group_idx}.png", dpi=300)


# plot a histogram of the population-level accuracy
plt.clf()
plt.figure(1000)
plt.hist(np.round(all_group_accuracies, 1), 30, label='group_accuracies')
plt.xlabel("clustering accuracy")
plt.ylabel("number of groups with that accuracy")
plt.title("Histogram of cluster assignment accuracies")
plt.savefig(CLUSTER_ACC_PATH)
plt.clf()
print(f"group_accuracy mean: {np.mean(all_group_accuracies)}")
print(f"group_accuracy std: {np.std(all_group_accuracies)}")


