import json
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from matplotlib import pyplot as plt
from utils import _decode_message
from collections import defaultdict, Counter

contact_histories = json.load(open('contact_histories.json', 'r'))
all_groups = []
all_total_num_contacts = []
all_unique_people_contacted = []

for history in contact_histories:
    groups = defaultdict(list)
    unique_people_contacted = set()
    total_num_contacts = 0
    for message, group in history.items():
        obs_uid, obs_risk, m_sent, unobs_uid = _decode_message(message)
        groups[group].append(unobs_uid)
        unique_people_contacted.add(unobs_uid)
        total_num_contacts += 1
    all_groups.append(dict(groups))
    all_unique_people_contacted.append(unique_people_contacted)
    all_total_num_contacts.append(total_num_contacts)
print(all_unique_people_contacted)
print(all_groups)
print(all_total_num_contacts)

all_count_people_in_group = []
all_within_group_percentage = []
all_number_of_groups = [len(groups) for groups in all_groups]

for group in all_groups:
    count_people_in_group = []
    within_group_percentage = []
    for g, ps in group.items():
        cnt = Counter()
        num_people_in_group = len(ps)
        for p in ps:
            cnt[p] += 1
        count_people_in_group.append(num_people_in_group)
        within_group_percentage.append(cnt.most_common()[0][1]/num_people_in_group)
    all_within_group_percentage.extend(within_group_percentage)
    all_count_people_in_group.extend(count_people_in_group)
# import pdb; pdb.set_trace()

fig, ax = plt.subplots(1, 1)
plt.hist(np.array(all_count_people_in_group).flatten(), 100, label='num_people_in_group')
plt.xlabel("Number of Messages in a Group")
plt.ylabel("Frequency Count of Messages")
plt.title("Histogram of Group Size")
plt.savefig("plots/group_size_hist.png")

fig, ax = plt.subplots(1, 1)
plt.hist(np.array(all_within_group_percentage).flatten(), 50, label='unobs message ids in group')
plt.xlabel("% of messages from most frequent unobs id")
plt.ylabel("Frequency Count of Outcome")
plt.title("Histogram of Group Dominant unobs id")
plt.savefig("plots/within_group_percentage.png")

fig, ax = plt.subplots(1, 1)
plt.hist(np.array(all_number_of_groups).flatten(), 10, label='groups per user')
plt.xlabel("number of groups")
plt.ylabel("people with that number of groups")
plt.title("Histogram of person <> group number frequency")
plt.savefig("plots/group_number_freq.png")

fig, ax = plt.subplots(1, 1)
plt.hist(np.array(all_total_num_contacts).flatten(), 10, label='messages per user')
plt.xlabel("number of messages")
plt.ylabel("people with that number of messages")
plt.title("Histogram of person <> message number frequency")
plt.savefig("plots/message_number_freq.png")
