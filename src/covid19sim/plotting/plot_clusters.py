"""
Running this file will produce plots of the cluster statistics and sample graph
"""

import json
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from covid19sim.frozen.utils import decode_message
from collections import defaultdict, Counter
import networkx as nx
import plotly.graph_objects as go
from collections import namedtuple


np.random.seed(0)

CLUSTER_ACC_PATH = "plots/cluster/hist_cluster_accuracies.png"
CLUSTER_PATH = "output/clusters.json"
CLUSTER_SIZE_PATH = "plots/cluster/cluster_size_hist.png"
CLUSTER_NUMBER_PATH = "plots/cluster/cluster_number_freq.png"
MESSAGE_NUMBER_PATH = "plots/cluster/message_number_freq.png"
INDIVIDUAL_CLUSTER_PATH = "plots/cluster/"

if not os.path.isdir(INDIVIDUAL_CLUSTER_PATH):
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
    for assignment, m_encs in someones_clustered_messages.items():
        for m_enc in m_encs:
            obs_uid, obs_risk, m_sent, unobs_uid = decode_message(m_enc)
            DebugMessage = namedtuple("DebugMessage", "obs_uid obs_risk m_sent unobs_id assignment")
            groups[assignment].append(DebugMessage(obs_uid, obs_risk, m_sent, unobs_uid, assignment))
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
matplotlib.use('Agg')

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

# create and plot networkx graphs for the clusterings of individual's contact histories
for group_idx, groups in enumerate(all_groups):
    G = nx.Graph()

    for group_id, messages in groups.items():
        num_right = 0
        # Add all the nodes
        for idx, message in enumerate(messages):
            G.add_node(message)

        # Add all the edges
        for idx1, message1 in enumerate(messages):
            for idx2, message2 in enumerate(messages):
                G.add_edge(message1, message2)

    if group_idx < 3:
        plt.figure(4+group_idx, figsize=(12,12))

        # Add all the edges
        for idx1, message1 in enumerate(messages):
            uid1 = message1.unobs_id.split(":")[1]
            for idx2, message2 in enumerate(messages):
                uid2 = message2.unobs_id.split(":")[1]
                G.add_edge(message1, message2)

    if group_idx < 3:
        plt.figure(4+group_idx, figsize=(12,12))
        node_color = []
        for node in G.nodes():
            node_color.append(int(node.unobs_id.split(":")[1]))
        pos = nx.spring_layout(G, weight=.3, k=0.5, iterations=50)
        # nx.draw(G, pos, with_labels=True, node_color=node_color, cmap=plt.cm.hsv)
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append('# of connections: ' + str(len(adjacencies[1])))

        node_trace.marker.color = node_color
        node_trace.text = node_text
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Network graph made with Python',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()

        # plt.savefig(f"{INDIVIDUAL_CLUSTER_PATH}clusters_{group_idx}.png", dpi=300)
