import networkx as nx
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def plot_intervention(filenames, title, percent_threshold=1.0, end_day=None, colormap=None, ax=None):
    cases_data, mobility_data, labels, mobility_label = [], [], [], []

    if type(filenames[0]) == list:
        assert "unmitigated" in filenames[0][0]
        filename_0 = filenames[0][0]
    else:
        assert "unmitigated" in filenames[0]
        filename_0 = filenames[0]

    # unmitigated filenames
    data = pickle.load(open(filename_0, "rb"))
    n_init_infected = data['n_init_infected']
    n_humans = data['n_humans']
    cum_cases_per_day = np.cumsum(data['cases_per_day'])
    cum_cases_per_day += n_init_infected
    percent_infected = cum_cases_per_day / n_humans
    start_idx = np.where(percent_infected >= 0.00)[0][0]

    if percent_threshold < 1:
        print("keeping the termination criterion as percent_threshold")
        end_idx = np.where(percent_infected <= percent_threshold)[0][-1]

    if end_day:
        print("keeping the termination criterion as end_day")
        end_idx = end_day

    cases_data.append(percent_infected[start_idx:end_idx+1].tolist())
    labels.append("unmitigated")

    if colormap is None:
        colormap = ['red', 'orange', 'blue', 'green', 'gray']

    i_days = set()
    for x in filenames[1:]:
        if type(x) == list:
            label = x[1]
            data = pickle.load(open(x[0], "rb"))
        else:
            label = x[:-3].split("_")[-1].split(".")[0]
            data = pickle.load(open(x, "rb"))

        cum_cases_per_day = np.cumsum(data['cases_per_day'])
        cum_cases_per_day += n_init_infected
        percent_infected = cum_cases_per_day / n_humans

        cases_data.append(percent_infected[start_idx:end_idx+1].tolist())
        labels.append(label)

        # mobility
        m = np.array(data['expected_mobility'])[start_idx:end_idx+1]
        m = m.tolist()
        mobility_data.append(m)
        mobility_label.append(label)
        i_days.add(data.get('intervention_day', -1))

    assert len(i_days) == 1, "found different intervention days..."
    intervention_day = i_days.pop()

    # make all equal
    data = cases_data
    if percent_threshold == 1.0:
        max_length = max(len(x) for x in data)
        data = [x + [x[-1]]*(max_length - len(x)) for x in data]

    data = pd.DataFrame(data).transpose()
    data.columns = labels
    data.index += start_idx
    color = colormap[:len(filenames)]

    # mobility
    mobility = pd.DataFrame(mobility_data).transpose()
    mobility.columns = mobility_label
    mobility.index += 1

    # ref curve : doubling rate of 3 days
    y_vals = [1.0*min(n_init_infected * pow(2, y/3), n_humans)/n_humans for y in range(0, len(data))]

    # plot
    ax = data.plot(figsize=(20,10), color = color, linewidth=2, ax=ax)
    mobility.plot(linestyle='--', color = color[1:], ax = ax, alpha=0.3, linewidth=2)
    if intervention_day >= 0:
        line = ax.axvline(x=intervention_day, linestyle="-.", linewidth=3)
        ax.annotate("Intervention", xy=(intervention_day, 0.5), xytext=(intervention_day-1.5, 0.6), size=30, rotation="vertical")
    plt.plot(range(1, len(y_vals)+1), y_vals, '-.', color="gray", alpha=0.3)

    ax.set_title(title, size=25)
    ax.grid(True, axis='x', alpha=0.3)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tick_params(labelsize=25)
    plt.xlabel("Days since outbreak", fontsize=30)
    plt.ylabel("% Infected / Mobile", fontsize=30, labelpad=20)

    # legend
    legends = ax.get_legend_handles_labels()
    legends, legend_labels = legends[0][:len(labels)], legends[1][:len(labels)]
    line1 = Line2D([0], [0], color="black", linewidth=2, linestyle='--', label="mobility")
    line2 = Line2D([0], [0], color="black", linewidth=2, linestyle='-', label="% infected")
    legends = (legends + [line1, line2], legend_labels + ["mobility", "% infected"])
    # plt.legend(legends + [line1, line2], legend_labels + ["mobility", "% infected"], prop={"size":20}, loc="upper left")

    return (ax,legends), (data, mobility, labels)
