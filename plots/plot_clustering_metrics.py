import collections
import os
import pickle

import pandas as pd
import plotly.express as px

SHOW_FULL_SCATTER = False  # toggle between scatter and density

cluster_data_paths = [
    # tuple of clustering algo name + actual pkl directory path
    ("blind", "output/clusters-500-blind"),
    ("naive", "output/clusters-500-naive"),
    ("perfect", "output/clusters-500-perfect"),
    ("simple", "output/clusters-500-simple"),
]

# will plot x = % error on cluster count, and y = homogeneity
# ...this means that top-left of 2d graph will show the best method

metrics_data = {}

# first, extract the raw metrics for all humans/timeslots
for algo_name, dir_path in cluster_data_paths:
    pickle_paths = [os.path.join(dir_path, p) for p in os.listdir(dir_path)]
    pickle_paths.sort()
    data_tracks = collections.defaultdict(list)  # one track = one human; these will evolve
    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as fd:
            raw_data = pickle.load(fd)
        for human_id, clusters in raw_data.items():
            cluster_count = clusters.get_cluster_count()
            if cluster_count:
                data_tracks[human_id].append({
                    "homogeneity": clusters._get_average_homogeneity(),
                    "abs_count_error": clusters._get_cluster_count_error(),
                    "cluster_count": cluster_count,
                })
            else:
                data_tracks[human_id].append({})
    metrics_data[algo_name] = data_tracks

# now, use the reference cluster count to determine the algo's relative count error, and
# prepare <x, y, t> points for display (only where gt is available!)
data_points = []
track_length = None
for algo_name, data_tracks in metrics_data.items():
    for human_id, stats_array in data_tracks.items():
        assert human_id in metrics_data["perfect"]
        assert len(metrics_data["perfect"][human_id]) == len(stats_array)
        if track_length is None:
            track_length = len(stats_array)
        else:
            assert track_length == len(stats_array)
        for sim_step, stats in enumerate(stats_array):
            if not stats:
                continue
            data_points.append({
                "Abs. Cluster # Error": stats["abs_count_error"],
                "Homogeneity": stats["homogeneity"],
                "Cluster count": stats["cluster_count"],
                "Simulation step": sim_step,  # would be nice to use date, but it's not exported...
                "Algorithm": algo_name,
                "Human": human_id,
            })

df = pd.DataFrame(data_points)
if SHOW_FULL_SCATTER:
    # will display all clustering results for all humans and timestamps...
    # remember: top-left is better
    fig = px.scatter(
        data_frame=df,
        x="Abs. Cluster # Error",
        y="Homogeneity",
        animation_frame="Simulation step",
        # animation_group="Algorithm",
        color="Algorithm",
        size="Cluster count",
        hover_name="Human",
        # size_max=100,
        range_x=[0, None],
        range_y=[0, 1],
    )
    fig["layout"].pop("updatemenus")  # optional, drop animation buttons
else:
    fig = px.density_contour(
        data_frame=df,
        x="Abs. Cluster # Error",
        y="Homogeneity",
        facet_col="Algorithm",
        range_x=[0, None],
        range_y=[0, 1],
    )
    fig.update_traces(
        contours_coloring="fill",
        contours_showlabels=True,
    )
fig.show()
