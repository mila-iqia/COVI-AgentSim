import pickle

d = "sim_v2_people-1000_days-30_init-0.001_uptake--1.0_seed-0_20201111-122550_660171"
trackers = []
for i in range(4):
    path = f"output/{d}/tracker_data_n_1000_seed_0_20201111-122614_{i}.pkl"
    tracker = pickle.load(open(path, "rb"))
    trackers.append(tracker)

print(trackers[0].keys())
import pdb; pdb.set_trace()
