import argparse
import pickle
import os
import datetime
from pathlib import Path
import covid19sim.plotting.plot_infection_chains as infection_chains
from covid19sim.plotting.baseball_cards import DebugDataLoader, generate_debug_plots

def main(data_path, output_path, num_chains=10):

    data_path = Path(data_path).resolve()
    assert data_path.exists()

    # Setup paths
    human_backups_path = os.path.join(data_path, "human_backups.hdf5")
    tracker_path = next(data_path.glob("tracker*.pkl"))

    with open(tracker_path, "rb") as f:
        pkl = pickle.load(f)

    init_infected = [x['name'] for x in pkl['human_monitor'][datetime.date(2020, 2, 28)] if x['infection_timestamp'] == datetime.datetime(2020, 2, 28, 0, 0)]
    ids = infection_chains.plot(pkl, output_path, num_chains=num_chains, init_infected=init_infected)

    print(ids)
    # Baseball Cards
    data_loader = DebugDataLoader(human_backups_path)
    generate_debug_plots(data_loader, output_path, ids=ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_data")
    parser.add_argument("--output_folder")
    args = parser.parse_args()

    main(args.debug_data, args.output_folder)
