from pathlib import Path
import argparse
import pickle


def compute_early_warning(data) -> dict:
    pass  # TODO


def compute_true_false_quarantine(data) -> tuple:
    pass  # TODO


def compute_exposed_to_infectious_interval(data) -> float:
    pass  # TODO


def get_metrics(data) -> dict:
    # early warning: how much time after exposure put in non-green bucket
    early_warning_per_rec_level: dict = compute_early_warning(data)
    # red but not even exposed (not infected nor infectious) and red and truely infectious
    true_quarantine, false_quarantine = compute_true_false_quarantine(data)
    # was the rec_level increased between exposure and infectiousness (good) or not (bad)
    exposed_to_infectious_interval = compute_exposed_to_infectious_interval(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type="str", default=".", help="experimental directory")
    opts = parser.parse_args()

    path = Path(opts.path).resolve()
    model = path.parent.name

    runs = [d for d in path.iterdir() if d.is_dir()]

    for run in runs:
        data_path = list(run.glob("tracker*.pkl"))[0]
        with data_path.open("rb") as f:
            data = pickle.load(f)
            metrics: dict = get_metrics(data)
