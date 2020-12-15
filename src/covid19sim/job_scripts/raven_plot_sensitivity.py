import pathlib
from raven.core import RavenJob


PLOTTING_SCRIPT = (
    "/lustre/home/nrahaman/python/covi-simulator/src/covid19sim/plotting/main.py"
)
ARMED = False


def run(sensitivity_dir):
    sensitivity_dir = pathlib.Path(sensitivity_dir)
    for scenario_dir in sensitivity_dir.iterdir():
        for scatter_dir in scenario_dir.iterdir():
            args = [
                f"plot=normalized_mobility",
                f"path={str(scatter_dir / 'normalized_mobility')}",
                f"load_cache=False",
                f"use_cache=False",
                f"normalized_mobility_use_extracted_data=False",
            ]
            if ARMED:
                RavenJob().set_script_path(PLOTTING_SCRIPT).set_script_args(
                    args
                ).launch(verbose=True)
            else:
                print(f"Would have ran: python {PLOTTING_SCRIPT} {' '.join(args)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory to sensitivity experiments.")
    args = parser.parse_args()
    run(args.dir)
