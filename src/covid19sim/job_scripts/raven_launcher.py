import time
import argparse
try:
    from raven.core import RavenJob
except ImportError:
    from unittest.mock import Mock
    RavenJob = Mock()
import shlex


def parse_args():
    parsey = argparse.ArgumentParser()
    parsey.add_argument("-f", "--file", type=str, required=True, help="Job file.")
    parsey.add_argument(
        "-d",
        "--disarm",
        action="store_true",
        default=False,
        help="Whether to just print stuff.",
    )
    # parsey.add_argument("-r", "--run", type=str, required=True, help="Run file.")
    return parsey.parse_args()


def main(args=None):
    args = args or parse_args()
    with open(args.file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith("python run.py"):
            continue
        components = shlex.split(line)[2:]
        components[-1] = components[-1].replace(";", "")
        raven_job = RavenJob().set_script_path("run.py").set_script_args(components)
        if args.disarm:
            print(raven_job.build_submission(write=False))
            print("-------------------------------------")
        else:
            raven_job.launch(verbose=True)
            time.sleep(1)


if __name__ == '__main__':
    main()