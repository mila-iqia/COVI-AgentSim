import os
import time
import argparse
from copy import deepcopy
from pathlib import Path

try:
    from raven.core import RavenJob
except ImportError:
    from unittest.mock import Mock

    RavenJob = Mock()
import shlex


def parse_args():
    parsey = argparse.ArgumentParser()
    parsey.add_argument("file", type=str, help="Job file.")
    parsey.add_argument(
        "-n", "--num-jobs", type=int, default=float("inf"), help="How many jobs to run."
    )
    parsey.add_argument(
        "-d",
        "--disarm",
        action="store_true",
        default=False,
        help="Whether to just print stuff.",
    )
    parsey.add_argument(
        "-s",
        "--no-space",
        action="store_true",
        default=False,
        help="Whether to get rid of spaces.",
    )
    parsey.add_argument(
        "-p",
        "--plot",
        action="store_true",
        default=False,
        help="Whether to run the plotting scripts.",
    )
    return parsey.parse_args()


def validate_components(components, no_space=False):
    components[-1] = components[-1].replace(";", "")
    for idx, component in enumerate(components):
        if " " in component:
            if no_space:
                components[idx] = component.replace(" ", "")
            else:
                # hydra doesn't like spaces without quotes, but shlex likes to get rid
                # of the quotes. :|
                assert "=" in component, "Can't parse hydra command with a space."
                key, value = component.split("=")
                components[idx] = f'{key}=\\"{value}\\"'
    return components


def parse_job_type(line):
    if line.startswith("python run.py"):
        return "run"
    elif line.startswith("python main.py"):
        return "plot"
    else:
        return None


def get_script_path(job_type):
    if job_type == "run":
        path = (Path(__file__) / ".." / ".." / "run.py").resolve()
        assert path.exists(), f"{path} does not exist."
        return str(path)
    elif job_type == "plot":
        path = (Path(__file__) / ".." / ".." / "plotting" / "main.py").resolve()
        assert path.exists(), f"{path} does not exist."
        return str(path)
    else:
        raise ValueError(f"Unknown job_type: {job_type}")


def main(args=None):
    args = args or parse_args()
    # If file is a directory, run recursively
    if os.path.isdir(args.file):
        for path in os.listdir(args.file):
            if path.endswith(".txt"):
                recurse_args = deepcopy(args)
                recurse_args.file = path
                main(args=recurse_args)
            else:
                continue
    else:
        assert args.file.endswith(".txt")
    # Read up and launch the jobs
    with open(args.file, "r") as f:
        lines = f.readlines()
    num_jobs_ran = 0
    for line in lines:
        job_type = parse_job_type(line)
        if job_type is None:
            continue
        if num_jobs_ran >= args.num_jobs:
            break
        if args.plot:
            assert job_type == "plot", (
                "To launch plotting jobs, append a -p to " "the command line argument"
            )
        # Get script path and components
        script_path = get_script_path(job_type)
        components = shlex.split(line)[2:]
        components[-1] = components[-1].replace(";", "")
        components = validate_components(components, no_space=args.no_space)
        # Launch the job
        raven_job = RavenJob().set_script_path(script_path).set_script_args(components)
        if args.disarm:
            print(raven_job.build_submission(write=False))
            print("-------------------------------------")
        else:
            raven_job.launch(verbose=True)
            time.sleep(1)
        num_jobs_ran += 1


if __name__ == "__main__":
    main()
