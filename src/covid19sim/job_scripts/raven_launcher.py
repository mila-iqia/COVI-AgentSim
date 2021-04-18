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
    parsey.add_argument(
        "-t",
        "--transformer-exp-path",
        type=str,
        default=None,
        help="Path to transformer exp, to be filled in the template if required.",
    )
    parsey.add_argument(
        "-rlt",
        "--rec-level-threshold",
        type=str,
        default=None,
        help="Path to transformer exp, to be filled in the template if required.",
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


def fill_transformer_spec(components, args):
    new_components = components[:]
    if args.transformer_exp_path is not None:
        for idx, component in enumerate(components):
            key, value = component.split("=")
            if key == "TRANSFORMER_EXP_PATH":
                # Sub in the exp path
                new_component = f"{key}={args.transformer_exp_path}"
                new_components[idx] = new_component
            elif key == "outdir":
                transformer_folder_name = os.path.basename(args.transformer_exp_path)
                value = value.replace(
                    "TRANSFORMER_FOLDER_NAME", transformer_folder_name
                )
                new_component = f"{key}={value}"
                new_components[idx] = new_component
    if args.rec_level_threshold is not None:
        for idx, component in enumerate(components):
            key, value = component.split("=")
            if key == "REC_LEVEL_THRESHOLDS":
                # Expecting something like 012 or 234 passed as a string
                assert len(args.rec_level_threshold) == 3
                value = "[" + ",".join(list(args.rec_level_threshold)) + "]"
                new_component = f"{key}={value}"
                new_components[idx] = new_component
    return new_components


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
        for filename in os.listdir(args.file):
            if filename.endswith(".txt"):
                recurse_args = deepcopy(args)
                recurse_args.file = os.path.join(args.file, filename)
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
        components = fill_transformer_spec(components, args)
        components = validate_components(components, no_space=args.no_space)
        # Launch the job
        if args.disarm:
            print(f"python {script_path} " + " ".join(components))
            print("-------------------------------------")
        else:
            raven_job = (
                RavenJob().set_script_path(script_path).set_script_args(components)
            )
            raven_job.launch(verbose=True)
            time.sleep(1)
        num_jobs_ran += 1


if __name__ == "__main__":
    main()
