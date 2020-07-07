"""
Script to creates the transition matrices to apply the mobility patterns from a
`source` experiment (e.g. Binary Digital Tracing) to a `target` experiment (e.g.
Transformer). The recommendation levels of source are updated as usual, following
the tracing method of source (e.g. Binary Digital Tracing), but the interventions
on the mobility follow the recommendations (in expectation) from the target (e.g.
Transformer).

How to use:
    python src/covid19sim/other/get_transition_matrix_rec_levels.py --source path/to/binary_tracing/experiment --target path/to/transformer/experiment

This script returns a new configuration file that can be run to apply the updates
of the recommendation levels of source, but with the interventions from target.
"""
import numpy as np
import os
import glob
import yaml
import pickle
import warnings
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path


def generate_name(source_config, target_config):
    source_model = source_config["RISK_MODEL"]
    seed = source_config["seed"]
    if args.name_as_seed:
        return f"seed-{seed}.yaml"
    target_model = target_config["RISK_MODEL"]
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")

    return f"{source_model}_to_{target_model}_seed-{seed}_{timenow}.yaml"


def get_config_and_data(folder):
    config_filenames = glob.glob(os.path.join(folder, "*.yaml"))
    data_filenames = glob.glob(os.path.join(folder, "*.pkl"))

    # Configuration
    if len(config_filenames) == 0:
        raise IOError(
            "There is no configuration file (*.yaml) in folder `{0}" "`.".format(folder)
        )
    if len(config_filenames) > 1:
        warnings.warn(
            "There are multiple configuration files (*.yaml) in "
            "folder `{0}`. Taking the first configuration file `{1}"
            "`.".format(folder, config_filenames[0])
        )

    with open(config_filenames[0], "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Data
    if len(data_filenames) == 0:
        raise IOError(
            "There is no data (*.pkl) in folder `{0}`. Make sure to "
            "run the experiment with `tune=True`.".format(folder)
        )
    if len(data_filenames) > 1:
        warnings.warn(
            "There are multiple data files (*.pkl) in folder `{0}`. "
            "Taking the first data file `{1}`.".format(folder, data_filenames[0])
        )

    with open(data_filenames[0], "rb") as f:
        data = pickle.load(f)

    if ("intervention_day" not in data) or (data["intervention_day"] < 0):
        raise ValueError(
            "The `intervention_day` is missing. Make sure there "
            "was an intervention in the experiment with "
            "configuration `{0}`.".format(config_filenames[0])
        )

    if "humans_rec_level" not in data:
        raise KeyError(
            "The `humans_rec_level` is missing in the data. The "
            "experiment was performed before this data was added "
            "to the tracker. Please re-run the experiment with "
            "configuration `{0}`.".format(config_filenames[0])
        )

    return config, data


def get_rec_levels_distributions(data, config, num_rec_levels=4):
    rec_levels = data["humans_rec_level"]
    intervention_day = data["intervention_day"]
    num_days = len(next(iter(rec_levels.values())))
    if "humans_has_app" in data:
        has_app = data["humans_has_app"]
        rec_levels_with_app = [
            value for (key, value) in rec_levels.items() if has_app[key]
        ]
    else:
        if config.get("APP_UPTAKE", -1) >= 0:
            logging.warning(
                "`humans_has_app` is not available even though "
                "APP_UPTAKE is not -1 (APP_UPTAKE={0}).".format(
                    config.get("APP_UPTAKE", -1)
                )
            )
        rec_levels_with_app = rec_levels.values()

    rec_levels_per_day = np.zeros((num_days, len(rec_levels_with_app)), dtype=np.int_)

    for index, recommendations in enumerate(rec_levels_with_app):
        rec_levels_per_day[:, index] = np.asarray(recommendations, dtype=np.int_)

    # Remove the days before intervention (without recommendation)
    rec_levels_per_day = rec_levels_per_day[intervention_day:]
    is_valid = np.logical_and(
        rec_levels_per_day >= 0, rec_levels_per_day < num_rec_levels
    )
    assert np.all(is_valid), "Some recommendation levels are invalid"

    bincount = lambda x: np.bincount(x, minlength=num_rec_levels)
    counts = np.apply_along_axis(bincount, axis=1, arr=rec_levels_per_day)

    return counts / np.sum(counts, axis=1, keepdims=True)


def generate_single(args, source, target):
    source_config, source_data = get_config_and_data(source)
    target_config, target_data = get_config_and_data(target)

    if source_config["seed"] != target_config["seed"]:
        warnings.warn(
            "The seed of the source experiment is different from the "
            "seed of the target experiment. source.seed={0}, "
            "target.seed={1}.".format(source_config["seed"], target_config["seed"])
        )

    if source_data["intervention_day"] != target_data["intervention_day"]:
        raise ValueError(
            "The intervention day of the source experiment is "
            "different from the intervention day of the "
            "target experiment. source.intervention_day={0}, "
            "target.intervention_day={1}.".format(
                source_data["intervention_day"], target_data["intervention_day"]
            )
        )

    # Compute the distributions of recommendation levels for
    # target tracing method (e.g. Transformer)
    target_dists = get_rec_levels_distributions(
        target_data, target_config, num_rec_levels=args.num_rec_levels
    )

    # Update the source configuration file
    source_config["DAILY_TARGET_REC_LEVEL_DIST"] = target_dists.flatten().tolist()

    # Save the new source configuration
    config_folder = os.path.join(
        os.path.dirname(__file__), "../configs/simulation", args.config_folder
    )
    config_folder = os.path.relpath(config_folder)
    os.makedirs(config_folder, exist_ok=True)

    output_filename = generate_name(source_config, target_config)
    output_config_name, _ = os.path.splitext(output_filename)
    output_path = os.path.join(config_folder, output_filename)

    logging.debug("Saving new configuration to `{0}`...".format(output_path))
    with open(output_path, "w") as f:
        yaml.dump(source_config, f, Dumper=yaml.Dumper)

    # Save the source configuration as a comment
    with open(output_path, "a") as f:
        target_config_yaml = yaml.dump(target_config, Dumper=yaml.Dumper)
        target_config_lines = target_config_yaml.split("\n")
        f.write(f"\n# Target configuration: {target}\n#\n")
        for line in target_config_lines:
            f.write(f"# {line}\n")

    logging.info("New configuration file saved: `{0}`".format(output_path))
    logging.info(
        f"To run the experiment with the new mobility:\n\tpython "
        f"src/covid19sim/run.py {args.config_folder}={output_config_name}"
    )

    return output_config_name


def get_bulk_folders(folder, keys):
    config_filenames = glob.glob(os.path.join(folder, "*/*.yaml"))
    bulk_folders = defaultdict(dict)

    for filename in config_filenames:
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
        key = tuple(config[k] for k in keys)
        seed = config["seed"]
        bulk_folders[key][seed] = os.path.dirname(filename)

    return bulk_folders


def get_model(conf):
    if conf["RISK_MODEL"] == "":
        return "unmitigated"

    if conf["RISK_MODEL"] == "digital":
        if conf["TRACING_ORDER"] == 1:
            return "bdt1"
        elif conf["TRACING_ORDER"] == 2:
            return "bdt2"
        else:
            raise ValueError(
                "Unknown binary digital tracing order: {}".format(conf["TRACING_ORDER"])
            )
    if conf["RISK_MODEL"] == "transformer":
        if conf["USE_ORACLE"]:
            return "oracle"
        return "transformer"

    if conf["RISK_MODEL"] == "heuristicv1":
        return "heuristicv1"

    if conf["RISK_MODEL"] == "heuristicv2":
        return "heuristicv2"

    raise ValueError("Unknown RISK_MODEL {}".format(conf["RISK_MODEL"]))


def main(args):
    if args.discover is None:
        run(args)

    parent = Path(args.discover).resolve()
    run_conf = None
    assert (
        parent.exists() and parent.is_dir()
    ), "Unknown parent folder to discover {}".format(str(parent))

    global_dict = {}

    print("Discovering runs to normalize in {}".format(str(parent)))

    to_normalize = []
    for method_dir in parent.iterdir():
        if not method_dir.is_dir():
            continue
        for run_dir in method_dir.iterdir():
            if not run_dir.is_dir():
                continue
            conf_path = run_dir / "full_configuration.yaml"
            if conf_path.exists():
                with (run_dir / "full_configuration.yaml").open("r") as f:
                    run_conf = yaml.safe_load(f)
                break

        assert (
            run_conf is not None
        ), "run_conf is None: could not find full config in subdirs of {}".format(
            str(method_dir)
        )
        tracing_method = get_model(run_conf)
        if (
            tracing_method != "unmitigated"
            and "DAILY_TARGET_REC_LEVEL_DIST" not in run_conf
        ):
            print(f"Found config for {method_dir.name}")
            to_normalize.append(
                {"tracing_method": tracing_method, "method_dir": method_dir}
            )

    target_name = Path(args.target).name
    for i, norm_dict in enumerate(to_normalize):
        tracing_method = norm_dict["tracing_method"]
        method_dir = norm_dict["method_dir"]
        print("\n#" + "#" * 39)
        print(f"Normalizing {method_dir.name} ({i + 1}/{len(to_normalize)})")
        print("#" * 39 + "#\n")
        if tracing_method != "transformer":
            args.source = str(method_dir)
            args.config_folder = f"{target_name}_{tracing_method}"
            output_dict = run(args)
            for k, v in output_dict.items():
                if isinstance(v, dict) and "sample" in v:
                    output_dict[k]["normalized"] = True
            global_dict.update(output_dict)

    global_dict["USE_INFERENCE_SERVER"] = False
    global_dict["base_dir"] = str(parent)
    global_dict["n_search"] = -1

    search_path = Path(__file__).parent.parent / "configs" / "search"
    global_conf_path = search_path / f"normalize_{parent.name}.yaml"
    with global_conf_path.open("w") as f:
        yaml.safe_dump(global_dict, f)

    print(
        "Writing in {}:\n{}".format(str(global_conf_path), yaml.safe_dump(global_dict))
    )


def run(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.bulk_keys is None:
        if args.output is not None:
            logging.warning(
                "The script is used to normalize only a single pair "
                "of simulations, and therefore will only return one "
                "single new configuration. The output filename for "
                "combining the new configuration files (--ouput) is "
                "therefore ignored."
            )
        generate_single(args, args.source, args.target)
    else:
        source_folders = get_bulk_folders(args.source, args.bulk_keys)
        target_folders = get_bulk_folders(args.target, args.bulk_keys)
        new_configs = []

        for key, folders in target_folders.items():
            for seed, target_folder in folders.items():
                try:
                    source_folder = source_folders[key][seed]
                except KeyError:
                    logging.warning(
                        "The configuration `{0}` with seed `{1}` exists "
                        "in the bulk target folder `{2}` but not in the "
                        "bulk source folder `{3}`. Ignoring this "
                        "configuration.".format(key, seed, args.target, args.source)
                    )
                    continue
                new_config = generate_single(args, source_folder, target_folder)
                new_configs.append(new_config)

        # Return the list of new configuration files as a *.yaml file
        if new_configs:
            output_dict = {args.config_folder: {"sample": "chain", "from": new_configs}}
            logging.info(
                "Use the following snippet inside a configuration file "
                "to use in combination with `random_search.py`\n"
                "{0}.".format(yaml.dump(output_dict, Dumper=yaml.Dumper, indent=4))
            )

            if args.output is not None:
                logging.info(
                    "Saving this snippet of code in `{0}`.".format(args.output)
                )
                with open(args.output, "w") as f:
                    yaml.dump(output_dict, f, Dumper=yaml.Dumper)
            return output_dict


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Creates the transition "
        "matrices to apply the mobility patterns from a `source` experiment "
        "(e.g. Binary Digital Tracing) to a `target` experiment (e.g. "
        "Transformer). The recommendation levels of source are updated as "
        "usual, following the tracing method of source (e.g. Binary Digital "
        "Tracing), but the interventions on the mobility follow the "
        "recommendations (in expectation) from the target (e.g. Transformer)."
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Path to the folder of the source experiment (e.g. Binary Digital "
        "Tracing), i.e. the tracing method to use for the update of the "
        "recommendation levels.",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Path to the folder of the target experiment (e.g. "
        "Transformer), i.e. the tracing method which we apply the mobility "
        "intervention of.",
    )
    parser.add_argument(
        "--config-folder",
        type=str,
        default="transport",
        help="Name of the folder where the new configuration files are placed. "
        "The folder is created automatically inside `configs/simulation`.",
    )
    parser.add_argument(
        "--bulk-keys",
        type=json.loads,
        default=None,
        help="The keys in the configuration to loop over for bulk creation. "
        "If not provided, then only a single pair of configuration files "
        "is merged.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        required=False,
        help="The output *.yaml file used to store the new configuration files "
        "to be compiled into a new configuration file that can be launched "
        "with `random_search.py`.",
    )
    parser.add_argument(
        "--num-rec-levels",
        type=int,
        default=4,
        help="Number of possible recommendation levels (default: 4)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--name_as_seed",
        action="store_true",
        help="stores the configuration file with the name seed-{seed}.yaml",
    )
    parser.add_argument(
        "--discover",
        help="Discover subfolders for which to run the normalization as if they were `target`",
        default=None,
    )

    args = parser.parse_args()

    main(args)
