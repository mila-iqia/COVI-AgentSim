"""
Main file to run the simulations
"""
import os
import datetime

import click
import numpy as np
from covid19sim.base import City, Env

from covid19sim.frozen.helper import SYMPTOMS_META_IDMAP
from covid19sim.monitors import EventMonitor, SEIRMonitor, TimeMonitor
from covid19sim.simulator import Human
from covid19sim.utils import (
    dump_tracker_data,
    extract_tracker_data,
    parse_configuration,
)
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="hydra-configs/config.yaml")
def main(conf: DictConfig) -> None:
    """
    [summary]

    Args:
        conf (DictConfig): yaml configuration file
    """

    # Load the experimental configuration
    print(conf.pretty())
    conf = parse_configuration(conf)
    if conf["outdir"] is None:
        conf["outdir"] = "output"
    os.makedirs(f"{conf['outdir']}", exist_ok=True)
    conf["outdir"] = "{}/sim_v2_people-{}_days-{}_init-{}_seed-{}_{}".format(
        conf["outdir"],
        conf["n_people"],
        conf["simulation_days"],
        conf["init_percent_sick"],
        conf["seed"],
        datetime.datetime.now().strftime("'%Y%m%d-%H%M%S"),
    )
    os.makedirs(conf["outdir"])
    outfile = os.path.join(conf["outdir"], "data")

    if conf["tune"]:
        print("Using Tune")
        import warnings

        warnings.filterwarnings("ignore")
        outfile = None

    conf["outfile"] = outfile

    print("n_people:", conf["n_people"])
    print("seed:", conf["seed"])
    print("n_jobs:", conf["n_jobs"])

    monitors, tracker = simulate(
        n_people=conf["n_people"],
        init_percent_sick=conf["init_percent_sick"],
        start_time=conf["start_time"],
        simulation_days=conf["simulation_days"],
        outfile=conf["outfile"],
        out_chunk_size=conf["out_chunk_size"],
        print_progress=conf["print_progress"],
        seed=conf["seed"],
        n_jobs=conf["n_jobs"],
        port=conf["port"],
        conf=conf,
    )
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if not conf["tune"]:
        monitors[0].dump()
        monitors[0].join_iothread()
        # write metrics
        logfile = os.path.join(f"{conf['outdir']}/logs.txt")
        tracker.write_metrics(logfile)
    else:
        filename = f"tracker_data_n_{conf['n_people']}_seed_{conf['seed']}_{timenow}_{conf['name']}.pkl"
        data = extract_tracker_data(tracker, conf)
        dump_tracker_data(data, conf["outdir"], filename)


def simulate(
    n_people=None,
    init_percent_sick=0.01,
    start_time=datetime.datetime(2020, 2, 28, 0, 0),
    simulation_days=10,
    outfile=None,
    out_chunk_size=None,
    print_progress=False,
    seed=0,
    port=6688,
    n_jobs=1,
    other_monitors=[],
    conf={},
):
    """
    [summary]

    Args:
        n_people ([type], optional): [description]. Defaults to None.
        init_percent_sick (float, optional): [description]. Defaults to 0.0.
        start_time ([type], optional): [description]. Defaults to datetime.datetime(2020, 2, 28, 0, 0).
        simulation_days (int, optional): [description]. Defaults to 10.
        outfile (str, optional): [description]. Defaults to None.
        out_chunk_size ([type], optional): [description]. Defaults to None.
        print_progress (bool, optional): [description]. Defaults to False.
        seed (int, optional): [description]. Defaults to 0.
        port (int, optional): [description]. Defaults to 6688.
        n_jobs (int, optional): [description]. Defaults to 1.
        other_monitors (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """

    conf["n_people"] = n_people
    conf["init_percent_sick"] = init_percent_sick
    conf["start_time"] = start_time
    conf["simulation_days"] = simulation_days
    conf["outfile"] = outfile
    conf["out_chunk_size"] = out_chunk_size
    conf["print_progress"] = print_progress
    conf["seed"] = seed
    conf["port"] = port
    conf["n_jobs"] = n_jobs
    conf["other_monitors"] = other_monitors

    rng = np.random.RandomState(seed)
    env = Env(start_time, conf.get("TICK_MINUTE"))
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    city = City(
        env,
        n_people,
        init_percent_sick,
        rng,
        city_x_range,
        city_y_range,
        start_time,
        Human,
        conf,
    )
    monitors = [
        EventMonitor(f=1800, dest=outfile, chunk_size=out_chunk_size),
        SEIRMonitor(f=1440),
    ]

    # run the simulation
    if print_progress:
        monitors.append(TimeMonitor(1440))  # print every day

    if other_monitors:
        monitors += other_monitors

    # run city
    monitors[0].dump()
    monitors[0].join_iothread()
    # run this every hour
    env.process(
        city.run(1440 / 24, outfile, start_time, SYMPTOMS_META_IDMAP, port, n_jobs)
    )

    # run humans
    for human in city.humans:
        env.process(human.run(city=city))

    # run monitors
    for m in monitors:
        env.process(m.run(env, city=city))

    env.run(until=simulation_days * 24 * 60 / city.conf.get("TICK_MINUTE"))

    return monitors, city.tracker


if __name__ == "__main__":
    main()
