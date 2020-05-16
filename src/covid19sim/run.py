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
from covid19sim.utils import dump_tracker_data, extract_tracker_data, load_conf


@click.command()
@click.option("--n_people", help="population of the city", type=int, default=100)
@click.option(
    "--init_percent_sick",
    help="initial percentage of sick people",
    type=float,
    default=0.01,
)
@click.option(
    "--simulation_days",
    help="number of days to run the simulation for",
    type=int,
    default=30,
)
@click.option(
    "--out_chunk_size",
    help="minimum number of events per dump in outfile",
    type=int,
    default=1,
    required=False,
)
@click.option(
    "--outdir",
    help="the directory to write data to",
    type=str,
    default="output",
    required=False,
)
@click.option("--seed", help="seed for the process", type=int, default=0)
@click.option(
    "--n_jobs",
    help="number of parallel procs to query the risk servers with",
    type=int,
    default=1,
)
@click.option(
    "--port",
    help="which port should we look for inference servers on",
    type=int,
    default=6688,
)
@click.option(
    "--config",
    help="where is the configuration file for this experiment",
    type=str,
    default="configs/naive_config.yml",
)
@click.option(
    "--tune",
    help="track additional specific metrics to plot and explore",
    is_flag=True,
    default=False,
)
@click.option(
    "--name", help="name of the file to append metrics file", type=str, default=""
)
def main(
    n_people=None,
    init_percent_sick=0.01,
    start_time=datetime.datetime(2020, 2, 28, 0, 0),
    simulation_days=30,
    outdir=None,
    out_chunk_size=None,
    seed=0,
    n_jobs=1,
    port=6688,
    config="configs/naive_config.yml",
    tune=False,
    name="",
):
    """
    [summary]

    Args:
        n_people ([type], optional): [description]. Defaults to None.
        init_percent_sick (int, optional): [description]. Defaults to 0.
        start_time ([type], optional): [description]. Defaults to datetime.datetime(2020, 2, 28, 0, 0).
        simulation_days (int, optional): [description]. Defaults to 30.
        outdir ([type], optional): [description]. Defaults to None.
        out_chunk_size ([type], optional): [description]. Defaults to None.
        seed (int, optional): [description]. Defaults to 0.
        n_jobs (int, optional): [description]. Defaults to 1.
        port (int, optional): [description]. Defaults to 6688.
        config (str, optional): [description]. Defaults to "configs/naive_config.yml".
    """

    # Load the experimental configuration
    conf = load_conf(config)
    if outdir is None:
        outdir = "output"
    os.makedirs(f"{outdir}", exist_ok=True)
    outdir = f"{outdir}/sim_v2_people-{n_people}_days-{simulation_days}_init-{init_percent_sick}_seed-{seed}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(outdir)
    outfile = os.path.join(outdir, "data")

    if tune:
        import warnings

        warnings.filterwarnings("ignore")
        outfile = None

    monitors, tracker = simulate(
        n_people=n_people,
        init_percent_sick=init_percent_sick,
        start_time=start_time,
        simulation_days=simulation_days,
        outfile=outfile,
        out_chunk_size=out_chunk_size,
        print_progress=True,
        seed=seed,
        n_jobs=n_jobs,
        port=port,
        conf=conf,
    )
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if not tune:
        monitors[0].dump()
        monitors[0].join_iothread()
        # write metrics
        logfile = os.path.join(f"{outdir}/logs.txt")
        tracker.write_metrics(logfile)
    else:
        filename = f"tracker_data_n_{n_people}_seed_{seed}_{timenow}_{name}.pkl"
        data = extract_tracker_data(tracker, conf)
        dump_tracker_data(data, outdir, filename)


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
    rng = np.random.RandomState(seed)
    env = Env(start_time)
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
