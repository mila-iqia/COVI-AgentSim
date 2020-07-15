"""
Main entrypoint for the execution of simulations.

The experimental settings of the simulations are managed via [Hydra](https://github.com/facebookresearch/hydra).
The root configuration file is located at `src/covid19sim/configs/simulation/config.yaml`. All settings
provided via commandline will override the ones loaded through the configuration files.
"""
import datetime
import logging
import os
import shutil
import time
import typing
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from covid19sim.locations.city import City
from covid19sim.utils.env import Env
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from covid19sim.log.monitors import EventMonitor, SEIRMonitor, TimeMonitor
from covid19sim.utils.utils import (dump_conf, dump_tracker_data,
                                    extract_tracker_data, parse_configuration,
                                    zip_outdir)


@hydra.main(config_path="configs/simulation/config.yaml")
def main(conf: DictConfig):
    """
    Enables command line execution of the simulator.

    Args:
        conf (DictConfig): yaml configuration file
    """

    # -------------------------------------------------
    # -----  Load the experimental configuration  -----
    # -------------------------------------------------
    conf = parse_configuration(conf)

    # -------------------------------------
    # -----  Create Output Directory  -----
    # -------------------------------------
    if conf["outdir"] is None:
        conf["outdir"] = str(Path(__file__) / "output")
    conf[
        "outdir"
    ] = "{}/sim_v2_people-{}_days-{}_init-{}_uptake-{}_seed-{}_{}_{}".format(
        conf["outdir"],
        conf["n_people"],
        conf["simulation_days"],
        conf["init_percent_sick"],
        conf["APP_UPTAKE"],
        conf["seed"],
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        str(time.time_ns())[-6:],
    )
    if Path(conf["outdir"]).exists():
        out_path = Path(conf["outdir"])
        out_idx = 1
        while (out_path.parent / (out_path.name + f"_{out_idx}")).exists():
            out_idx += 1
        conf["outdir"] = str(out_path.parent / (out_path.name + f"_{out_idx}"))

    os.makedirs(conf["outdir"])

    if not conf["tune"]:
        outfile = os.path.join(conf["outdir"], "data")

    # ---------------------------------
    # -----  Filter-Out Warnings  -----
    # ---------------------------------
    import warnings

    warnings.filterwarnings("ignore")
    if conf["tune"]:
        print("Using Tune")
        outfile = None

    # ----------------------------
    # -----  Run Simulation  -----
    # ----------------------------
    conf["outfile"] = outfile

    print("RISK_MODEL ==> ", conf.get("RISK_MODEL"))

    city, monitors, tracker = simulate(
        n_people=conf["n_people"],
        init_percent_sick=conf["init_percent_sick"],
        start_time=conf["start_time"],
        simulation_days=conf["simulation_days"],
        outfile=conf["outfile"],
        out_chunk_size=conf["out_chunk_size"],
        print_progress=conf["print_progress"],
        seed=conf["seed"],
        return_city=True,
        conf=conf,
    )

    # ----------------------------------------
    # -----  Compute Effective Contacts  -----
    # ----------------------------------------
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    all_effective_contacts = 0
    all_contacts = 0
    for human in city.humans:
        all_effective_contacts += human.effective_contacts
        all_contacts += human.num_contacts
    print(f"all_effective_contacts: {all_effective_contacts}")
    print(
        f"all_effective_contacts/(sim days * len(city.humans)): {all_effective_contacts / (conf['simulation_days'] * len(city.humans))}"
    )
    if all_contacts != 0:
        print(
            f"effective contacts per contacts (GLOBAL_MOBILITY_SCALING_FACTOR): {all_effective_contacts / all_contacts}"
        )

    dump_conf(city.conf, "{}/full_configuration.yaml".format(city.conf["outdir"]))

    if not conf["tune"]:
        # ----------------------------------------------
        # -----  Not Tune: Write Logs And Metrics  -----
        # ----------------------------------------------
        monitors[0].dump()
        monitors[0].join_iothread()
        # write metrics
        logfile = os.path.join(f"{conf['outdir']}/logs.txt")
        tracker.write_metrics(logfile)

        # write values to train with
        train_priors = os.path.join(f"{conf['outdir']}/train_priors.pkl")
        tracker.write_for_training(city.humans, train_priors, conf)

        if conf["zip_outdir"]:
            zip_outdir(conf["outdir"])
            if conf["delete_outdir"]:
                shutil.rmtree(conf["outdir"])
    else:
        # ------------------------------------------------------
        # -----  Tune: Create Plots And Write Tacker Data  -----
        # ------------------------------------------------------
        from covid19sim.plotting.plot_rt import PlotRt

        cases_per_day = tracker.cases_per_day
        serial_interval = tracker.get_generation_time()
        if serial_interval == 0:
            serial_interval = 7.0
            print("WARNING: serial_interval is 0")
        print(f"using serial interval :{serial_interval}")
        plotrt = PlotRt(R_T_MAX=4, sigma=0.25, GAMMA=1.0 / serial_interval)
        most_likely, _ = plotrt.compute(cases_per_day, r0_estimate=2.5)
        print("Rt", most_likely[:20])

        print("Dumping Tracker Data in", conf["outdir"])
        Path(conf["outdir"]).mkdir(parents=True, exist_ok=True)
        filename = f"tracker_data_n_{conf['n_people']}_seed_{conf['seed']}_{timenow}_{conf['name']}.pkl"
        data = extract_tracker_data(tracker, conf)
        dump_tracker_data(data, conf["outdir"], filename)
        tracker.write_metrics(f"{conf['outdir']}/log_{timenow}_{conf['name']}.txt")
    return conf


def simulate(
    n_people: int = 1000,
    init_percent_sick: float = 0.01,
    start_time: datetime.datetime = datetime.datetime(2020, 2, 28, 0, 0),
    simulation_days: int = 30,
    outfile: typing.Optional[typing.AnyStr] = None,
    out_chunk_size: typing.Optional[int] = None,
    print_progress: bool = False,
    seed: int = 0,
    other_monitors: typing.Optional[typing.List] = None,
    return_city: bool = False,
    conf: typing.Optional[typing.Dict] = None,
):
    """
    Runs a simulation.

    Args:
        n_people: total number of humans (agents) to simulate.
        init_percent_sick: initial percentage of the population that will be exposed to Covid-19.
        start_time: initial starting day of the simulation.
        simulation_days: number of days to run the simulation for.
        outfile: output file/folder path where data should be saved.
        out_chunk_size: TODO @@@@ DOCUMENT ME
        print_progress: toggles whether to print monitoring results to console or not.
        seed: seed used to initialize the global RNG for the simulation.
        other_monitors: TODO @@@@ DOCUMENT ME
        return_city: toggles whether to return the city object as the output of this function or not.
        conf: global configuration dictionary for the simulation.

    Returns:
        A tuple of the monitors and tracker, with the city as extra (if requested).
    """

    if other_monitors is None:
        other_monitors = []
    if conf is None:
        conf = {}

    conf["n_people"] = n_people
    conf["init_percent_sick"] = init_percent_sick
    conf["start_time"] = start_time
    conf["simulation_days"] = simulation_days
    conf["outfile"] = outfile
    conf["out_chunk_size"] = out_chunk_size
    conf["print_progress"] = print_progress
    conf["seed"] = seed
    conf["other_monitors"] = other_monitors

    logging.root.setLevel(getattr(logging, conf["LOGGING_LEVEL"].upper()))

    rng = np.random.RandomState(seed)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    city = City(
        env, n_people, init_percent_sick, rng, city_x_range, city_y_range, conf
    )

    # Add monitors
    monitors = [
        EventMonitor(f=SECONDS_PER_HOUR * 30, dest=outfile, chunk_size=out_chunk_size),
        SEIRMonitor(f=SECONDS_PER_DAY),
    ]
    if print_progress:
        monitors.append(TimeMonitor(SECONDS_PER_DAY))
    if other_monitors:
        monitors += other_monitors

    # Kickstart EventMonitor
    monitors[0].dump()
    monitors[0].join_iothread()

    # we might need to reset the state of the clusters held in shared memory (server or not)
    if conf.get("RESET_INFERENCE_SERVER", False):
        if conf.get("USE_INFERENCE_SERVER"):
            inference_frontend_address = conf.get("INFERENCE_SERVER_ADDRESS", None)
            print("requesting cluster reset from inference server...")
            from covid19sim.inference.server_utils import InferenceClient

            temporary_client = InferenceClient(
                server_address=inference_frontend_address
            )
            temporary_client.request_reset()
        else:
            from covid19sim.inference.heavy_jobs import DummyMemManager

            DummyMemManager.global_cluster_map = {}

    # Initiate city process, which runs every hour
    env.process(city.run(SECONDS_PER_HOUR, outfile))

    # Initiate human processes
    for human in city.humans:
        env.process(human.run(city=city))

    # Initiate monitor processes
    for m in monitors:
        env.process(m.run(env, city=city))

    # Run simulation until termination
    env.run(until=env.ts_initial + simulation_days * SECONDS_PER_DAY)

    if not return_city:
        return monitors, city.tracker
    else:
        return city, monitors, city.tracker


if __name__ == "__main__":
    main()
