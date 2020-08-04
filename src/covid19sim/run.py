"""
Main entrypoint for the execution of simulations.

The experimental settings of the simulations are managed via [Hydra](https://github.com/facebookresearch/hydra).
The root configuration file is located at `src/covid19sim/configs/simulation/config.yaml`. All settings
provided via commandline will override the ones loaded through the configuration files.
"""
import datetime
import logging
import os
import time
import typing
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from covid19sim.locations.city import City
from covid19sim.utils.env import Env
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from covid19sim.log.monitors import EventMonitor, SimulationMonitor
from covid19sim.inference.server_utils import DataCollectionServer
from covid19sim.utils.utils import dump_conf, dump_tracker_data, extract_tracker_data, parse_configuration, log


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

    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    conf[
        "outdir"
    ] = "{}/sim_v2_people-{}_days-{}_init-{}_uptake-{}_seed-{}_{}_{}".format(
        conf["outdir"],
        conf["n_people"],
        conf["simulation_days"],
        conf["init_fraction_sick"],
        conf["APP_UPTAKE"],
        conf["seed"],
        timenow,
        str(time.time_ns())[-6:],
    )

    if Path(conf["outdir"]).exists():
        out_path = Path(conf["outdir"])
        out_idx = 1
        while (out_path.parent / (out_path.name + f"_{out_idx}")).exists():
            out_idx += 1
        conf["outdir"] = str(out_path.parent / (out_path.name + f"_{out_idx}"))

    os.makedirs(conf["outdir"])
    logfile = f"{conf['outdir']}/log_{timenow}.txt"

    outfile = os.path.join(conf["outdir"], "data")

    # ---------------------------------
    # -----  Filter-Out Warnings  -----
    # ---------------------------------
    import warnings
    # warnings.filterwarnings("ignore")

    # ----------------------------
    # -----  Run Simulation  -----
    # ----------------------------
    # correctness of configuration file
    assert not conf['RISK_MODEL'] != "" or conf['INTERVENTION_DAY'] >= 0, "risk model is given, but no intervnetion day specified"
    assert conf['N_BEHAVIOR_LEVELS'] >= 2, "At least 2 behavior levels are required to model behavior changes"
    assert not conf['RISK_MODEL'] == "" or conf['N_BEHAVIOR_LEVELS'] == 2, "number of behavior levels (N_BEHAVIOR_LEVELS) in unmitigated or lockdown scenario should be 2"

    if conf['RISK_MODEL'] == "":
        type_of_run = "Unmitigated"
        if conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']:
            type_of_run = "Lockdown"
    else:
        type_of_run = f"{conf['RISK_MODEL']} with {conf['N_BEHAVIOR_LEVELS']} behavior levels (Only relevant ones are used)"

    log(f"RISK_MODEL = {conf['RISK_MODEL']}", logfile)
    log(f"INTERVENTION_DAY = {conf['INTERVENTION_DAY']}", logfile)
    log(f"Type of run: {type_of_run}", logfile)

    conf["outfile"] = outfile
    city, monitors, tracker = simulate(
        n_people=conf["n_people"],
        init_fraction_sick=conf["init_fraction_sick"],
        start_time=conf["start_time"],
        simulation_days=conf["simulation_days"],
        outfile=conf["outfile"],
        out_chunk_size=conf["out_chunk_size"],
        print_progress=conf["print_progress"],
        seed=conf["seed"],
        conf=conf,
        logfile=logfile
    )

    dump_conf(city.conf, "{}/full_configuration.yaml".format(city.conf["outdir"]))
    tracker.write_metrics()

    monitors[0].dump()
    monitors[0].join_iothread()

    if hasattr(city, "tracker") and \
            hasattr(city.tracker, "collection_server") and \
            isinstance(city.tracker.collection_server, DataCollectionServer) and \
            city.tracker.collection_server is not None:
        city.tracker.collection_server.stop_gracefully()
        city.tracker.collection_server.join()

    if not conf["tune"]:
        # ----------------------------------------------
        # -----  Not Tune: Collect Training Data   -----
        # ----------------------------------------------
        # write values to train with
        train_priors = os.path.join(f"{conf['outdir']}/train_priors.pkl")
        tracker.write_for_training(city.humans, train_priors, conf)

    else:
        # ------------------------------------------------------
        # -----     Tune: Write logs And Tacker Data       -----
        # ------------------------------------------------------
        timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log("Dumping Tracker Data in {}".format(conf["outdir"]), logfile)

        Path(conf["outdir"]).mkdir(parents=True, exist_ok=True)
        filename = f"tracker_data_n_{conf['n_people']}_seed_{conf['seed']}_{timenow}.pkl"
        data = extract_tracker_data(tracker, conf)
        dump_tracker_data(data, conf["outdir"], filename)
    return conf


def simulate(
    n_people: int = 1000,
    init_fraction_sick: float = 0.01,
    start_time: datetime.datetime = datetime.datetime(2020, 2, 28, 0, 0),
    simulation_days: int = 30,
    outfile: typing.Optional[typing.AnyStr] = None,
    out_chunk_size: typing.Optional[int] = None,
    print_progress: bool = False,
    seed: int = 0,
    other_monitors: typing.Optional[typing.List] = None,
    conf: typing.Optional[typing.Dict] = None,
    logfile: str = None,
):
    """
    Runs a simulation.

    Args:
        n_people ([type], optional): [description]. Defaults to None.
        init_fraction_sick (float, optional): fraction of population initialized as sick. Defaults to 0.01.
        start_time ([type], optional): [description]. Defaults to datetime.datetime(2020, 2, 28, 0, 0).
        simulation_days (int, optional): [description]. Defaults to 10.
        outfile (str, optional): [description]. Defaults to None.
        out_chunk_size ([type], optional): [description]. Defaults to None.
        seed (int, optional): [description]. Defaults to 0.
        other_monitors (list, optional): [description]. Defaults to [].
        conf (dict): yaml configuration of the experiment
        logfile (str): filepath where the console output and final tracked metrics will be logged. Prints to the console only if None.

    Returns:
        city (covid19sim.locations.city.City): [description]
        monitors (list):
        tracker (covid19sim.log.track.Tracker):
    """

    if other_monitors is None:
        other_monitors = []
    if conf is None:
        conf = {}

    conf["n_people"] = n_people
    conf["init_fraction_sick"] = init_fraction_sick
    conf["start_time"] = start_time
    conf["simulation_days"] = simulation_days
    conf["outfile"] = outfile
    conf["out_chunk_size"] = out_chunk_size
    conf["print_progress"] = print_progress
    conf["seed"] = seed
    conf["other_monitors"] = other_monitors
    conf['logfile'] = logfile

    # set days and mixing constants
    conf['_CURRENT_PREFERENTIAL_ATTACHMENT_FACTOR'] = conf['BEGIN_PREFERENTIAL_ATTACHMENT_FACTOR']
    start_time_offset_days = conf['COVID_START_DAY']
    intervention_start_days = conf['INTERVENTION_DAY']

    # start of COVID spread
    conf['COVID_SPREAD_START_TIME'] = start_time

    # start of intervention
    conf['INTERVENTION_START_TIME'] = None
    if intervention_start_days >= 0:
        conf['INTERVENTION_START_TIME'] = start_time + datetime.timedelta(days=intervention_start_days)

    # start of simulation without COVID
    start_time -= datetime.timedelta(days=start_time_offset_days)
    conf['SIMULATION_START_TIME'] = str(start_time)

    # adjust the simulation days
    conf['simulation_days'] += conf['COVID_START_DAY']
    simulation_days = conf['simulation_days']

    logging.root.setLevel(getattr(logging, conf["LOGGING_LEVEL"].upper()))

    rng = np.random.RandomState(seed)
    env = Env(start_time)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    city = City(
        env, n_people, init_fraction_sick, rng, city_x_range, city_y_range, conf, logfile
    )

    # Add monitors
    monitors = [
        EventMonitor(f=SECONDS_PER_HOUR * 30, dest=outfile, chunk_size=out_chunk_size),
        SimulationMonitor(frequency=SECONDS_PER_DAY, logfile=logfile, conf=conf),
    ]

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

    # initiate humans
    for human in city.humans:
        env.process(human.run())

    # Initiate monitor processes
    for m in monitors:
        env.process(m.run(env, city=city))

    # Run simulation until termination
    env.run(until=env.ts_initial + simulation_days * SECONDS_PER_DAY)

    return city, monitors, city.tracker


if __name__ == "__main__":
    main()
