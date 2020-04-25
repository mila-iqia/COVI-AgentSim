import datetime
import click
import numpy as np
import math
import pickle
import os
import zipfile

from config import TICK_MINUTE
from simulator import Human
from base import *
from utils import log, _draw_random_discreet_gaussian, _get_random_age, _get_random_area
from monitors import EventMonitor, TimeMonitor, SEIRMonitor


@click.group()
def simu():
    pass


@simu.command()
@click.option('--n_people', help='population of the city', type=int, default=100)
@click.option('--init_percent_sick', help='% of population initially sick', type=float, default=0.01)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--out_chunk_size', help='number of events per dump in outfile', type=int, default=2500, required=False)
@click.option('--outdir', help='the directory to write data to', type=str, default="output", required=False)
@click.option('--seed', help='seed for the process', type=int, default=0)
def sim(n_people=None,
        init_percent_sick=0,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=30,
        outdir=None, out_chunk_size=None,
        seed=0):

    import config
    config.COLLECT_LOGS = True

    if outdir is None:
        outdir = "output"

    os.makedirs(f"{outdir}", exist_ok=True)
    outdir = f"{outdir}/sim_v2_people-{n_people}_days-{simulation_days}_init-{init_percent_sick}_seed-{seed}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(outdir)

    outfile = os.path.join(outdir, "data")
    monitors, tracker = run_simu(
        n_people=n_people,
        init_percent_sick=init_percent_sick,
        start_time=start_time,
        simulation_days=simulation_days,
        outfile=outfile, out_chunk_size=out_chunk_size,
        print_progress=True,
        seed=seed
    )
    monitors[0].dump()
    monitors[0].join_iothread()

    # write metrics
    logfile = os.path.join(f"{outdir}/logs.txt")
    tracker.write_metrics(logfile)

@simu.command()
def base():
    import pandas as pd
    import cufflinks as cf
    cf.go_offline()
    import config
    config.COLLECT_LOGS = False
    monitors, tracker = run_simu(
        n_people=100,
        init_percent_sick=0.01,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=30,
        outfile=None,
        print_progress=False, seed=0,
    )
    stats = monitors[1].data
    x = pd.DataFrame.from_dict(stats).set_index('time')
    fig = x[['susceptible', 'exposed', 'infectious', 'removed']].iplot(asFigure=True, title="SEIR")
    fig.show()

    fig = x['R'].iplot(asFigure=True, title="R0")
    fig.show()


@simu.command()
@click.option('--n_people', help='population of the city', type=int, default=1000)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--seed', help='seed for the process', type=int, default=0)
def tune(n_people, simulation_days, seed):
    # Force COLLECT_LOGS=False
    import config
    config.COLLECT_LOGS = False

    # extra packages required  - plotly-orca psutil networkx glob seaborn
    from simulator import Human
    import pandas as pd
    # import cufflinks as cf
    import matplotlib.pyplot as plt
    # cf.go_offline()
    monitors, tracker = run_simu(n_people=n_people, init_percent_sick=0.01,
                            start_time=datetime.datetime(2020, 2, 28, 0, 0),
                            simulation_days=simulation_days,
                            outfile=None,
                            print_progress=True, seed=seed, other_monitors=[]
                            )
    # stats = monitors[1].data
    # x = pd.DataFrame.from_dict(stats).set_index('time')
    # fig = x[['susceptible', 'exposed', 'infectious', 'removed']].iplot(asFigure=True, title="SEIR")
    # fig.write_image("plots/tune/seir.png")
    timenow = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    data = dict()
    data['contacts'] = dict(tracker.contacts)
    data['cases_per_day'] = tracker.cases_per_day
    data['r_0'] = tracker.r_0
    data['r'] = tracker.r
    data['dist_encounters'] = dict(tracker.dist_encounters)
    data['time_encounters'] = dict(tracker.time_encounters)
    data['day_encounters'] = dict(tracker.day_encounters)
    data['hour_encounters'] = dict(tracker.hour_encounters)
    data['daily_age_group_encounters'] = dict(tracker.daily_age_group_encounters)
    data['age_distribution'] = tracker.age_distribution
    data['sex_distribution'] = tracker.sex_distribution
    data['house_size'] = tracker.house_size
    data['house_age'] = tracker.house_age
    data['symptoms'] = dict(tracker.symptoms)
    data['transition_probability'] = dict(tracker.transition_probability)

    import dill
    with open(f"logs/tracker_data_n_{n_people}_seed_{seed}_{timenow}.pkl", 'wb') as f:
        dill.dump(data, f)

    logfile = os.path.join(f"logs/log_n_{n_people}_seed_{seed}_{timenow}.txt")
    tracker.write_metrics(logfile)


    # fig = x['R'].iplot(asFigure=True, title="R0")
    # fig.write_image("plots/tune/R.png")
    #
    # x = pd.DataFrame.from_dict(stats).set_index('time')
    # x = pd.DataFrame.from_dict(tracker.contacts['all'])
    # x = x[sorted(x.columns)]
    # x = x + x.transpose()
    # x /= x.sum(1)
    #
    # x = pd.DataFrame.from_dict(tracker.contacts['human_infection'])
    # x = x[sorted(x.columns)]
    # fig = x.iplot(kind='heatmap', asFigure=True)
    # fig.write_image("plots/tune/human_infection_contacts.png")
    #
    # tracker.plot_metrics(dirname="plots/tune")


@simu.command()
def test():
    import unittest
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='*_test.py')

    runner = unittest.TextTestRunner()
    assert runner.run(suite).wasSuccessful()



def run_simu(n_people=None, init_percent_sick=0,
             start_time=datetime.datetime(2020, 2, 28, 0, 0),
             simulation_days=10,
             outfile=None, out_chunk_size=None,
             print_progress=False, seed=0, other_monitors=[]):

    rng = np.random.RandomState(seed)
    env = Env(start_time)
    city_x_range = (0,1000)
    city_y_range = (0,1000)
    city = City(env, n_people, rng, city_x_range, city_y_range, start_time, init_percent_sick, Human)
    monitors = [EventMonitor(f=120, dest=outfile, chunk_size=out_chunk_size), SEIRMonitor(f=1440)]

    # run the simulation
    if print_progress:
        monitors.append(TimeMonitor(1440)) # print every day

    if other_monitors:
        monitors += other_monitors

    for human in city.humans:
        env.process(human.run(city=city))

    for m in monitors:
        env.process(m.run(env, city=city))
    env.run(until=simulation_days * 24 * 60 / TICK_MINUTE)

    return monitors, city.tracker


if __name__ == "__main__":
    simu()
