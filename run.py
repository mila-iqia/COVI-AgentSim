from monitors import EventMonitor, TimeMonitor, SEIRMonitor
from base import *
from utils import _draw_random_discreet_gaussian, _get_random_age, _get_random_area
import datetime
import click
from config import TICK_MINUTE
import numpy as np
import math
import pickle


@click.group()
def simu():
    pass


@simu.command()
@click.option('--n_people', help='population of the city', type=int, default=100)
@click.option('--init_percent_sick', help='% of population initially sick', type=float, default=0.01)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--outfile', help='filename of the output (file format: .pkl)', type=str, default="output/data", required=False)
@click.option('--out_humans', help='filename of the output (file format: .pkl)', type=str, default="output/humans.pkl", required=False)
@click.option('--print_progress', is_flag=True, help='print the evolution of days', default=False)
@click.option('--seed', help='seed for the process', type=int, default=0)
def sim(n_people=None,
        init_percent_sick=0, store_capacity=30, misc_capacity=30,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=10,
        outfile=None, out_humans=None,
        print_progress=False, seed=0):
    from simulator import Human
    monitors, _= run_simu(
        n_people=n_people,
        init_percent_sick=init_percent_sick, store_capacity=store_capacity, misc_capacity=misc_capacity,
        start_time=start_time,
        simulation_days=simulation_days,
        outfile=outfile,
        out_humans=out_humans,
        print_progress=print_progress,
        seed=seed
    )
    monitors[0].dump(outfile)
    return monitors[0].data


@simu.command()
@click.option('--toy_human', is_flag=True, help='run the Human from toy.py')
def base(toy_human):
    if toy_human:
        from toy import Human
    else:
        from simulator import Human
    import pandas as pd
    import cufflinks as cf
    cf.go_offline()

    monitors, tracker = run_simu(
        n_stores=20, n_people=100, n_parks=10, n_misc=20, n_hospitals=2,
        init_percent_sick=0.01, store_capacity=30, misc_capacity=30,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=30,
        outfile=None,
        print_progress=False, seed=0, Human=Human,
    )
    stats = monitors[1].data
    x = pd.DataFrame.from_dict(stats).set_index('time')
    fig = x[['susceptible', 'exposed', 'infectious', 'removed']].iplot(asFigure=True, title="SEIR")
    fig.show()

    fig = x['R'].iplot(asFigure=True, title="R0")
    fig.show()


@simu.command()
def tune():
    # extra packages required  - plotly-orca psutil networkx glob seaborn
    from simulator import Human
    import pandas as pd
    import cufflinks as cf
    import matplotlib.pyplot as plt
    cf.go_offline()

    monitors, tracker = run_simu(n_people=1000, init_percent_sick=0.02,
        store_capacity=30, misc_capacity=30,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=30,
        outfile=None,
        print_progress=True, seed=0, Human=Human, other_monitors=[]
    )
    stats = monitors[1].data
    x = pd.DataFrame.from_dict(stats).set_index('time')
    fig = x[['susceptible', 'exposed', 'infectious', 'removed']].iplot(asFigure=True, title="SEIR")
    fig.write_image("plots/tune/seir.png")

    tracker.write_metrics()
    import pdb; pdb.set_trace()
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
    runner.run(suite)



def run_simu(n_people=None, init_percent_sick=0, store_capacity=30, misc_capacity=30,
             start_time=datetime.datetime(2020, 2, 28, 0, 0),
             simulation_days=10,
             outfile=None, out_humans=None,
             print_progress=False, seed=0, Human=None, other_monitors=[]):

    if Human is None:
        from simulator import Human

    rng = np.random.RandomState(seed)
    env = Env(start_time)

    city_x_range = (0,1000)
    city_y_range = (0,1000)
    city = City(env, n_people, rng, city_x_range, city_y_range, start_time, init_percent_sick, Human)

    monitors = [EventMonitor(f=120), SEIRMonitor(f=1440)]
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
