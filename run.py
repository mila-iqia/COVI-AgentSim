from monitors import EventMonitor, TimeMonitor, SEIRMonitor
from base import *
from utils import _draw_random_discreet_gaussian, _get_random_age, _get_random_area
import datetime
import click
from config import TICK_MINUTE
import numpy as np


@click.group()
def simu():
    pass


@simu.command()
@click.option('--n_people', help='population of the city', type=int, default=100)
@click.option('--n_stores', help='number of grocery stores in the city', type=int, default=100)
@click.option('--n_parks', help='number of parks in the city', type=int, default=20)
@click.option('--n_misc', help='number of non-essential establishments in the city', type=int, default=100)
@click.option('--init_percent_sick', help='% of population initially sick', type=float, default=0.01)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--outfile', help='filename of the output (file format: .pkl)', type=str, required=False)
@click.option('--print_progress', is_flag=True, help='print the evolution of days', default=False)
@click.option('--seed', help='seed for the process', type=int, default=0)
def sim(n_stores=None, n_people=None, n_parks=None, n_misc=None,
        init_percent_sick=0, store_capacity=30, misc_capacity=30,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=10,
        outfile=None,
        print_progress=False, seed=0):
    from simulator import Human
    monitors = run_simu(
        n_stores=n_stores, n_people=n_people, n_parks=n_parks, n_misc=n_misc,
        init_percent_sick=init_percent_sick, store_capacity=store_capacity, misc_capacity=misc_capacity,
        start_time=start_time,
        simulation_days=simulation_days,
        outfile=outfile,
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

    monitors = run_simu(
        n_stores=20, n_people=100, n_parks=10, n_misc=20,
        init_percent_sick=0.01, store_capacity=30, misc_capacity=30,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=60,
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
def test():
    import unittest
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='*_test.py')

    runner = unittest.TextTestRunner()
    runner.run(suite)



def run_simu(n_stores=None, n_people=None, n_parks=None, n_misc=None,
             init_percent_sick=0, store_capacity=30, misc_capacity=30,
             start_time=datetime.datetime(2020, 2, 28, 0, 0),
             simulation_days=10,
             outfile=None,
             print_progress=False, seed=0, Human=None):

    if Human is None:
        from simulator import Human

    rng = np.random.RandomState(seed)
    env = Env(start_time)
    city_limit = ((0, 1000), (0, 1000))
    total_area = (city_limit[0][1]-city_limit[0][0])*(city_limit[1][1]-city_limit[1][0])
    area_dict = {'store':_get_random_area('store', n_stores, total_area, rng), 
                 'park':_get_random_area('park',n_parks, total_area, rng),
                 'misc':_get_random_area('misc',n_misc, total_area, rng),
                 'household':_get_random_area('household', int(n_people/2), total_area, rng),
                 'workplace':_get_random_area('workplace', int(n_people/30), total_area, rng)}
    
    stores = [
        Location(
            env, rng,
            capacity=_draw_random_discreet_gaussian(store_capacity, int(0.5 * store_capacity), rng),
            cont_prob=0.6,
            location_type='store',
            name=f'store{i}',
            area = area_dict['store'][i],
            lat=rng.randint(*city_limit[0]),
            lon=rng.randint(*city_limit[1]),
            surface_prob=[0.1, 0.1, 0.3, 0.2, 0.3]
        )
        for i in range(n_stores)]

    parks = [
        Location(
            env, rng,
            cont_prob=0.05,
            name=f'park{i}',
            area = area_dict['park'][i],
            location_type='park',
            lat=rng.randint(*city_limit[0]),
            lon=rng.randint(*city_limit[1]),
            surface_prob=[0.7, 0.05, 0.05, 0.1, 0.1]
        )
        for i in range(n_parks)
    ]
    households = [
        Location(
            env, rng,
            cont_prob=1,
            name=f'household{i}',
            location_type='household',
            area = area_dict['household'][i],
            lat=rng.randint(*city_limit[0]),
            lon=rng.randint(*city_limit[1]),
            surface_prob=[0.05, 0.05, 0.05, 0.05, 0.8]
        )
        for i in range(int(n_people / 2))
    ]
    workplaces = [
        Location(
            env, rng,
            cont_prob=0.3,
            name=f'workplace{i}',
            location_type='workplace',
            area = area_dict['workplace'][i],
            lat=rng.randint(*city_limit[0]),
            lon=rng.randint(*city_limit[1]),
            surface_prob=[0.1, 0.1, 0.3, 0.2, 0.3]
        )
        for i in range(int(n_people / 30))
    ]
    miscs = [
        Location(
            env, rng,
            cont_prob=1,
            capacity=_draw_random_discreet_gaussian(misc_capacity, int(0.5 * misc_capacity), rng),
            name=f'misc{i}',
            area = area_dict['misc'][i],
            location_type='misc',
            lat=rng.randint(*city_limit[0]),
            lon=rng.randint(*city_limit[1]),
            surface_prob=[0.1, 0.1, 0.3, 0.2, 0.3]
        ) for i in range(n_misc)
    ]

    humans = [
        Human(
            env=env,
            name=i,
            rng=rng,
            age=_get_random_age(rng),
            infection_timestamp=start_time if i < n_people * init_percent_sick else None,
            household=rng.choice(households),
            workplace=rng.choice(workplaces)
        )
        for i in range(n_people)]

    city = City(stores=stores, parks=parks, humans=humans, miscs=miscs)
    monitors = [EventMonitor(f=120), SEIRMonitor(f=1440)]

    # run the simulation
    if print_progress:
        monitors.append(TimeMonitor(60))

    for human in humans:
        env.process(human.run(city=city))

    for m in monitors:
        env.process(m.run(env, city=city))
    env.run(until=simulation_days * 24 * 60 / TICK_MINUTE)
    return monitors


if __name__ == "__main__":
    simu()
