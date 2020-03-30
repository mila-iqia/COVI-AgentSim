from monitors import EventMonitor, TimeMonitor
from simulator import *
from utils import _draw_random_discreet_gaussian
import datetime
import click


@click.group()
def simu():
    pass


@simu.command()
@click.option('--n_people', help='population of the city', type=int, default=1000)
@click.option('--n_stores', help='number of grocery stores in the city', type=int, default=100)
@click.option('--n_parks', help='number of parks in the city', type=int, default=20)
@click.option('--n_miscs', help='number of non-essential establishments in the city', type=int, default=100)
@click.option('--init_percent_sick', help='% of population initially sick', type=float, default=0.01)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--outfile', help='filename of the output (file format: .pkl)', type=str, required=False)
@click.option('--print_progress', is_flag=True, help='print the evolution of days', default=False)
def sim(n_stores=None, n_people=None, n_parks=None, n_miscs=None,
        init_percent_sick=0, store_capacity=30, misc_capacity=30,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=10,
        outfile=None,
        print_progress=False):
    run_simu(
        n_stores=n_stores, n_people=n_people, n_parks=n_parks, n_miscs=n_miscs,
        init_percent_sick=init_percent_sick, store_capacity=store_capacity, misc_capacity=misc_capacity,
        start_time=start_time,
        simulation_days=simulation_days,
        outfile=outfile,
        print_progress=print_progress
    )


def run_simu(n_stores=None, n_people=None, n_parks=None, n_miscs=None,
             init_percent_sick=0, store_capacity=30, misc_capacity=30,
             start_time=datetime.datetime(2020, 2, 28, 0, 0),
             simulation_days=10,
             outfile=None,
             print_progress=False):
    env = Env(start_time)
    city_limit = ((0, 1000), (0, 1000))
    stores = [
        Location(
            env,
            capacity=_draw_random_discreet_gaussian(store_capacity, int(0.5 * store_capacity)),
            cont_prob=0.1,
            location_type='store',
            name=f'store{i}',
            lat=random.randint(*city_limit[0]),
            lon=random.randint(*city_limit[1]),
        )
        for i in range(n_stores)]
    parks = [
        Location(
            env, cont_prob=0.02,
            name=f'park{i}',
            location_type='park',
            lat=random.randint(*city_limit[0]),
            lon=random.randint(*city_limit[1])
        )
        for i in range(n_parks)
    ]
    households = [
        Location(
            env, cont_prob=1,
            name=f'household{i}',
            location_type='household',
            lat=random.randint(*city_limit[0]),
            lon=random.randint(*city_limit[1]),
        )
        for i in range(int(n_people / 2))
    ]
    workplaces = [
        Location(
            env, cont_prob=1,
            name=f'workplace{i}',
            location_type='workplace',
            lat=random.randint(*city_limit[0]),
            lon=random.randint(*city_limit[1]),
        )
        for i in range(int(n_people / 30))
    ]
    miscs = [
        Location(
            env, cont_prob=1,
            capacity=_draw_random_discreet_gaussian(misc_capacity, int(0.5 * misc_capacity)),
            name=f'misc{i}',
            location_type='misc',
            lat=random.randint(*city_limit[0]),
            lon=random.randint(*city_limit[1])
        ) for i in range(n_miscs)
    ]

    humans = [
        Human(
            env=env,
            name=i,
            infection_timestamp=start_time if i < n_people * init_percent_sick else None,
            household=np.random.choice(households),
            workplace=np.random.choice(workplaces)
        )
        for i in range(n_people)]

    city = City(stores=stores, parks=parks, humans=humans, miscs=miscs)
    monitors = [EventMonitor(f=120)]

    # run the simulation
    if print_progress:
        monitors.append(TimeMonitor(60))

    for human in humans:
        env.process(human.run(city=city))

    for m in monitors:
        env.process(m.run(env, city=city))
    env.run(until=simulation_days * 24 * 60 / TICK_MINUTE)

    monitors[0].dump(outfile)
    return monitors[0].data


@simu.command()
def test():
    import unittest
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='*_test.py')

    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    simu()
