from simulator import *
from utils import _draw_random_discreet_gaussian
import argparse

def sim(n_stores, n_people, n_parks, n_misc, init_percent_sick=0, store_capacity=30, misc_capacity=30, outfile=None, print_progress=False):
    env = simpy.Environment()
    city_limit = ((0, 1000), (0, 1000))
    stores = [
              Location(
                  env,
                  capacity=_draw_random_discreet_gaussian(store_capacity, int(0.5 * store_capacity)),
                  cont_prob=0.1,
                  type='store',
                  name=f'store{i}',
                  lat=random.randint(*city_limit[0]),
                  lon=random.randint(*city_limit[1]),
              )
              for i in range(n_stores)]
    parks = [
             Location(
                 env, cont_prob=0.02,
                 name=f'park{i}',
                 type='park',
                 lat=random.randint(*city_limit[0]),
                  lon=random.randint(*city_limit[1])
             )
             for i in range(n_parks)
             ]
    households = [
             Location(
                 env, cont_prob=1,
                 name=f'household{i}',
                 type='household',
                lat=random.randint(*city_limit[0]),
                  lon=random.randint(*city_limit[1]),
            )
             for i in range(int(n_people/2))
             ]
    workplaces = [
             Location(
                 env, cont_prob=1,
                 name=f'workplace{i}',
                 type='workplace',
                lat=random.randint(*city_limit[0]),
                  lon=random.randint(*city_limit[1]),
            )
             for i in range(int(n_people/30))
             ]
    miscs = [
        Location(
            env, cont_prob=1,
            capacity=_draw_random_discreet_gaussian(misc_capacity, int(0.5 * misc_capacity)),
            name=f'misc{i}',
            type='misc',
            lat=random.randint(*city_limit[0]),
            lon=random.randint(*city_limit[1])
        ) for i in range(n_misc)
    ]

    humans = [
        Human(
            i, is_sick= i < n_people * init_percent_sick,
            household=np.random.choice(households),
            workplace=np.random.choice(workplaces)
            )
    for i in range(n_people)]

    clock=Clock(env)
    city = City(stores=stores, parks=parks, humans=humans, miscs=miscs, clock=clock)
    monitor = EventMonitor(f=120)

    # run the simulation
    if print_progress:
        env.process(clock.run()) # to monitor progress

    for human in humans:
      env.process(human.run(env, city=city))
    env.process(monitor.run(env, city=city))
    env.run(until=SIMULATION_DAYS*24*60/TICK_MINUTE)

    monitor.dump(outfile)
    return monitor.data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--n_people', help='population of the city', type=int, default=1000)
    parser.add_argument( '--n_stores', help='number of grocery stores in the city', type=int, default=100)
    parser.add_argument( '--n_parks', help='number of parks in the city', type=int, default=20)
    parser.add_argument( '--n_miscs', help='number of non-essential establishments in the city', type=int, default=100)
    parser.add_argument( '--init_percent_sick', help='% of population initially sick', type=float, default=0.01)
    parser.add_argument( '--outfile', help='filename of the output (file format: .pkl)', type=str, default="data")
    parser.add_argument( '--print_progress', help='print the evolution of days', action='store_tree')
    args = parser.parse_args()

    data = sim(n_stores=args.n_stores, n_parks=args.n_parks,
                    n_people=args.n_people, n_misc=args.n_miscs,
                    init_percent_sick=args.init_percent_sick, outfile=args.outfile, print_progress=args.print_progress)
