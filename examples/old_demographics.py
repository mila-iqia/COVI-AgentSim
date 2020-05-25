import os
"""
Main file to run the simulations
"""
import os
import datetime

import numpy as np
from covid19sim.base import City, Env

from covid19sim.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from covid19sim.monitors import EventMonitor, SEIRMonitor, TimeMonitor
from covid19sim.simulator import Human
from covid19sim.utils import (
    dump_tracker_data,
    extract_tracker_data,
    parse_configuration,
    dump_conf
)
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../src/covid19sim/hydra-configs/config.yaml")
def age_mixing(conf: DictConfig) -> None:
    """ Run a simulation of 1 infection in a seniors residence, and perform some sanity checks """
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


    rng = np.random.RandomState(seed)

    outfile = os.path.join(outdir, "data")

    ExpConfig.load_config(Path.cwd().parent.joinpath('src', 'covid19sim', 'configs', 'naive_config.yml'))

    start_date = datetime.fromisoformat(start_date)
    env = Env(start_date)

    city_x_range = (0, 1000)
    city_y_range = (0, 1000)
    city = City(
        env=env,
        n_people=n_people,
        init_percent_sick=init_percent_sick,
        rng=rng,
        x_range=city_x_range,
        y_range=city_y_range,
        start_time=start_date,
        Human=Human)

    # Demographics
    population = []
    for human in city.humans:
        population.append([
            human.age,
            human.sex,
            human.has_app,
            human.profession,
            human.workplace
        ])

    df = pd.DataFrame.from_records(
        data=population,
        columns=['age', 'sex', 'has_app', 'profession', 'workplace']
    )
    print(df)

    age_groups = [(low, up+1) for low, up in APP_USERS_FRACTION_BY_AGE.keys()]
    intervals = pd.IntervalIndex.from_tuples(age_groups, closed='left')
    age_grouped = df.groupby(pd.cut(df['age'], intervals))
    stats = age_grouped.agg({
        'age': 'count',
        'has_app': ['sum', 'mean']
    })
    assert(stats.age.sum() == n_people)
    stats = stats.age.apply(lambda x: x/n_people)

    print(stats)

    # -----------------------------------

    monitors = [
        EventMonitor(f=1800, dest=outfile, chunk_size=1),
        SEIRMonitor(f=1440),
        TimeMonitor(1440),
    ]
    all_possible_symptoms = [""] * len(SYMPTOMS_META)
    for k, v in SYMPTOMS_META.items():
        all_possible_symptoms[v] = k
    env.process(city.run(1440 / 24, outfile, start_date, all_possible_symptoms, 6688, 1))

    # run humans
    for human in city.humans:
        env.process(human.run(city=city))

    # run monitors
    for m in monitors:
        env.process(m.run(env, city=city))

    env.run(until=simulation_days * 24 * 60 / TICK_MINUTE)



    df_female = df.loc[df['sex'] == 'female']
    #print(df_female['age'].to_numpy())


    # sex_age_distribution = {'male': [], 'female': []}
    # app_sex_age_distribution = {'male': [], 'female': []}

    # professions = ['healthcare', 'school', 'others', 'retired']
    # profession_age_distribution = {key: [] for key in professions}
    # for human in city.humans:
    #     sex_age_distribution[human.sex].append(human.age)
    #     if human.has_app:
    #         app_sex_age_distribution.append(human.age)

    # # Location age distribution

    contacts = city.tracker.contacts
    contact_matrix = contacts['duration']['n']

    contact_pklfile = open('contact10k.pkl', 'wb')
    pickle.dump(contacts, contact_pklfile)
    contact_pklfile.close()

    pt = PlotTracker()
    plot_age_mixing(contact_matrix/np.sum(contact_matrix))


if __name__ == '__main__':
    age_mixing()
