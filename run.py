import datetime
import click
import numpy as np
import math
import pickle
import os
import sys
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
@click.option('--n_jobs', help='number of parallel procs to query the risk servers with', type=int, default=1)
@click.option('--port', help='which port should we look for inference servers on', type=int, default=6688)
def sim(n_people=None,
        init_percent_sick=0,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=30,
        outdir=None, out_chunk_size=None,
        seed=0, n_jobs=1, port=6688):

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
        seed=seed, n_jobs=n_jobs, port=port,
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
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=50)
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
    data['intervention_day'] = config.INTERVENTION_DAY

    data['mobility'] = tracker.mobility
    data['n_init_infected'] = tracker.n_infected_init
    data['risk_precision'] = tracker.risk_precision_daily
    data['contacts'] = dict(tracker.contacts)
    data['cases_per_day'] = tracker.cases_per_day
    data['ei_per_day'] = tracker.ei_per_day
    data['r_0'] = tracker.r_0
    data['r'] = tracker.r
    data['n_humans'] = tracker.n_humans
    data['s'] = tracker.s_per_day
    data['e'] = tracker.e_per_day
    data['i'] = tracker.i_per_day
    data['r'] = tracker.r_per_day
    data['avg_infectiousness_per_day'] = tracker.avg_infectiousness_per_day
    data['risk_precision'] = tracker.compute_risk_precision(False)
    # data['dist_encounters'] = dict(tracker.dist_encounters)
    # data['time_encounters'] = dict(tracker.time_encounters)
    # data['day_encounters'] = dict(tracker.day_encounters)
    # data['hour_encounters'] = dict(tracker.hour_encounters)
    # data['daily_age_group_encounters'] = dict(tracker.daily_age_group_encounters)
    # data['age_distribution'] = tracker.age_distribution
    # data['sex_distribution'] = tracker.sex_distribution
    # data['house_size'] = tracker.house_size
    # data['house_age'] = tracker.house_age
    # data['symptoms'] = dict(tracker.symptoms)
    # data['transition_probability'] = dict(tracker.transition_probability)
    #
    # import dill
    # filename = f"tracker_data_n_{n_people}_seed_{seed}_{timenow}.pkl"
    # with open(f"logs/{filename}", 'wb') as f:
    #     dill.dump(data, f)
    #
    # logfile = os.path.join(f"logs/log_n_{n_people}_seed_{seed}_{timenow}.txt")
    # tracker.write_metrics(logfile)
    tracker.write_metrics(None)

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
@click.option('--n_people', help='population of the city', type=int, default=2000)
@click.option('--days', help='number of days to run the simulation for', type=int, default=60)
@click.option('--tracing', help='which tracing method', type=str, default="")
@click.option('--order', help='trace to which depth?', type=int, default=1)
@click.option('--symptoms', help='trace symptoms?', type=bool, default=False)
@click.option('--risk', help='trace risk updates?', type=bool, default=False)
@click.option('--noise', help='noise', type=float, default=0.5)
def tracing(n_people, days, tracing, order, symptoms, risk, noise):
    import config
    config.COLLECT_LOGS = False

    # switch off
    config.COLLECT_TRAINING_DATA = False
    config.USE_INFERENCE_SERVER = False

    if tracing != "":

        config.INTERVENTION_DAY = 25 # approx 512 will be infected by then
        config.INTERVENTION = "Tracing"
        config.RISK_MODEL = tracing

        # noise
        if tracing == "manual":
            config.MANUAL_TRACING_NOISE = noise
        else:
            config.P_HAS_APP = noise

        #symptoms
        config.TRACE_SYMPTOMS = symptoms

        #risk
        config.TRACE_SYMPTOMS = risk

        # order
        config.TRACING_ORDER = order
        name = f"{tracing}-s{1*symptoms}-r{risk}-o{order}"

    else:
        # no intervention
        config.INTERVENTION_DAY = -1
        name = "unmitigated"

    monitors, tracker = run_simu(n_people=n_people, init_percent_sick=0.0025,
                        start_time=datetime.datetime(2020, 2, 28, 0, 0),
                        simulation_days=days,
                        outfile=None,
                        print_progress=True, seed=1234, other_monitors=[]
                        )

    timenow = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    data = dict()
    data['tracing'] = tracing
    data['symptoms'] = symptoms
    data['order'] = order
    data['intervention_day'] = config.INTERVENTION_DAY
    data['noise'] = noise

    data['mobility'] = tracker.mobility
    data['n_init_infected'] = tracker.n_infected_init
    data['risk_precision'] = tracker.risk_precision_daily
    data['contacts'] = dict(tracker.contacts)
    data['cases_per_day'] = tracker.cases_per_day
    data['ei_per_day'] = tracker.ei_per_day
    data['r_0'] = tracker.r_0
    data['r'] = tracker.r
    data['n_humans'] = tracker.n_humans
    data['s'] = tracker.s_per_day
    data['e'] = tracker.e_per_day
    data['i'] = tracker.i_per_day
    data['r'] = tracker.r_per_day
    data['avg_infectiousness_per_day'] = tracker.avg_infectiousness_per_day
    data['risk_precision'] = tracker.compute_risk_precision(False)

    import dill
    timenow = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = f"tracing_data_n_{n_people}_{timenow}_{name}.pkl"
    with open(f"logs/compare/{filename}", 'wb') as f:
        dill.dump(data, f)

    # logfile = os.path.join(f"logs/log_n_{n_people}_seed_{seed}_{timenow}.txt")
    # tracker.write_metrics(None)

def run_simu(n_people=None, init_percent_sick=0.0,
             start_time=datetime.datetime(2020, 2, 28, 0, 0),
             simulation_days=10,
             outfile=None, out_chunk_size=None,
             print_progress=False, seed=0, port=6688, n_jobs=1, other_monitors=[]):

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

    # run city
    all_possible_symptoms = ['moderate', 'mild', 'severe', 'extremely-severe', 'fever',
                             'chills', 'gastro', 'diarrhea', 'nausea_vomiting', 'fatigue',
                             'unusual', 'hard_time_waking_up', 'headache', 'confused',
                             'lost_consciousness', 'trouble_breathing', 'sneezing',
                             'cough', 'runny_nose', 'aches', 'sore_throat', 'severe_chest_pain',
                             'loss_of_taste', 'mild_trouble_breathing', 'light_trouble_breathing', 'moderate_trouble_breathing',
                             'heavy_trouble_breathing']
    monitors[0].dump()
    monitors[0].join_iothread()
    env.process(city.run(1440, outfile, start_time, all_possible_symptoms, port, n_jobs))

    # run humans
    for human in city.humans:
        env.process(human.run(city=city))

    # run monitors
    for m in monitors:
        env.process(m.run(env, city=city))

    env.run(until=simulation_days * 24 * 60 / TICK_MINUTE)

    return monitors, city.tracker


if __name__ == "__main__":
    simu()
