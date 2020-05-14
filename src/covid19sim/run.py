"""
[summary]
"""
import click
import os

from covid19sim.frozen.helper import SYMPTOMS_META
from covid19sim.simulator import Human
from covid19sim.base import *
from covid19sim.monitors import EventMonitor, TimeMonitor, SEIRMonitor
from covid19sim.configs.exp_config import ExpConfig
from covid19sim.configs.constants import *

@click.group()
def simu():
    """
    [summary]
    """
    pass


@simu.command()
@click.option('--n_people', help='population of the city', type=int, default=100)
@click.option('--init_percent_sick', help='initial percentage of sick people', type=float, default=0.01)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--out_chunk_size', help='minimum number of events per dump in outfile', type=int, default=1, required=False)
@click.option('--outdir', help='the directory to write data to', type=str, default="output", required=False)
@click.option('--seed', help='seed for the process', type=int, default=0)
@click.option('--n_jobs', help='number of parallel procs to query the risk servers with', type=int, default=1)
@click.option('--port', help='which port should we look for inference servers on', type=int, default=6688)
@click.option('--config', help='where is the configuration file for this experiment', type=str, default="configs/naive_config.yml")
def sim(n_people=None,
        init_percent_sick=0.01,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=30,
        outdir=None, out_chunk_size=None,
        seed=0, n_jobs=1, port=6688, config="configs/naive_config.yml"):
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
    ExpConfig.load_config(config)

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
        seed=seed, n_jobs=n_jobs, port=port
    )
    monitors[0].dump()
    monitors[0].join_iothread()

    # write metrics
    logfile = os.path.join(f"{outdir}/logs.txt")
    tracker.write_metrics(logfile)


@simu.command()
@click.option('--n_people', help='population of the city', type=int, default=1000)
@click.option('--init_percent_sick', help='initial percentage of sick people', type=float, default=0.01)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=50)
@click.option('--seed', help='seed for the process', type=int, default=0)
@click.option('--outdir', help='the directory to write data to', type=str, default="tune", required=False)
@click.option('--config', help='where is the configuration file for this experiment', type=str, default="configs/no_intervention.yml")
@click.option('--n_jobs', help='number of parallel procs to query the risk servers with', type=int, default=1)
@click.option('--name', help='name of the file to append metrics file', type=str, default="")
def tune(n_people, init_percent_sick, simulation_days, seed, outdir, config, n_jobs, name):
    """
    [summary]

    Args:
        n_people ([type]): [description]
        simulation_days ([type]): [description]
        seed ([type]): [description]
        config ([type]): [description]
    """

    # Load the experimental configuration
    ExpConfig.load_config(config)

    # extra packages required  - plotly-orca psutil networkx glob seaborn
    import pandas as pd
    # import cufflinks as cf
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    # cf.go_offline()
    monitors, tracker = run_simu(n_people=n_people,
                            init_percent_sick=init_percent_sick,
                            start_time=datetime.datetime(2020, 2, 28, 0, 0),
                            simulation_days=simulation_days,
                            outfile=None,
                            print_progress=True, n_jobs=n_jobs, seed=seed, other_monitors=[],
                            )

    # stats = monitors[1].data
    # x = pd.DataFrame.from_dict(stats).set_index('time')
    # fig = x[['susceptible', 'exposed', 'infectious', 'removed']].iplot(asFigure=True, title="SEIR")
    # fig.write_image("plots/tune/seir.png")
    timenow = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    data = dict()
    data['intervention_day'] = ExpConfig.get('INTERVENTION_DAY')
    data['intervention'] = ExpConfig.get('INTERVENTION')
    data['risk_model'] = ExpConfig.get('RISK_MODEL')

    data['expected_mobility'] = tracker.expected_mobility
    data['serial_interval'] = tracker.get_generation_time()
    data['mobility'] = tracker.mobility
    data['n_init_infected'] = tracker.n_infected_init
    data['contacts'] = dict(tracker.contacts)
    data['cases_per_day'] = tracker.cases_per_day
    data['ei_per_day'] = tracker.ei_per_day
    data['r_0'] = tracker.r_0
    data['R'] = tracker.r
    data['n_humans'] = tracker.n_humans
    data['s'] = tracker.s_per_day
    data['e'] = tracker.e_per_day
    data['i'] = tracker.i_per_day
    data['r'] = tracker.r_per_day
    data['avg_infectiousness_per_day'] = tracker.avg_infectiousness_per_day
    data['risk_precision_global'] = tracker.compute_risk_precision(False)
    data['risk_precision'] = tracker.risk_precision_daily
    data['human_monitor'] = tracker.human_monitor
    data['infection_monitor'] = tracker.infection_monitor
    data['infector_infectee_update_messages'] = tracker.infector_infectee_update_messages
    data['risk_attributes'] = tracker.risk_attributes
    data['feelings'] = tracker.feelings
    data['rec_feelings'] = tracker.rec_feelings
    data['outside_daily_contacts'] = tracker.outside_daily_contacts
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
    if name:
        import dill
        filename = f"tracker_data_n_{n_people}_seed_{seed}_{timenow}_{name}.pkl"
        with open(f"{outdir}/{filename}", 'wb') as f:
            dill.dump(data, f)
    #
    # logfile = os.path.join(f"logs3/log_n_{n_people}_seed_{seed}_{timenow}_{name}.txt")
    # tracker.write_metrics(logfile)
    # tracker.write_metrics(None)

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

def run_simu(n_people=None,
             init_percent_sick=0.01,
             start_time=datetime.datetime(2020, 2, 28, 0, 0),
             simulation_days=10,
             outfile=None, out_chunk_size=None,
             print_progress=False, seed=0, port=6688, n_jobs=1, other_monitors=[]):
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
    city_x_range = (0,1000)
    city_y_range = (0,1000)
    city = City(env, n_people, init_percent_sick, rng,
                city_x_range, city_y_range, Human)

    # Add monitors
    monitors = [
        EventMonitor(f=SECONDS_PER_HOUR*30, dest=outfile, chunk_size=out_chunk_size),
        SEIRMonitor (f=SECONDS_PER_DAY),
    ]
    if print_progress:
        monitors.append(TimeMonitor(SECONDS_PER_DAY))
    if other_monitors:
        monitors += other_monitors

    # Kickstart EventMonitor
    monitors[0].dump()
    monitors[0].join_iothread()

    # Initiate city process, which runs every hour
    all_possible_symptoms = [""] * len(SYMPTOMS_META)
    for k, v in SYMPTOMS_META.items():
        all_possible_symptoms[v] = k
    env.process(city.run(SECONDS_PER_HOUR, outfile, all_possible_symptoms, port, n_jobs))

    # Initiate human processes
    for human in city.humans:
        env.process(human.run(city=city))

    # Initiate monitor processes
    for m in monitors:
        env.process(m.run(env, city=city))

    # Run simulation until termination
    env.run(until=env.ts_initial+simulation_days*SECONDS_PER_DAY)

    # Return
    return monitors, city.tracker


if __name__ == "__main__":
    simu()
