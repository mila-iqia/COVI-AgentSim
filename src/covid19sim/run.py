import click
import os
import yaml

from covid19sim.frozen.helper import SYMPTOMS_META
from covid19sim.simulator import Human
from covid19sim.base import *
from covid19sim.monitors import EventMonitor, TimeMonitor, SEIRMonitor
from covid19sim.configs import config
from covid19sim.configs.exp_config import ExpConfig
from covid19sim.configs.constants import TICK_MINUTE

@click.group()
def simu():
    pass


@simu.command()
@click.option('--n_people', help='population of the city', type=int, default=100)
@click.option('--init_percent_sick', help='% of population initially sick', type=float, default=0.01)
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=30)
@click.option('--out_chunk_size', help='minimum number of events per dump in outfile', type=int, default=1, required=False)
@click.option('--outdir', help='the directory to write data to', type=str, default="output", required=False)
@click.option('--seed', help='seed for the process', type=int, default=0)
@click.option('--n_jobs', help='number of parallel procs to query the risk servers with', type=int, default=1)
@click.option('--port', help='which port should we look for inference servers on', type=int, default=6688)
@click.option('--exp_config_path', help='where is the configuration file for this experiment', type=str, default="configs/naive_config.yml")
def sim(n_people=None,
        init_percent_sick=0,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=30,
        outdir=None, out_chunk_size=None,
        seed=0, n_jobs=1, port=6688, exp_config_path="configs/naive_config.yml"):

    # Load the experimental configuration
    ExpConfig.load_config(exp_config_path)

    ExpConfig.config['COLLECT_LOGS'] = True

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
@click.option('--simulation_days', help='number of days to run the simulation for', type=int, default=50)
@click.option('--seed', help='seed for the process', type=int, default=0)
@click.option('--exp_config_path', help='where is the configuration file for this experiment', type=str, default="configs/naive_config.yml")
def tune(n_people, simulation_days, seed, exp_config_path):

    # Load the experimental configuration
    ExpConfig.load_config(exp_config_path)

    # Force COLLECT_LOGS=False
    ExpConfig.set('COLLECT_LOGS', False)

    # extra packages required  - plotly-orca psutil networkx glob seaborn
    import pandas as pd
    # import cufflinks as cf
    import matplotlib.pyplot as plt
    # cf.go_offline()

    monitors, tracker = run_simu(n_people=n_people, init_percent_sick=0.0025,
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
    data['intervention'] = config.INTERVENTION

    data['expected_mobility'] = tracker.expected_mobility
    data['mobility'] = tracker.mobility
    data['n_init_infected'] = tracker.n_infected_init
    data['risk_precision'] = tracker.risk_precision_daily
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
    import dill
    filename = f"tracker_data_n_{n_people}_seed_{seed}_{timenow}.pkl"
    with open(f"logs2/{filename}", 'wb') as f:
        dill.dump(data, f)
    #
    logfile = os.path.join(f"logs/log_n_{n_people}_seed_{seed}_{timenow}.txt")
    tracker.write_metrics(logfile)
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
@click.option('--exp_config_path', help='where is the configuration file for this experiment', type=str, default="configs/naive_config.yml")
def tracing(n_people, days, tracing, order, symptoms, risk, noise, exp_config_path):
    ExpConfig.load_config(exp_config_path)

    # TODO: we should have a specific config for this and not be setting them manually.
    ExpConfig.set('COLLECT_LOGS', False)
    ExpConfig.set('COLLECT_TRAINING_DATA', False)
    ExpConfig.set('USE_INFERENCE_SERVER', False)

    if tracing != "":
        ExpConfig.set('INTERVENTION_DAY', 20) # approx 512 will be infected by then
        ExpConfig.set('INTERVENTION', "Tracing")
        ExpConfig.set('RISK_MODEL', tracing)

        # noise
        if tracing == "manual":
            ExpConfig.set('MANUAL_TRACING_NOISE', noise)
        else:
            ExpConfig.set('P_HAS_APP', noise)

        #symptoms (not used in risk_model = transformer)
        ExpConfig.set('TRACE_SYMPTOMS', symptoms)

        #risk (not used in risk_model = transformer)
        ExpConfig.set('TRACE_RISK_UPDATE', risk)

        # order (not used in risk_model = transformer)
        ExpConfig.set('TRACING_ORDER', order)

        # set filename
        if tracing != "transformer":
            name = f"{tracing}-s{1*symptoms}-r{risk}-o{order}"
        else:
            name = "transformer"

    else:
        # no intervention
        ExpConfig.set('INTERVENTION_DAY', -1)
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
    data['intervention_day'] = ExpConfig.get['INTERVENTION_DAY']
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
    monitors = [EventMonitor(f=1800, dest=outfile, chunk_size=out_chunk_size), SEIRMonitor(f=1440)]

    # run the simulation
    if print_progress:
        monitors.append(TimeMonitor(1440)) # print every day

    if other_monitors:
        monitors += other_monitors

    # run city
    all_possible_symptoms = [""] * len(SYMPTOMS_META)
    for k, v in SYMPTOMS_META.items():
        all_possible_symptoms[v] = k
    monitors[0].dump()
    monitors[0].join_iothread()
    # run this every hour
    env.process(city.run(1440/24, outfile, start_time, all_possible_symptoms, port, n_jobs))

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
