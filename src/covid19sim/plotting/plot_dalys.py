import pickle
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

retirement_age = 65
MEAN_HOURLY_WAGE = 27.67
sexes = ['male', 'female']

pd.set_option('display.float_format', '{:.6f}'.format)

disability_weights = {
    'no_hospitalization': 0.051,  # moderate lower respiratory infection
    'hospitalized': 0.133,  # severe respiratory infection
    'critical': 0.408  # severe COPD without heart failure
    }

# 95% Uncertainty Level, lower bound
dw_low = {
    'no_hospitalization': 0.032,  # moderate lower respiratory infection
    'hospitalized': 0.088,  # severe respiratory infection
    'critical': 0.273  # severe COPD without heart failure
    }

# 95% Uncertainty Level, upper bound
dw_high = {
    'no_hospitalization': 0.074,  # moderate lower respiratory infection
    'hospitalized': 0.190,  # severe respiratory infection
    'critical': 0.556  # severe COPD without heart failure
    }

dw_dicts = {"": disability_weights,
            "_low": dw_low,
            "_high": dw_high
            }


def load_tracker(path):
    with open(path, 'rb') as f:
        tracker_data = pickle.load(f)
    return tracker_data


def load_life_expectancies(path):
    """
    Loads life expectancy values from a csv into a DataFrame.
    Attr: path to life expectancy csv
    Returns: DataFrame of life expectancies by age and sex
    """

    # specific to STATS CAN CSV file
    # https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1310011401
    life_expectancies = pd.read_csv(path, header=[7])
    life_expectancies = life_expectancies.rename(columns={
        '2016 to 2018': 'other',  # other -> both sexes
        '2016 to 2018.1': 'male',
        '2016 to 2018.2': 'female',
        })
    life_expectancies = life_expectancies.dropna(subset=['female'])
    life_expectancies.loc[2, 'Age group'] = '1 years'
    # life_expectancies.iloc[-1, 'Age group'] = '110 years'
    life_expectancies.loc[life_expectancies.index[-1],
                          'Age group'] = '110 years'
    life_expectancies = life_expectancies.set_index('Age group')

    return life_expectancies


def get_daly_data(demographics,
                  human_monitor_data,
                  life_expectancies,
                  ):
    """
    Gathers all data required for DALY calculations into one dataFrame.
    Attrs:
        - demographics data from tracker file
        - human monitor data from tracker file
        - life expectancies from output of load_life_expectancies()
    Returns:
        - Dataframe where rows are individuals and
            columns are attributes of individuals
        - columns:
                "days_in_hospital",
                "days_in_ICU",
                "has_died",
                "days_symptoms_and_infection",
                "days_sick_not_in_hospital",
                "was_infected",
                "life_expectancy",
    """

    human_names = [human_monitor_data[datetime.date(2020, 2, 28)][i]['name']
                   for i in range(len(
                       human_monitor_data[datetime.date(2020, 2, 28)]
                                      )
                                  )
                   ]

    symptom_status = {i: {} for i in human_names}
    hospitalization_status = {i: {} for i in human_names}
    ICU_status = {i: {} for i in human_names}
    death_status = {i: {} for i in human_names}
    outpatient_status = {i: {} for i in human_names}
    has_had_infection = {i: {} for i in human_names}

    days_in_hospital = {}
    days_in_ICU = {}
    has_died = {}
    days_with_symptoms = {}
    days_as_outpatient = {}
    was_infected = {}

    for day in human_monitor_data.keys():
        for human in range(len(
            human_monitor_data[datetime.date(2020, 2, 28)]
                               )
                           ):

            human_name = human_monitor_data[day][human]['name']
            is_in_hospital = human_monitor_data[day][human]['is_in_hospital']
            is_in_ICU = human_monitor_data[day][human]['is_in_ICU']
            is_dead = human_monitor_data[day][human]['dead']
            is_symptomatic = human_monitor_data[day][human]['n_symptoms'] > 0
            is_infected = human_monitor_data[
                day][human]['infection_timestamp'] is not None
            # positive_test = human_monitor_data[day][human]['test_result']

            hospitalization_status[human_name][day] = is_in_hospital
            ICU_status[human_name][day] = is_in_ICU
            death_status[human_name][day] = is_dead
            symptom_status[human_name][day] = is_symptomatic and is_infected
            outpatient_status[human_name][day] = is_symptomatic \
                and is_infected \
                and not is_in_hospital \
                and not is_in_ICU
            has_had_infection[human_name][day] = is_infected

    for human in human_names:

        days_in_hospital[human] = sum(hospitalization_status[human].values())
        days_in_ICU[human] = sum(ICU_status[human].values())
        has_died[human] = sum(death_status[human].values()) > 0
        days_with_symptoms[human] = sum(symptom_status[human].values())
        days_as_outpatient[human] = sum(outpatient_status[human].values())
        was_infected[human] = sum(has_had_infection[human].values()) > 0

    daly_df = pd.DataFrame([days_in_hospital,
                            days_in_ICU,
                            has_died,
                            days_with_symptoms,
                            days_as_outpatient,
                            was_infected]).transpose()

    daly_df.columns = ["days_in_hospital",
                       "days_in_ICU",
                       "has_died",
                       "days_symptoms_and_infection",
                       "days_sick_not_in_hospital",
                       "was_infected"
                       ]

    daly_df = pd.merge(pd.DataFrame(demographics).set_index('name'), daly_df,
                       left_index=True,
                       right_index=True
                       )

    daly_df['life_expectancy'] = ""
    # set life expectancy as a function of age and sex
    for human in human_names:

        daly_df.loc[human, 'life_expectancy'] = float(
            life_expectancies[
                daly_df['sex'][human]][str(daly_df['age'][human])+" years"]
            )

    return daly_df


def compute_yll(daly_df,
                discounting=False,
                discount_rate=None):
    """
    Computes Years of Life Lost (YLL)
    If `discounting` is set to True, then discounting will be applied
        according to `discount_rate`.
    """

    # if no discounting, simply take initial life expectancy of those that died
    if discounting is False:
        yll = daly_df['was_infected'] * (
            daly_df['has_died'] * daly_df['life_expectancy']
                                     )

    # if discounting, apply according to discount rate
    if discounting is True:
        raise NotImplementedError('Coming Soon')

    return yll


def compute_yld(daly_df,
                dis_weights=disability_weights,
                discounting=False,
                discount_rate=None):
    """
    Computes Years of Life Disabled (YLD)
    If `discounting` is set to True, then discounting will be applied
        according to `discount_rate`.
    """

    # if no discounting, simply multiply time in state i by disability i
    if discounting is False:
        yld = daly_df['was_infected'] * (
                daly_df['days_sick_not_in_hospital']/365 *
                dis_weights['no_hospitalization'] +

                daly_df['days_in_hospital']/365 *
                dis_weights['hospitalized'] +

                daly_df['days_in_ICU']/365 *
                dis_weights['critical'])

    # if discounting, apply according to discount rate
    if discounting is True:
        raise NotImplementedError("Coming soon")

    return yld


def compute_dalys(daly_df,
                  dis_weights=disability_weights,
                  discounting=False,
                  discount_rate=None):
    """
        Adds YLL and YLD to obtain DALYs.
        Returns yll, yld and DALYs.
    """

    yll = compute_yll(daly_df,
                      discounting=discounting,
                      discount_rate=discount_rate)

    yld = compute_yld(daly_df,
                      dis_weights=disability_weights,
                      discounting=discounting,
                      discount_rate=discount_rate)

    dalys = yll + yld

    return yll, yld, dalys


def lost_work_hours_total(tracker_data, wfh_prod=0.51):
    """
        Adds up lost work hours from all sources (kid, ill, quarantine).
        Sums across age bins 25 to 64.
        The simulator assumes full employment among agents aged 25 to 64.

        Attrs:
            - tracker_data: loaded tracker data file
            - wfh_prod: productivity of agents forced to work from home.
        Returns:
            - float for total lost work hours across all agents aged 25 to 64
    """

    lost_work_hours = (tracker_data['work_hours']['WORK-CANCEL--KID'][5:13] +
                       tracker_data['work_hours']['WORK-CANCEL--ILL'][5:13] +
                       tracker_data['work_hours'][
                           'WORK-CANCEL--QUARANTINE'][5:13] * (1-wfh_prod)
                       )

    return lost_work_hours.sum()


def multiple_seeds_get_data(intervention, l_e_path):
    """
        loads data from multiple runs of the simulator into a dict

        Attrs:
            - intervention: method used for contact tracing
            - l_e_path: path to life_expectancies csv

        Returns:
            - dict where intervention[data_subset][seed] = loaded data
                - data_subset = tracker,
                                demographics,
                                human_monitor,
                                or daly_data
    """
    tracker_dict = {}
    demog_dict = {}
    human_monitor_dict = {}
    daly_data_dict = {}

    for root, subdir, filenames in \
            os.walk(os.path.join('../output/results', intervention)):

        for filename in [f for f in filenames if '.pkl' in f]:

            tracker_dict[filename] = load_tracker(os.path.join(root, filename))
            demog_dict[filename] = tracker_dict[
                filename]['humans_demographics']
            human_monitor_dict[filename] = tracker_dict[
                filename]['human_monitor']
            life_expectancies = load_life_expectancies(path=l_e_path)
            daly_data_dict[filename] = get_daly_data(
                                      demog_dict[filename],
                                      human_monitor_dict[filename],
                                      life_expectancies
                                                     )

    return {'tracker_dict': tracker_dict,
            'demographics_dict': demog_dict,
            'human_monitor_dict': human_monitor_dict,
            'daly_data_dict': daly_data_dict}


def find_life_expectancies(directory):
    """
    Find the life_expectancies csv file
    Attrs: directory to look in
    Returns: path to csv file
    """
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if "1310011401" in f:  # string unique to life expectancy file
                return os.path.abspath(os.path.join(dirpath, f))
    raise ValueError("Can't find life expectancies csv")


def run(data, path, compare="app_adoption"):
    """
    Under construction.
    Outputs the necessary plots and tables for the ArXiv version of the paper.
    attrs:
        - data: data from plotting/main.py
        - path: currently unused
        - compare: currently unused
    Returns
        - Figure 9: work hours vs total DALYs
            for different CT methods
        - Figure 10: DALYs per person by 10-year age bin
            for different CT methods
        - table 2: ICER table
            for different CT methods
        - table L4: Foregone Work hours and TPL
            for different CT methods
        - table L5: YLL, YLD and DALYs stratified by age or sex
            for different CT methods
    """
    label2pkls = list()
    for method in data:
        for key in data[method]:
            label = f"{method}_{key}"
            pkls = [r["pkl"] for r in data[method][key].values()]
            label2pkls.append((label, pkls))

    method_to_labels = {}
    method_to_colors = {}
    for label, pkls in label2pkls:
        if 'post-lockdown-no-tracing' in label:
            method_to_labels[label] = "No Tracing"
            method_to_colors[label] = "#34495E"
        elif 'no_intervention' in label:
            method_to_labels[label] = "No Tracing"
            method_to_colors[label] = "#34495E"
        elif 'bdt1' in label:
            method_to_labels[label] = "Test-based BCT1"
            method_to_colors[label] = "mediumvioletred"
        elif 'heuristic' in label:
            method_to_labels[label] = "Heuristic-FCT"
            method_to_colors[label] = "darkorange"
        else:
            print(label)
            raise ValueError('Method not recognized for daly plotting')

    # Define aggregate output variables
    agg_dalys = {label: {} for label, _ in label2pkls}
    agg_dalys_low = {label: {} for label, _ in label2pkls}
    agg_dalys_high = {label: {} for label, _ in label2pkls}
    agg_work_hours = {label: {} for label, _ in label2pkls}

    s = {label: {} for label, _ in label2pkls}
    e = {label: {} for label, _ in label2pkls}
    i = {label: {} for label, _ in label2pkls}
    r = {label: {} for label, _ in label2pkls}

    # [method][run][age group] per person
    dalys_pp_age = {label: {} for label, _ in label2pkls}
    dalys_pp_age_sex_metrics = {label: {} for label, _ in label2pkls}

    dalys_pp_age_low = {label: {} for label, _ in label2pkls}
    dalys_pp_age_sex_metrics_low = {label: {} for label, _ in label2pkls}

    dalys_pp_age_high = {label: {} for label, _ in label2pkls}
    dalys_pp_age_sex_metrics_high = {label: {} for label, _ in label2pkls}

    # get life expectancy data
    le_data_path = os.path.join(
        pathlib.Path(__file__).resolve().parent.parent.parent.parent,
        'daly_data/life_expectancies/1310011401-eng.csv')
    le_data = load_life_expectancies(le_data_path)

    # daly calculations
    for label, pkls in label2pkls:
        for idx, pkl in enumerate(pkls):
            # get human_monitor data
            monitor_data = pkl['human_monitor']

            # get their demographic information (pre-existing conditions, age)
            demog_data = pkl['humans_demographics']

            # get daly_data for each individual
            daly_df_seed = get_daly_data(demog_data, monitor_data, le_data)
            daly_df_seed_low = daly_df_seed
            daly_df_seed_high = daly_df_seed

            yll, yld, dalys = compute_dalys(daly_df_seed,
                                            dis_weights=disability_weights,
                                            discounting=False,
                                            discount_rate=None)
            yll_low, yld_low, dalys_low = compute_dalys(daly_df_seed,
                                                        disability_weights,
                                                        False,
                                                        None)
            yll_high, yld_high, dalys_high = compute_dalys(daly_df_seed,
                                                           disability_weights,
                                                           False,
                                                           None)

            # calculate metrics for this pickle file
            daly_df_seed['YLL'] = yll
            daly_df_seed['YLD'] = yld
            daly_df_seed['DALYs'] = dalys

            # repeat for low and high values
            daly_df_seed_low['YLL'] = yll_low
            daly_df_seed_low['YLD'] = yld_low
            daly_df_seed_low['DALYs'] = dalys_low

            daly_df_seed_high['YLL'] = yll_high
            daly_df_seed_high['YLD'] = yld_high
            daly_df_seed_high['DALYs'] = dalys_high

            # store metrics for this pickle file
            agg_dalys[label][idx] = daly_df_seed['DALYs'].sum()
            agg_dalys_low[label][idx] = daly_df_seed_low['DALYs'].sum()
            agg_dalys_high[label][idx] = daly_df_seed_high['DALYs'].sum()

            assert daly_df_seed[daly_df_seed.was_infected == False
                                ].DALYs.sum() == 0,\
                'uninfected should not contribute DALYs'

            # calculate total foregone work hours for this pickle file
            agg_work_hours[label][idx] = lost_work_hours_total(pkl)

            # calculate dalys per person by age
            dalys_pp_age, dalys_pp_age_sex_metrics = metrics_sex_age(
                daly_df_seed=daly_df_seed,
                dalys_pp_age=dalys_pp_age,
                dalys_pp_age_sex_metrics=dalys_pp_age_sex_metrics,
                label=label,
                idx=idx,
                )

            # calculate dalys per person by age for low and high DW
            dalys_pp_age_low, dalys_pp_age_sex_metrics_low = metrics_sex_age(
                daly_df_seed=daly_df_seed_low,
                dalys_pp_age=dalys_pp_age_low,
                dalys_pp_age_sex_metrics=dalys_pp_age_sex_metrics_low,
                label=label,
                idx=idx,
                )

            dalys_pp_age_high, dalys_pp_age_sex_metrics_high = metrics_sex_age(
                daly_df_seed=daly_df_seed_high,
                dalys_pp_age=dalys_pp_age_high,
                dalys_pp_age_sex_metrics=dalys_pp_age_sex_metrics_high,
                label=label,
                idx=idx,
                )

            # grab SEIR data for each method and seed
            s[label][idx] = pkl['s']
            e[label][idx] = pkl['e']
            i[label][idx] = pkl['i']
            r[label][idx] = pkl['r']

    # seir[metric][method][seed]
    seir = {'s': s, 'e': e, 'i': i, 'r': r}

    # aggregate dalys across seeds, get mean and stderr
    agg_daly_df, agg_daly_mean, agg_daly_stderr = agg_daly_summary(
        agg_dalys=agg_dalys)

    # repeat for low and high disability weights
    agg_daly_df_low, agg_daly_mean_low, agg_daly_stderr_low = agg_daly_summary(
        agg_dalys=agg_dalys_low)

    agg_daly_df_high, \
        agg_daly_mean_high, agg_daly_stderr_high = agg_daly_summary(
            agg_dalys=agg_dalys)

    # add low and high DW dalys to dict
    daly_sensitivity = {
        'low': {
            'mean': agg_daly_mean_low,
            'stderr': agg_daly_stderr_low
            },
        'high': {
            'mean': agg_daly_mean_high,
            'stderr': agg_daly_stderr_high
            }
        }

    # aggregate work hours across seeds, get mean and stderr
    work_hours_df, work_hours_mean, work_hours_stderr = work_hours_summary(
        agg_work_hours=agg_work_hours)

    # convert work hours to TPL by multiplying by mean hourly wage
    tpl_mean, tpl_stderr = tpl_summary(
        work_hours_df=work_hours_df,
        MEAN_HOURLY_WAGE=MEAN_HOURLY_WAGE)

    # link full keys (that specify params like APP_UPTAKE)
    # to shorthand (e.g. "bdt1")
    method_keys = get_daly_df_keys(agg_daly_df=agg_daly_df)

    # generate figure 9
    plot_dalys_tpl(agg_daly_mean=agg_daly_mean,
                   agg_daly_stderr=agg_daly_stderr,
                   work_hours_mean=work_hours_mean,
                   work_hours_stderr=work_hours_stderr,
                   method_to_labels=method_to_labels,
                   method_to_colors=method_to_colors,
                   path=path)

    # generate figure 10
    dalys_by_age_bin(dalys_pp_age=dalys_pp_age,
                     method_to_labels=method_to_labels,
                     method_to_colors=method_to_colors,
                     path=path)

    # generate table 2
    ICER_table(agg_daly_mean=agg_daly_mean,
               agg_daly_stderr=agg_daly_stderr,
               work_hours_mean=work_hours_mean,
               work_hours_stderr=work_hours_stderr,
               method_keys=method_keys,
               MEAN_HOURLY_WAGE=MEAN_HOURLY_WAGE,
               path=path)

    # generate table L4
    work_hours_table(work_hours_mean,
                     work_hours_stderr,
                     tpl_mean,
                     tpl_stderr,
                     method_keys,
                     path=path)

    # generate table L5
    dalys_age_sex_breakdown(dalys_pp_age_sex_metrics,
                            method_keys,
                            path=path)

    # plot SEIR curves
    seir_curves(seir=seir,
                method_to_labels=method_to_labels,
                method_to_colors=method_to_colors,
                path=path)

    # make sensitivity tables
    sensitivity_tables(daly_sensitivity,
                       work_hours_mean,
                       work_hours_stderr,
                       method_keys,
                       dalys_pp_age_sex_metrics,
                       MEAN_HOURLY_WAGE=MEAN_HOURLY_WAGE,
                       path=None)


def plot_dalys_tpl(agg_daly_mean,
                   agg_daly_stderr,
                   work_hours_mean,
                   work_hours_stderr,
                   method_to_labels,
                   method_to_colors,
                   path=None):
    """generate figure 9 (work hours and total DALYs)"""

    if path is None:
        raise ValueError('Must provide path to save fig')

    fig = plt.figure(figsize=(15, 10))

    for method in agg_daly_mean.keys():

        plt.scatter(work_hours_mean[method],
                    agg_daly_mean[method],
                    label=method_to_labels[method],
                    color=method_to_colors[method],
                    s=100)
        plt.errorbar(work_hours_mean[method],
                     agg_daly_mean[method],
                     xerr=work_hours_stderr[method],
                     yerr=agg_daly_stderr[method],
                     color=method_to_colors[method])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # grids
    plt.grid(True, axis='x', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xlabel('Total Work Hours Foregone', fontsize=20)
    plt.ylabel('Total DALYs', fontsize=20)
    plt.legend(prop={'size': 30})
    plt.title('Health-Economic costs of CT methods @ 60% Adoption Rate',
              fontsize=24)
    plt.tight_layout()

    # save in daly_data folder
    save_path = ('DALYs_vs_TPL.png')
    fig.savefig(os.path.join(path, save_path))
    print('Figure 9 saved to DALYs_vs_TPL.png')


def dalys_by_age_bin(dalys_pp_age,
                     method_to_labels,
                     method_to_colors,
                     path=None):
    """Plot dalys per person stratified by age groups of size 10."""

    fig = plt.figure(figsize=(15, 10))

    for method in dalys_pp_age.keys():

        # get mean and stderr across seeds
        daly_age_df = pd.DataFrame(dalys_pp_age[method])
        daly_age_df['mean'] = daly_age_df.mean(axis=1)
        daly_age_df['stderr'] = daly_age_df.sem(axis=1)

        # last age group is 100+
        daly_age_df.rename(index={'100-119': '100+'}, inplace=True)

        # x-axis is age groups, y-axis is dalys per person
        plt.plot(daly_age_df.index, daly_age_df['mean'],
                 label=method_to_labels[method],
                 color=method_to_colors[method])
        # standard error across runs
        plt.fill_between(daly_age_df.index,
                         daly_age_df['mean'] - daly_age_df['stderr'],
                         daly_age_df['mean'] + daly_age_df['stderr'],
                         alpha=0.2,
                         color=method_to_colors[method]
                         )

    # plot ticks
    plt.xticks([i for i in range(0, 11)],
               [str(i) + ' to ' + str(i+9) +
               ' years' for i in range(0, 100, 10)] +
               ['100+ years'],
               rotation=60,
               ha='right',
               fontsize=16)
    plt.yticks(fontsize=16)

    # grid
    plt.grid(True, axis='x', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)

    # labels
    plt.xlabel('Age group', fontsize=20)
    plt.ylabel('DALYs per person', fontsize=20)

    # legend and title
    plt.legend(prop={'size': 30})
    plt.title(('Age-stratified DALYs per person '
              'for CT methods @ 60% Adoption rate'), fontsize=24)
    plt.tight_layout()

    # save in daly_data folder
    save_path = 'dalys_per_person_age_stratified.png'
    fig.savefig(os.path.join(path, save_path))
    print('Figure 10 saved to dalys_per_person_age_stratified.png')


def ICER_table(agg_daly_mean,
               agg_daly_stderr,
               work_hours_mean,
               work_hours_stderr,
               method_keys,
               MEAN_HOURLY_WAGE=MEAN_HOURLY_WAGE,
               path=None,
               save_path=('ICER_table.csv')
               ):
    print('############################################################')
    print('Table 2: difference in DALYs and TPL, '
          'as well as ICER for different CT methods')
    print('saved to ICER_table.csv')
    # get full DataFrame keys according to CT method used
    methods = ['no-tracing', 'bdt1', 'heuristic']
    no_tracing_key = method_keys['no-tracing']
    bct_key = method_keys['bdt1']
    heuristic_key = method_keys['heuristic']

    # Difference in DALYs compared to No Tracing
    agg_daly_mean['diff_bct'] = agg_daly_mean[no_tracing_key] - \
        agg_daly_mean[bct_key]
    agg_daly_mean['diff_heuristic'] = agg_daly_mean[no_tracing_key] - \
        agg_daly_mean[heuristic_key]

    agg_daly_stderr['diff_bct'] = np.sqrt(
        agg_daly_stderr[no_tracing_key]**2
        + agg_daly_stderr[bct_key]**2
                                           )

    agg_daly_stderr['diff_heuristic'] = np.sqrt(
        agg_daly_stderr[no_tracing_key]**2
        + agg_daly_stderr[heuristic_key]**2
                                                 )

    agg_daly_diff = pd.DataFrame([agg_daly_mean['diff_bct'],
                                  agg_daly_stderr['diff_bct'],
                                  agg_daly_mean['diff_heuristic'],
                                  agg_daly_stderr['diff_heuristic']
                                  ],
                                 columns=['diff_dalys'],
                                 index=pd.MultiIndex.from_product(
                                     [methods[1:],
                                      ['mean', 'stderr']
                                      ]
                                                                  )
                                 )

    work_hours_mean['diff_bct'] = work_hours_mean[bct_key] - \
        work_hours_mean[no_tracing_key]
    work_hours_mean['diff_heuristic'] = work_hours_mean[heuristic_key] - \
        work_hours_mean[no_tracing_key]

    work_hours_stderr['diff_bct'] = np.sqrt(
        work_hours_stderr[no_tracing_key]**2
        + work_hours_stderr[bct_key]**2
                                           )

    work_hours_stderr['diff_heuristic'] = np.sqrt(
        work_hours_stderr[no_tracing_key]**2
        + work_hours_stderr[heuristic_key]**2
                                                 )

    agg_daly_diff['diff_hours'] = [work_hours_mean['diff_bct'],
                                   work_hours_stderr['diff_bct'],
                                   work_hours_mean['diff_heuristic'],
                                   work_hours_stderr['diff_heuristic']
                                   ]

    # ICER over no-tracing
    agg_daly_diff['ICER'] = (agg_daly_diff['diff_hours']
                             / agg_daly_diff['diff_dalys']
                             * MEAN_HOURLY_WAGE
                             )

    diff_DALYs_df = agg_daly_diff['diff_dalys'].unstack()
    diff_TPL_df = agg_daly_diff['diff_hours'].unstack()
    ICER_df = agg_daly_diff['ICER'].unstack()
    ICER_df['stderr'] = (np.abs(ICER_df['mean']) *
                         np.sqrt(
        (diff_DALYs_df['stderr']/diff_DALYs_df['mean'])**2
        +
        (diff_TPL_df['stderr']/diff_TPL_df['mean'])**2

                                 )
                         )

    agg_daly_diff['ICER'] = ICER_df.stack()
    agg_daly_diff = agg_daly_diff.transpose()
    print(agg_daly_diff)
    save_path_icer = save_path
    agg_daly_diff.to_csv(os.path.join(path, save_path_icer))


def work_hours_table(work_hours_mean,
                     work_hours_stderr,
                     tpl_mean,
                     tpl_stderr,
                     method_keys,
                     path=None):
    # generate table L4 (work hours, TPL)
    tpl_df = pd.DataFrame([work_hours_mean[method_keys.values()],
                           work_hours_stderr[method_keys.values()],
                           tpl_mean[method_keys.values()],
                           tpl_stderr[method_keys.values()]
                           ],
                          index=pd.MultiIndex.from_product(
                            [['Foregone Work', 'TPL'],
                             ['mean', 'stderr']])
                          )

    print('############################################################')
    print('Table L4: Foregone work hours and TPL for different CT methods')
    print('Saved to tpl_table.csv')
    print(tpl_df)
    # parent_path = pathlib.Path(
    # __file__).resolve().parent.parent.parent.parent
    save_path_tpl = ('tpl_table.csv')
    tpl_df.to_csv(os.path.join(path, save_path_tpl))


def dalys_age_sex_breakdown(dalys_pp_age_sex_metrics,
                            method_keys,
                            path=None,
                            save_path=('stratified_metrics.csv')
                            ):
    metrics_table = {}
    for method in method_keys.values():
        method_df_concat = pd.concat([
            pd.DataFrame(dalys_pp_age_sex_metrics[method][run])
            for run in dalys_pp_age_sex_metrics[method].keys()
                               ]
                              )
        method_df = method_df_concat.groupby(method_df_concat.index).mean()
        metrics_table[method] = pd.DataFrame.from_dict(method_df)

    metrics_df = pd.concat(metrics_table.values(), keys=metrics_table.keys()
                           ).transpose()
    metrics_df.rename(index={'100-119': '100+'})
    print('############################################################')
    print('Table L5: health metrics stratified by age and sex '
          'for different CT methods')
    print('Saved to health_metrics_stratified_table.csv')
    print(metrics_df)
    save_path_metrics = save_path
    metrics_df.to_csv(os.path.join(path, save_path_metrics))


def seir_curves(seir,
                method_to_labels,
                method_to_colors,
                path=None):

    fig, axes = plt.subplots(2, 2)
    subpl = axes.ravel()

    for idx, curve in enumerate(seir):  # find better name than curve
        current_plot = idx  # one plot each of s, e, i, r
        for method in seir[curve]:
            # get data for one method, one curve, 10 seeds
            curve_df = pd.DataFrame(seir[curve][method])
            curve_mean = curve_df.mean(axis=1)
            curve_stderr = curve_df.sem(axis=1)

            # flatten indexing of subplots
            subpl[current_plot].plot(curve_mean.index,
                                     curve_mean,
                                     label=method_to_labels[method],
                                     color=method_to_colors[method])
            subpl[current_plot].fill_between(curve_mean.index,
                                             y1=curve_mean - curve_stderr,
                                             y2=curve_mean + curve_stderr,
                                             color=method_to_colors[method],
                                             alpha=0.2)
        subpl[current_plot].set_title(str(curve))

    # grid
    plt.grid(True, axis='x', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)

    # legend
    plt.legend()

    # save figure
    save_path = 'seir.png'
    fig.savefig(os.path.join(path, save_path))
    print('Figure 10 saved to seir.png')


def metrics_sex_age(daly_df_seed,
                    dalys_pp_age,
                    dalys_pp_age_sex_metrics,
                    label,
                    idx):

    age_ranges = [range(i, i+10) for i in
                  range(0, 100, 10)] + [range(100, 120)]

    age_range_dict = {str(np.min(i)) + '-' + str(np.max(i)): {}
                      for i in age_ranges}
    by_age = {}

    for age_range, age_key in zip(age_ranges, age_range_dict.keys()):
        by_age[age_key] = daly_df_seed[daly_df_seed.age.isin(age_range)
                                       ]['DALYs'].mean(axis=0)
        dalys_pp_age[label][idx] = by_age

    by_sex = {}
    for age_range, age_key in zip(age_ranges, age_range_dict.keys()):
        by_sex[age_key] = daly_df_seed[
            daly_df_seed.age.isin(age_range)
                                        ][
            ['DALYs', 'YLL', 'YLD']].sum(axis=0)
        dalys_pp_age_sex_metrics[label][idx] = by_sex

    for sex in sexes:
        dalys_pp_age_sex_metrics[label][idx][sex] = daly_df_seed[
            daly_df_seed.sex.eq(sex)][
                ['DALYs', 'YLL', 'YLD']].sum(axis=0)

    return dalys_pp_age, dalys_pp_age_sex_metrics


def get_daly_df_keys(agg_daly_df):
    for key in agg_daly_df.keys():
        # for methods...
        if 'no-tracing' in key:
            no_tracing_key = key
        elif 'bdt1' in key:
            bct_key = key
        elif 'heuristic' in key:
            heuristic_key = key
        else:
            raise ValueError('unknown key')

    method_keys = {'no-tracing': no_tracing_key,
                   'bdt1': bct_key,
                   'heuristic': heuristic_key
                   }
    return method_keys


def agg_daly_summary(agg_dalys):
    agg_daly_df = pd.DataFrame(agg_dalys)
    agg_daly_mean = agg_daly_df.mean(axis=0)
    agg_daly_stderr = agg_daly_df.sem(axis=0)

    return agg_daly_df, agg_daly_mean, agg_daly_stderr


def work_hours_summary(agg_work_hours):
    work_hours_df = pd.DataFrame(agg_work_hours)
    work_hours_mean = work_hours_df.mean(axis=0)
    work_hours_stderr = work_hours_df.sem(axis=0)

    return work_hours_df, work_hours_mean, work_hours_stderr


def tpl_summary(work_hours_df,
                MEAN_HOURLY_WAGE):
    tpl_mean = (work_hours_df*MEAN_HOURLY_WAGE).mean(axis=0)
    tpl_stderr = (work_hours_df*MEAN_HOURLY_WAGE).sem(axis=0)

    return tpl_mean, tpl_stderr


def sensitivity_tables(daly_sensitivity,
                       work_hours_mean,
                       work_hours_stderr,
                       method_keys,
                       dalys_pp_age_sex_metrics,
                       MEAN_HOURLY_WAGE=MEAN_HOURLY_WAGE,
                       path=None):
    for bound in daly_sensitivity:
        mean = daly_sensitivity[bound]['mean']
        stderr = daly_sensitivity[bound]['stderr']

        ICER_table(mean,
                   stderr,
                   work_hours_mean,
                   work_hours_stderr,
                   method_keys,
                   MEAN_HOURLY_WAGE,
                   path=path,
                   save_path=('ICER_table_' + bound + '.csv')
                   )

        dalys_age_sex_breakdown(dalys_pp_age_sex_metrics,
                                method_keys,
                                path=path,
                                save_path=('stratified_metrics' +
                                           bound + '.csv')
                                )
