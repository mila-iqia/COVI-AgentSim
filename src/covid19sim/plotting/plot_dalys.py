import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

# pd.set_option('display.float_format', '{:.6f}'.format)

MEAN_HOURLY_WAGE = 27.67


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
                  disability_weights={
                    'no_hospitalization': 0.051,  # moderate resp infection
                    'hospitalized': 0.133,  # severe respiratory infection
                    'critical': 0.408  # severe COPD without heart failure
                    }
                  ):
    """
    Gathers all data required for DALY calculations into one dataFrame.
    Attrs:
        - demographics data from tracker file
        - human monitor data from tracker file
        - life expectancies from output of load_life_expectancies()
    Returns:
        - Dataframe where rows are individuals,
          columns are attributes of individuals
        - columns:
                "days_in_hospital",
                "days_in_ICU",
                "has_died",
                "days_symptoms_and_infection",
                "days_sick_not_in_hospital",
                "was_infected",
                "life_expectancy",
                -------- following columns are
                         calculated from the previous ones --------
                "YLL",
                "YLD",
                "DALYs
    """

    human_names = human_monitor_data['days_in_hospital'].keys()

    daly_df = pd.DataFrame(list(human_monitor_data.values())).transpose()

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
                daly_df['sex'][human]][str(daly_df['age'][human]) + " years"]
            )

    # add YLL
    daly_df['YLL'] = daly_df['was_infected'] * (
        daly_df['has_died'] * daly_df['life_expectancy'])

    # add YLD
    daly_df['YLD'] = daly_df['was_infected'] * (
        daly_df['days_sick_not_in_hospital'] / 365 *
        disability_weights['no_hospitalization'] +

        daly_df['days_in_hospital'] / 365 *
        disability_weights['hospitalized'] +

        daly_df['days_in_ICU'] / 365 *
        disability_weights['critical']
                                                )
    # add DALYs
    daly_df['DALYs'] = daly_df['YLL'] + daly_df['YLD']

    assert daly_df[daly_df.was_infected == False
                   ].DALYs.sum() == 0,\
        'uninfected should not contribute DALYs'

    return daly_df


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
    # bins are 5-year age bins: 5 is 25-29, 13 is 65-69
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


def get_normalized_runs(path,
                        desired_r=1.2):

    # OBSOLETE. Keeping for return to previous mobility normalization method.

    norm_mob_path = os.path.join(path, 'normalized_mobility')
    norm_csvs = [csv for csv in os.listdir(norm_mob_path)
                 if 'full_extracted_data_AR' in csv]
    # hotfix: make a dict of pct to uptake keys in future
    # there SHOULD only be one element in the list
    norm_csv = os.path.join(norm_mob_path, norm_csvs[0])
    norm_mob_csv = pd.read_csv(norm_csv)
    plnt = norm_mob_csv.loc[(norm_mob_csv['method'] ==
                             'post-lockdown-no-tracing')]

    desired_r_idx = np.argmin(np.abs(plnt['r'] - desired_r))
    mob_factor_lower = norm_mob_csv.iloc[desired_r_idx]['mobility_factor'] - 0.25
    mob_factor_upper = norm_mob_csv.iloc[desired_r_idx]['mobility_factor'] + 0.25

    desired_runs = norm_mob_csv.loc[(norm_mob_csv['mobility_factor'] >
                                     mob_factor_lower) &
                                    (norm_mob_csv['mobility_factor'] <
                                     mob_factor_upper)]

    return desired_runs['dir']


def run(data, path, compare="app_adoption"):
    """
    Outputs the necessary plots and tables for the ArXiv version of the paper.
    attrs:
        - data: data from plotting/main.py
        - path: provides output path for plots
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

    # Extract data from pickle files
    adoption_rate = None
    label2pkls = list()
    for method in data:
        for uptake in data[method]:
            pkls = []
            for filepath in data[method][uptake].keys():
                pkls.append(data[method][uptake][filepath]['pkl'])
            label = f"{method}_{uptake}"
            label2pkls.append((label, pkls))

    # Set colors and labels based on intervention
    label, method_to_labels, method_to_colors = labels_and_colors(label2pkls)

    # Define aggregate output variables
    agg_dalys = {label: {} for label, _ in label2pkls}
    agg_work_hours = {label: {} for label, _ in label2pkls}

    # [method][run][age group]
    dalys_pp_age = {label: {} for label, _ in label2pkls}
    dalys_pp_age_sex_metrics = {label: {} for label, _ in label2pkls}

    # get life expectancy data
    le_data_path = os.path.join(
        pathlib.Path(__file__).resolve().parent.parent.parent.parent,
        'daly_data/life_expectancies/1310011401-eng.csv')
    le_data = load_life_expectancies(le_data_path)

    # daly calculations
    for label, pkls in label2pkls:
        for idx, pkl in enumerate(pkls):

            adoption_rate = pkl['adoption_rate']
            monitor_data = pkl['human_monitor']
            demog_data = pkl['humans_demographics']

            # calculate YLL, YLD and DALYs for each individual in one pkl file
            daly_df_seed = get_daly_data(demog_data, monitor_data, le_data)
            pop_size = len(daly_df_seed.index)

            # calculate total DALYs for this pickle file
            agg_dalys[label][idx] = daly_df_seed['DALYs'].sum()

            # calculate total foregone work hours for this pickle file
            agg_work_hours[label][idx] = lost_work_hours_total(pkl)

            # calculate dalys per person by age for this pickle file
            dalys_age(daly_df_seed, dalys_pp_age, label, idx)
            health_age_sex_sum(daly_df_seed, dalys_pp_age_sex_metrics, label,
                               idx)

    # add to aggregate output variables
    agg_daly_df = pd.DataFrame(agg_dalys)
    agg_daly_mean = agg_daly_df.mean(axis=0)
    agg_daly_stderr = agg_daly_df.sem(axis=0)

    work_hours_df = pd.DataFrame(agg_work_hours)
    work_hours_mean = work_hours_df.mean(axis=0)
    work_hours_stderr = work_hours_df.sem(axis=0)

    # mobility factor plot
    fig = plt.figure(figsize=(15, 10))

    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gp_regressors = {}
    max_work_hours = work_hours_df.max().max()

    for method in agg_daly_df.columns:
        gp_regressors[method] = GaussianProcessRegressor(
            kernel=kernel, random_state=0).fit(
                work_hours_df[method], agg_daly_df[method])
        X_plot = np.linspace(0, max_work_hours, 10000)[:, None]
        y_gpr, y_std = gp_regressors[method].predict(X_plot, return_std=True)

        plt.plot(X_plot, y_gpr, label=method_to_labels[method],
                 color=method_to_colors[method])
        plt.fill_between(X_plot[:, 0], y_gpr - y_std, y_gpr + y_std)
        plt.scatter(work_hours_df[method], agg_daly_df[method],
                    label=method_to_labels[method],
                    color=method_to_colors[method])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # grids
    plt.grid(True, axis='x', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xlabel('Work Hours Foregone Per Person', fontsize=20)
    plt.ylabel('DALYs Foregone Per Person', fontsize=20)
    plt.legend(prop={'size': 15}, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Health-Economic costs of CT methods @ ' + str(adoption_rate) +
              '% Adoption Rate, scatter',
              fontsize=24)
    plt.tight_layout()

    # save in daly_data folder
    save_path = ('plotting_pareto_mobility.png')
    fig.savefig(os.path.join(path, save_path))
    print('Figure saved to plotting_pareto_mobility.png')

    # save daly and work hours dfs
    save_path_csv_dalys = ('pareto_mobility_dalys' + str(adoption_rate))
    agg_daly_df.to_csv(os.path.join(path, save_path_csv_dalys))
    save_path_csv_work = ('pareto_mobility_work' + str(adoption_rate))
    work_hours_df.to_csv(os.path.join(path, save_path_csv_work))

    # generate figure 9 (work hours and total DALYs)
    plot_figure_nine(agg_daly_mean, work_hours_mean, pop_size,
                     method_to_labels, method_to_colors, work_hours_stderr,
                     agg_daly_stderr, adoption_rate, path)

    # generate figure 10
    plot_figure_ten(dalys_pp_age, method_to_labels, method_to_colors,
                    adoption_rate, path)

    # get full DataFrame keys according to CT method used
    method_keys = get_keys(agg_daly_df)

    # Delta DALYs means and stderrs
    for key in method_keys:
        if key != 'no-tracing':
            agg_daly_mean['delta_' + str(key)] = \
                agg_daly_mean[method_keys['no-tracing']] - \
                agg_daly_mean[method_keys[key]]

            agg_daly_stderr['delta_' + str(key)] = np.sqrt(
                agg_daly_stderr[method_keys['no-tracing']]**2 +
                agg_daly_stderr[method_keys[key]]**2
                )

    # agg_daly_mean['delta_bct'] = agg_daly_mean[no_tracing_key] - \
    #     agg_daly_mean[bct_key]
    # agg_daly_mean['delta_heuristic'] = agg_daly_mean[no_tracing_key] - \
    #     agg_daly_mean[heuristic_key]
    # agg_daly_mean['delta_microwave'] = agg_daly_mean[no_tracing_key] - \
    #     agg_daly_mean[microwave_key]
    # agg_daly_mean['delta_galaxy'] = agg_daly_mean[no_tracing_key] - \
    #     agg_daly_mean[galaxy_key]
    # agg_daly_mean['delta_galaxy_000'] = agg_daly_mean[no_tracing_key] - \
    #     agg_daly_mean[galaxy_000_key]
    # agg_daly_mean['delta_galaxy_001'] = agg_daly_mean[no_tracing_key] - \
    #     agg_daly_mean[galaxy_001_key]

    # agg_daly_stderr['delta_bct'] = np.sqrt(
    #     agg_daly_stderr[no_tracing_key]**2
    #     + agg_daly_stderr[bct_key]**2
    #                                        )
    # agg_daly_stderr['delta_heuristic'] = np.sqrt(
    #     agg_daly_stderr[no_tracing_key]**2
    #     + agg_daly_stderr[heuristic_key]**2
    #                                              )
    # agg_daly_stderr['delta_microwave'] = np.sqrt(
    #     agg_daly_stderr[no_tracing_key]**2
    #     + agg_daly_stderr[microwave_key]**2
    #                                              )
    # agg_daly_stderr['delta_galaxy'] = np.sqrt(
    #     agg_daly_stderr[no_tracing_key]**2
    #     + agg_daly_stderr[galaxy_key]**2
    #                                              )
    # agg_daly_stderr['delta_galaxy_000'] = np.sqrt(
    #     agg_daly_stderr[no_tracing_key]**2
    #     + agg_daly_stderr[galaxy_000_key]**2
    #                                              )
    # agg_daly_stderr['delta_galaxy_001'] = np.sqrt(
    #     agg_daly_stderr[no_tracing_key]**2
    #     + agg_daly_stderr[galaxy_001_key]**2
    #                                              )

    # list of alternating mean and stderr
    methods = method_keys.values()
    print("###################################################################")
    print("###################################################################")
    print('methods: ')
    print(methods)

    print("###################################################################")
    print("###################################################################")
    print('agg_daly_mean keys: ')
    print(agg_daly_mean.keys())

    delta_keys = [k for k in agg_daly_mean.keys() if 'delta' in k]

    agg_daly_cols = [(agg_daly_mean[method], agg_daly_stderr[method]) for
                     method in delta_keys]
    agg_daly_cols = [col for tup in agg_daly_cols for col in tup]
    print("###################################################################")
    print("###################################################################")
    print('agg_daly_cols: ')
    print(agg_daly_cols)

    agg_daly_diff = pd.DataFrame(
        agg_daly_cols,
        columns=['delta DALYs'],
        index=pd.MultiIndex.from_product(
            [[val for val in delta_keys],
             ['mean', 'stderr']]
                                         )
                                 )
    print("###################################################################")
    print("###################################################################")
    print('agg_daly_diff keys: ')
    print(agg_daly_diff.keys())

    # agg_daly_diff = pd.DataFrame([agg_daly_mean['delta_bct'],
    #                               agg_daly_stderr['delta_bct'],
    #                               agg_daly_mean['delta_heuristic'],
    #                               agg_daly_stderr['delta_heuristic'],
    #                               agg_daly_mean['delta_microwave'],
    #                               agg_daly_stderr['delta_microwave'],
    #                               agg_daly_mean['delta_galaxy'],
    #                               agg_daly_stderr['delta_galaxy'],
    #                               agg_daly_mean['delta_galaxy_000'],
    #                               agg_daly_stderr['delta_galaxy_000'],
    #                               agg_daly_mean['delta_galaxy_001'],
    #                               agg_daly_stderr['delta_galaxy_001']
    #                               ],
    #                              columns=['delta DALYs'],
    #                              index=pd.MultiIndex.from_product(
    #                                  [methods[1:],
    #                                   ['mean', 'stderr']
    #                                   ]
    #                                                               )
    #                              )

    # delta TPL
    work_hours_mean = work_hours_df.mean(axis=0)
    work_hours_stderr = work_hours_df.sem(axis=0)
    tpl_mean = (work_hours_df*MEAN_HOURLY_WAGE).mean(axis=0)
    tpl_stderr = (work_hours_df*MEAN_HOURLY_WAGE).sem(axis=0)

    for key in method_keys:
        if key != 'no-tracing':
            work_hours_mean['delta_' + str(key)] = \
                work_hours_mean[method_keys['no-tracing']] - \
                work_hours_mean[method_keys[key]]

            work_hours_stderr['delta_' + str(key)] = np.sqrt(
                work_hours_stderr[method_keys['no-tracing']]**2 +
                work_hours_stderr[method_keys[key]]**2
                )

    # work_hours_mean['delta_bct'] = work_hours_mean[bct_key] - \
    #     work_hours_mean[no_tracing_key]
    # work_hours_mean['delta_heuristic'] = work_hours_mean[heuristic_key] - \
    #     work_hours_mean[no_tracing_key]
    # work_hours_mean['delta_microwave'] = work_hours_mean[microwave_key] - \
    #     work_hours_mean[no_tracing_key]
    # work_hours_mean['delta_galaxy'] = work_hours_mean[galaxy_key] - \
    #     work_hours_mean[no_tracing_key]
    # work_hours_mean['delta_galaxy_000'] = work_hours_mean[galaxy_000_key] - \
    #     work_hours_mean[no_tracing_key]
    # work_hours_mean['delta_galaxy_001'] = work_hours_mean[galaxy_001_key] - \
    #     work_hours_mean[no_tracing_key]

    # work_hours_stderr['delta_bct'] = np.sqrt(
    #     work_hours_stderr[no_tracing_key]**2
    #     + work_hours_stderr[bct_key]**2
    #                                        )
    # work_hours_stderr['delta_heuristic'] = np.sqrt(
    #     work_hours_stderr[no_tracing_key]**2
    #     + work_hours_stderr[heuristic_key]**2
    #                                              )
    # work_hours_stderr['delta_microwave'] = np.sqrt(
    #     work_hours_stderr[no_tracing_key]**2
    #     + work_hours_stderr[microwave_key]**2
    #                                              )
    # work_hours_stderr['delta_galaxy'] = np.sqrt(
    #     work_hours_stderr[no_tracing_key]**2
    #     + work_hours_stderr[galaxy_key]**2
    #                                              )
    # work_hours_stderr['delta_galaxy_000'] = np.sqrt(
    #     work_hours_stderr[no_tracing_key]**2
    #     + work_hours_stderr[galaxy_000_key]**2
    #                                              )
    # work_hours_stderr['delta_galaxy_001'] = np.sqrt(
    #     work_hours_stderr[no_tracing_key]**2
    #     + work_hours_stderr[galaxy_001_key]**2
    #
    #                                               )

    # list of columns alternating between mean and stderr
    work_hours_cols = [(work_hours_mean[method], work_hours_stderr[method]) for
                       method in delta_keys]
    work_hours_cols = [col for tup in work_hours_cols for col in tup]
    agg_daly_diff['delta TPL'] = work_hours_cols

    # agg_daly_diff['delta TPL'] = [work_hours_mean['delta_bct'],
    #                               work_hours_stderr['delta_bct'],
    #                               work_hours_mean['delta_heuristic'],
    #                               work_hours_stderr['delta_heuristic'],
    #                               work_hours_mean['delta_microwave'],
    #                               work_hours_stderr['delta_microwave'],
    #                               work_hours_mean['delta_galaxy'],
    #                               work_hours_stderr['delta_galaxy'],
    #                               work_hours_mean['delta_galaxy_000'],
    #                               work_hours_stderr['delta_galaxy_000'],
    #                               work_hours_mean['delta_galaxy_001'],
    #                               work_hours_stderr['delta_galaxy_001']
    #                               ]

    # generate table 2
    table_icer(agg_daly_diff, path)

    # generate table L4 (work hours, TPL)
    table_tpl(work_hours_mean, delta_keys, work_hours_stderr, tpl_mean,
              tpl_stderr, path)

    # generate table L5 (metrics for age, sex breakdown)
    table_age_sex_breakdown(delta_keys, dalys_pp_age_sex_metrics, path)


def table_age_sex_breakdown(method_keys, dalys_pp_age_sex_metrics, path):
    metrics_table = {}
    for method in method_keys:
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
    save_path_metrics = ('health_metrics_stratified_table.csv')
    metrics_df.to_csv(os.path.join(path, save_path_metrics))


def table_tpl(work_hours_mean, method_keys, work_hours_stderr, tpl_mean,
              tpl_stderr, path):
    tpl_df = pd.DataFrame([work_hours_mean[method_keys],
                           work_hours_stderr[method_keys],
                           tpl_mean[method_keys],
                           tpl_stderr[method_keys]
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


def table_icer(agg_daly_diff, path):
    # generate table 2
    print('############################################################')
    print('Table 2: difference in DALYs and TPL,'
          'as well as ICER for different CT methods')
    print('saved to ICER_table.csv')
    agg_daly_diff['ICER'] = (agg_daly_diff['delta TPL']
                             / agg_daly_diff['delta DALYs']
                             * MEAN_HOURLY_WAGE
                             )

    delta_DALYs_df = agg_daly_diff['delta DALYs'].unstack()
    delta_TPL_df = agg_daly_diff['delta TPL'].unstack()
    ICER_df = agg_daly_diff['ICER'].unstack()
    ICER_df['stderr'] = (np.abs(ICER_df['mean']) *
                         np.sqrt(
        (delta_DALYs_df['stderr']/delta_DALYs_df['mean'])**2
        +
        (delta_TPL_df['stderr']/delta_TPL_df['mean'])**2

                                 )
                         )

    agg_daly_diff['ICER'] = ICER_df.stack()
    agg_daly_diff = agg_daly_diff.transpose()
    print(agg_daly_diff)
    save_path_icer = ('ICER_table.csv')
    agg_daly_diff.to_csv(os.path.join(path, save_path_icer))


def get_keys(agg_daly_df):
    method_keys = {}
    for key in agg_daly_df.keys():
        # for methods...
        if 'no-tracing' in key:
            method_keys['no-tracing'] = key
        elif 'bdt1' in key:
            method_keys['bdt1'] = key
        elif 'heuristic' in key:
            method_keys['heuristic'] = key
        elif 'MICROWAVE' in key:
            method_keys['whole-microwave'] = key
        elif 'GALAXY' in key:
            if '000' in key:
                method_keys['wordly-galaxy_000'] = key
            elif '001' in key:
                method_keys['wordly-galaxy_001'] = key
            else:
                method_keys['wordly-galaxy'] = key
        else:
            raise ValueError('unknown key')
    return method_keys


def labels_and_colors(label2pkls):
    # Set colors and labels based on intervention
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
        elif 'MICROWAVE' in label:
            method_to_labels[label] = "WHOLE-MICROWAVE"
            method_to_colors[label] = "royalblue"
        elif 'GALAXY' in label:
            if '801_000_60' in label:
                method_to_labels[label] = "WORDLY-GALAXY_000"
                method_to_colors[label] = "#A4C61A"
            elif '801_001_60' in label:
                method_to_labels[label] = "WORDLY-GALAXY_001"
                method_to_colors[label] = "#00FF1E"
            else:
                method_to_labels[label] = "WORDLY-GALAXY"
                method_to_colors[label] = "#A4C61A"
        else:
            print(label)
            raise ValueError('Method not recognized for daly plotting')
    return label, method_to_labels, method_to_colors


def health_age_sex_sum(daly_df_seed, dalys_pp_age_sex_metrics, label, idx):
    age_ranges = [range(i, i+10) for i in range(0, 100, 10)] + [range(100, 120)]
    age_range_dict = {str(np.min(i)) + '-' + str(np.max(i)): {}
                      for i in age_ranges}
    sexes = ['male', 'female']
    by_age = {}
    for age_range, age_key in zip(age_ranges, age_range_dict.keys()):
        by_age[age_key] = daly_df_seed[daly_df_seed.age.isin(age_range)
                                       ][
            ['DALYs', 'YLL', 'YLD']].sum(axis=0)
    dalys_pp_age_sex_metrics[label][idx] = by_age

    for sex in sexes:
        dalys_pp_age_sex_metrics[label][idx][sex] = daly_df_seed[
            daly_df_seed.sex.eq(sex)][
                ['DALYs', 'YLL', 'YLD']].sum(axis=0)


def dalys_age(daly_df_seed, dalys_pp_age, label, idx):
    age_ranges = [range(i, i+10) for i in range(0, 100, 10)] + [range(100, 120)]
    age_range_dict = {str(np.min(i)) + '-' + str(np.max(i)): {}
                      for i in age_ranges}
    by_age = {}
    for age_range, age_key in zip(age_ranges, age_range_dict.keys()):
        by_age[age_key] = daly_df_seed[daly_df_seed.age.isin(age_range)
                                       ]['DALYs'].mean(axis=0)
    dalys_pp_age[label][idx] = by_age  # [method][run][age group]


def plot_figure_ten(dalys_pp_age, method_to_labels, method_to_colors,
                    adoption_rate, path):

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
              'for CT methods @ ' + str(adoption_rate) + '% Adoption Rate'),
              fontsize=24)
    plt.tight_layout()

    # save in daly_data folder
    save_path = 'dalys_per_person_age_stratified.png'
    fig.savefig(os.path.join(path, save_path))
    print('Figure 10 saved to dalys_per_person_age_stratified.png')

    save_path_csv_age = 'age_stratification_' + str(adoption_rate)

    with open(os.path.join(path, save_path_csv_age) + '.pkl', 'wb') as f:
        pickle.dump(dalys_pp_age, f)


def plot_figure_nine(agg_daly_mean, work_hours_mean, pop_size,
                     method_to_labels, method_to_colors, work_hours_stderr,
                     agg_daly_stderr, adoption_rate, path):
    # generate figure 9 (work hours and total DALYs)
    csv_df = None
    fig = plt.figure(figsize=(15, 10))

    for method in agg_daly_mean.keys():

        plt.scatter(work_hours_mean[method]/pop_size,
                    agg_daly_mean[method]/pop_size,
                    label=method_to_labels[method],
                    color=method_to_colors[method],
                    s=100)
        plt.errorbar(work_hours_mean[method]/pop_size,
                     agg_daly_mean[method]/pop_size,
                     xerr=work_hours_stderr[method]/pop_size,
                     yerr=agg_daly_stderr[method]/pop_size,
                     color=method_to_colors[method])

        data = {
            'method': method,
            'adoption rate': adoption_rate,
            'mean work hours': work_hours_mean[method]/pop_size,
            'stderr work hours': work_hours_stderr[method]/pop_size,
            'mean dalys': agg_daly_mean[method]/pop_size,
            'stderr dalys': agg_daly_stderr[method]/pop_size,
            'label': method_to_labels[method],
            'color': method_to_colors[method]
        }

        if csv_df is None:
            csv_df = pd.DataFrame(data, index=[0])
        else:
            csv_df = csv_df.append(data, ignore_index=True)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # grids
    plt.grid(True, axis='x', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xlabel('Work Hours Foregone Per Person', fontsize=20)
    plt.ylabel('DALYs Foregone Per Person', fontsize=20)
    plt.legend(prop={'size': 30})
    plt.title('Health-Economic costs of CT methods @ ' + str(adoption_rate) +
              '% Adoption Rate',
              fontsize=24)
    plt.tight_layout()

    # save in daly_data folder
    save_path = ('plotting_pareto_comparison_with_replacement.png')
    fig.savefig(os.path.join(path, save_path))
    print('Figure 9 saved to plotting_pareto_comparison_with_replacement.png')

    save_path_csv = ('pareto_comparison_' + str(adoption_rate))
    csv_df.to_csv(os.path.join(path, save_path_csv))
