import pickle
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

social_discount = 0.03
age_weighting_constant = 0.04
modulation_constant = 1 
adjustment_constant = 0.1658
retirement_age = 65

disability_weights = {
    'no_hospitalization':0.051, # moderate lower respiratory infection
    'hospitalized':0.133, # severe respiratory infection
    'critical':0.408 #severe COPD without heart failure
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
    life_expectancies = life_expectancies.rename(columns = {
        '2016 to 2018':'other', # other -> both sexes
        '2016 to 2018.1':'male',
        '2016 to 2018.2':'female',
        })
    life_expectancies = life_expectancies.dropna(subset = ['female'])
    life_expectancies.loc[2, 'Age group'] = '1 years'
    # life_expectancies.iloc[-1, 'Age group'] = '110 years'
    life_expectancies.loc[life_expectancies.index[-1], 'Age group'] = '110 years'
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
        - Dataframe where rows are individuals, columns are attributes of individuals
        - columns:
                "days_in_hospital",
                "days_in_ICU",
                "has_died",
                "days_symptoms_and_infection",
                "days_sick_not_in_hospital",
                "was_infected",
                "life_expectancy",
                -------- The following columns are calculated from the previous ones --------
                "YLL",
                "YLD",
                "DALYs
    """

    human_names = [human_monitor_data[datetime.date(2020,2,28)][i]['name'] 
                   for i in range(len(human_monitor_data[datetime.date(2020,2,28)]))]
    
    symptom_status = {i:{} for i in human_names}
    hospitalization_status = {i:{} for i in human_names}
    ICU_status = {i:{} for i in human_names}
    death_status = {i:{} for i in human_names}
    outpatient_status = {i:{} for i in human_names}
    has_had_infection = {i:{} for i in human_names}

    days_in_hospital = {}
    days_in_ICU = {}
    has_died = {}
    days_with_symptoms = {}
    days_as_outpatient = {}
    was_infected = {}

    for day in human_monitor_data.keys():
        for human in range(len(human_monitor_data[datetime.date(2020, 2, 28)])):
            
            human_name = human_monitor_data[day][human]['name'] 
            is_in_hospital = human_monitor_data[day][human]['is_in_hospital'] 
            is_in_ICU = human_monitor_data[day][human]['is_in_ICU'] 
            is_dead = human_monitor_data[day][human]['dead']
            is_symptomatic = human_monitor_data[day][human]['n_symptoms'] > 0
            is_infected = human_monitor_data[day][human]['infection_timestamp'] is not None
            positive_test = human_monitor_data[day][human]['test_result']

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
             right_index=True)
    
    daly_df['life_expectancy'] = ""
    # set life expectancy as a function of age and sex
    for human in human_names:
        
        daly_df.loc[human,'life_expectancy'] = float(
            life_expectancies[daly_df['sex'][human]][str(daly_df['age'][human])+" years"]
            )
    
    # add YLL 
    daly_df['YLL'] = daly_df['has_died'] * daly_df['life_expectancy']

    # add YLD 
    daly_df['YLD'] = \
                    daly_df['days_sick_not_in_hospital']/365 * disability_weights['no_hospitalization'] + \
                    daly_df['days_in_hospital']/365 * disability_weights['hospitalized'] + \
                    daly_df['days_in_ICU']/365 * disability_weights['critical']

    # add DALYs
    daly_df['DALYs'] = daly_df['YLL'] + daly_df['YLD']
    
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
    
    lost_work_hours = ( tracker_data['work_hours']['WORK-CANCEL--KID'][5:13] + \
                        tracker_data['work_hours']['WORK-CANCEL--ILL'][5:13] + \
                        tracker_data['work_hours']['WORK-CANCEL--QUARANTINE'][5:13]*(1-wfh_prod)
                       )
    
    return lost_work_hours.sum()

def multiple_seeds_get_data(intervention,l_e_path):
    """
        loads data from multiple runs of the simulator into a dict 

        Attrs: 
            - intervention: method used for contact tracing
            - l_e_path: path to life_expectancies csv

        Returns: 
            - dict where intervention[data_subset][seed] = loaded data
                - data_subset = tracker, demographics, human_monitor or daly_data
    """
    tracker_dict = {}
    demographics_dict = {}
    human_monitor_dict = {}
    daly_data_dict = {}
    
    for root, subdir, filenames in os.walk(os.path.join('../output/results',intervention)):
        for filename in [f for f in filenames if '.pkl' in f]:
            
            tracker_dict[filename] = load_tracker(os.path.join(root,filename))
            demographics_dict[filename] = load_demographics(tracker_dict[filename])
            human_monitor_dict[filename] = load_human_monitor(tracker_dict[filename])
            life_expectancies = load_life_expectancies(path = l_e_path)
            daly_data_dict[filename] = get_daly_data(demographics_dict[filename], 
                                      human_monitor_dict[filename], 
                                      life_expectancies)
            
    return {'tracker_dict' : tracker_dict, 
            'demographics_dict' : demographics_dict, 
            'human_monitor_dict' : human_monitor_dict, 
            'daly_data_dict' : daly_data_dict}

def find_life_expectancies(directory):
    """
    Find the life_expectancies csv file
    Attrs: directory to look in
    Returns: path to csv file
    """
    to_return = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if "1310011401" in f: #string unique to life expectancy file
                return os.path.abspath(os.path.join(dirpath, f))
    raise ValueError("Can't find life expectancies csv")
    
# TODO: add plots and tables to run function
def run(data, path, compare="app_adoption"):\
    """
    Under construction.
    Outputs the necessary plots and tables for the ArXiv version of the paper. 
    attrs:
        - data: data from plotting/main.py
        - path: currently unused
        - compare: currently unused
    Returns: 
        - Figure 9: work hours vs total DALYs for different CT methods
        - Figure 10: DALYs per person by 10-year age bin for different CT methods
        - table 2: ICER table for different CT methods
        - table L4: Foregone Work hours and TPL for different CT methods
        - table L5: YLL, YLD and DALYs stratified by age or sex for different CT methods
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
    agg_dalys = {label:{} for label, _ in label2pkls}
    agg_work_hours = {label:{} for label, _ in label2pkls}

    #get life expectancy data
    le_data_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent.parent, 'daly_data/life_expectancies/1310011401-eng.csv')
    le_data = load_life_expectancies(le_data_path)

    # daly calculations
    for label, pkls in label2pkls:
        for idx, pkl in enumerate(pkls):
            # get human_monitor data
            monitor_data = pkl['human_monitor']

            # get their demographic information (pre-existing conditions, age)
            demog_data = pkl['humans_demographics']

            # get calculate YLL, YLD and DALYs for each individual
            daly_df_seed = get_daly_data(demog_data, monitor_data, le_data)

            assert daly_df_seed[daly_df_seed.was_infected == False].DALYs.sum() == 0, 'uninfected should not contribute DALYs'



            # calculate total DALYs for this pickle file
            agg_dalys[label][idx] = daly_df_seed['DALYs'].sum()

            # calculate total foregone work hours for this pickle file
            agg_work_hours[label][idx] = lost_work_hours_total(pkl)

    # add to aggregate output variables
    # agg_daly_df = pd.DataFrame(agg_dalys, keys = agg_dalys.keys())
    agg_daly_df = pd.DataFrame(agg_dalys)
    agg_daly_mean = agg_daly_df.mean(axis = 0)
    agg_daly_stderr = agg_daly_df.sem(axis = 0)

    work_hours_df = pd.DataFrame(agg_work_hours)
    work_hours_mean = work_hours_df.mean(axis = 0)
    work_hours_stderr = work_hours_df.sem(axis = 0)

    # print a table with mean & std
    print(agg_daly_df)

    # generate figure 9 (work hours and total DALYs)
    fig = plt.figure(figsize=(15,10))

    for method in agg_daly_mean.keys():
        
        plt.scatter(work_hours_mean[method], 
                    agg_daly_mean[method],
                    label = method_to_labels[method],
                    color = method_to_colors[method],
                    s=100)
        plt.errorbar(work_hours_mean[method],
                     agg_daly_mean[method],
                     xerr = work_hours_stderr[method],
                     yerr = agg_daly_stderr[method],
                     color = method_to_colors[method])

    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    # grids
    plt.grid(True, axis='x', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xlabel('Total Work Hours Foregone', fontsize = 20)
    plt.ylabel('Total DALYs', fontsize = 20)
    # plt.xticks(plt.xticks()[0],[""]+[str(i*10) + 'K' for i in range(5,12)]+[""])
    plt.legend(prop={'size': 30})
    plt.title('Health-Economic costs of CT methods @ 60% Adoption Rate', fontsize=24)
    plt.tight_layout()
    
    # save in daly_data folder
    parent_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    save_path = 'daly_data/output/graphs/plotting_pareto_comparison_with_replacement.png'
    plt.savefig(os.path.join(parent_path,save_path))

    # generate figure 10

    # generate table 2
    
    # generate table L4 (work hours, TPL)

    # generate table L5 (metrics for age, sex breakdown)