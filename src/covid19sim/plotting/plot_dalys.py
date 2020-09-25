import pickle
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

social_discount = 0.03
age_weighting_constant = 0.04
modulation_constant = 1 
adjustment_constant = 0.1658
population_size = 3000 #hotfix

def load_tracker(path):
    with open(path, 'rb') as f:
        tracker_data = pickle.load(f)
    return tracker_data

def load_demographics(tracker_data):
    return tracker_data['humans_demographics']

def load_human_monitor(tracker_data):
    return tracker_data['human_monitor']

def load_life_expectancies(path):
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
#     life_expectancies.iloc[-1, 'Age group'] = '110 years'
    life_expectancies.loc[life_expectancies.index[-1], 'Age group'] = '110 years'
    life_expectancies = life_expectancies.set_index('Age group')

    return life_expectancies

def load_life_expectancies_preexisting(path):
    ### data does not exist yet
    ### takes as input a path pointing to life expectancies with preexisting conditions
    life_expectancies_preexisting = pd.read_csv(path)    


def get_daly_data(demographics, human_monitor_data, life_expectancies):
    
    human_names = [human_monitor_data[datetime.date(2020,2,28)][i]['name'] 
                   for i in range(population_size)]
    
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
    for human in human_names:
        daly_df.loc[human,'life_expectancy'] = float(life_expectancies[daly_df['sex'][human]][str(daly_df['age'][human])+" years"])
    
    return daly_df

def set_life_expectancies_preexisting(daly_data, life_expectancies_preexisting):
    pass
    
    
def yll(human_name,
        daly_data,
        social_discount = social_discount,
        age_weighting_constant = age_weighting_constant,
        modulation_constant = modulation_constant,
        adjustment_constant = adjustment_constant,
        discounting = False
        ):
    '''
        Computes Years of Life Lost (YLL)
        Without discounting: sum up years of life lost per human

        With discounting: 
        YLL and YLD formulas
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7345321/#B10-ijerph-17-04233 
        HRQL scores
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320437/
    '''
    age = daly_data['age'][human_name]
    life_expectancy = daly_data['life_expectancy'][human_name]
    human_data = daly_data.loc[human_name]

    if daly_data['has_died'][human_name] == False:
        return 0

    if discounting == True:
        
        # TODO: make equation its own function
        yll = (
                modulation_constant * adjustment_constant * np.exp(social_discount * age)
                /
                (social_discount + age_weighting_constant)**2
            ) \
            * \
            (
                np.exp(- (social_discount + age_weighting_constant) * (life_expectancy + age))
                *
                (
                    - (social_discount + age_weighting_constant) 
                    * (life_expectancy + age)
                    - 1
                )
                - np.exp(- (social_discount + age_weighting_constant) * age) 
                * 
                (
                    - (social_discount + age_weighting_constant) 
                    * age 
                    - 1
                )
            ) 
            # the last part is 0 since modulation_constant = 1
        return yll

    elif discounting == False:

        yll = daly_data['life_expectancy'][human_name]
        return yll


    else:
        raise NotImplementedError

def yld(method, 
        human_name,
        daly_data,
        social_discount = social_discount,
        age_weighting_constant = age_weighting_constant,
        modulation_constant = modulation_constant,
        adjustment_constant = adjustment_constant,
        discounting = False
        ):
    
    age = daly_data['age'][human_name]
    life_expectancy = daly_data['life_expectancy'][human_name]
    human_data = daly_data.loc[human_name]
    age_of_death = age + life_expectancy 
    
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7345321/#B20-ijerph-17-04233
    if method == 'infection_and_symptomatic':
        
        disability_weight = 0.133 # everyone has severe DW, which is wrong
        duration_disability = daly_data['days_symptoms_and_infection'][human_name] / 365

        if discounting == True: 
        
            yld_total = yld_formula(
                            social_discount, 
                            age_weighting_constant, 
                            modulation_constant,
                            adjustment_constant,
                            age_of_death,
                            disability_weight,
                            duration_disability
                        )
            return yld_total
        
        elif discounting == False:
            yld_total = disability_weight*duration_disability

            return yld_total
    
    elif method == 'hospitalization': # outpatients = sick and not in hospital
        
        disability_weights = {
                              'no_hospitalization':0.051, # moderate lower respiratory infection
                              'hospitalized':0.133, # severe respiratory infection
                              'critical':0.408 #severe COPD without heart failure
                             }
        duration_disability = {
                               'no_hospitalization':daly_data['days_sick_not_in_hospital'][human_name]/365,
                               'hospitalized':daly_data['days_in_hospital'][human_name]/365,
                               'critical':daly_data['days_in_ICU'][human_name]/365
                              }
        
        if discounting == True:
            yld_total = 0

            for state in disability_weights.keys():
                
                yld_total +=yld_formula(social_discount, 
                                        age_weighting_constant, 
                                        modulation_constant,
                                        adjustment_constant,
                                        age_of_death,
                                        disability_weights[state],
                                        duration_disability[state])
            
            return yld_total

        elif discounting == False:
            yld_total = 0

            for state in disability_weights.keys():
                yld_total += disability_weights[state]*duration_disability[state]
            
            return yld_total

    elif method == 'symptoms': #### TODO maybe, disability weight by symptom
        
        raise NotImplementedError
    
    

def yld_formula(social_discount, 
                age_weighting_constant, 
                modulation_constant,
                adjustment_constant,
                age_of_death,
                disability_weight,
                duration_disability):
    
    yld = (
              modulation_constant * adjustment_constant * np.exp(social_discount * age_of_death)
              /
              (social_discount + age_weighting_constant)**2 
              *
              (
                  np.exp(- (social_discount + age_weighting_constant) 
                         * (duration_disability + age_of_death)) 
                  *
                  (
                      - (social_discount + age_weighting_constant) 
                      * (duration_disability + age_of_death)
                      - 1
                  )
                  - np.exp(- (social_discount+age_weighting_constant) * age_of_death) 
                  *
                  (
                      - (social_discount+age_weighting_constant) 
                      * (age_of_death) 
                      - 1
                  )
              )
          )*disability_weight
    # the last bit is 0 since modulation_constant = 1

    return yld

def total_yll(daly_data):

    total_yll = sum([yll(i, daly_data) for i in daly_data.index])
    return total_yll

def total_yld(daly_data, method = 'hospitalization'):
    
    total_yld = sum([yld(method, i, daly_data) for i in daly_data.index])
    return total_yld

def total_dalys(daly_data, method = 'hospitalization'):
    
    total_dalys = total_yll(daly_data) + total_yld(daly_data, method)
    return total_dalys

def dalys_per_thousand(daly_data, method = 'hospitalization'):
    
    n_humans = len(daly_data.index)
    return total_dalys(daly_data, method)/(population_size/1000)

def total_metrics_sex_age(daly_data):
    
    daly_data['yll'] = ""
    daly_data['yld'] = ""
    daly_data['DALYS'] = ""
    
    for human in daly_data.index:
        daly_data.loc[human, 'yll']= yll(human, daly_data)
        daly_data.loc[human, 'yld'] = yld('hospitalization', human, daly_data)
        daly_data.loc[human, 'DALYS'] = daly_data['yll'][human] + daly_data['yld'][human]

    # iterables = [['male', 'female', 'other'],
    #              ['total YLL','total YLD','total DALYS']]
    iterables = [['male', 'female'],
                 ['total YLL','total YLD','total DALYS']]
    
    daly_dfs = {i:{} for i in iterables[0]}
    
    iterables_to_functions = {'total YLL':total_yll,
                              'total YLD':total_yld,
                              'total DALYS':total_dalys}
    
    for sex in iterables[0]:
        for metric in iterables[1]:
            if metric == 'total YLL':
                
                args = [daly_data[(daly_data.age.isin(range(i * 10,(i + 1) * 10))) 
                                  & (daly_data.sex == sex)
                                 ] 
                        for i in range(0,12)]
                
                daly_dfs[sex][metric] = [iterables_to_functions[metric](arg) for arg in args]

            else:
                
                args = ((daly_data[(daly_data.age.isin(range(i * 10,(i + 1) * 10))) 
                                   & (daly_data.sex == sex)
                                  ], 
                         'hospitalization'
                        )
                        for i in range(0,12))
                
                daly_dfs[sex][metric] = [iterables_to_functions[metric](*arg) for arg in args]

    dalys_sex = pd.DataFrame([daly_dfs[sex][metric] for sex in iterables[0] \
                              for metric in iterables[1]]
                            ).transpose()
    
    dalys_sex.index = [str(i) + ' - ' + str(i+9) for i in range(0,111,10)]
    dalys_sex.index.name = 'Age'
    dalys_sex.rename(index={'110 - 119': '110+'})
    columns = pd.MultiIndex.from_product(iterables, names = ['sex','metric'])
    dalys_sex.columns = columns
    
    return dalys_sex

def total_dalys_sex_age(daly_data):
    
    # sexes = ['male','female','other']
    sexes = ['male','female']

    total_dalys_sex_age = {}
    
    for sex in sexes:
        
        total_dalys_sex_age[sex] = pd.DataFrame([total_dalys( 
                                         daly_data[(daly_data.age.isin(range(i*10,(i+1)*10))) 
                                         & 
                                         (daly_data.sex == sex)
                                                ], 
                                        'hospitalization') 
                                                for i in range(0,12)])
    
    total_daly_df = pd.concat(total_dalys_sex_age, axis = 1)
    total_daly_df.index = [str(i) + ' - ' + str(i+9) for i in range(0,111,10)]
    total_daly_df.index.name = 'Age'
    total_daly_df.rename(index={'110 - 119': '110+'})
    # total_daly_df.columns = pd.MultiIndex.from_product([['Total DALYs'],
    #                                                    ['male','female','other']])
    total_daly_df.columns = pd.MultiIndex.from_product([['Total DALYs'],
                                                       ['male','female']])
    
    return total_daly_df

def dalys_per_thousand_sex_age(daly_data):
    
    # sexes = ['male','female','other']
    sexes = ['male','female']
    daly_1000 = {}
    
    for sex in sexes:
        
        daly_1000[sex] = pd.DataFrame([dalys_per_thousand( 
                                         daly_data[(daly_data.age.isin(range(i*10,(i+1)*10))) 
                                         & 
                                         (daly_data.sex == sex)
                                      ], 'hospitalization') for i in range(0,12)])
    
    daly_1000_df = pd.concat(daly_1000, axis = 1)
    daly_1000_df.index = [str(i) + ' - ' + str(i+9) for i in range(0,111,10)]
    daly_1000_df.index.name = 'Age'
    daly_1000_df.rename(index={'110 - 119': '110+'})
    # daly_1000_df.columns = pd.MultiIndex.from_product([['DALYs per thousand'],
    #                                                    ['male','female','other']])
    daly_1000_df.columns = pd.MultiIndex.from_product([['DALYs per thousand'],
                                                       ['male','female']])
    
    return daly_1000_df



def dalys_per_thousand_sex(daly_data):
    
    per_sex =  dalys_per_thousand_sex_age(daly_data) ### terrible implementation, just use dalys_per_thousand
    return per_sex.sum(axis=0)

def dalys_per_thousand_age(daly_data):
    
    per_sex =  dalys_per_thousand_sex_age(daly_data) ### terrible implementation, just use dalys_per_thousand
    return per_sex.sum(axis=1)

def total_dalys_preexisting_conditions(daly_data):
    
    conditions =['COPD',
                 'allergies', #no
                 'asthma',
                 'cancer', #no
                 'diabetes',
                 'heart_disease',
                 'immuno-suppressed',
                 'lung_disease',
                 'pregnant', #no
                 'smoker', #no
                 'stroke']
    total_dalys_preexisting_conditions = {}
    
    for condition in conditions:
        
        mask = daly_data.preexisting_conditions.apply(lambda x: any(item for item in [condition] if item in x))
        total_dalys_preexisting_conditions[condition] = total_dalys(daly_data[mask], 'hospitalization')
        
    return total_dalys_preexisting_conditions

def lost_work_hours_age_bins(tracker_data):
    """
    Adds up lost work hours from all sources (kid, ill, quarantine).
    Sums across time. 
    Returns an array of shape (16,)
    """
    
    lost_work_hours = ( tracker_data['work_hours']['WORK-CANCEL--KID'] + \
                        tracker_data['work_hours']['WORK-CANCEL--ILL'] + \
                        tracker_data['work_hours']['WORK-CANCEL--QUARANTINE']
                       )
    age_bins = [str(i) + ' - ' + str(i+9) for i in range(0,80,10)]
    age_bins[-1] = '70+'
    
    return dict(zip(age_bins,lost_work_hours.sum(axis=1)))

def lost_work_hours_total(tracker_data):
    """
        Adds up lost work hours from all sources (kid, ill, quarantine).
        Sums across age bins.
        Returns a scalar value.
    """
    
    lost_work_hours = ( tracker_data['work_hours']['WORK-CANCEL--KID'] + \
                        tracker_data['work_hours']['WORK-CANCEL--ILL'] + \
                        tracker_data['work_hours']['WORK-CANCEL--QUARANTINE']
                       )
    
    return lost_work_hours.sum()
    

def multiple_seeds_get_data(intervention,l_e_path):
    """
        load data into a nested dict 
        intervention[data_subset][seed] = loaded data
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

def multiple_seeds_metric(metric, data_dict):
    """
        Calculates metrics across multiple seeds.
        Returns mean, standard deviation and standard error across seeds.
        Takes a input a metric function and a data dict defined above.
        
        Different metrics (grouped by metric output data type):
            scalar:
            - total_yll
            - total_yld
            - total_dalys
            - dalys_per_thousand
            
            dataframe:
            - total_metrics_sex
            - dalys_per_thousand_sex_age
            - dalys_per_thousand_sex
            - dalys_per_thousand_age
            
            dict:
            - total_dalys_preexisting_conditions
            
    """
    metric_outputs = [metric(data) for data in data_dict.values()]
    
    if isinstance(metric_outputs[0], np.float64):
        
        mean = np.mean(metric_outputs)
        std = np.std(metric_outputs)
        std_err = std/np.sqrt(len(metric_outputs))
        
    elif isinstance(metric_outputs[0], pd.DataFrame):
        
        df_concat = pd.concat(metric_outputs)
        by_row_index = df_concat.groupby(df_concat.index)
        mean = by_row_index.mean()
        std = by_row_index.std()
        std_err = std/np.sqrt(len(metric_outputs))
        
    else:
        
        metric_df = pd.DataFrame(metric_outputs)
        mean = metric_df.mean()
        std = metric_df.std()
        std_err = std/np.sqrt(len(metric_outputs))

    return {'mean': mean, 
            'std':  std,
            'std_err': std_err
           }    

def total_dalys_sex(daly_data):
    
    per_sex_age =  total_dalys_sex_age(daly_data) ### terrible implementation, just use total_dalys
    return per_sex_age.sum(axis=0)

def total_dalys_age(daly_data):
    
    per_sex_age =  total_dalys_sex_age(daly_data) ### terrible implementation, just use total_dalys
    return per_sex_age.sum(axis=1)

def per_person_metrics_sex_age(daly_data):
    
    daly_data['yll'] = ""
    daly_data['yld'] = ""
    daly_data['DALYS'] = ""
    
    for human in daly_data.index:
        daly_data.loc[human, 'yll']= yll(human, daly_data)
        daly_data.loc[human, 'yld'] = yld('hospitalization', human, daly_data)
        daly_data.loc[human, 'DALYS'] = daly_data['yll'][human] + daly_data['yld'][human]

    # iterables = [['male', 'female', 'other'],
    #              ['total YLL','total YLD','total DALYS']]
    iterables = [['male', 'female'],
                 ['YLL per person','YLD per person','DALYs per person']]
    
    daly_dfs = {i:{} for i in iterables[0]}
    
    iterables_to_functions = {'YLL per person':total_yll,
                              'YLD per person':total_yld,
                              'DALYs per person':total_dalys}
    
    for sex in iterables[0]:
        for metric in iterables[1]:
            if metric == 'YLL per person':
                
                args = [daly_data[(daly_data.age.isin(range(i * 10,(i + 1) * 10))) 
                                  & (daly_data.sex == sex)
                                 ] 
                        for i in range(0,10)]
                        
                args.append(daly_data[(daly_data.age > 99) 
                                  & (daly_data.sex == sex)])
                
                n_agents = [len(arg.index) for arg in args]

                args_with_n_agents = zip(args, n_agents)

                daly_dfs[sex][metric] = [iterables_to_functions[metric](arg[0])/arg[1] if arg[1]>0 else 0 for arg in args_with_n_agents]

            else:
                
                args = [(daly_data[(daly_data.age.isin(range(i * 10,(i + 1) * 10))) 
                                   & (daly_data.sex == sex)
                                  ], 
                         'hospitalization',
                        len(daly_data[(daly_data.age.isin(range(i * 10,(i + 1) * 10))) 
                                   & (daly_data.sex == sex)
                                  ].index)
                        )
                        for i in range(0,10)]
                
                args.append((daly_data[(daly_data.age >99) 
                                   & (daly_data.sex == sex)
                                  ], 
                            'hospitalization',
                            len(daly_data[(daly_data.age >99) 
                                   & (daly_data.sex == sex)
                                  ].index)
                            ))
                
                daly_dfs[sex][metric] = [iterables_to_functions[metric](*arg[0:2])/arg[2] if arg[2]>0 else 0 for arg in args]

    dalys_sex = pd.DataFrame([daly_dfs[sex][metric] for sex in iterables[0] \
                              for metric in iterables[1]]
                            ).transpose()
    
    dalys_sex.index = [str(i) + ' - ' + str(i+9) for i in range(0,101,10)]
    dalys_sex.index.name = 'Age'
    dalys_sex = dalys_sex.rename(index={'100 - 109': '100+'})
    columns = pd.MultiIndex.from_product(iterables, names = ['sex','metric'])
    dalys_sex.columns = columns
    
    return dalys_sex

# TODO: add plots and tables to run function
def run(data, path, compare="app_adoption"):
    label2pkls = list()
    for method in data:
        for key in data[method]:
            label = f"{method}_{key}"
            pkls = [r["pkl"] for r in data[method][key].values()]
            label2pkls.append((label, pkls))

    # Define aggregate output variables
    agg_qaly = []

    for label, pkls in label2pkls:
        for pkl in pkls:

            # find the people who were afflicted (infection_monitor)

            # get their demographic information (pre-existing conditions, age)
            demos = pkl['humans_demographics']

            # get the symptoms they experienced and their hospitalization (or ICU) (also human_monitor)
            for date, humans in pkl['human_monitor']:
                for human in humans:
                    pass

                    # determine if they died (it's in the human_monitor)

            # put into the same QALY unit

            # add to aggregate output variables

    # print a table with mean / std
    print(agg_qaly)