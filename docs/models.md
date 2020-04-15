# Risk Prediction, Message Passing, and Clustering

## Overview
This document describes the code in models. By running `python models/run.py`, you will 
initiate a message passing algorithm which iterates over the logs output by the mobility simulator. We provide more details
in the message passing section below. Those logs are described more in `events.md`, but briefly there are four kinds: encounters, symptom_start, contamination,
and recovery. Our goal is to pass messages between people such that each person can accurately predict their own 
risk of being infectious while maintaining the highest level of privacy. 

## Privacy
* Risk values are quantized to 4-bit codes such that they can take on one of 16 levels (0-15). 
* User ids are also quantized to 4-bit codes into and left-shifted each day, with a new random bit appended to the right. 

## Message Passing
When you run `python models/run.py`, we load the logs contained in `output/data.pkl`, then initialize the people in the sim:
* A User is initialized with either: risk = population level risk, or if they were already diagnosed with COVID (in the initial self-report), with maximum level (15).
* When user self-reports a positive diagnosis, they update their risk level to 15 and broadcast that information to all their contacts of the past 14 days.

For each day in the simulation, for each person, we do simulate the workings of their contact tracing app.
* Every day,
    * autorotating code is updated 
    * user checks for new received messages and extracts risk levels in a table with (risk level, day) keeping only those of the last 14 days
    * user adds up the risk probabilities (risk levels are converted to probabilities between 0 and 1 by the inverse of the probability encoding table) of last 14 days
    * this is plugged into a Risk Model (Tristan's formula in app V1) to compute a new risk probability
    * this new risk probability is quantized by the probability encoding table to obtain a 4-bit code (0 to F)
    * if the new code is different from the old code, it is broadcast to all the contacts of the past 14 days (with the usual format (old risk, new risk, autorotating code)
    * the user purges messages older than 14 days
    
## Risk Models
There are many candidate risk models. Some of them require message passing, while others can be trained on the data which 
results from the message passing algorithm. These latter models are useful because we will be getting real-world data similar
to the output of the message passing algorithm with Tristan's risk model (naive contact tracing). Therefore, writing a 
model which can be trained on that data and loaded into V2 is a worthwhile goal.

Model's which require a message passing step can be developed using the code in `models/risk_models.py`. 
Classes in that file implement several functions and can be added to the args of `models/run.py` to be incorporated in the message
passing algorithm. The performance of these algorithms is then reported at the end of execution, with daily outputs provided if the 
arg `--plot_daily` is provided.

The `RiskModelBase` class provides three functions, which are called at the appropriate times during the message passing
algorithm:
* `update_risk_daily` is called every day for every human, and implements some of the basic logic (e.g., if they got a positive test result, their risk=1) 
as well as adding a risk for their reported symptoms.
* `update_risk_encounter` is called for each encounter message (optional)
* `update_risk_risk_update` is called for each risk update message the person currently has (optional)

There is one additional function which may be required for some algorithms: `add_message_to_cluster`. 
Certain algorithms perform better when we can attribute messages to individuals. The intuition is that each encounter with an 
individual takes some of their risk, and if we update the same amount for encounters with the same individual, we will overestimate the risk.
The `add_message_to_cluster` performs a noisy de-anonymization on the messages, but currently only has a ~55% accuracy rate. There are many ways in which it may be improved.

## Model development for V2
If you wish to write data out from this simulator, use the `--save_training_data` arg. This writes `output/output.pkl`.
For each person, for each day, this file contains data in the form: 

```
{"current_day": current_day,
"observed":
    {
        "reported_symptoms": np.array((rolling_num_days, num_possible_symptoms)), #rolling_num_days is 14
        "messages": np.array((num_messages, msg_dim)), #msg_dim = 4bit uid + 4bit risk
        "update_messages": np.array((num_messages, update_msg_dim), #update_msg_dim = 4bit uid + 4bit risk + 4bit oldrisk
        "test_results": np.array(rolling_num_days), # binary test_results on one of the last 14 days
     },
"unobserved":
    {
        "true_symptoms": np.array((rolling_num_days, num_possible_symptoms)), # this is the same as reported symptoms, but not lossy
        "state": np.array((rolling_num_days, num_possible_states)), # is_susceptible, is_exposed, is_infectious, is_recovered, is_dead
    }
}
```

That data is loaded into a PyTorch dataloader in `models/dataloader.py`.