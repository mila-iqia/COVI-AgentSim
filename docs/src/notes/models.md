# Risk Prediction, Message Passing, and Clustering

## Overview
This document describes the code in models. By running `python run.py model`, you will
initiate a message passing algorithm which iterates over the logs output by the mobility simulator. We provide more details
in the message passing section below. Those logs are described more in `events.md`, but briefly there are seven kinds: encounters, symptom_start, contamination, recovery, daily, test_results, and static_info. Our goal is to pass messages between people such that each person can accurately predict their own
risk of being infectious while maintaining the highest level of privacy.

## Privacy
* Risk values are quantized to 4-bit codes such that they can take on one of 16 levels (0-15).
* User ids are also quantized to 4-bit codes into and left-shifted each day, with a new random bit appended to the right.
* De-anonymized User Ids - the message clustering algorithm attempts to cluster messages by which "real person" sent them.
These de-anonymized user ids are integer values.

## Message Passing
When you run `python run.py model`, we load the logs contained in `output/data.zip` (or `--data_path`), then initialize the people in the sim:
* A DummyHuman is initialized with either: risk = population level risk, or if they were already diagnosed with COVID (in the initial self-report), with maximum level (15).
* When a DummyHuman self-reports a positive diagnosis, they update their risk level to 15 and broadcast that information to all their contacts of the past 14 days.

For each day in the simulation, for each person who has the app, we simulate its behavior like this:
* Every day, exactly once per day
    * the autorotating 4-bit user id is left shifted by 1 bit and a random bit is appended to the right
    * user checks for newly received messages and clusters them into "signatures" of possible contacts (i.e., that two contacts may have been from the same person)
    * risk levels from the last 14 days of messages are converted to probabilities between 0 and 1 by the inverse of the probability encoding table in `models/log_risk_mapping.npy` which was constructed by XXX.
    * user updates their risk based on their new messages with the `update_risk_encounter` function to determine this user's current risk
    * user checks whether they have any risk `update_messages`, indicating that a previous encounter was either more or less risky, then updates the old encounter message that is most likely the referenced encounter message
    * this new risk probability is quantized by the probability encoding table to obtain a 4-bit code (0 to F)
    * if the new code is different from the old code, an `update_message` is broadcast for each message received over the past 14 days
    * the user purges messages older than 14 days

## Risk Models
There are many candidate risk models. Some of them require message passing, while others can be trained on in a supervised fashion on the data which
results from the message passing algorithm. These latter models are useful because we will be getting real-world data similar
to the output of the message passing algorithm with XXX's risk model (naive contact tracing). Therefore, writing a
model which can be trained on that data and loaded into V2 is a worthwhile goal.

Risk models which require a message passing step can be developed using the code in `models/risk_models.py`.
Classes in that file implement several functions and can be added to the args of `models/run.py` to be incorporated in the message
passing algorithm. The performance of these algorithms can be viewed by runnin `plots/plot_risk.py` is provided.

The `RiskModelBase` class provides four functions, which are called at the appropriate times during the message passing
algorithm:
* `update_risk_daily` is called every day for every human, and implements some of the basic logic (e.g., if they got a positive test result, their risk=1)
as well as adding a risk for their reported symptoms.
* `update_risk_encounter` is called for each encounter message
* `update_risk_risk_update` is called for each risk update message the person currently has
* `add_message_to_cluster` is called for each encounter message, which attempts to 'de-anonymize' the sender, thereby enabling us to apply risk update messages

Certain algorithms perform better when we can attribute messages to individuals. The intuition is that each encounter with an
individual takes some of their risk, and if we update the same amount for encounters with the same individual, we will overestimate the risk.
The `add_message_to_cluster` performs a noisy de-anonymization on the messages, but has a relatively low accuracy. It is a work in progress.

## Model development for V2
If you wish to write data out from this simulator, use the `--save_training_data` arg. This writes data out to the path provided in `<data_path>/daily_outputs/<day>/<person>/daily_human.pkl`.
As a post-processing step, we must run `python models/merge_outputs.py --data_path <data_path> --output_path <output_path>` to get the final zip file.
For each person, for each day, this file contains data in the form:

```
daily_output = {"current_day": current_day,
                "observed":
                    {
                        "reported_symptoms": symptoms_to_np(
                            (todays_date - human.symptoms_start).days,
                            human.symptoms_at_time(todays_date, human.all_reported_symptoms),
                            all_possible_symptoms),
                        "candidate_encounters": candidate_encounters,
                        "test_results": human.get_test_result_array(todays_date),
                        "preexisting_conditions": conditions_to_np(human.obs_preexisting_conditions),
                        "age": encode_age(human.obs_age),
                        "sex": encode_sex(human.obs_sex)
                    },
                "unobserved":
                    {
                        "true_symptoms": symptoms_to_np((todays_date - human.symptoms_start).days,
                                                        human.symptoms_at_time(todays_date,
                                                                               human.all_symptoms),
                                                        all_possible_symptoms),
                        "is_exposed": is_exposed,
                        "exposure_day": exposure_day,
                        "is_infectious": is_infectious,
                        "infectious_day": infectious_day,
                        "is_recovered": is_recovered,
                        "recovery_day": recovery_day,
                        "exposure_encounter": exposure_encounter,
                        "infectiousness": infectiousness,
                        "true_preexisting_conditions": conditions_to_np(human.preexisting_conditions),
                        "true_age": encode_age(human.age),
                        "true_sex": encode_sex(human.sex)
                    }
                }
```
Rolling arrays should always be of the same dimension, with the content shifting over by 1 every day.

- `reported_symptoms` should be a rolling numpy array of dimension 14 (days) by the total number of symptoms observed.
It should be a strict subset of the data contained in `true_symptoms`. I.e., if you performed dropout on `true_symptoms`, you could
get `reported_symptoms`.
- `true_symptoms` is the same dimensionality as `reported_symptoms`, but contains the complete set of symptoms for the Human's illness observed within the last 14 days.
`reported_symptoms` and `true_symptoms` are computed using the information in `Log.symptom_start`. We set attributes `all_symptoms`, `reported_symptoms`, and `symptom_start_time` on the Human.
The first of these contain the entire progression of the illness, starting at `symptom_start_time`.
- `candidate_encounters` is a rolling set of messages received by the user containing updated risks, noisily predicted de-anonymized user ids, and the day of receipt. The date of receipt should never be more than 14 days in the past.
Unlike other rolling arrays, the number of elements in `candidate_encounters` will change often as new messages are appended and old messages are purged.
- `exposure_encounter` is an array of zeros with the same first dimension as `candidate_encounters`. If the Human was infected by an interaction
with another Human, there is a specific encounter (message) which was responsible for their infection. This happens in roughly 85% of infections.
The index of the exposure message, if it is in `candidate_encounters`, is set to 1.
- `test_results` is a rolling array of length 14. If the Human receives a positive test result, then the index of that day is set to 1.
- `is_exposed` is a binary value representing whether the Human has been exposed to the virus, and the virus is incubating within them.
- `exposure_day` is an integer value between 0 and -13 which represents the day when the exposure took place. 0 represents today.
- `is_infectious` is a binary value representing whether the Human is infectious. If True, then this human may infect other humans or locations. It cannot be True while `is_exposed` or `is_recovered` are True.
- `infectious_day`is an integer value between 0 and -13 which represents the day when the person became infectious.
- `is_recovered` is a binary value indicating whether the Human has had Covid-19 and is now recovered.
- `recovery_day` is an integer value between 0 and -13 which represents the day when the person recovered. If the person has not recovered, the value is None.
- `infectiousness` is an array of length 14 which contains floating values between 0 and 1 representing how infectious the person is. The infectiousness for the current day is at index 0. The infectiousness for yesterday is at index 1, and so on.
If a person becomes infectious, the value is non-zero for every day until it reaches zero again, after which it does not become non-zero.
- `[true_]preexisting_conditions` is a numpy array or encoded precondicitons. A present precondition index will have a value of 1. In order or index, preconditions are: immuno-suppressed, diabetes, heart_disease, COPD, asthma.
- `[true_]age` is an integer valued at -1 for `undefined` or age of the human, 0 included.
- `[true_]sex` is an integer valued at -1 for `undefined`, 0 for 'other', 1 for 'female' and 2 for 'male'.

That data is loaded into an absurdly bad PyTorch dataloader in `models/dataloader.py`.
