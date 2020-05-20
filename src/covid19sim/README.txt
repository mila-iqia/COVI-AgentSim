Simulator Git Commit Hash: 4e7e85c98c8c0e6a6ec75fe36eda6dcc77a5bfae
Risk Prediction Git Commit Hash: 4e7e85c98c8c0e6a6ec75fe36eda6dcc77a5bfae
This dataset is structured as follows:
- each directory contains the data for an experiment. The experiments differ only by random seed.
- Within that directory, there is data contained in zip files.
- Each zip archive contains the data for 1,000 people over the course of the mobility simulation. If there are only 1k people, then there is 1 zip archive.
- The zip archive for 1k people for 1 experiment is structured as follows:
- for each day, there is a directory, for each person there is a subdirectory containing a number of pickle files equal to the number of times that human updated their risk.
- In that pickle file, we have data in a format specified in the docs/models.md of this repo: https://github.com/pg2455/covid_p2p_simulation/blob/develop/docs/models.md
Essentially each "sample" here is the data on specific phone on a specific day. The task is to predict the unobserved variables from the observed variables. The most important unobserved variable is that person's infectiousness over the last 14 days. This variable contains information about the risk of exposure for this person with respect to their contacts. As input to your model, you have information about their encounters, whether this person got a test result, and what their symptoms are (if any). Most of these are structured as a rolling numpy array over the last 14 days.
If you have any questions or you think you've found a bug -- please reach out to martin.clyde.weiss@gmail.com or post an issue to the relevant repository:
- https://github.com/pg2455/covid_p2p_simulation
- https://github.com/mila-iqia/covid_p2p_risk_prediction
