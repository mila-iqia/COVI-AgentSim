#!/bin/bash
#path="/dev/shm/output/batch"
root_path="/dev/shm/output"
batch_path="batch"
outzip_name="1k_app_1.zip"
config_dir="configs"
config_file="naive_config.yml"
n_people=1000
simulation_days=30
n_jobs=10
num_seeds=10
sim_git_hash=$(cd covid_p2p_simulation; git rev-parse HEAD)
ctt_git_hash=$(cd ctt; git rev-parse HEAD)

for (( i=0; i<$num_seeds; i++ ))
  do
    echo "running seed ${i}"
    mkdir "${root_path}/${batch_path}/${i}"

    # Run the simulations
    python run.py sim --n_people $n_people --seed $i --outdir "$root_path/$batch_path/$i" --simulation_days $simulation_days --n_jobs $n_jobs --config $config_dir/$config_file &
  done

wait

for (( i=0; i<$num_seeds; i++ ))
  do
    # Merge the outputs
    path2="$(ls $root_path/$batch_path/$i | grep sim)"
    echo "merge"
    echo "$root_path/$batch_path/$i/$path2/$path2-output.zip"
    python models/merge_outputs.py --data_path "$root_path/$batch_path/$i/$path2/daily_outputs" --output_path "$root_path/$path2-output.zip"
  done

# Copy the experimental config into the output zip
cp $config_dir/$config_file $root_path/$config_file

# Add a dataset README
cat > README.txt <<EOL
Simulator Git Commit Hash: ${sim_git_hash}
Risk Prediction Git Commit Hash: ${ctt_git_hash}
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
EOL

# Cleanup the output
rm -fr $root_path/$batch_path

# Zip the merged outputs and config
zip $root_path/$outzip_name $root_path/*

# Copy the merged outputs to AWS
aws s3 cp  $root_path/$outzip_name s3://covid-p2p-simulation

