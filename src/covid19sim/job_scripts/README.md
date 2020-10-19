# How to run experiments and plot the results?

## Pareto Adoption
To get the app-adoption pareto plots, launch jobs using the following command -
```bash
python experiment.py exp_file=app_adoption base_dir=/your/folder/path/followed_by/folder_name track=light env_name=your_env
```
Note: Above is customized for SLURM jobs.

To get the plots, launch the following command,
```bash
python plotting/main.py plot=pareto_adoption path=/your/folder/path/followed_by/folder_name
```


## Normalized Mobility Frontier
Launch jobs using the following command (check `configs/experiment/normalized_mobility.yaml` for more details)

NOTE: default APP_UPTAKE = 0.8415 or adoption rate of 60%.
```bash
# heuristicv1
python experiment.py exp_file=normalized_mobility base_dir=/your/folder/path/followed_by/folder_name \
      track=light env_name=your_env \
      intervention=heuristicv1 \
      global_mobility_scaling_factor_range=cartesian_high

# bdt1
python experiment.py exp_file=normalized_mobility base_dir=/your/folder/path/followed_by/folder_name \
      track=light env_name=your_env \
      intervention=bdt1 \
      global_mobility_scaling_factor_range=cartesian_medium

# no-tracing
python experiment.py exp_file=normalized_mobility base_dir=/your/folder/path/followed_by/folder_name \
      track=light env_name=your_env \
      intervention=post-lockdown-no-tracing \
      global_mobility_scaling_factor_range=cartesian_low
```

Plot the frontier using the following command  -

```bash
python plotting/main.py plot=normalized_mobility path=/your/folder/path/followed_by/folder_name
```


## Generating domain randomized datasets to train machine learning algorithms
In order to generate domain randomized datasets, we must define the parameters we wish to vary. We provide an example in `configs/experiment/randomization.yaml` which may be used to generate a suite of datasets. In particular, running `experiment.py` with `exp_file=randomization` will generate 120 datasets with different random seeds, dropout rates, app adoption rates, etc, and write data to the location specified by the `outdir` parameter. This data is in the form of `zarr` arrays containing dictionaries. Each dictionary contains observable data from an individual in that simulation's cellphone, as well as their unobserved state (e.g. infectiousness). 
