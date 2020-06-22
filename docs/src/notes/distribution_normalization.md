# Distribution normalization

One efficient strategy to reduce the number of infectious cases is to implement mass quarantine policies. While this looks optimal from the point of view of limiting the propagation of the virus, it is also very restrictive from a mobility point of view. Better strategy therefore would be to selectively isolate people with the highest risk, while allowing the rest of the population to move freely.

To compare different tracing methods, it is therefore crucial to control for the mobility: using the same expected mobility in the population, how efficient are two different methods to reduce the number of cases? We can do this kind of analysis by _normalizing_ the daily distributions of recommendations (e.g. levels 0-3) across the whole population.

## How to normalize a single run?

Let's take an example where you want to normalize a run based on **1st-order Binary Contact Tracing (BDT1)** using the mobility of **Transformer**. You first need to have two original runs: one for BDT1 (source) and one for Transformer (target); both of those runs need to have the corresponding pickle file available (i.e. they were created `tune=True`). Then you can simply call the following script:

```
python src/covid19sim/other/get_transition_matrix_rec_levels.py --source path/to/bdt1/experiment --target path/to/transformer/experiment --config-folder normalized_bdt1
```

This script takes as input the paths of the **source (BDT1)** and the **target (Transformer)**, which contains the `*.pkl` file, as well as the `*.yaml` configuration file. This script generates a new configuration file for **BDT1, using the mobility of Transformer**, which is placed into the **config-folder** (under `src/covid19sim/configs/simulation`).

>/!\ Both runs must have the same seed (the script checks for this), and preferrably the same settings (`APP_UPTAKE`, `init_percent_sick`, etc...), although the script **does not** check for matching settings.

For example

```
$ python src/covid19sim/other/get_transition_matrix_rec_levels.py --source $SLURM_TMPDIR/bdt1/sim_v2_people-1000_days-30_init-0.01_uptake-0.8415_seed-5001_20200611-001905_771610 --target $SLURM_TMPDIR/transMI/sim_v2_people-1000_days-30_init-0.01_uptake-0.8415_seed-5001_20200611-001505_996537 --config-folder normalized_bdt1

INFO:root:New configuration file saved: `src/covid19sim/configs/simulation/normalized_bdt1/digital_to_transformer_seed-5001_20200617-085711.yaml`
INFO:root:To run the experiment with the new mobility:
    python src/covid19sim/run.py normalized_bdt1=digital_to_transformer_seed-5001_20200617-085711
```

To run the new normalized simulation, you can simply run the command returned by the script:

```
$ python src/covid19sim/run.py normalized_bdt1=digital_to_transformer_seed-5001_20200617-085711 outdir=$SLURM_TMPDIR/bdt1_normalized

...
(0s)     Day  0: 2020-02-28 | Ro: -1.00 S:990  E:10   I:0    E+I+R:10   +Test:0/0 | P3:10.00 RiskP:0.00 F:1.00 EM:0.99 TestQueue:0 | cold:0 allergies:0 | G:0 B:0 O:0 R:0
(10s)    Day  1: 2020-02-29 | Ro: 0.00 S:990  E:10   I:0    E+I+R:10   +Test:0/0 | P3:12.60 RiskP:0.00 F:1.00 EM:0.99 TestQueue:0 | cold:0 allergies:0 | G:0 B:0 O:0 R:0
(28s)    Day  2: 2020-03-01 | Ro: 0.00 S:990  E:10   I:0    E+I+R:10   +Test:0/0 | P3:15.87 RiskP:0.00 F:1.00 EM:0.99 TestQueue:0 | cold:0 allergies:0 | G:0 B:0 O:0 R:0
...
```

>/!\ You might want to specify the output folder with `outdir=/path/to/output_folder` for the output of the normalized simulation, since the original runs might have been using a different infrastructure.

>/!\ Make sure you are using the same version of the code as the one used to run the original simulations. You can check the commit hash by looking at `GIT_COMMIT_HASH` inside the configuration file `*.yaml` for both of your original runs.

## How to normalize multiple runs?

The above script works well when you have a single run you want to normalize. But sometimes you might want to normalize multiple runs at once, where the different runs might be using different settings (`APP_UPTAKE`, `init_percent_sick`, etc...) or different seeds. Let's say for example that you have many runs for **BDT1** and many runs for **Transformer**, e.g. the result of a *random search*, where you vary the `seed`, `APP_UPTAKE` and `init_percent_sick`. Then you can call the following script:

```
python src/covid19sim/other/get_transition_matrix_rec_levels.py --source path/to/bdt1 --target path/to/transformer --bulk-keys '["APP_UPTAKE","init_percent_sick"]' --config-folder normalized_bdt1
```

Where `path/to/bdt1` is the path to the folder containing multiple folders for the original experiments. For example if we have

```
$ ls -l $SLURM_TMPDIR/bdt1

sim_v2_people-1000_days-30_init-0.002_uptake-0.5618_seed-5000_20200611-001822_228992
sim_v2_people-1000_days-30_init-0.002_uptake-0.5618_seed-5001_20200611-001904_503582
sim_v2_people-1000_days-30_init-0.002_uptake-0.5618_seed-5002_20200611-001934_508323
sim_v2_people-1000_days-30_init-0.002_uptake-0.8415_seed-5000_20200611-001934_449872
...
```

Then we can call the following script:

```
$ python src/covid19sim/other/get_transition_matrix_rec_levels.py --source $SLURM_TMPDIR/bdt1 --target $SLURM_TMPDIR/transMI --bulk-keys '["APP_UPTAKE","init_percent_sick"]' --config-folder normalized_bdt1


INFO:root:New configuration file saved: `src/covid19sim/configs/simulation/normalized_bdt1/digital_to_transformer_seed-5001_20200617-091412.yaml`
INFO:root:To run the experiment with the new mobility:
    python src/covid19sim/run.py normalized_bdt1=digital_to_transformer_seed-5001_20200617-091412
INFO:root:New configuration file saved: `src/covid19sim/configs/simulation/normalized_bdt1/digital_to_transformer_seed-5000_20200617-091422.yaml`
INFO:root:To run the experiment with the new mobility:
    python src/covid19sim/run.py normalized_bdt1=digital_to_transformer_seed-5000_20200617-091422
INFO:root:New configuration file saved: `src/covid19sim/configs/simulation/normalized_bdt1/digital_to_transformer_seed-5002_20200617-091433.yaml`
INFO:root:To run the experiment with the new mobility:
    python src/covid19sim/run.py normalized_bdt1=digital_to_transformer_seed-5002_20200617-091433
...
```

>/!\ The single and double quotes in `--bulk-keys` are important and are not exchangeable. Also there can't be a space inside the `''` (specially after the `,`).

This will give you all the commands you have to run to get the new simulations on **BDT1, with the mobility of Transformer**. You can manually launch them individually, or have another script loop over all the configuration files returned by the script (e.g. inside the `src/covid19sim/configs/simulation/normalized_bdt1/` folder) and launch the corresponding jobs using `sbatch`.

>/!\ There is not automatic process to launch all the jobs at once for now. This might be possible using `random_search.py`.

>/!\ The comments regarding the outdir and using the same version of the code mentioned in the case of normalizing a single run still when launching multiple simulations.

## How to verify that the normalization works as expected?

The pickle files returned by the simulation contains, among other things, two keys: `humans_rec_level` and `humans_intervention_level`, both of which are dictionaries with the name of the `Human` as keys (e.g. `human:1`) and a list of recommendation levels (one for each day) as values.

 - The recommendation levels (values in 0-3) contained in `humans_rec_level` correspond to the recommendation levels **given** to the user.
 - The recommendation levels contained in `humans_intervention_level` correspond to the recommendation levels **followed** by the user.

To verify that the normalization works as expected, you can check if the daily distributions of recommendation levels in `humans_intervention_level` for the **normalized simulation** (e.g. BDT1 with Transformer mobility) matches the daily distributions of recommendation levels in `humans_rec_level` for the **target simulation** (e.g. Transformer).