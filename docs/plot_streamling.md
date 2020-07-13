# Streamlining plots

- [Streamlining plots](#streamlining-plots)
  - [Overview](#overview)
    - [Intent](#intent)
    - [How to use](#how-to-use)
      - [Folder structure](#folder-structure)
      - [Basic Command line](#basic-command-line)
  - [Script Structure](#script-structure)
    - [Outline](#outline)
    - [Data](#data)
  - [Adding a plot](#adding-a-plot)
    - [Create a `run()` function](#create-a-run-function)
    - [Add to `main.py`](#add-to-mainpy)
    - [Options](#options)
  - [Advanced](#advanced)
    - [Cache](#cache)
    - [`map_conf_to_models()`](#map_conf_to_models)


## Overview


### Intent

To systematically and consistently evaluate the simulator and machine learning models, we developped a simple python tool to run create many plots and aggregated outputs from various simulation runs: across different *tracing methods* and *app adoption* (for instance, but we'll see how we could compare across other parameters).

Given a *parent* folder holding all the data organised in tracing method subfolders, one simply has to run:

```
(src/covid19sim) $ python plotting/main.py path=path/to/parent plot=all
```

This will create `png` figures in `parent/` alongside the tracing method subfolders.



### How to use

#### Folder structure

The data should be organized as follows:

```
parent/
    method1/
        sim1/
            tracker.pkl
            full_configuration.yaml
        sim2/
            tracker.pkl
            full_configuration.yaml
        sim3/
            tracker.pkl
            full_configuration.yaml
        ...

    method2/
        sim1/
            tracker.pkl
            full_configuration.yaml
        sim2/
            tracker.pkl
            full_configuration.yaml
        sim3/
            tracker.pkl
            full_configuration.yaml
        ...
    ...
```

`plotting/main.py` will create:

* a `png` per plot as specified by `plot=`
* a `cache.pkl` file to hold all but only the required data from the runs in `parent/` (see [Cache Section](#Cache))

*NOTE:* This folder structure is the default output of `random_search.py` when running a validation script as `app_adoption.yaml` with a `base_dir=` argument

#### Basic Command line

`plotting/main.py`'s command line arguments are managed by Hydra. It's configuration file is `configs/plot/config.yaml`.

If you don't know about Hydra, you just need to know that arguments are passed as `key=value` where value will be automatically parsed to string or int or list (`key="[val1, val2]" # <- note the quotes`).

There are 3 main arguments:

* **`path=`** which points to the parent folder holding the data structured as explained above
* **`plot=`** which can be:
    * a string `plot=jellybeans`
    * a list `plot="[pareto_adoption, presymptomatic]"`
    * all plots `plot=all`
* (optional) **`compare=`** which is a simulation parameter which we want to compare across methods. The default value is `APP_UPTAKE`  (so if that's what you want, you don't need to specify the `compare` argument) meaning the plots will be run to compare the effect of `APP_UPTAKE` on various metrics across `tracing_method`. Any shallow simulation parameter (int, str, bool or flat lists) should theoretically be usable to compare tracing-methods though bear in mind most have not been tested.

Additionnaly:
* `exclude="[plot1, plot2]"` in conjunction with `plot=all` not to plot some specific plots but all others
* `help=true` will print `plotting/main.py`'s help
* [options](#Options)

## Script Structure

### Outline

1. Define all available plotting functions
2. Parse command-line arguments
3. Selects plots to run according to `plot=` and `exclude=`
4. Create a set `keep_pkl_keys` of keys (and therefore values) to keep from trackers' data
5. Loads the data either from the cache or by loading each `tracker*.pkl` in `parent/method/run/`
6. Exectues plotting scripts' `run(...)` functions with arguments:
    a. `data` -> the data loaded and filtered from the trackers' data
    b. `path` -> the path to the parent folder where the script will write figures
    c. `compare` -> the simulation parameter which is being compared (`APP_UPTAKE` by default)
    d. `**options` -> additional script-specific options

### Data

The data passed to `run(...)` functions is a dictionnary which holds, for each tracing method, a sub-dictionnary holding, for each compared value, a sub-sub-dictionnary holding, for each seed, a sub-sub-sub-dictionnary `{"conf": simulation_configuration_dict, "pkl": tracker_data_dict}`:

```
data = {
    method1:{
        compare_value1:{
            method1_compare_value1_seed1:{
                "conf": simulation_configuration_dict,
                "pkl": tracker_data_dict
            },
            method1_compare_value1_seed2:{
                "conf": simulation_configuration_dict,
                "pkl": tracker_data_dict
            },
            ...
        }
        compare_value2:{
            method1_compare_value2_seed1:{
                "conf": simulation_configuration_dict,
                "pkl": tracker_data_dict
            },
            method1_compare_value2_seed2:{
                "conf": simulation_configuration_dict,
                "pkl": tracker_data_dict
            },
            ...
        }
        ...
    },
    method2:{
        compare_value1:{
            method2_compare_value1_seed1:{
                "conf": simulation_configuration_dict,
                "pkl": tracker_data_dict
            },
            method2_compare_value1_seed2:{
                "conf": simulation_configuration_dict,
                "pkl": tracker_data_dict
            },
            ...
        }
        compare_value2:{
            method2_compare_value2_seed1:{
                "conf": simulation_configuration_dict,
                "pkl": tracker_data_dict
            },
            method2_compare_value2_seed2:{
                "conf": simulation_configuration_dict,
                "pkl": tracker_data_dict
            },
            ...
        }
        ...
    },
    ...
}
```

## Adding a plot

### Create a `run()` function

1. Create a script in `plotting/` like `plot_*.py`.
2. It **must** have a `run()` function whith positional arguments `data, path, compare_key` as per [outline/6](#Outline)
3. `run(...)` **can** have keyword arguments of type `str`, `int`, `float`, `bool` or `list`. Those will be called *`options`*
4. It **should** save figures in `path` in `png` format

### Add to `main.py`

1. import the script at the top of `main.py`:
    ```python
    import covid19sim.plotting.plot_something as something
    ```
2. add it to the `all_plots` dictionnary in `main.py/main(...)`:
    ```python
    all_plots = {
        ...,
        "something": something
    }
    ```
3. update `keep_pkl_keys` with *all* the keys your script requires (even if they are already listed by other scripts, as the user might not want to run both):
    ```python
    if "something" in plots:
        keep_pkl_keys.update(["your_key_1", "your_key_2", ...])
    ```

### Options

If a plotting script's `run()` function has keyword arguments they will be parsed from the commandline as `plotname_keyword=value`.

For instance `plot_presymptomatic.py/run()` has keyword arguments `mode` and `times`. One can overwrite default values for those with:

```
$ python plotting/main.py path=path/to/parent plot=all \
    presymptomatic_mode=rec \
    presymptomatic_times="[–2, -3, -4]"
```

You don't need to do anything here when adding a plot, `main.py/parse_options()` will automatically do that for you.

## Advanced

### Cache

`plotting/main.py` saves a cache in `parent/` so that re-running plots (because some code changed or with different options) doesn't require reading all the data all over again.

However, because different plots require different values from trackers, the cache may not hold the data required if you change the `plot=` argument. In this case, it will *have* to read from all the trackers' data all over again, updating the cache.

You can disable caching with those arguments:

* `dump_cache=False` (defaults to `True`) which will prevent the dump of a new cache (not creating it if it did not exist, not updating it if it did). This will **not** however delete existing cache data
* `use_cache=False` (defaults to `True`) which will prevent the use of pre-computed cache for the plots you required (for the current execution only).


### `map_conf_to_models()`

`plotting/main.py` does not rely on the folders' names to figure out what kind of tracing method or comparison value a given tracker's data comes from. It relies on **`full_configuration.yaml`** and uses `map_conf_to_models()` to map folders to standard models like `bdt1`, `bdt2`, `heuristicv1`, `transformer`, `oracl`, `unmitigated`, `bdt1` and their normalized counter-parts `*_norm`