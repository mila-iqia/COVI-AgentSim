# Documentation for [plotting/compare_real_and_sim.py](/src/covid19sim/plotting/compare_real_and_sim.py)

## Comparing data from the simulator to Quebec COVID-19 statistics

This script plots the Quebec COVID-19 data with the corresponding average simulation statistics (hospitalizations and mortalities) across 1 
or more simulations located in the same directory for the same time period.
The generated plot can be used to compare how the simulator hospitalizations and mortalities compare to the Quebec data.

## Usage
```
                                      HOW TO USE

    csv_path:
        This script expects the path to the .csv file containing Quebec data in order to plot it.

    sims_dir_path:
        This script expects the path to the directory containing the simulations in order to
        plot the average hospitalizations and mortalities per day across those simulations.
        This path can be specified directly in the script file or as a command line argument.

    Usage:

        $ python compare_real_and_sim.py
        or
        $ python compare_real_and_sim.py path/to/simulations_directory/
```