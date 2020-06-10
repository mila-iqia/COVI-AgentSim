"""
Script to creates the transition matrices to apply the mobility patterns from a
`source` experiment (e.g. Binary Digital Tracing) to a `target` experiment (e.g.
Transformer). The recommendation levels of source are updated as usual, following
the tracing method of source (e.g. Binary Digital Tracing), but the interventions
on the mobility follow the recommendations (in expectation) from the target (e.g.
Transformer).

How to use:
    python src/covid19sim/other/get_transition_matrix_rec_levels.py --source path/to/binary_tracing/experiment --target path/to/transformer/experiment

This script returns a new configuration file that can be run to apply the updates
of the recommendation levels of source, but with the interventions from target.
"""
import numpy as np
import os
import glob
import yaml
import pickle
import warnings
import logging
from datetime import datetime
from collections import defaultdict


def generate_name(source_config, target_config):
    source_model = source_config['RISK_MODEL']
    seed = source_config['seed']
    target_model = target_config['RISK_MODEL']
    timenow = datetime.now().strftime('%Y%m%d-%H%M%S')

    return f'{source_model}_to_{target_model}_seed-{seed}_{timenow}.yaml'


def get_config_and_data(folder):
    config_filenames = glob.glob(os.path.join(folder, '*.yaml'))
    data_filenames = glob.glob(os.path.join(folder, '*.pkl'))

    # Configuration
    if len(config_filenames) == 0:
        raise IOError('There is no configuration file (*.yaml) in folder `{0}'
                      '`.'.format(folder))
    if len(config_filenames) > 1:
        warnings.warn('There are multiple configuration files (*.yaml) in '
                      'folder `{0}`. Taking the first configuration file `{1}'
                      '`.'.format(folder, config_filenames[0]))

    with open(config_filenames[0], 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Data
    if len(data_filenames) == 0:
        raise IOError('There is no data (*.pkl) in folder `{0}`. Make sure to '
                      'run the experiment with `tune=True`.'.format(folder))
    if len(data_filenames) > 1:
        warnings.warn('There are multiple data files (*.pkl) in folder `{0}`. '
                      'Taking the first data file `{1}`.'.format(
                      folder, data_filenames[0]))

    with open(data_filenames[0], 'rb') as f:
        data = pickle.load(f)

    if ('intervention_day' not in data) or (data['intervention_day'] < 0):
        raise ValueError('The `intervention_day` is missing. Make sure there '
                         'was an intervention in the experiment with '
                         'configuration `{0}`.'.format(config_filenames[0]))

    if 'humans_rec_level' not in data:
        raise KeyError('The `humans_rec_level` is missing in the data. The '
                       'experiment was performed before this data was added '
                       'to the tracker. Please re-run the experiment with '
                       'configuration `{0}`.'.format(config_filenames[0]))

    return config, data


def get_rec_levels_distributions(data, config, num_rec_levels=4):
    rec_levels = data['humans_rec_level']
    intervention_day = data['intervention_day']
    num_days = len(next(iter(rec_levels.values())))
    if 'humans_has_app' in data:
        has_app = data['humans_has_app']
        rec_levels_with_app = [value for (key, value) in rec_levels.items() if has_app[key]]
    else:
        if config.get('APP_UPTAKE', -1) >= 0:
            logging.warning('`humans_has_app` is not available even though '
                'APP_UPTAKE is not -1 (APP_UPTAKE={0}).'.format(config.get('APP_UPTAKE', -1)))
        rec_levels_with_app = rec_levels.values()

    rec_levels_per_day = np.zeros((num_days, len(rec_levels_with_app)), dtype=np.int_)

    for index, recommendations in enumerate(rec_levels_with_app):
        rec_levels_per_day[:, index] = np.asarray(recommendations, dtype=np.int_)

    # Remove the days before intervention (without recommendation)
    rec_levels_per_day = rec_levels_per_day[intervention_day:]
    is_valid = np.logical_and(rec_levels_per_day >= 0,
                              rec_levels_per_day < num_rec_levels)
    assert np.all(is_valid), 'Some recommendation levels are invalid'

    bincount = lambda x: np.bincount(x, minlength=num_rec_levels)
    counts = np.apply_along_axis(bincount, axis=1, arr=rec_levels_per_day)

    return counts / np.sum(counts, axis=1, keepdims=True)


def generate_single(args, source, target):
    source_config, source_data = get_config_and_data(source)
    target_config, target_data = get_config_and_data(target)

    if source_config['seed'] != target_config['seed']:
        warnings.warn('The seed of the source experiment is different from the '
                      'seed of the target experiment. source.seed={0}, '
                      'target.seed={1}.'.format(source_config['seed'],
                      target_config['seed']))

    if source_data['intervention_day'] != target_data['intervention_day']:
        raise ValueError('The intervention day of the source experiment is '
                         'different from the intervention day of the '
                         'target experiment. source.intervention_day={0}, '
                         'target.intervention_day={1}.'.format(
                         source_data['intervention_day'],
                         target_data['intervention_day']))

    # Compute the distributions of recommendation levels for
    # target tracing method (e.g. Transformer)
    target_dists = get_rec_levels_distributions(target_data, target_config,
                                                num_rec_levels=args.num_rec_levels)

    # Update the source configuration file
    source_config['DAILY_TARGET_REC_LEVEL_DIST'] = target_dists.flatten().tolist()

    # Save the new source configuration
    config_folder = os.path.join(os.path.dirname(__file__),
                                 '../hydra-configs/simulation',
                                 args.config_folder)
    config_folder = os.path.relpath(config_folder)
    if not os.path.exists(config_folder):
        os.mkdir(config_folder)

    output_filename = generate_name(source_config, target_config)
    output_config_name, _ = os.path.splitext(output_filename)
    output_path = os.path.join(config_folder, output_filename)

    logging.debug('Saving new configuration to `{0}`...'.format(output_path))
    with open(output_path, 'w') as f:
        yaml.dump(source_config, f, Dumper=yaml.Dumper)

    # Save the source configuration as a comment
    with open(output_path, 'a') as f:
        target_config_yaml = yaml.dump(target_config, Dumper=yaml.Dumper)
        target_config_lines = target_config_yaml.split('\n')
        f.write(f'\n# Target configuration: {target}\n#\n')
        for line in target_config_lines:
            f.write(f'# {line}\n')

    logging.info('New configuration file saved: `{0}`'.format(output_path))
    logging.info(f'To run the experiment with the new mobility:\n\tpython '
                 f'src/covid19sim/run.py {args.config_folder}={output_config_name}')


def get_bulk_folders(folder, keys):
    config_filenames = glob.glob(os.path.join(folder, '*/*.yaml'))
    bulk_folders = defaultdict(dict)

    for filename in config_filenames:
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        key = (config[k] for k in keys)
        seed = config['seed']
        bulk_folders[key][seed] = os.path.dirname(filename)

    return bulk_folders


def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.bulk_keys is None:
        generate_single(args, args.source, args.target)
    else:
        source_folders = get_bulk_folders(args.source, args.bulk_keys)
        target_folders = get_bulk_folders(args.target, args.bulk_keys)

        for key, folders in target_folders.items():
            for seed, target_folder in folders.items():
                try:
                    source_folder = source_folders[key][seed]
                except KeyError:
                    logging.warn('The configuration `{0}` with seed `{1}` exists '
                                 'in the bulk target folder `{2}` but not in the '
                                 'bulk source folder `{3}`. Ignoring this '
                                 'configuration.'.format(key, seed, args.target, args.source))
                    continue
                generate_single(args, source_folder, target_folder)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Creates the transition '
        'matrices to apply the mobility patterns from a `source` experiment '
        '(e.g. Binary Digital Tracing) to a `target` experiment (e.g. '
        'Transformer). The recommendation levels of source are updated as '
        'usual, following the tracing method of source (e.g. Binary Digital '
        'Tracing), but the interventions on the mobility follow the '
        'recommendations (in expectation) from the target (e.g. Transformer).')
    parser.add_argument('--source', type=str,
        help='Path to the folder of the source experiment (e.g. Binary Digital '
             'Tracing), i.e. the tracing method to use for the update of the '
             'recommendation levels.')
    parser.add_argument('--target', type=str,
        help='Path to the folder of the target experiment (e.g. '
             'Transformer), i.e. the tracing method which we apply the mobility '
             'intervention of.')
    parser.add_argument('--config-folder', type=str, default='transport',
        help='Name of the folder where the new configuration files are placed. '
             'The folder is created automatically inside `hydra-configs/simulation`.')
    parser.add_argument('--bulk-keys', type=json.loads, default=None,
        help='The keys in the configuration to loop over for bulk creation. '
             'If not provided, then only a single pair of configuration files '
             'is merged.')
    parser.add_argument('--num-rec-levels', type=int, default=4,
        help='Number of possible recommendation levels (default: 4)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
