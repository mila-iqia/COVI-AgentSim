import numpy as np
import os
import glob
import yaml
import pickle
import warnings
import logging

from covid19sim.utils import get_rec_level_transition_matrix


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


def get_rec_levels_distributions(data, num_rec_levels=4):
    rec_levels = data['humans_rec_level']
    intervention_day = data['intervention_day']
    num_days = len(next(iter(rec_levels.values())))

    rec_levels_per_day = np.zeros((num_days, len(rec_levels)), dtype=np.int_)

    for index, recommendations in enumerate(rec_levels.values()):
        rec_levels_per_day[:, index] = np.asarray(recommendations, dtype=np.int_)

    # Remove the days before intervention (without recommendation)
    rec_levels_per_day = rec_levels_per_day[intervention_day:]
    is_valid = np.logical_and(rec_levels_per_day >= 0,
                              rec_levels_per_day < num_rec_levels)
    assert np.all(is_valid), 'Some recommendation levels are invalid'

    bincount = lambda x: np.bincount(x, minlength=num_rec_levels)
    counts = np.apply_along_axis(bincount, axis=1, arr=rec_levels_per_day)

    return counts / np.sum(counts, axis=1, keepdims=True)


def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    origin_config, origin_data = get_config_and_data(args.origin)
    destination_config, destination_data = get_config_and_data(args.destination)

    if origin_config['seed'] != destination_config['seed']:
        warnings.warn('The seed of the origin experiment is different from the '
                      'seed of the destination experiment. origin.seed={0}, '
                      'destination.seed={1}.'.format(origin_config['seed'],
                      destination_config['seed']))

    if origin_data['intervention_day'] != destination_data['intervention_day']:
        raise ValueError('The intervention day of the origin experiment is '
                         'different from the intervention day of the '
                         'destination experiment. origin.intervention_day={0}, '
                         'destination.intervention_day={1}.'.format(
                         origin_data['intervention_day'],
                         destination_data['intervention_day']))

    origin_dists = get_rec_levels_distributions(origin_data,
                                                num_rec_levels=args.num_rec_levels)
    destination_dists = get_rec_levels_distributions(destination_data,
                                                     num_rec_levels=args.num_rec_levels)

    transition_matrices = np.zeros((origin_dists.shape[0], args.num_rec_levels,
                                   args.num_rec_levels), dtype=np.float_)

    for index, (origin_dist, destination_dist) in enumerate(zip(origin_dists, destination_dists)):
        logging.debug('Building the transition matrix for day {0}'.format(
                      origin_data['intervention_day'] + index))
        transition_matrices[index] = get_rec_level_transition_matrix(
            origin_dist, destination_dist)
    logging.info('Transition matrices successfully created.')

    # Update the destination configuration file
    destination_config['DAILY_REC_LEVEL_MAPPING'] = transition_matrices.flatten().tolist()

    # Save the new destination configuration
    logging.debug('Saving new configuration to `{0}`...'.format(args.output_config))
    with open(args.output_config, 'w') as f:
        yaml.dump(destination_config, f, Dumper=yaml.Dumper)

    # Save the original configuration as a comment
    with open(args.output_config, 'a') as f:
        origin_config_yaml = yaml.dump(origin_config, Dumper=yaml.Dumper)
        origin_config_lines = origin_config_yaml.split('\n')
        f.write(f'\n# Original configuration: {args.origin}\n#\n')
        for line in origin_config_lines:
            f.write(f'# {line}\n')

    logging.info('New configuration file saved: `{0}`'.format(args.output_config))
    # Requires Hydra > 1.0
    # logging.info('To run the experiment with the new mobility:')
    # logging.info(f'\tpython src/covid19sim/run.py --config-path {args.output_config}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Creates the transition '
        'matrices to apply the mobility patterns from an `origin` experiment '
        '(e.g. Transformer) to a `destination` experiment (e.g. Digital Binary '
        'Tracing). The recommendation levels of destination are updated as '
        'usual, following the tracing method of destination (e.g. Digital '
        'Binary Tracing), but the interventions on the mobility follow the '
        'recommendations (in expectation) from the origin (e.g. Transformer).')
    parser.add_argument('--origin', type=str,
        help='Path to the folder of the origin experiment (e.g. Transformer).')
    parser.add_argument('--destination', type=str,
        help='Path to the folder of the destination experiment (e.g. Binary Digital Tracing).')
    parser.add_argument('--output-config', type=str, required=True,
        help='Path to save the updated configuration file (destination '
             'configuration file + additional mapping from origin to destination.')
    parser.add_argument('--num-rec-levels', type=int, default=4,
        help='Number of possible recommendation levels (default: 4)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
