import numpy as np
import os
import glob
import yaml
import pickle
import warnings
import logging
from datetime import datetime

from covid19sim.utils import get_rec_level_transition_matrix


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

    source_config, source_data = get_config_and_data(args.source)
    target_config, target_data = get_config_and_data(args.target)

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

    source_dists = get_rec_levels_distributions(source_data,
                                                num_rec_levels=args.num_rec_levels)
    target_dists = get_rec_levels_distributions(target_data,
                                                     num_rec_levels=args.num_rec_levels)

    transition_matrices = np.zeros((source_dists.shape[0], args.num_rec_levels,
                                   args.num_rec_levels), dtype=np.float_)

    for index, (source_dist, target_dist) in enumerate(zip(source_dists, target_dists)):
        logging.debug('Building the transition matrix for day {0}'.format(
                      source_data['intervention_day'] + index))
        transition_matrices[index] = get_rec_level_transition_matrix(
            source_dist, target_dist)
    logging.info('Transition matrices successfully created.')

    # Update the source configuration file
    source_config['DAILY_REC_LEVEL_MAPPING'] = transition_matrices.flatten().tolist()

    # Save the new source configuration
    config_folder = os.path.join(os.path.dirname(__file__), '../hydra-configs/simulation', 'transport')
    config_folder = os.path.relpath(config_folder)
    if not os.path.exists(config_folder):
        os.mkdir(config_folder)

    output_filename = generate_name(source_config, target_config)
    output_path = os.path.join(config_folder, output_filename)

    logging.debug('Saving new configuration to `{0}`...'.format(output_path))
    with open(output_path, 'w') as f:
        yaml.dump(source_config, f, Dumper=yaml.Dumper)

    # Save the source configuration as a comment
    with open(output_path, 'a') as f:
        target_config_yaml = yaml.dump(target_config, Dumper=yaml.Dumper)
        target_config_lines = target_config_yaml.split('\n')
        f.write(f'\n# Target configuration: {args.target}\n#\n')
        for line in target_config_lines:
            f.write(f'# {line}\n')

    logging.info('New configuration file saved: `{0}`'.format(output_path))
    logging.info(f'To run the experiment with the new mobility:\n\tpython '
                 f'src/covid19sim/run.py transport={output_filename}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Creates the transition '
        'matrices to apply the mobility patterns from an `source` experiment '
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
    parser.add_argument('--num-rec-levels', type=int, default=4,
        help='Number of possible recommendation levels (default: 4)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
