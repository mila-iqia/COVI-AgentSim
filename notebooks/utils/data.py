import numpy as np
import pickle


def get_states(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    mapping = {'S': 0, 'E': 1, 'I': 2, 'R': 3, 'N/A': -1}

    humans_state = data['humans_state']
    has_app = data['humans_has_app']
    names = list(humans_state.keys())
    num_days = len(humans_state[names[0]])
    states = np.zeros((len(names), num_days), dtype=np.int_)
    
    for i, name in enumerate(names):
        states[i] = np.asarray([mapping[state] for state in humans_state[name]], dtype=np.int_)
    assert np.all(states >= 0)
    
    bincount = lambda x: np.bincount(x, minlength=4)
    return np.apply_along_axis(bincount, axis=0, arr=states) / len(names)

def get_all_states(filenames):
    states = get_states(filenames[0])
    all_states = np.zeros((len(filenames),) + states.shape, dtype=states.dtype)
    all_states[0] = states
    
    for i, filename in enumerate(filenames[1:]):
        all_states[i + 1] = get_states(filename)

    return all_states

def get_rec_levels(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    humans_rec_level = data['humans_rec_level']
    intervention_day = data['intervention_day']
    has_app = data['humans_has_app']
    names = [name for name in humans_rec_level if has_app[name]]
    num_days = len(humans_rec_level[names[0]])
    rec_levels = np.zeros((len(names), num_days), dtype=np.int_)

    for i, name in enumerate(names):
        rec_levels[i] = np.asarray(humans_rec_level[name], dtype=np.int_)
    rec_levels = rec_levels[:, intervention_day:]
    
    bincount = lambda x: np.bincount(x, minlength=4)
    return np.apply_along_axis(bincount, axis=0, arr=rec_levels) / len(names)

def get_all_rec_levels(filenames):
    rec_levels = get_rec_levels(filenames[0])
    all_rec_levels = np.zeros((len(filenames),) + rec_levels.shape, dtype=rec_levels.dtype)
    all_rec_levels[0] = rec_levels
    
    for i, filename in enumerate(filenames[1:]):
        all_rec_levels[i + 1] = get_rec_levels(filename)

    return all_rec_levels

def get_intervention_levels(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    humans_rec_level = data['humans_intervention_level']
    intervention_day = data['intervention_day']
    has_app = data['humans_has_app']
    names = [name for name in humans_rec_level if has_app[name]]
    num_days = len(humans_rec_level[names[0]])
    rec_levels = np.zeros((len(names), num_days), dtype=np.int_)

    for i, name in enumerate(names):
        rec_levels[i] = np.asarray(humans_rec_level[name], dtype=np.int_)
    rec_levels = rec_levels[:, intervention_day:]
    # QKFIX: Recommendation levels -1 are 0
    rec_levels[rec_levels < 0] = 0

    bincount = lambda x: np.bincount(x, minlength=4)
    return np.apply_along_axis(bincount, axis=0, arr=rec_levels) / len(names)

def get_all_intervention_levels(filenames):
    intervention_levels = get_intervention_levels(filenames[0])
    all_intervention_levels = np.zeros((len(filenames),) + intervention_levels.shape, dtype=intervention_levels.dtype)
    all_intervention_levels[0] = intervention_levels
    
    for i, filename in enumerate(filenames[1:]):
        all_intervention_levels[i + 1] = get_intervention_levels(filename)

    return all_intervention_levels
