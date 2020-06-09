import pytest
import numpy as np

from covid19sim.utils import lp_solve_wasserstein, lp_solution_to_transport_plan, get_rec_level_transition_matrix


def test_lp_solve_wasserstein():
    dist_0 = np.array([0.8, 0.0, 0.0, 0.2])
    dist_1 = np.array([0.4, 0.2, 0.3, 0.1])

    expected_solution = np.array([0.2, 0.2, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    solution = lp_solve_wasserstein(dist_0, dist_1)

    np.testing.assert_allclose(solution, expected_solution, atol=1e-7)


def test_lp_solution_to_transport_plan():
    dist_0 = np.array([0.8, 0.0, 0.0, 0.2])
    dist_1 = np.array([0.4, 0.2, 0.3, 0.1])
    solution = np.array([0.2, 0.2, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.1])

    expected_plan = np.array([
        [0.4, 0.2, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.1]
    ])
    optmial_plan = lp_solution_to_transport_plan(dist_0, dist_1, solution)

    np.testing.assert_allclose(optmial_plan, expected_plan, atol=1e-7)


def test_get_rec_level_transition_matrix():
    source = np.array([0.8, 0.0, 0.0, 0.2])
    target = np.array([0.4, 0.2, 0.3, 0.1])

    expected_transition_matrix = np.array([
        [0.5, 0.25, 0.25, 0.0],
        [0.0,  1.0,  0.0, 0.0],
        [0.0,  0.0,  1.0, 0.0],
        [0.0,  0.0,  0.5, 0.5]
    ])
    transition_matrix = get_rec_level_transition_matrix(source, target)

    np.testing.assert_allclose(expected_transition_matrix, transition_matrix, atol=1e-7)
