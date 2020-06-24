from scipy.optimize import linprog
import numpy as np


def get_rec_level_transition_matrix(source, target):
    """Compute the transition matrix to go from one distribution of
    recommendation levels (e.g. given by Digital Binary Tracing) to another
    distribution of recommendation levels (e.g. given by a Transformer).

    Args:
        source (np.ndarray): The source distribution. This distribution does
            not need to be normalized (i.e. array of counts).
        target (np.ndarray): The target distribution. This distribution does
        not need to be normalized (i.e. array of counts).

    Returns:
        np.ndarray: Transition matrix containing, where the value {i, j}
            corresponds to
            P(target recommendation level = j | source recommendation level = i)
    """
    # Normalize the distributions (in case we got counts)
    if np.isclose(source.sum(), 0) or np.isclose(target.sum(), 0):
        raise ValueError('The function `get_rec_level_transition_matrix` expects '
                         'two distributions, but got an array full of zeros. '
                         'source={0}, target={1}.'.format(source, target))
    dist_0 = source / source.sum()
    dist_1 = target / target.sum()

    solution = lp_solve_wasserstein(dist_0, dist_1)
    transport_plan = lp_solution_to_transport_plan(dist_0, dist_1, solution)

    # Leave the bins with no mass untouched (this ensures the transition matrix
    # is well defined everywhere, i.e. rows all sum to 1)
    diagonal = np.diag(transport_plan)
    np.fill_diagonal(transport_plan, np.where(diagonal == 0., 1., diagonal))

    return transport_plan / np.sum(transport_plan, axis=1, keepdims=True)


def lp_solve_wasserstein(dist_0, dist_1):
    """Solve the optimal transport problem between two distributions as a
    Linear Program by minimizing the (squared) Wasserstein distance between the
    two distributions.

    The problem to be solved is [1, Equation 2.5]

        min_T   sum_{ij} T_{ij} * |i - j|^{2}
        st.     sum_{i} T_{ij} = dist1_{j}
                sum_{j} T_{ij} = dist0_{i}
                T >= 0

    Here we use the following heuristic first: we keep as much mass as possible
    fixed (i.e. priority is to stay in the same recommendation level). Using
    this heuristic, we then have a LP formulation with 12 variables (if dist0
    and dist1 take each 4 values, e.g. 4 recommendation levels) and 7 constraints.
    The variable T is encoded as a vector with the upper triangular values of the
    transport plan in the first half and the lower triangular values in the second half.

        T = [T_{01}, T_{02}, T_{03}, T_{12}, T_{13}, T_{23},
             T_{10}, T_{20}, T_{21}, T_{30}, T_{31}, T_{32}]

    The (equality) constraints are

        T_{01} + T_{02} + T_{03} = dist0_{0} - min(dist0_{0}, dist1_{0})
        T_{10} + T_{12} + T_{13} = dist0_{1} - min(dist0_{1}, dist1_{1})
        T_{20} + T_{21} + T_{23} = dist0_{2} - min(dist0_{2}, dist1_{2})
        T_{10} + T_{20} + T_{30} = dist1_{0} - min(dist0_{0}, dist1_{0})
        T_{01} + T_{21} + T_{31} = dist1_{1} - min(dist0_{1}, dist1_{1})
        T_{02} + T_{12} + T_{32} = dist1_{2} - min(dist0_{2}, dist1_{2})
                 sum_{ij} T_{ij} = 1 - sum_{i} min(dist0_{i}, dist1_{i})

    Note:
        [1] Justin Solomon, Optimal Transport on Discrete Domains
            (https://arxiv.org/abs/1801.07745)

    Args:
        dist_0 (np.ndarray): The distribution to move from. This array should
            have non-negative values, be normalized (i.e. sum to 1), and have
            the same shape as dist_1.
        dist_1 (np.ndarray): The distribution to move to. This array should
            have non-negative values, be normalized (i.e. sum to 1), and have
            the same shape as dist_0.

    Returns:
        np.ndarray: Array containing the solution of the Linear Program.
            This array contains the off-diagonal values of the optimal
            transport plan.
    """
    min_dist = np.minimum(dist_0, dist_1)

    # LP formulation
    c = np.array([1, 2, 3, 1, 2, 1, 1, 2, 1, 3, 2, 1], dtype=np.float_)
    A_eq = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float_)
    b_eq = np.hstack([dist_0[:-1] - min_dist[:-1],
                      dist_1[:-1] - min_dist[:-1],
                      1 - min_dist.sum()])

    # Solve LP
    result = linprog(c ** 2, A_eq=A_eq, b_eq=b_eq)

    return result.x

def lp_solution_to_transport_plan(dist_0, dist_1, solution):
    """Converts the solution of the LP given by lp_solve_wasserstein
    (off-diagonal values) into the full transport plan.

    Args:
        dist_0 (np.ndarray): The distribution to move from. This array should
            have non-negative values, be normalized (i.e. sum to 1), and have
            the same shape as dist_1.
        dist_1 (np.ndarray): The distribution to move to. This array should
            have non-negative values, be normalized (i.e. sum to 1), and have
            the same shape as dist_0.
        solution (np.ndarray): Array containing the solution of the Linear
            Program. This array contains the off-diagonal values of the optimal
            transport plan (upper triangular values in the first half, lower
            triangular values in the second half).

    Returns:
        np.ndarray: An array containing the full optimal transport plan. The
            transition matrix is a normalized version of the optimal transport plan.
    """
    # The diagonal of the transition matrix contains the minimum values
    # of both distributions, i.e. as much mass as possible is kept fixed.
    min_dist = np.minimum(dist_0, dist_1)
    transition = np.diag(min_dist)

    # Zero out small values of the solution
    solution[np.isclose(solution, 0.)] = 0.
    if not np.isclose(min_dist.sum(), 1):
        solution /= solution.sum() / (1 - min_dist.sum())

    # The solution contains the upper triangular values in the first half
    # of the solution, and the lower triangular values in the second half.
    upper, lower = solution[:solution.size // 2], solution[solution.size // 2:]
    transition[np.triu_indices_from(transition, k=1)] = upper
    transition[np.tril_indices_from(transition, k=-1)] = lower

    return transition
