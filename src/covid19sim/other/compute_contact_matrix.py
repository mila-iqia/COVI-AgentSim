from pathlib import PosixPath

import numpy as np


def generate_contact_matrix(
        contact_matrix: np.array,
        out_file: PosixPath):
    """

    This function takes an empirical contact matrix, which is usually not symmetric, symmetrizes it by averaging
    and save the results in a csv file. The empirical contact matrix is not symmetric because people in the study
    meet with people that are not part of the study.

    Args:
        contact_matrix (np.array): an empirical contact matrix
        out_file (PosixPath): the 
    """
    # Symmetrize with averaging the contact matrix
    sym_contact_matrix = (contact_matrix + contact_matrix.T) / 2.

    np.savetxt(out_file, sym_contact_matrix, delimiter=',', fmt='%1.3f')


if __name__ == '__main__':
    # This is the contact matrix associated with Great Britain from the study TODO Table. S8.4 from TODO
    cm = np.array([
        [1.92, 0.65, 0.41, 0.24, 0.46, 0.73, 0.67, 0.83, 0.24, 0.22, 0.36, 0.2, 0.2, 0.26, 0.13],
        [0.95, 6.64, 1.09, 0.73, 0.61, 0.75, 0.95, 1.39, 0.9, 0.16, 0.3, 0.22, 0.5, 0.48, 0.2],
        [0.48, 1.31, 6.85, 1.52, 0.27, 0.31, 0.48, 0.76, 1, 0.69, 0.32, 0.44, 0.27, 0.41, 0.33],
        [0.33, 0.34, 1.03, 6.71, 1.58, 0.73, 0.42, 0.56, 0.85, 1.16, 0.7, 0.3, 0.2, 0.48, 0.63],
        [0.45, 0.3, 0.22, 0.93, 2.59, 1.49, 0.75, 0.63, 0.77, 0.87, 0.88, 0.61, 0.53, 0.37, 0.33],
        [0.79, 0.66, 0.44, 0.74, 1.29, 1.83, 0.97, 0.71, 0.74, 0.85, 0.88, 0.87, 0.67, 0.74, 0.33],
        [0.97, 1.07, 0.62, 0.5, 0.88, 1.19, 1.67, 0.89, 1.02, 0.91, 0.92, 0.61, 0.76, 0.63, 0.27],
        [1.02, 0.98, 1.26, 1.09, 0.76, 0.95, 1.53, 1.5, 1.32, 1.09, 0.83, 0.69, 1.02, 0.96, 0.2],
        [0.55, 1, 1.14, 0.94, 0.73, 0.88, 0.82, 1.23, 1.35, 1.27, 0.89, 0.67, 0.94, 0.81, 0.8],
        [0.29, 0.54, 0.57, 0.77, 0.97, 0.93, 0.57, 0.8, 1.32, 1.87, 0.61, 0.8, 0.61, 0.59, 0.57],
        [0.33, 0.38, 0.4, 0.41, 0.44, 0.85, 0.6, 0.61, 0.71, 0.95, 0.74, 1.06, 0.59, 0.56, 0.57],
        [0.31, 0.21, 0.25, 0.33, 0.39, 0.53, 0.68, 0.53, 0.55, 0.51, 0.82, 1.17, 0.85, 0.85, 0.33],
        [0.26, 0.25, 0.19, 0.24, 0.19, 0.34, 0.4, 0.39, 0.47, 0.55, 0.41, 0.78, 0.65, 0.85, 0.57],
        [0.09, 0.11, 0.12, 0.2, 0.19, 0.22, 0.13, 0.3, 0.23, 0.13, 0.21, 0.28, 0.36, 0.7, 0.6],
        [0.14, 0.15, 0.21, 0.1, 0.24, 0.17, 0.15, 0.41, 0.5, 0.71, 0.53, 0.76, 0.47, 0.74, 1.47]
    ])

    cm_file = PosixPath('contact_matrix.csv')

    generate_contact_matrix(contact_matrix=cm, out_file=cm_file)
