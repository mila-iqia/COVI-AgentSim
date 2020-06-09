import matplotlib.pyplot as plt
import numpy as np


def plot_age_mixing(age_mix_matrix: np.array,
                    show_figure: bool = False):
    """

    This function takes a 150x150 matrix C where each C_ij is the number of contacts between people
    of age i with people of age j. The matrix should be symmetric.


    Args:
        age_mix_matrix (np.array): a 150x150 np.array with all values being non-negatives.
        show_figure (bool): a flag to display the figure on screen
    """

    assert np.all(age_mix_matrix >= 0), f'There is at least a negative value in the matrix.'

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_xlabel('Age')
    axes.set_ylabel('Age')
    caxes = axes.matshow(age_mix_matrix, interpolation='nearest', origin='lower', cmap=plt.get_cmap('viridis'))

    axes.set_xlim(left=0, right=105)
    axes.set_ylim(bottom=0, top=105)
    axes.xaxis.set_label_position('bottom')
    axes.xaxis.tick_bottom()

    cbar = fig.colorbar(caxes)
    cbar.set_label('Ratio of contacts', rotation=270, labelpad=12)

    if show_figure:
        plt.show()
