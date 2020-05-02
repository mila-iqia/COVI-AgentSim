# Graphics class definition
# Bogdan Hlevca, Markham, Ontario, Canada
# April 2020
#
# Graphics class that handles visualizations for track.py

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import MONDAY
from datetime import date, datetime, timedelta
import sys
# config is needed by dill.load
sys.path.append('../')
import config

# --------------------------------------------------------------------------------------------------------------
class PlotTracker(object):
    """
    Class to visualize data for covid_p2p_simulation runs:
    """

    def __init__(self, data, fontsize=14, dateformat=None):
        """
        PEP8 initialize here
        """

        self._data = data
        self._fontsize = fontsize
        self._params = {'xtick.labelsize': fontsize,
                        'ytick.labelsize': fontsize - 2,
                        'axes.titlesize': fontsize + 2,
                        'axes.labelsize': fontsize + 2,
                        'text.usetex': False,
                        'lines.linewidth': 1,
                        'lines.markersize': 3,
                        }
        self.labels = {
            'all_encounters': {'title': 'All Encounters', 'xlabel': 'xlabel', 'ylabel': 'ylabel'},
            'location_all_encounters': {'title': 'Location All Encounters', 'xlabel': 'xlabel', 'ylabel': 'ylabel'},
            'human_infection': {'title': 'Human Infection', 'xlabel': 'xlabel', 'ylabel': 'ylabel'},
            'env_infection': {'title': 'Environmental Infection', 'xlabel': 'xlabel', 'ylabel': 'ylabel'},
            'location_env_infection': {'title': 'Location Environmental Infection', 'xlabel': 'xlabel',
                                       'ylabel': 'ylabel'},
            'location_human_infection': {'title': 'Location Human Infection', 'xlabel': 'xlabel', 'ylabel': 'ylabel'},
            'duration': {'title': 'Contact Duration [min]', 'xlabel': 'Age Human 1', 'ylabel': 'Age Human 2'},
            'histogram_duration': {'title': 'Human Infection', 'xlabel': 'Intervals - 15 [min]',
                                   'ylabel': 'Number of encounters'},
            'location_duration': {'title': 'Contact Duration [min]', 'xlabel': 'Location', 'ylabel': 'Duration'},
            'n_contacts': {'title': 'Number of contacts Duration', 'xlabel': 'Age Human 1', 'ylabel': 'Age Human 2'},
        }
        self.years = matplotlib.dates.YearLocator()  # every year
        self.months = matplotlib.dates.MonthLocator()  # every month
        self.years_fmt = matplotlib.dates.DateFormatter('%Y')
        # every monday
        self.mondays = matplotlib.dates.WeekdayLocator(MONDAY)
        if dateformat == None:
            self.formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
        else:
            self.formatter = matplotlib.dates.DateFormatter(dateformat)

    @staticmethod
    def load_data(fname):
        """
        PARAMETERS
            file_name (string) :
                the file name containing data for display
        """

        import dill, os, sys

        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                try:
                    return dill.load(f)
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))
                    exit(0)
                except ModuleNotFoundError as e:  # handle other exceptions such as attribute errors
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))
                    exit(1)
                except:# handle other exceptions such as attribute errors
                    print("Dill load, unexpected error:", sys.exc_info()[0])
                    exit(2)
        else:
            print('Graphics: Error file: {} does not exist'.format(self._fname))
            sys.exit(3)
        return None



    def get_data(self):
        return self._data


    def _plot_heatmap(self, xarr, yarr, vals,
                     xyscale="linear", xlabel="", ylabel="", title="",
                     x_type=None, invert_y=False,
                     numticks_x=None, numticks_y=None,
                     crange=None, extend=None, cscale="linear",
                     cnum=7, cmap='jet', cdecimals=2):
        """
        Plots the the heatmap on x,y coordinates for the vals matrix  of
        function.

        PARAMETERS
            xarr (array like) :
                the coordinates on the x-axis
            yarr (array like) :
                the coordinates on the x-axis
            vals (matrix like of size x*y - can be pandas.DataFrame)
                the values to be represented by color
            xyscale (string) :
                scale for axes , one of 'linear', 'log'
            xlabel (string):
                label on the x axis
            ylabel (string):
                label on the y axis
            title (string):
                title of the graph
            x_type (string) :
                None, 'date'
            invert_y (boolean):
                True if Y axis needs to be inverted
            numticks_x: (int):
                Will divide the X axis in numticks_x intervals
            numticks_y: (int):
                Will divide the X axis in numticks_x intervals
            crange (touple min, max values)
                Define the levels that are going to be contoured and ticks on the cbar
            extend {'neither', 'both', 'min', 'max'}, optional, default: 'neither'
                Determines the contourf-coloring of values that are outside the levels range.
                If 'neither', values outside the levels range are not colored. If 'min', 'max' or 'both', color
                    the values below, above or below and above the levels range.
                Values below min(levels) and above max(levels) are mapped to the under/over values of the Colormap.
                Note, that most colormaps do not have dedicated colors for these by default, so that the over and
                    under values are the edge values of the colormap. You may want to set these values explicitly
                    using Colormap.set_under and Colormap.set_over.
            cscale (string) :
                scale for values, one of 'linear', 'loglog', 'semilog'
            cnum (int):
                number of different color levels. if cnum>20 ticks on the colorbar will be reduced to 10
            cmap (string):
                colormap for the plot
            cdecimals (int):
                the number of decimals to be printed on th cbar
        RETURNS
            A list with the figure and axis objects for the plot.
        """

        def downsample(array, npts):
            from scipy.interpolate import interp1d
            interpolated = interp1d(np.arange(len(array)), array, axis=0, fill_value='extrapolate')
            downsampled = interpolated(np.linspace(0, len(array), npts))
            return downsampled

        plt.rcParams.update(self._params)  # Plot parameters
        figprops = dict(figsize = (11, 8), dpi = 96)
        fig = plt.figure(**figprops)
        ax = fig.add_subplot(1, 1, 1)

        if xyscale == 'loglog':
            X = np.log10(xarr)
            Y = np.log10(yarr)
        elif xyscale == 'semilog':
            X = xarr
            Y = np.log10(yarr)
        else:
            X = xarr
            Y = yarr

        if x_type == 'date':
            x_label = 'Time (days)'
            ax.xaxis.set_major_formatter(self.formatter)
            ax.xaxis.set_minor_locator(self.mondays)
            ax.xaxis.grid(True, 'minor')
            fig.autofmt_xdate()
            # endif
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # determine the ticks on the cbar
        if crange is not None:
            levels = np.linspace(crange[0], crange[1], num=cnum)
            labels = levels.astype(np.str)
        else:
            levels = np.linspace(np.nanmin(vals), np.nanmax(vals), num=cnum)
            labels = levels.astype(np.str)

        if cscale == 'log':
            Levels = np.log10(levels)
        else:
            Levels = levels

        cmin, cmax = np.nanmin(vals), np.nanmax(vals)
        rmin, rmax = min(Levels), max(Levels)
        if extend is not None:
            extend = extend
        elif (cmin < rmin) & (cmax > rmax):
            extend = 'both'
        elif (cmin < rmin) & (cmax <= rmax):
            extend = 'min'
        elif (cmin >= rmin) & (cmax > rmax):
            extend = 'max'
        elif (cmin >= rmin) & (cmax <= rmax):
            extend = 'neither'

        cf = ax.contourf(X, Y, vals, Levels, cmap=plt.get_cmap(cmap), extend = extend)
        ax.contour(X, Y, vals, Levels, colors = 'k', linewidths = 0.7)
        if numticks_y is not None:
            Yticks = np.arange(np.ceil(Y.min()), np.ceil(Y.max()),
                               step= (np.ceil(Y.max())- np.ceil(Y.min())) / numticks_y )
            ax.set_yticks(Yticks)
        if numticks_x is not None:
            Xticks = np.arange(np.ceil(X.min()), np.ceil(X.max()),
                               step=(np.ceil(X.max()) - np.ceil(X.min())) / numticks_x)
            ax.set_xticks(Xticks)

        ax.set_xlim([X.min(), X.max()])
        ax.set_ylim([Y.min(), Y.max()])
        if invert_y:
            ax.invert_yaxis()
        if len(Levels) > 20:
            Levels = downsample(Levels, 10)
        cticks = np.around(Levels, cdecimals)
        clabels = cticks.astype(np.str)
        cbar = fig.colorbar(cf, ticks = cticks, extend = extend)
        cbar.ax.set_yticklabels(clabels)

        plt.show()
        return fig, ax


    def _plot_histogram(self, x, height, xlabel='', ylabel='', title='', color='red'):
        """
        Args:
            x (array like):
                bar categories
            height (array like):
                height of each category
            xlabel (string):
                label of the x axis
            ylabel (string):
                label of the y axis
            title (string):
                title of plot
            color (string):
                color of the histogram bars: 'red', 'blue', 'yellow', etc.
        Returns:

        """
        plt.figure(figsize=(13, 7))
        plt.title(title)
        plt.bar(x, height, color=color, edgecolor='k')
        plt.xticks(rotation=45, fontsize=14)
        plt.xlabel(xlabel, fontsize=self._fontsize)
        plt.ylabel(ylabel, fontsize=self._fontsize)
        plt.subplots_adjust(bottom=0.22)
        plt.grid()
        plt.show()


    def plot_heatmap(self, metrics, crange=None):
        """
        Prepares data from the dictionary to be plotted
        Args:
            metrics (string):
                one of 'all_encounters', 'location_all_encounters', 'human_infection', 'env_infection'
                        'location_env_infection', 'location_human_infection', 'duration', 'histogram_duration',
                        'location_duration', 'n_contacts':
            crange (list of 2):
                [min max] to be represented on the cbar of the heatmap. values > max will have the same colour
        Returns:
        """
        contacts = self.get_data()['contacts']
        # extract the appropriate data from the dictionary
        if metrics == 'duration' or metrics == 'n_contacts':
            to_plot = contacts[metrics]['avg'][1]
            if crange:
                to_plot = to_plot[crange[0]:crange[1], crange[0]:crange[1]]

        x = y = np.linspace(start=0, stop=len(to_plot) - 1, num=len(to_plot))
        self._plot_heatmap(x, y, to_plot,
                            xlabel=self.labels[metrics]['xlabel'], ylabel=self.labels[metrics]['ylabel'],
                            title=self.labels[metrics]['title'],
                            crange=crange, cnum=100)


    def plot_histogram(self, metrics, color='red'):
        """
        Prepares data from the dictionary to be plotted
        Args:
            metrics (string):
                one of 'all_encounters', 'location_all_encounters', 'human_infection', 'env_infection'
                        'location_env_infection', 'location_human_infection', 'duration', 'histogram_duration',
                        'location_duration', 'n_contacts':
            color (string):
                color of the histogram bars: 'red', 'blue', 'yellow', etc.
        Returns:
        """
        contacts = self.get_data()['contacts']
        # extract the appropriate data from the dictionary
        # needs development as more types are added
        if metrics == 'duration':
            height = contacts[metrics]
            x = np.arange(len(height))
        else:
            height = contacts[metrics]
            x = np.arange(len(height))

        self._plot_histogram(x, height,
                             xlabel=self.labels[metrics]['xlabel'],
                             ylabel=self.labels[metrics]['ylabel'],
                             title=self.labels[metrics]['title'],
                             color=color
                            )

    @staticmethod
    def plot_heatmap_fromdata(metrics, data, crange,  fontsize=14, dateformat=None):
        """
        Convenince method to avoid knowing too much about the PlotTracker class
        Args:
            metrics (string):
                one of 'all_encounters', 'location_all_encounters', 'human_infection', 'env_infection'
                        'location_env_infection', 'location_human_infection', 'duration', 'histogram_duration',
                        'location_duration', 'n_contacts'
            data (dictionary):
                containing 'contacts'
            crange (list of 2):
                [min max] to be represented on the cbar of the heatmap. values > max will have the same colour
            fontsize (int):
                font size used in the labeds and ticks and title
            dateformat (string):
                default '%Y-%m-%d'
        Returns:
        """
        gr = PlotTracker(data, fontsize, fontsize, dateformat)
        gr.plot_heatmap(metrics, crange)


    @staticmethod
    def plot_heatmap_fromfile(metrics, fname, crange=None, fontsize=14, dateformat=None):
        """
        Convenince method to avoid knowing too much about the PlotTracker class
        Args:
            metrics (string):
                one of 'all_encounters', 'location_all_encounters', 'human_infection', 'env_infection'
                        'location_env_infection', 'location_human_infection', 'duration', 'histogram_duration',
                        'location_duration', 'n_contacts':
            fname (string):
                name of the file that holds pickled data
            crange (list of 2):
                [min max] to be represented on the cbar of the heatmap. values > max will have the same colour
            fontsize (int):
                font size used in the labeds and ticks and title

            dateformat (string):
                default '%Y-%m-%d'
        Returns:
        """
        gr = PlotTracker(PlotTracker.load_data(fname), fontsize, dateformat)
        gr.plot_heatmap(metrics, crange)

    @staticmethod
    def plot_histogram_fromfile(metrics, fname, color='red', fontsize=14, dateformat=None):
        """
        Convenince method to avoid knowing too much about the PlotTracker class
        Args:
            metrics (string):
                one of 'all_encounters', 'location_all_encounters', 'human_infection', 'env_infection'
                        'location_env_infection', 'location_human_infection', 'duration', 'histogram_duration',
                        'location_duration', 'n_contacts':
            fname (string):
                name of the file that holds pickled data
            color (string):
                color of the histogram bars: 'red', 'blue', 'yellow', etc.
            fontsize (int):
                font size used in the labeds and ticks and title

            dateformat (string):
                default '%Y-%m-%d'
        Returns:
        """
        gr = PlotTracker(PlotTracker.load_data(fname), fontsize, dateformat)
        gr.plot_histogram(metrics, color)


    @staticmethod
    def test_graphics():
        """
        test the varius Graphics methods
        """
        origin = 'lower'
        delta = 0.025

        x = y = np.arange(-3.0, 3.01, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
        Z = (Z1 - Z2) * 2

        graph = PlotTracker("test")
        graph._plot_heatmap(X, Y, Z,
                           xlabel="X axis", ylabel="Y axis", title="T I T L E",
                           cnum=100, cdecimals=2)


# unit tests for the current class
if __name__ == '__main__':
    PlotTracker.test_graphics()
    PlotTracker.plot_heatmap_fromfile("duration", "tracker_data_n_1000_seed_0_test.pkl", [0,41])
    PlotTracker.plot_heatmap_fromfile("n_contacts", "tracker_data_n_1000_seed_0_test.pkl")
    PlotTracker.plot_histogram_fromfile("histogram_duration", "tracker_data_n_1000_seed_0_test.pkl", color="red")
