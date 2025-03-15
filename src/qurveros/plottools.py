"""
This module contains plotting functionalities for the curves and the fields.
"""

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from matplotlib import colors

from qurveros.settings import settings


def plot_curve(vector, curve_param, ax=None):

    """
    Plots a curve based on the provided parametrization and inserts
    a colorbar if no Axes object is provided.

    Args:
        vector (array): An array with dimensions n_samples x 3.
        curve_param (array): The parameterization of the curve.
        ax(matplotlib.axes.Axes): An axes object to plot the curve.

    Returns:
        A Line3DCollection object.
    """

    curve_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                                    'SCQC', settings.options['CURVE_COLORS'])

    line_segs = [np.vstack([vector[i], vector[i+1]])
                 for i in range(len(vector)-1)]
    line_segs.append(line_segs[-1])

    fig = None
    if ax is None:
        fig = plt.figure(layout='constrained')
        ax = fig.add_subplot(projection='3d')
        ax.set_axis_off()

    line_collection = Line3DCollection(line_segs, cmap=curve_cmap,
                                       array=curve_param)

    ax.add_collection3d(Line3DCollection([line_segs[0]],
                                         color=curve_cmap(curve_param[0]),
                                         capstyle='round'))

    ax.add_collection3d(line_collection)

    ax.add_collection3d(Line3DCollection([line_segs[-1]],
                                         color=curve_cmap(curve_param[-1]),
                                         capstyle='round'))
    if fig is not None:
        cb = fig.colorbar(line_collection,
                          location='top',
                          ax=ax,
                          pad=0,
                          extend='both',
                          shrink=0.65)
        cb.set_label(r'Time $(t/T_g)$',
                     labelpad=matplotlib.rcParams['xtick.major.pad'])

        ax.set_aspect('equalxz')

    return line_collection


def plot_fields(control_dict, plot_mode='full', axs=None):

    """
    Plots the control fields found in the control dictionary.

    Args:
        control_dict (dict): Contains arrays of the control fields with keys
        described in the controltools.py.
        plot_format (str): The mode of the plotting:
            mode == 'full' plots all the fields with the associated labels.
            mode == 'compact' plots the fields as shown in the paper.
        axs (matplotlib.axes.Axes or List of matplotlib.axes.Axes):
        An axes object or a list of axes objects, depending on the mode,
        to plot the fields.

    Returns:
        A matplotlib Axes instance.

    Raises:
        NotImplementedError: This exception is raised when the plot mode is
        not implemented.
    """

    linewidth = matplotlib.rcParams['lines.linewidth']
    width = plt.rcParams['figure.figsize'][0]
    height = plt.rcParams['figure.figsize'][1]

    Tg = control_dict['times'][-1]

    if plot_mode == 'full':

        if axs is None:
            fig, axs = plt.subplots(3, sharex=True,
                                    gridspec_kw={'wspace': 0,
                                                 'hspace': 0},
                                    figsize=(width, 3*height))
            fig.align_ylabels()

        axs[1].plot(control_dict['times']/Tg, control_dict['phi'],
                    color=settings.options['FIELD_COLORS'][0],
                    linewidth=linewidth)

        axs[1].set_ylabel(r' $\Phi$: Phase field (rad)')

        axs[2].plot(control_dict['times']/Tg, Tg*control_dict['delta'],
                    color=settings.options['FIELD_COLORS'][0],
                    linewidth=linewidth)

        axs[2].set_ylabel(r'$T_g\Delta$: Detuning (rad)')

        envelope_ax = axs[0]
        x_label_ax = axs[2]

    elif plot_mode == 'compact':

        if axs is None:
            _, axs = plt.subplots(1)
        envelope_ax = axs
        x_label_ax = axs

    else:
        raise NotImplementedError('Plot mode not implemented')

    envelope_ax.plot(control_dict['times']/Tg,
                     Tg*control_dict['omega'],
                     color=settings.options['FIELD_COLORS'][0],
                     linewidth=linewidth,
                     zorder=2,
                     label=r'$T_g\Omega$')

    envelope_ax.plot(control_dict['times']/Tg,
                     Tg*control_dict['omega']*np.cos(control_dict['phi']),
                     color=settings.options['FIELD_COLORS'][1],
                     linewidth=linewidth,
                     zorder=1,
                     label=r'$T_g\Omega_x$')

    envelope_ax.plot(control_dict['times']/Tg,
                     Tg*control_dict['omega']*np.sin(control_dict['phi']),
                     color=settings.options['FIELD_COLORS'][2],
                     linewidth=linewidth,
                     zorder=0,
                     label=r'$T_g\Omega_y$')

    x_label_ax.set_xlabel(r'Time $(t/T_g)$')
    envelope_ax.set_ylabel(r'Pulse amplitude (rad)')
    envelope_ax.legend(loc=[0.12, 1], ncol=3)

    return axs


def plot_noise_contour(sim_dict, levels=None, ax=None):

    """
    Plots the noise contour diagrams for multiplicative static driving error
    and additive static dephasing error.

    Args:
        sim_dict (dict): A dictionary as returned from the functions in
        noise_experiments.py.
        levels (list): A list containing the levels to be used for the
        countour plot.
        ax (matplotlib.axes.Axes): An axes object to plot the contour.

    Returns:
        A QuadContourSet object.
    """

    if levels is None:
        levels = settings.options['COUNTOUR_LEVELS']

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, layout='constrained')

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                                    'Contour',
                                    settings.options['COUNTOUR_COLORS'])

    epsilon_log = np.log10(sim_dict['epsilon'])
    tg_delta_z_log = np.log10(sim_dict['tg_delta_z'])
    infidelity_log_matrix = np.emath.log10(sim_dict['infidelity_matrix']).real

    contour_set = ax.contourf(epsilon_log,
                              tg_delta_z_log,
                              infidelity_log_matrix,
                              cmap=cmap, levels=levels, extend='both')

    ax.set_xlabel(r'Pulse error $[\text{log}_{10}(\varepsilon)]$')
    ax.set_ylabel(r'Dephasing error $[\text{log}_{10}(T_g\delta_z)]$')

    if fig is not None:
        cb = fig.colorbar(contour_set, location='top', ax=ax, pad=0.0)
        cb.set_label(r'Infidelity  $[\text{log}_{10}(\mathcal{I})]$',
                     labelpad=matplotlib.rcParams['xtick.major.pad'])

        ax.locator_params(axis='y', nbins=7)
        ax.locator_params(axis='x', nbins=7)

    return contour_set
