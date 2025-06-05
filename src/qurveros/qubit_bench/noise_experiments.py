"""
This module automates some commonly used experiments to assess
the robustness properties of a pulse.
"""

import numpy as np
import logging

from qurveros.misctools import progbar_range
from qurveros.settings import settings
from qurveros.qubit_bench import simulator, noisetools


def static_dephasing_experiment(control_dict, u_target, max_points=None):

    r"""
    Implements a static additive dephasing error experiment.

    The noise Hamiltonian enters additively as
    $H_{\text{n}} = \frac{\delta_z}{2}\sigma_z$.

    The error range is found in qurveros.settings.

    Args:
        control_dict (dict): The control_dict of the quantum evolution.
        u_target (array): The SU(2) gate.
        max_points (int): The maximum number of delta_z points
        for the simulation.

    Returns:
        A dictionary with keys:
        {
        experiment (str): The description of the experiment.
        tg_delta_z (array): An array for the values of the error in
        dimensionless units.
        infidelity (array): An array for the gate infidelity.
        }
    """

    EXPERIMENT_TITLE = 'static additive dephasing experiment'

    if max_points is None:
        max_points = settings.options['MAX_POINTS']

    Tg = control_dict['times'][-1]
    delta_z_vec = np.logspace(*settings.options['DELTA_Z_INTERVAL'],
                              max_points)/Tg

    fidelity_results = np.zeros(delta_z_vec.shape)

    delta_no_error = np.copy(control_dict['delta'])
    const_vector = np.ones(delta_no_error.shape)

    if settings._DEPHASING_PROGBAR_DEPTH == 0:
        print(EXPERIMENT_TITLE.capitalize())

    for delta_z_idx in progbar_range(max_points,
                                     title='Static additive dephasing error',
                                     depth=settings._DEPHASING_PROGBAR_DEPTH):

        delta_z = delta_z_vec[delta_z_idx]
        control_dict['delta'] = delta_no_error + delta_z*const_vector

        sim_noise_dict = simulator.simulate_control_dict(control_dict,
                                                         u_target)

        fidelity_results[delta_z_idx] = sim_noise_dict['avg_gate_fidelity']

    control_dict['delta'] = delta_no_error

    sim_dict = {}

    sim_dict['experiment'] = EXPERIMENT_TITLE

    sim_dict['infidelity'] = 1 - fidelity_results
    sim_dict['tg_delta_z'] = Tg*delta_z_vec

    return sim_dict


def simult_source_experiment(control_dict, u_target, max_points=None):

    r"""
    Implements a simultaneous error source experiment with
    static additive dephasing error and multiplicative static driving error.

    The noise Hamiltonian is defined as:
    $H_{\text{n}} = \frac{\varepsilon\, \Omega}{2}[\cos\Phi\sigma_x +
    \sin\Phi\sigma_y] +\frac{\delta_z}{2}\sigma_z$

    The error range is found in qurveros.settings.

    Args:
        control_dict (dict): The control_dict of the quantum evolution.
        u_target (array): The SU(2) gate.
        max_points (int): The maximum number of delta_z points
        for the simulation.

    Returns:
        A dictionary with keys:
        {
        experiment (str): The description of the experiment.
        epsilon (array): An array for the values
        of the driving field error.
        tg_delta_z (array): An array for the values
        of the dephasing error (dimensionless units).
        infidelity_matrix (array): An array for the gate infidelity.
        Each column provides the infidelity of a static dephasing
        experiment under a fixed driving error.
        }
    """

    EXPERIMENT_TITLE = ('static additive dephasing and multiplicative '
                        'driving field experiment')

    if max_points is None:
        max_points = settings.options['MAX_POINTS']

    pulse_error_vec = np.logspace(*settings.options['EPSILON_INTERVAL'],
                                  max_points)

    infidelity_matrix = np.zeros([max_points, max_points])
    omega_no_error = np.copy(control_dict['omega'])
    settings._DEPHASING_PROGBAR_DEPTH = 1

    print(EXPERIMENT_TITLE.capitalize())

    for pulse_idx in progbar_range(max_points,
                                   title='Multiplicative driving field error'):

        pulse_error = pulse_error_vec[pulse_idx]

        control_dict['omega'] = (1 + pulse_error)*omega_no_error

        dephasing_sim = static_dephasing_experiment(control_dict,
                                                    u_target,
                                                    max_points)

        infidelity_matrix[:, pulse_idx] = dephasing_sim['infidelity']

    control_dict['omega'] = omega_no_error

    sim_dict = {}

    sim_dict['experiment'] = EXPERIMENT_TITLE

    sim_dict['epsilon'] = pulse_error_vec
    sim_dict['tg_delta_z'] = dephasing_sim['tg_delta_z']
    sim_dict['infidelity_matrix'] = infidelity_matrix

    settings._DEPHASING_PROGBAR_DEPTH = 0

    return sim_dict


def td_dephasing_experiment(control_dict, u_target, alpha, rng,
                            num_realizations=None,
                            max_points=None):

    r"""
    Implements a time-dependent additive dephasing error experiment.

    The noise PSD in the simulation is S = T_g/((omega/omega_B))^alpha
    with omega_B = 2\pi/Tg.

    The error range is found in qurveros.settings.

    Args:
        control_dict (dict): The control_dict of the quantum evolution.
        u_target (array): The SU(2) ideal gate.
        alpha (float): The power law noise exponent 1/f^alpha.
        rng (np.random): The random number generator.
        num_realizations (int): The number of realizations to average over.
        max_points (int): The maximum number of delta_z points
        for the simulation.

    Returns:
        A dictionary with keys:
        {
        experiment (str): The description of the experiment.
        tg_delta_z (array): An array for the values of the error in
        dimensionless units.
        infidelity_matrix (array): An array for the gate infidelity.
        Every column contains the infidelity of the noise realizations at
        a fixed noise strength.
        alpha (float): The value of alpha used for the simulation.
        }
    """

    EXPERIMENT_NAME = f'Time-dependent additive dephasing (alpha = {alpha})'

    logger = logging.getLogger('TDsim')

    if max_points is None:
        max_points = settings.options['MAX_POINTS']

    if num_realizations is None:
        num_realizations = settings.options['NUM_REALIZATIONS']

    Tg = control_dict['times'][-1]
    delta_z_vec = np.logspace(*settings.options['DELTA_Z_INTERVAL'],
                              max_points)/Tg

    infidelity_matrix = np.zeros([num_realizations, max_points])

    delta_no_error = np.copy(control_dict['delta'])
    n_points = len(control_dict['omega'])

    for delta_z_idx in progbar_range(max_points, title=EXPERIMENT_NAME):

        delta_z = delta_z_vec[delta_z_idx]

        logger.info(f'Generating noise for Tg_delta_z={Tg*delta_z}.')

        if alpha == 0:
            noise_in = noisetools.get_white_noise_array(
                                    num_realizations=num_realizations,
                                    n_points=n_points,
                                    rng=rng)

        else:
            noise_in = noisetools.get_colored_noise_array(
                                    num_realizations=num_realizations,
                                    n_points=n_points,
                                    alpha=alpha,
                                    rng=rng)

        logger.info(
         f'Total number of noise realizations generated: {num_realizations}')

        # Removes the N-dependence from the white noise
        noise_in = noise_in*np.sqrt(n_points)

        # Create omega_B for the simulated spectra.
        omega_b_scale = (2*np.pi/(n_points))**(alpha)
        noise_in = noise_in*np.sqrt(omega_b_scale)

        logger.info('Noise generation done.')

        for noise_run in progbar_range(num_realizations,
                                       title='Noise realizations',
                                       depth=1):

            noise_vector = delta_z * noise_in[noise_run, :]
            control_dict['delta'] = delta_no_error + noise_vector

            try:

                sim_noise_dict = simulator.simulate_control_dict(
                    control_dict,
                    u_target)

                fidelity = sim_noise_dict['avg_gate_fidelity']

            except Exception:

                logger.info(
                    f'Error in noise realization: {noise_run}\n'
                    f' and delta_z: {delta_z}\n'
                    f' Check index [{noise_run}, {delta_z_idx}] for nan.')

                logger.exception('')
                fidelity = np.nan

            infidelity_matrix[noise_run, delta_z_idx] = 1-fidelity

    control_dict['delta'] = delta_no_error

    sim_dict = {}

    sim_dict['experiment'] = EXPERIMENT_NAME
    sim_dict['tg_delta_z'] = Tg*delta_z_vec

    sim_dict['infidelity_matrix'] = infidelity_matrix
    sim_dict['alpha'] = alpha

    return sim_dict
