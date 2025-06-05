"""
This module contains all operations relevant to the control field dictionary.
"""

import scipy
import numpy as np
import pandas as pd

from qurveros import frametools
from qurveros.settings import settings


def calculate_control_dict(frenet_dict, control_mode, n_points):

    r"""
    Calculates the control dictionary for the quantum evolution.

    The control Hamiltonian is of the form:

    H = \frac{1}{2}(\Omega cos \Phi \sigma_x +
        \frac{1}{2}(\Omega \sin Phi \sigma_y +
        \frac{1}{2}(\Delta) \sigma_z

    The quantum evolution is rescaled to the unitless time t' = t/T_g where
    T_g is the gate time found as the total length of the curve.

    Unless the tangent vector is unit-speed, a non-linear relationship
    arises between time and fields. To produce uniformly-spaced samples,
    the fields are interpolated and the corresponding linear-time values are
    returned.

    Args:
        frenet_dict (dict): The frenet dictionary (see SpaceCurve class).

        control_mode (str): The control type depending on the control
                            Hamiltonian.

            control_mode == 'XY': Control along sigma_x and sigma_y.
            In this case, the phase field is the integral of the torsion,
            with resonant control.

            control_mode == 'XZ': Control along sigma_x and sigma_z.
            In this case, the phase field is zero,
            and we drive off-resonantly with \Delta = -torsion.

            control_mode == 'TTC': Total Torsion Compensation (TTC) control.

            control_mode == 'resTTC': TTC with resonant drive. When the
            detuning is minimized, the system can be driven with nearly
            on-resonant control. The deviation from the ideal control can be
            suppressed if the closed curve condition is satisfied.

        n_points (int): The number of samples for the elements of the
                        control dictionary.

    Returns:
        control_dict (dict): The control dictionary with keys:
            {
            'times' (array): A uniformly-spaced time array in [0,1].
            'omega' (array): The envelope field (Omega)
            of the control Hamiltonian in normalized units.
            'phi' (array): The phase field of the control Hamiltonian.
            'delta' (array): The detuning of the control Hamiltonian in
            normalized units.
            'adj_curve' (array): The adjoint presentation of the quantum
            evolution at the final time, depending on the control mode used.
            'tg_envelope_max' (float): The maximum of the envelope in
            normalized units.
            'singularity_idxs': The indices were singular points occur.
            }

    Raises:
        NotImplementedError: Raised when the control mode is
        not implemented.
        ValueError: Raised when a barq angle is not supplied for TTC.
    """

    control_dict = {}

    # Detect singularities

    singularity_idxs = frametools.calculate_singularity_indices(frenet_dict)

    # Filter the 0 values
    singularity_idxs = singularity_idxs[singularity_idxs > 0]

    sign_tracker_deriv = np.zeros_like(frenet_dict['curvature'])
    sign_tracker_deriv[singularity_idxs] = \
        -2*(-1)**np.arange(len(singularity_idxs))
    sign_tracker_deriv[0] = 1

    sign_tracker = np.cumsum(sign_tracker_deriv)

    # Fields and time

    times = frametools.calculate_cumulative_length(frenet_dict)

    Tg = times[-1]
    times = times/Tg

    omega = Tg*sign_tracker*frenet_dict['curvature']

    phi = scipy.integrate.cumulative_simpson(
                frenet_dict['speed']*frenet_dict['torsion'],
                x=frenet_dict['x_values'],
                initial=0)

    if control_mode == 'XY':
        delta = np.zeros_like(omega)

    elif control_mode == 'XZ':

        phi = np.zeros_like(omega)
        delta = -Tg*frenet_dict['torsion']

    elif 'TTC' in control_mode:

        # See barqtools for details.

        if 'pgf_params' in frenet_dict['params']:
            barq_angle = frenet_dict['params']['pgf_params']['barq_angle'][0]
        elif 'barq_angle' in frenet_dict:
            barq_angle = frenet_dict['barq_angle']
        else:
            raise ValueError('TTC requires a barq angle to calculate'
                             ' the detuning value.')

        delta_value = frametools.calculate_ttc_detuning(frenet_dict,
                                                        barq_angle)
        delta = delta_value*np.ones_like(omega)

        phi = phi + delta*times

        if 'res' in control_mode:
            # Enforces zero detuning while maintaining the correct value of the
            # phase field, as if the detuning was non-zero.
            delta = np.zeros_like(omega)

    else:
        raise NotImplementedError('Control mode not implemented.')

    # Boundary value of the adjoint representation.
    # The left endpoint limit does not introduce a discontinuity.
    # The right endpoint limit contains a discontinuity depending on the order.
    # The final angle of the adjoint representation is \Phi(t=Tg) +
    # \pi times the number of singular points.
    # As defined in the paper, this is an effective phase field that captures
    # the sign changes of the envelope.

    gauge_angle = phi[-1] + np.pi*len(singularity_idxs)

    control_dict['times'] = np.array(times)
    control_dict['omega'] = np.array(omega)
    control_dict['phi'] = np.array(phi)
    control_dict['delta'] = np.array(delta)
    control_dict['adj_curve'] = frametools.calculate_frenet_adj(
        frenet_dict, gauge_angle)
    control_dict['tg_envelope_max'] = np.max(np.abs(omega))
    control_dict['singularity_idxs'] = singularity_idxs

    # Interpolates the fields for uniformly spaced samples
    control_dict = interpolate_control_dict(control_dict, n_points)

    return control_dict


def interpolate_control_dict(control_dict, n_points):

    """
    Interpolates the control fields for uniformly-spaced samples.
    The current interpolation scheme uses the CubicSpline method.

    Args:
        control_dict (dict): A dictionary that contains the entries of the
        fields as described from the settings' variable FIELD_NAMES.
        n_points (int): The number of fields' samples.

    Returns:
        The interpolated control dict.
    """

    times_linear = np.linspace(0, 1, n_points)

    for field in settings.options['FIELD_NAMES']:

        control_dict[field] = scipy.interpolate.CubicSpline(
                                     control_dict['times'],
                                     control_dict[field])(times_linear)

    control_dict['times'] = times_linear

    return control_dict


def save_control_dict(control_dict, filename):

    """
    Stores the control fields in a csv or similarly formatted file.
    """

    df = pd.DataFrame({key: control_dict[key]
                       for key in ['times', *settings.options['FIELD_NAMES']]
                       })

    df.to_csv(filename, index=False, float_format='%.12f')


def load_control_dict(filename):

    """
    Loads the control fields from a csv file or similarly formatted file and
    returns a control dictionary.
    """

    load_df = pd.read_csv(filename)
    loaded_dict = load_df.to_dict('list')

    for field in loaded_dict:
        loaded_dict[field] = np.array(loaded_dict[field])

    return loaded_dict


def make_dummy_dict(n_points):

    """
    Creates a zero-field control dictionary for testing purposes.
    """

    times = np.linspace(0, 1, n_points)
    field = np.zeros(n_points)

    ctrl_dict = {}

    for field_name in settings.options['FIELD_NAMES']:

        ctrl_dict[field_name] = field

    ctrl_dict['times'] = times
    ctrl_dict['adj'] = np.eye(3)

    return ctrl_dict
