"""
This module calculates the necessary components that establish the curve to
quantum evolution mapping.
"""

import jax
import jax.numpy as jnp
import scipy

from qurveros.settings import settings

# Required for the TTC method.
# see calculate_ttc_detuning
K_RANGE = jnp.arange(-settings.options['ANGLE_K_MAX'],
                     settings.options['ANGLE_K_MAX']+1)


def make_deriv_array_fun(curve, order):

    """
    Args:
        curve (function): The function of the curve that accepts arguments
            in the form (x, params) and returns an array with the components
            of the curve.
        order (int): The order to consider for the derivative array function.
            order == 0 corresponds to the position vector.
            order == 1 corresponds to the tangent vector.

    Returns:
        The derivative array function that returns a two-dimensional array
        of size NUM_DERIVS x number of curve components.

    Raises:
        ValueError: Depending on the value of the 'order' argument.
    """

    if order == 0:

        tangent = jax.jit(jax.jacrev(curve, argnums=0))

    elif order == 1:

        tangent = jax.jit(curve)

    else:
        raise ValueError(
            'The curve function can only refer to the curve or its tangent.')

    deriv_funs = []

    fun = tangent

    for _ in range(settings.options['NUM_DERIVS']-1):

        deriv_funs.append(jax.jit(fun))
        fun = jax.jacrev(fun, argnums=0)

    deriv_funs.append(jax.jit(fun))

    @jax.jit
    def deriv_array_fun(x, params):

        deriv_list = [vec(x, params) for vec in deriv_funs]

        return jnp.array(deriv_list)

    return deriv_array_fun


@jax.jit
def calculate_frenet_dict(deriv_array):

    """
    Calculates the frenet dictionary which contains all the necessary
    geometric quantities for the quantum evolution.

    The frame vectors are calculated at both regular and inflection points.
    At an inflection point, the binormal vector is calculated up to the sign
    induced by the signed curvature and the order of the inflection point.
    The sign is then found numerically (see calculate_singularity_indices).

    Args:
        deriv_array (jax.Array): An array of at least three derivatives of the
        curve at a given point. The dimensions of this array is
        Number of derivatives x 3.

    Returns:
        The frenet dictionary with keys:
        {
         frame (jax.Array) : The moving frame evaluated at a point.
         Each row contains the 3D vectors in the sequence:
         Tangent, Normal, Binormal.

         speed (float) : The norm of the first derivative of the curve.

         curvature (float): The curvature of the curve.

         torsion (float): The torsion of the curve.

         deriv_array (jax.Array): The supplied argument for this function.
        }

    Reference:
        Torsion At An Inflection Point of a Space Curve by Richard A. Hord
        https://www.jstor.org/stable/2978088
    """

    speed = vector_norm(deriv_array[0])

    tangent = deriv_array[0]/speed

    # Find if the given point is an inflection point and its order.
    # The sequence of vectors starts from the second derivative. Hence, the
    # vec_index is subsequently increased to correctly address the elements
    # of the deriv_array variable.

    cross_prod_norm_array = jnp.array([
         vector_norm(jnp.cross(tangent, deriv_array[deriv_order]))
         for deriv_order in range(1, settings.options['NUM_DERIVS'])
    ])

    vec_idx = jnp.flatnonzero(
                cross_prod_norm_array > settings.options['INFLECTION_NORM'],
                size=settings.options['NUM_DERIVS'] - 1)[0] + 1

    binormal = jnp.cross(tangent, deriv_array[vec_idx])
    binormal = binormal/cross_prod_norm_array[vec_idx - 1]
    normal = jnp.cross(binormal, tangent)

    curvature = cross_prod_norm_array[0]/(speed**2)

    torsion_num = jnp.sum(binormal * deriv_array[vec_idx + 1])
    torsion_denum = (vec_idx)*(speed)*(cross_prod_norm_array[vec_idx-1])

    torsion = torsion_num/torsion_denum

    frenet_dict = {}

    frenet_dict['frame'] = jnp.array([tangent, normal, binormal])
    frenet_dict['speed'] = speed
    frenet_dict['curvature'] = curvature
    frenet_dict['torsion'] = torsion
    frenet_dict['deriv_array'] = deriv_array

    return frenet_dict


@jax.jit
def calculate_rot_gauge(angle):

    """
    Calculates the rotation matrix that arranges the moving frame vectors
    for the SCQC adjoint representation.

    Args:
        angle (float): The value of the phase field.

    Returns:
        The rotation matrix with the phase gauge angle.
    """

    R_gauge = jnp.array([[0, 0, 1],
                         [-jnp.sin(angle), jnp.cos(angle), 0],
                         [-jnp.cos(angle), -jnp.sin(angle), 0]]).T

    return R_gauge


@jax.jit
def calculate_frenet_adj(frenet_dict, angle=0):

    """
    Calculates the adjoint representation of SCQC.

    Args:
        frenet_dict (dict): The frenet dictionary (see SpaceCurve class).
        angle (float): The phase gauge parameter.

    Returns:
        The adjoint representation of the quantum evolution based on SCQC.
    """

    R_UF = calculate_rot_gauge(angle) @ frenet_dict['frame'][-1]
    R_UF_zero = calculate_rot_gauge(0) @ frenet_dict['frame'][0]

    return R_UF @ (R_UF_zero.T)


@jax.jit
def calculate_adj_fidelity(adj_curve, adj_target):

    """
    Calculates the average gate fidelity using the adjoint representation.
    A unitary gate is assumed.

    Args:
        adj_curve (array): The adjoint representation described by the curve.
        adj_target (array): The target SU(2) operator
        in the adjoint representation.

    Returns:
        The average gate fidelity of the quantum operation.

    Reference:
        A simple formula for the average gate fidelity of a
        quantum dynamical operation by Michael A. Nielsen, equation (18).

    """
    d = jnp.sqrt(adj_curve.shape[0] + 1)

    adj_M = adj_curve @ adj_target.T

    # The first term of the numerator comes from the M_00 element where under
    # unitary dynamics, M_{00} = 1.

    normalization = d*(d+1)
    avg_gate_fidelity = (d + 1 + jnp.trace(adj_M))/normalization

    return avg_gate_fidelity


@jax.jit
def calculate_singularity_indices(frenet_dict):

    """
    Finds the indices of the inflection points where the frame vectors are
    discontinuous (singular points).

    Args:
        frenet_dict (dict): The frenet dictionary (See calculate_frenet_dict).

    Returns:
        An array containing the indices of singular points.

    Note:
        In order to operate in a compiled version, an expected number of
        singular points is defined in the settings file,
        and zeros are inserted depending on the chosen value.

        By construction, there can be no sign to be taken into account at t=0,
        hence such values must be discarded in the post-processing phase.

        The procedure for detecting singular points inside the interval is
        the same as detecting the additional sign that emerges at t=Tg, if
        the lowest order of the non-zero derivative at the inflection point
        is odd. The returned set of indices contain both the indices of the
        singular points and the index where the potential sign appears
        at the end of the interval.
    """

    B = frenet_dict['frame'][:, 2, :]
    B_rolled = jnp.roll(B, 1, axis=0)
    B_prod = jnp.sum(B*B_rolled, axis=1)

    singularity_idxs = jnp.flatnonzero(B_prod[1:] < -0.9,
                                       size=settings.options['NUM_SINGLS'],
                                       fill_value=-1) + 1

    return singularity_idxs


@jax.jit
def calculate_total_twist(frenet_dict):

    """
    Calculates the total twist of the curve. For this implementation, it is
    defined as the sum of the total torsion and the sign changes
    introduced by singular points.

    Args:
        frenet_dict (dict): The frenet dictionary (See calculate_frenet_dict).

    Returns:
        Float value of the total twist.
    """

    total_torsion = calculate_total_torsion(frenet_dict)
    singl_idxs = calculate_singularity_indices(frenet_dict)

    total_twist = total_torsion + jnp.pi * jnp.count_nonzero(singl_idxs)

    return total_twist


@jax.jit
def calculate_ttc_detuning(frenet_dict, angle):

    r"""
    Computes the Total Torsion Compensation (TTC) detuning in units of
    T_g \Delta. The TTC detuning with the smallest norm is returned.

    Args:
        frenet_dict (dict): The frenet dictionary (See calculate_frenet_dict).
        angle (float): The angle variable in the TTC method.

    Returns:
        The float value of the (constant) TTC detuning.
    """

    total_twist = calculate_total_twist(frenet_dict)

    detuning_cand = angle - total_twist + 2*K_RANGE*jnp.pi

    detuning_min_idx = jnp.argmin(detuning_cand**2)

    return detuning_cand[detuning_min_idx]


@jax.jit
def vector_norm(vector):

    return jnp.sqrt(jnp.sum(vector**2))


@jax.jit
def vector_int(vector, x_values):

    return jnp.trapezoid(vector, x_values, axis=0)


@jax.jit
def calculate_curve_area(frenet_dict):

    """
    Calculates the total signed area of the curve normalized by squared length.

    Args:
        frenet_dict (dict): The frenet dictionary (See calculate_frenet_dict).

    Returns:
        An array with the curve's area in 3D Cartesian coordinates.

    Note:
        The current implementation assumes that the frenet_dict contains
        the position vector as frenet_dict['curve'].
    """

    Tnum = frenet_dict['frame'][:, 0, :]
    r_prime_num = frenet_dict['speed'].reshape(-1, 1)*Tnum

    curve_area_integrand = jnp.cross(frenet_dict['curve'], r_prime_num,
                                     axis=1)

    curve_area = vector_int(curve_area_integrand, frenet_dict['x_values'])

    Tg = calculate_total_length(frenet_dict)

    curve_area = curve_area/(Tg**2)

    return curve_area


@jax.jit
def calculate_tantrix_area(frenet_dict):

    """
    Calculates the total signed area of the tantrix.

    Args:
        frenet_dict (dict): The frenet dictionary (See calculate_frenet_dict).

    Returns:
       An array with the tantrix area in 3D Cartesian coordinates.
    """

    Bnum = frenet_dict['frame'][:, 2, :]

    speed_curvature_prod = frenet_dict['speed']*frenet_dict['curvature']
    tangent_area_integrand = speed_curvature_prod.reshape(-1, 1)*Bnum
    tantrix_area = vector_int(tangent_area_integrand, frenet_dict['x_values'])

    return tantrix_area


@jax.jit
def calculate_total_length(frenet_dict):

    total_length = jnp.trapezoid(frenet_dict['speed'],
                                 frenet_dict['x_values'])

    return total_length


@jax.jit
def calculate_total_curvature(frenet_dict):

    total_curvature = jnp.trapezoid(
                       frenet_dict['speed']*frenet_dict['curvature'],
                       frenet_dict['x_values'])

    return total_curvature


@jax.jit
def calculate_total_torsion(frenet_dict):

    total_torsion = jnp.trapezoid(
                     frenet_dict['speed']*frenet_dict['torsion'],
                     frenet_dict['x_values'])

    return total_torsion


@jax.jit
def calculate_cfi_value(frenet_dict):

    """
    Calculates the Curve Filtering Index related to the average gate infidelity
    when the system is perturbed by 1/f^2 noise.

    Args:
        frenet_dict (dict): The frenet dictionary (See calculate_frenet_dict).

    Returns:
        The value of the CFI.

    Note:
        The current implementation assumes that the frenet_dict contains
        the position vector as frenet_dict['curve'].
    """

    curve_energy_integrand = jnp.sum(frenet_dict['curve']**2, axis=1)

    curve_energy = jnp.trapezoid(frenet_dict['speed']*curve_energy_integrand,
                                 frenet_dict['x_values'])

    total_length = calculate_total_length(frenet_dict)

    return curve_energy/(total_length**3)

# Helper functions
# Non-differentiable transforms


def vector_int_cumul(vector, x_values):

    """
    Provides the cumulative integral for a vector function.
    Assumes that the vectors are arranged by rows: n_samples x 3.
    """

    return scipy.integrate.cumulative_simpson(y=vector, x=x_values, axis=0,
                                              initial=0)


def calculate_curve_from_tantrix(frenet_dict):

    """
    Integrates the tangent vector numerically, to recover the position vector.
    """

    Tnum = frenet_dict['frame'][:, 0, :]
    r_prime_num = frenet_dict['speed'].reshape(-1, 1)*Tnum
    curve_num = vector_int_cumul(r_prime_num, frenet_dict['x_values'])

    return curve_num


def calculate_cumulative_length(frenet_dict):

    """
    Integrates the speed of the curve to recover the arclength (time).
    """

    return scipy.integrate.cumulative_simpson(y=frenet_dict['speed'],
                                              x=frenet_dict['x_values'],
                                              initial=0)
